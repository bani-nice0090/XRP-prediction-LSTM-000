# src/data_engine.py

import asyncio
import time
from typing import AsyncGenerator, Dict, List, Tuple

import aiohttp
import numpy as np
import polars as pl
import redis
import websockets
from rich.console import Console
from rich.table import Table
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

console = Console()


class MarketData:
    """Represents a single piece of market data."""

    def __init__(
        self,
        timestamp: float,
        symbol: str,
        price: float,
        volume: float,
        bid: float,
        ask: float,
        spread: float,
        order_imbalance: float,
        quality_score: float,
    ):
        self.timestamp = timestamp
        self.symbol = symbol
        self.price = price
        self.volume = volume
        self.bid = bid
        self.ask = ask
        self.spread = spread
        self.order_imbalance = order_imbalance
        self.quality_score = quality_score

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "price": self.price,
            "volume": self.volume,
            "bid": self.bid,
            "ask": self.ask,
            "spread": self.spread,
            "order_imbalance": self.order_imbalance,
            "quality_score": self.quality_score,
        }


class IntelligentDataEngine:
    """
    Autonomous data collection, validation, and streaming.
    """

    def __init__(self, config: Dict):
        """
        Initializes the IntelligentDataEngine.

        Args:
            config: The configuration dictionary.
        """
        self.config = config
        self.sources = self._init_sources(config["data"]["sources"])
        self.cache = redis.Redis(
            host=config["data"]["cache"]["host"], port=config["data"]["cache"]["port"]
        )
        self.quality_model = self._train_quality_scorer()
        self.historical_data = self._load_historical_data()  # Placeholder for historical data
        self.scaler = StandardScaler()

    def _init_sources(self, sources_config: List[Dict]) -> Dict:
        """
        Initializes the data sources from the configuration.

        Args:
            sources_config: A list of data source configurations.

        Returns:
            A dictionary of initialized data sources.
        """
        sources = {}
        for source in sources_config:
            sources[source["name"]] = source
        return sources

    def _train_quality_scorer(self) -> IsolationForest:
        """
        Trains a model to score data quality.

        Returns:
            A trained IsolationForest model.
        """
        # Generate some dummy data for training
        X = np.random.rand(1000, 5)
        # Add some anomalies
        X = np.vstack([X, np.random.rand(50, 5) * 10])

        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(X)
        return model

    def _load_historical_data(self, n_samples: int = 10000, n_features: int = 5) -> np.ndarray:
        """
        Loads historical data for drift detection.

        In a real system, this would load data from a database or file.
        Here, we generate synthetic data for demonstration.

        Args:
            n_samples: The number of samples to generate.
            n_features: The number of features to generate.

        Returns:
            A NumPy array of historical data.
        """
        return np.random.randn(n_samples, n_features)

    async def stream(self) -> AsyncGenerator[MarketData, None]:
        """
        Main data stream. Yields normalized market data.

        Returns:
            An async generator of MarketData objects.
        """
        queues = {name: asyncio.Queue() for name in self.sources}
        tasks = []
        for name, source_config in self.sources.items():
            if "websocket_url" in source_config:
                task = asyncio.create_task(
                    self._stream_websocket(name, source_config["websocket_url"], queues[name])
                )
                tasks.append(task)
            elif "api_key" in source_config: # Assuming REST for polygon
                 task = asyncio.create_task(
                    self._stream_rest(name, source_config, queues[name])
                )
                 tasks.append(task)


        while True:
            for name in self.sources:
                if not queues[name].empty():
                    raw_data = await queues[name].get()
                    market_data = self._process_raw_data(raw_data)
                    if market_data:
                        yield market_data

            await asyncio.sleep(0.001)  # Prevent busy-waiting

    async def _stream_websocket(self, name: str, url: str, queue: asyncio.Queue):
        """
        Streams data from a WebSocket connection.

        Args:
            name: The name of the data source.
            url: The WebSocket URL.
            queue: The queue to put the received data into.
        """
        while True:
            try:
                async with websockets.connect(url) as websocket:
                    console.log(f"Connected to {name} WebSocket.")
                    # Subscription message might be needed for some exchanges
                    # For example, for Coinbase:
                    if name == "coinbase":
                         await websocket.send('{"type": "subscribe","product_ids": ["BTC-USD"],"channels": ["ticker"]}')
                    while True:
                        message = await websocket.recv()
                        await queue.put({"source": name, "data": message})
            except (websockets.ConnectionClosed, ConnectionRefusedError) as e:
                console.log(f"WebSocket connection for {name} closed: {e}. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)


    async def _stream_rest(self, name: str, config: Dict, queue: asyncio.Queue):
        """
        Streams data from a REST API by polling.

        Args:
            name: The name of the data source.
            config: The configuration for the data source.
            queue: The queue to put the received data into.
        """
        url = "https://api.polygon.io/v2/aggs/ticker/AAPL/prev?adjusted=true" # Example URL
        headers = {"Authorization": f"Bearer {config['api_key']}"}

        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            await queue.put({"source": name, "data": data})
                        else:
                            console.log(f"Error fetching data from {name}: {response.status}")
                except aiohttp.ClientError as e:
                    console.log(f"Error with {name} REST request: {e}")

                await asyncio.sleep(5) # Poll every 5 seconds


    def _process_raw_data(self, raw_data: Dict) -> MarketData:
        """
        Processes raw data from a source into a MarketData object.
        """
        source = raw_data.get("source")
        data_str = raw_data.get("data")

        try:
            import json
            data = json.loads(data_str)

            parsed = {}
            if source == "coinbase" and data.get("type") == "ticker":
                parsed = {
                    "symbol": data["product_id"],
                    "price": float(data["price"]),
                    "volume": float(data["last_size"]),
                    "bid": float(data["best_bid"]),
                    "ask": float(data["best_ask"]),
                }
            elif source == "binance" and data.get("e") == "trade":
                parsed = {
                    "symbol": data["s"],
                    "price": float(data["p"]),
                    "volume": float(data["q"]),
                }
            elif source == "polygon" and data.get("status") == "OK":
                 # This is a dummy parsing for polygon REST response
                 if data.get('results'):
                    res = data['results'][0]
                    parsed = {
                        "symbol": data['ticker'],
                        "price": res['c'],
                        "volume": res['v'],
                    }

            if not parsed:
                return None

            # Add placeholder for missing values
            parsed.setdefault("bid", parsed["price"] * 0.9999)
            parsed.setdefault("ask", parsed["price"] * 1.0001)

            df = pl.DataFrame([parsed])
            quality_score = self.validate_quality(df)

            if quality_score < 0.5:
                return None

            return MarketData(
                timestamp=time.time(),
                symbol=parsed["symbol"],
                price=parsed["price"],
                volume=parsed["volume"],
                bid=parsed["bid"],
                ask=parsed["ask"],
                spread=parsed["ask"] - parsed["bid"],
                order_imbalance=np.random.rand(),
                quality_score=quality_score,
            )

        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    def validate_quality(self, data: pl.DataFrame) -> float:
        """
        Scores data quality using a learned model.
        Checks: completeness, consistency, timeliness, anomalies.

        Args:
            data: A Polars DataFrame of market data.

        Returns:
            A quality score between 0 and 1.
        """
        # This is a simplified quality check.
        # A real system would have more sophisticated checks.
        features = data.select(pl.all().is_numeric()).to_numpy()
        if features.shape[1] < self.quality_model.n_features_in_:
            # Pad features if necessary (e.g., if some columns are missing)
             padding = np.zeros((features.shape[0], self.quality_model.n_features_in_ - features.shape[1]))
             features = np.hstack([features, padding])


        score = self.quality_model.score_samples(features)
        # Normalize score to be between 0 and 1
        return (score - score.min()) / (score.max() - score.min()) if (score.max() - score.min()) > 0 else 0.5


    def detect_drift(self, recent: np.ndarray, historical: np.ndarray) -> float:
        """
        Adversarial validation to detect data drift.
        Trains a classifier to distinguish old vs new data.
        High accuracy = drift detected.

        Args:
            recent: A NumPy array of recent data.
            historical: A NumPy array of historical data.

        Returns:
            A drift score (accuracy of the classifier).
        """
        # Create labels
        y_recent = np.ones(recent.shape[0])
        y_historical = np.zeros(historical.shape[0])

        # Combine data
        X = np.vstack([recent, historical])
        y = np.hstack([y_recent, y_historical])

        # Scale data
        X = self.scaler.fit_transform(X)

        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)

        return model.score(X_test, y_test)
