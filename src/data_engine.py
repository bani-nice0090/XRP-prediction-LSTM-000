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

    def _train_quality_scorer(self):
        """
        Placeholder for a quality scoring model.
        In this version, we use a rule-based validator instead.
        """
        return None

    def _load_historical_data(self) -> np.ndarray:
        """
        Loads historical data for drift detection from a CSV file.
        """
        filepath = self.config["data"]["historical_data_path"]
        try:
            df = pl.read_csv(filepath)
            # Assuming the CSV contains columns for features
            return df.to_numpy()
        except FileNotFoundError:
            console.log(f"[bold red]Warning:[/bold red] Historical data file not found at {filepath}. Drift detection will be impaired.")
            # Return a small, empty-like array to avoid crashes downstream
            return np.empty((0, 5)) # Assuming 5 features as before

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

            quality_score = self.validate_quality(parsed)

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

    def validate_quality(self, data: Dict) -> float:
        """
        Scores data quality using a rule-based system.
        Returns a score between 0.0 (bad) and 1.0 (good).
        """
        score = 1.0

        # Rule 1: Prices, bids, and asks must be positive
        if data.get("price", 0) <= 0 or data.get("bid", 0) <= 0 or data.get("ask", 0) <= 0:
            score -= 0.5

        # Rule 2: Spread should not be negative
        if data.get("ask", 0) < data.get("bid", 0):
            score -= 0.5

        # Rule 3: Volume must not be negative
        if data.get("volume", 0) < 0:
            score -= 0.2

        # Rule 4: Spread shouldn't be excessively large (e.g., >1% of price)
        if (data.get("ask", 0) - data.get("bid", 0)) > data.get("price", 1) * 0.01:
            score -= 0.2

        return max(0.0, score)


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
