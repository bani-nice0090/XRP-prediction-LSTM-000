# main.py

import asyncio
import os
import time
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import deque

# Import all the system components
from src.data_engine import IntelligentDataEngine
from src.nas_engine import NASEngine
from src.meta_controller import MetaController
from src.feature_learner import FeatureLearner
from src.online_learner import OnlineLearner
from src.risk_predictor import RiskAwarePredictor, PredictionOutput
from src.experience_memory import ExperienceMemory, Experience
from src.executor import TradingExecutor
from src.utils import display_system_status

class ElitePredictionSystem:
    """
    The main orchestrator for the elite prediction system.
    """
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # --- System State for UI & Performance Tracking (Initialize first) ---
        self.log_messages = deque(maxlen=10)
        self.last_prediction: PredictionOutput = None
        self.current_regime: str = "Initializing..."
        self.online_updates: int = 0
        self.latency: float = 0.0
        self.accuracy: float = 0.0
        self.error_count: int = 0
        self.pending_accuracy_checks = deque()
        self.accuracy_outcomes = deque(maxlen=100)
        self.ACCURACY_CHECK_DELAY = 60 # seconds

        # --- Initialize all components ---
        self.data_engine = IntelligentDataEngine(self.config)
        self.nas_engine = self._initialize_nas()
        self.meta_controller = MetaController(self.nas_engine, self.config)
        self.feature_learner = FeatureLearner(self.config)

        initial_model = self.meta_controller.select_model("regime_0")

        self.online_learner = OnlineLearner(initial_model, self.config)
        self.risk_predictor = RiskAwarePredictor(initial_model, self.config)
        self.memory = ExperienceMemory(self.config)
        self.executor = TradingExecutor(self.config)

        self.log("System initialized.")

    def _initialize_nas(self) -> NASEngine:
        """Initializes the NAS engine."""
        nas_engine = NASEngine(self.config)
        if not nas_engine.population:
            self.log("NAS population is empty. Populating with dummy architectures.")
            nas_engine.population = [
                ([('lstm', {'units': 256, 'bidirectional': True}), ('attention', {'heads': 8, 'dim': 256})], 2.1),
                ([('tcn', {'layers': 6, 'channels': 128}), ('conv1d', {'filters': 64, 'kernel': 5})], 1.9)
            ]
        return nas_engine

    def log(self, message: str, is_error: bool = False):
        """Adds a message to the system log."""
        prefix = "ERROR" if is_error else "INFO"
        self.log_messages.append(f"[{time.time():.2f}] [{prefix}] {message}")

    async def _check_accuracy(self, current_price: float):
        """Checks pending predictions for accuracy."""
        current_time = time.time()
        while self.pending_accuracy_checks and (current_time - self.pending_accuracy_checks[0]['time']) > self.ACCURACY_CHECK_DELAY:
            check = self.pending_accuracy_checks.popleft()

            actual_return = current_price - check['price']
            predicted_positive = check['predicted_return'] > 0
            actual_positive = actual_return > 0

            is_correct = 1 if predicted_positive == actual_positive else 0
            self.accuracy_outcomes.append(is_correct)

            if self.accuracy_outcomes:
                self.accuracy = sum(self.accuracy_outcomes) / len(self.accuracy_outcomes)

    async def _initial_setup(self):
        """Handles the initial training and setup of the models."""
        self.log("Performing initial system setup...")

        # 1. Train FeatureLearner
        historical_data_df = pd.DataFrame(
            self.data_engine.historical_data,
            columns=[f'feature_{i}' for i in range(self.data_engine.historical_data.shape[1])]
        )
        # Add required columns for TimeSeriesDataSet
        historical_data_df['time_idx'] = np.arange(len(historical_data_df))
        historical_data_df['group'] = 0
        historical_data_df['target'] = historical_data_df['feature_0'].shift(-1).fillna(0) # Dummy target

        self.feature_learner.train(historical_data_df)
        self.log("FeatureLearner trained.")

        # 2. Run NAS to find best architectures
        self.log("Starting NAS search...")
        # We need featurized data for NAS
        featurized_historical_data = self.feature_learner.extract_features(self.data_engine.historical_data)
        # Append target column for training
        nas_training_data = np.hstack([
            featurized_historical_data,
            historical_data_df['target'].values[len(historical_data_df) - len(featurized_historical_data):].reshape(-1, 1)
        ])

        self.nas_engine.search(nas_training_data, n_generations=self.config['nas']['n_generations'])
        self.log(f"NAS search complete. Top architectures found: {len(self.nas_engine.population)}")

        # 3. Re-initialize MetaController with the found architectures
        self.meta_controller = MetaController(self.nas_engine, self.config)
        self.log("MetaController updated with NAS population.")


    async def run(self):
        """The main execution loop of the system."""
        # Perform initial setup if NAS population is empty
        if not self.nas_engine.population:
            await self._initial_setup()

        self.log("Starting main system loop.")
        ui_task = asyncio.create_task(self._run_ui())
        data_buffer = []
        buffer_size = self.config['feature_learner']['tft']['input_size'] # Buffer size should match TFT input size

        async for market_data in self.data_engine.stream():
            try:
                if market_data is None: continue

                start_time = time.time()
                data_buffer.append(market_data)
                await self._check_accuracy(market_data.price)

                if len(data_buffer) < buffer_size:
                    continue

                # Prepare raw data for feature extraction
                raw_data_array = np.array([[md.price, md.volume, md.spread, md.order_imbalance, 0, 0, 0, 0] for md in data_buffer])
                data_buffer.pop(0)

                features = self.feature_learner.extract_features(raw_data_array)

                self.current_regime = self.meta_controller.detect_regime(features)
                model = self.meta_controller.select_model(self.current_regime)

                self.risk_predictor.base_model = model
                prediction = self.risk_predictor.predict(features)
                self.last_prediction = prediction

                self.pending_accuracy_checks.append({
                    'time': time.time(),
                    'price': market_data.price,
                    'predicted_return': prediction.expected_return
                })

                self.executor.execute_trade(prediction)
                self.latency = time.time() - start_time
                self.log(f"New prediction. Sharpe: {prediction.sharpe_prediction:.2f}")

            except Exception as e:
                self.error_count += 1
                self.log(f"An error occurred: {e}", is_error=True)

    async def _run_ui(self):
        """Runs the terminal display."""
        await asyncio.to_thread(display_system_status, self, self.log_messages)

if __name__ == "__main__":
    if "YOUR_POLYGON_API_KEY" in open("config.yaml").read():
        print("Warning: Polygon API key not set. Using a dummy key.")
        os.environ["POLYGON_API_KEY"] = "dummy_key_for_testing"

    try:
        system = ElitePredictionSystem()
        asyncio.run(system.run())
    except KeyboardInterrupt:
        print("\nSystem shutting down.")
    except Exception as e:
        print(f"An error occurred: {e}")
