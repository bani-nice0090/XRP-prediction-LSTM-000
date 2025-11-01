# main.py

import asyncio
import os
import yaml
import numpy as np
import torch
import torch.nn as nn
from collections import deque

# Import all the system components
from src.data_engine import IntelligentDataEngine, MarketData
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

        # --- Initialize all components ---
        self.data_engine = IntelligentDataEngine(self.config)
        self.nas_engine = self._initialize_nas()
        self.meta_controller = MetaController(self.nas_engine, self.config)
        self.feature_learner = FeatureLearner(self.config)

        # Select an initial model to start with
        initial_model = self.meta_controller.select_model("regime_0")

        self.online_learner = OnlineLearner(initial_model, self.config)
        self.risk_predictor = RiskAwarePredictor(initial_model, self.config)
        self.memory = ExperienceMemory(self.config)
        self.executor = TradingExecutor(self.config)

        # --- System State for UI ---
        self.last_prediction: PredictionOutput = None
        self.current_regime: str = "Initializing..."
        self.online_updates: int = 0
        self.log_messages = deque(maxlen=10)
        self.log("System initialized.")

    def _initialize_nas(self) -> NASEngine:
        """
        Initializes the NAS engine and loads/runs search as needed.
        """
        nas_engine = NASEngine(self.config)

        # In a real system, you'd save/load the population
        # For this simulation, we'll populate it with dummy architectures
        if not nas_engine.population:
            self.log("NAS population is empty. Populating with dummy architectures.")
            nas_engine.population = [
                (['lstm_256_True', 'attention_8_256_0.1'], 2.1),
                (['tcn_6_128', 'conv1d_64_5_2'], 1.9),
                (['attention_12_512_0.2'], 2.3)
            ]
        return nas_engine

    def log(self, message: str):
        """Adds a message to the system log."""
        self.log_messages.append(f"[{asyncio.get_event_loop().time():.2f}] {message}")

    async def run(self):
        """
        The main execution loop of the system.
        """
        self.log("Starting main system loop.")

        # In a separate task, run the UI
        ui_task = asyncio.create_task(self._run_ui())

        # Main prediction and learning loop
        data_buffer = []
        buffer_size = 40 # Should be >= max_encoder_length

        async for market_data in self.data_engine.stream():
            if market_data is None:
                continue

            # Add data to buffer
            data_buffer.append(market_data)

            # If buffer is not full, continue collecting data
            if len(data_buffer) < buffer_size:
                continue

            # 1. Convert buffer to a numpy array
            raw_data_array = np.array([[
                md.price, md.volume, md.spread,
                md.order_imbalance, 0, 0, 0, 0 # Pad to 8 features
            ] for md in data_buffer])

            # Trim buffer to maintain size
            data_buffer.pop(0)

            # 2. Extract features
            # This will now produce a single feature vector for the latest time step
            features = self.feature_learner.extract_features(raw_data_array)

            # 3. Detect regime and select model
            self.current_regime = self.meta_controller.detect_regime(features)
            model = self.meta_controller.select_model(self.current_regime)

            # 4. Fast-adapt model if needed (placeholder logic)
            # if self.current_regime is new:
            #   support_set = self.memory.recall_by_regime(self.current_regime)
            #   model = self.meta_controller.fast_adapt(model, support_set)

            # 5. Make a risk-aware prediction
            self.risk_predictor.base_model = model # Update predictor with the selected model
            prediction = self.risk_predictor.predict(features)
            self.last_prediction = prediction
            self.log(f"New prediction. Sharpe: {prediction.sharpe_prediction:.2f}")

            # 6. Execute trade (in a real system)
            self.executor.execute_trade(prediction)

            # 7. Online learning and memory storage (if we have a label)
            # This part is conceptual as we don't have real-time labels
            if hasattr(market_data, 'future_return'):
                label = np.array([market_data.future_return])

                # Update the online learner
                self.online_learner.update((features, label))
                self.online_updates += 1

                # Store the experience in memory
                exp = Experience(
                    market_state_embedding=features[0],
                    prediction=prediction,
                    outcome=label[0],
                    model_architecture=self.nas_engine.population[0][0], # Placeholder
                    regime=self.current_regime,
                    timestamp=market_data.timestamp
                )
                self.memory.store(exp)
                self.log("Online learner updated and experience stored.")

            await asyncio.sleep(1) # Control loop speed

    async def _run_ui(self):
        """Runs the terminal display."""
        # This function is not async, so we run it in a thread
        # to avoid blocking the main async loop.
        await asyncio.to_thread(display_system_status, self)


if __name__ == "__main__":
    # This check is to prevent issues with multiprocessing/ray on some platforms
    # It ensures the main block is only run when the script is executed directly

    # Set a dummy API key for Polygon if it's not set
    if "YOUR_POLYGON_API_KEY" in open("config.yaml").read():
        print("Warning: Polygon API key not set in config.yaml. Using a dummy key.")
        # This is not a real key
        os.environ["POLYGON_API_KEY"] = "dummy_key_for_testing"

    try:
        system = ElitePredictionSystem()
        asyncio.run(system.run())
    except KeyboardInterrupt:
        print("\nSystem shutting down.")
    except Exception as e:
        print(f"An error occurred: {e}")
