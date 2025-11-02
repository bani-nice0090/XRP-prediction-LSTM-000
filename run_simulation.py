# run_simulation.py

import asyncio
import time
import yaml
import numpy as np

# Import all the system components
from data_engine import IntelligentDataEngine
from nas_engine import NASEngine
from meta_controller import MetaController
from feature_learner import FeatureLearner
from online_learner import OnlineLearner
from risk_predictor import RiskAwarePredictor
from experience_memory import ExperienceMemory
from executor import TradingExecutor

async def run_simulation():
    """
    Simulates the main execution loop to test and validate the system's logic.
    """
    print("--- Starting System Logic Simulation ---")

    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # --- Initialize Components ---
    data_engine = IntelligentDataEngine(config)
    nas_engine = NASEngine(config)
    if not nas_engine.population: # Populate with dummy archs
        nas_engine.population = [
            ([('lstm', {'units': 256, 'bidirectional': True}), ('attention', {'heads': 8, 'dim': 256})], 2.1),
            ([('tcn', {'layers': 6, 'channels': 128}), ('conv1d', {'filters': 64, 'kernel': 5})], 1.9)
        ]
    meta_controller = MetaController(nas_engine, config)
    feature_learner = FeatureLearner(config)
    initial_model = meta_controller.select_model("regime_0")
    risk_predictor = RiskAwarePredictor(initial_model, config)

    print("All components initialized successfully.")

    # --- Simulation Loop ---
    data_buffer = []
    buffer_size = 40
    predictions_made = 0
    total_latency = 0.0

    print("Starting data stream processing...")

    async for market_data in data_engine.stream():
        if market_data is None:
            continue

        start_time = time.time()

        data_buffer.append(market_data)
        if len(data_buffer) < buffer_size:
            continue

        raw_data_array = np.array([[md.price, md.volume, md.spread, md.order_imbalance, 0, 0, 0, 0] for md in data_buffer])
        data_buffer.pop(0)

        features = feature_learner.extract_features(raw_data_array)
        if features.shape[0] == 0:
            continue

        prediction = risk_predictor.predict(features)

        latency = time.time() - start_time
        total_latency += latency
        predictions_made += 1

        print(f"Prediction #{predictions_made}: Expected Return={prediction.expected_return:.5f}, Latency={latency*1000:.2f}ms")

        # Stop the simulation after a few predictions to establish a baseline
        if predictions_made >= 10:
            break

    print("\n--- Simulation Complete ---")
    if predictions_made > 0:
        avg_latency = total_latency / predictions_made
        print(f"Total Predictions: {predictions_made}")
        print(f"Average Latency: {avg_latency * 1000:.2f} ms")
    else:
        print("No predictions were made. The data stream might have been too slow or empty.")

if __name__ == "__main__":
    try:
        asyncio.run(run_simulation())
    except Exception as e:
        print(f"An error occurred during the simulation: {e}")
