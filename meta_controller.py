# src/meta_controller.py

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.mixture import GaussianMixture
from nas_engine import NASEngine, DynamicModel


class RegimeDetector:
    """
    Discovers market regimes using unsupervised learning.
    """

    def __init__(self, n_components: int = 12):
        self.gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        # A simple neural network to extract features for regime detection
        self.feature_extractor = nn.Sequential(
            nn.Linear(32, 64), # Assuming input features are 32-dimensional from FeatureLearner
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

    def fit(self, historical_data: np.ndarray):
        """
        Discovers regimes from historical data.
        Features could include volatility, trend, volume patterns, etc.
        """
        # In a real scenario, you'd extract meaningful features first.
        # Here, we assume historical_data is already featurized.
        with torch.no_grad():
            features = self.feature_extractor(torch.from_numpy(historical_data).float())
        self.gmm.fit(features.numpy())

    def predict(self, recent_data: np.ndarray) -> Tuple[int, float]:
        """
        Predicts the current regime and the confidence of the prediction.
        """
        with torch.no_grad():
            features = self.feature_extractor(torch.from_numpy(recent_data).float())

        regime_probs = self.gmm.predict_proba(features.numpy())

        # For a single recent data point
        if len(regime_probs) == 1:
            regime_id = np.argmax(regime_probs[0])
            confidence = np.max(regime_probs[0])
            return int(regime_id), float(confidence)

        # For a batch of recent data, return the most common regime
        regime_ids = np.argmax(regime_probs, axis=1)
        most_common_regime = np.bincount(regime_ids).argmax()
        avg_confidence = np.mean(np.max(regime_probs, axis=1))

        return int(most_common_regime), float(avg_confidence)


class MetaController:
    """
    Learns which models work best in which market regimes.
    Uses Model-Agnostic Meta-Learning (MAML).
    """

    def __init__(self, nas_engine: NASEngine, config: Dict):
        # We need to build the models from the population descriptions
        self.models = [self._build_model_from_desc(arch) for arch, _ in nas_engine.population]
        self.regime_detector = RegimeDetector(n_components=config['meta_controller']['regime_detection']['n_components'])
        # Placeholder for historical performance data
        self.model_performance = {f"regime_{i}": [[] for _ in self.models] for i in range(12)}

    def _build_model_from_desc(self, architecture: List[Tuple[str, Dict]]) -> nn.Module:
        """
        Builds a PyTorch model from an architecture description using the DynamicModel class.
        """
        # Assuming 32 input features from FeatureLearner and 1 output for prediction
        return DynamicModel(architecture, input_features=32, output_dim=1)

    def detect_regime(self, recent_data: np.ndarray) -> str:
        """
        Detects the current market regime.
        """
        regime_id, confidence = self.regime_detector.predict(recent_data)
        return f"regime_{regime_id}"

    def select_model(self, regime: str) -> nn.Module:
        """
        Selects the best model for the current regime based on historical performance.
        """
        regime_perf = self.model_performance.get(regime)
        if not regime_perf or not any(regime_perf):
            # If no data, return the first model as default
            return self.models[0]

        avg_perf = [np.mean(p) if p else -np.inf for p in regime_perf]
        best_model_idx = np.argmax(avg_perf)
        return self.models[best_model_idx]

    def fast_adapt(self, model: nn.Module, support_set: Tuple[np.ndarray, np.ndarray], inner_lr=1e-3, inner_steps=5) -> nn.Module:
        """
        Adapts a model to a new regime with a few samples using MAML.
        """
        # 1. Clone the model to create a "fast" model for adaptation
        fast_model = self._build_model_from_desc([]) # Re-create a fresh model for simplicity
        fast_model.load_state_dict(model.state_dict())

        # Use a simple optimizer for the inner loop
        inner_optimizer = optim.SGD(fast_model.parameters(), lr=inner_lr)
        loss_fn = nn.MSELoss()

        X_support, y_support = support_set
        X_support = torch.from_numpy(X_support).float()
        y_support = torch.from_numpy(y_support).float().unsqueeze(1)

        # 2. Take K gradient steps on the support set
        for _ in range(inner_steps):
            preds = fast_model(X_support)
            loss = loss_fn(preds, y_support)

            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()

        return fast_model

# Example usage (for testing)
if __name__ == '__main__':
    import yaml
    from src.nas_engine import NASEngine

    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Create a dummy NASEngine with a population
    nas_engine = NASEngine(config)
    nas_engine.population = [(['lstm_256_True'], 2.1), (['attention_8_256_0.1'], 1.9)]

    meta_controller = MetaController(nas_engine, config)

    # Fit the regime detector
    dummy_historical_data = np.random.rand(1000, 32)
    meta_controller.regime_detector.fit(dummy_historical_data)

    # Detect a regime
    dummy_recent_data = np.random.rand(10, 32)
    current_regime = meta_controller.detect_regime(dummy_recent_data)
    print(f"Detected Regime: {current_regime}")

    # Select a model
    selected_model = meta_controller.select_model(current_regime)
    print(f"Selected Model: {selected_model}")

    # Fast adapt the model
    support_set = (np.random.rand(20, 32), np.random.rand(20))
    adapted_model = meta_controller.fast_adapt(selected_model, support_set)
    print(f"Adapted Model: {adapted_model}")
