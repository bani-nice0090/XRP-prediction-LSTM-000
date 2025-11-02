# src/risk_predictor.py

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

@dataclass
class PredictionOutput:
    """
    Holds the output of a risk-aware prediction.
    """
    expected_return: float
    volatility: float
    quantiles: Dict[float, float]
    var_95: float
    cvar_95: float
    sharpe_prediction: float
    optimal_position: float

class QuantileRegressionNetwork(nn.Module):
    """
    A neural network that predicts multiple quantiles of the return distribution.
    """
    def __init__(self, input_dim: int, quantiles: List[float]):
        super(QuantileRegressionNetwork, self).__init__()
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_quantiles)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

def quantile_loss(preds: torch.Tensor, targets: torch.Tensor, quantiles: List[float]) -> torch.Tensor:
    """
    Calculates the quantile loss (pinball loss).
    """
    assert preds.shape[1] == len(quantiles)
    targets = targets.unsqueeze(1).expand_as(preds)

    errors = targets - preds
    loss = torch.max((torch.tensor(quantiles) - 1) * errors, torch.tensor(quantiles) * errors)
    return loss.mean()

def sharpe_loss(predictions: torch.Tensor, targets: torch.Tensor, risk_free_rate: float = 0.0) -> torch.Tensor:
    """
    A custom loss function that aims to maximize the Sharpe ratio.
    Note: This is a simplified version. A numerically stable version is more complex.
    """
    # Assuming predictions are the returns
    returns = predictions.squeeze()

    mean_return = torch.mean(returns)
    std_return = torch.std(returns)

    # Add a small epsilon to avoid division by zero
    sharpe = (mean_return - risk_free_rate) / (std_return + 1e-6)

    # We want to maximize Sharpe, so we minimize its negative
    return -sharpe

class RiskAwarePredictor:
    """
    Predicts the full return distribution and optimizes for Sharpe ratio.
    """
    def __init__(self, model: nn.Module, config: Dict):
        self.base_model = model # This would be a model from the NASEngine
        self.quantiles = config['risk_predictor']['quantiles']

        # We'll assume the base_model's output is fed into the quantile head
        # In a more integrated setup, the quantile head would be part of the base model
        self.quantile_head = QuantileRegressionNetwork(
            input_dim=32, # Assuming 32 features from FeatureLearner
            quantiles=self.quantiles
        )

    def predict(self, features: np.ndarray) -> PredictionOutput:
        """
        Predicts the return distribution from input features.
        """
        features_tensor = torch.from_numpy(features).float()

        with torch.no_grad():
            quantile_preds = self.quantile_head(features_tensor)

        # For a single prediction, take the first row
        if quantile_preds.ndim > 1:
            quantile_preds = quantile_preds[0]

        quantile_values = quantile_preds.numpy()

        # --- Calculate risk metrics from quantiles ---

        # Expected return (approximated by the median)
        median_idx = self.quantiles.index(0.5)
        expected_return = quantile_values[median_idx]

        # Volatility (approximated by interquartile range)
        q75_idx = self.quantiles.index(0.75)
        q25_idx = self.quantiles.index(0.25)
        volatility = quantile_values[q75_idx] - quantile_values[q25_idx]

        # VaR 95 (Value at Risk)
        q05_idx = self.quantiles.index(0.05)
        var_95 = quantile_values[q05_idx] # This is the 5th percentile return

        # CVaR 95 (Conditional VaR) - average of returns below VaR
        # Simplified: just the 5th percentile value if we don't have the full distribution
        cvar_95 = var_95

        # Predicted Sharpe Ratio
        sharpe = expected_return / (volatility + 1e-6)

        # Optimal position size (Kelly Criterion - simplified)
        win_prob = np.mean(quantile_values > 0) # Prob of positive return
        avg_win = np.mean(quantile_values[quantile_values > 0]) if win_prob > 0 else 0
        avg_loss = np.abs(np.mean(quantile_values[quantile_values <= 0])) if win_prob < 1 else 1e-6
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0

        kelly_fraction = win_prob - ((1 - win_prob) / win_loss_ratio)

        return PredictionOutput(
            expected_return=float(expected_return),
            volatility=float(volatility),
            quantiles=dict(zip(self.quantiles, quantile_values)),
            var_95=float(var_95),
            cvar_95=float(cvar_95),
            sharpe_prediction=float(sharpe),
            optimal_position=float(max(0, min(1, kelly_fraction))) # Cap between 0 and 1
        )

# Example usage (for testing)
if __name__ == '__main__':
    import yaml

    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    dummy_base_model = nn.Sequential(nn.Linear(32, 32)) # Placeholder
    predictor = RiskAwarePredictor(dummy_base_model, config)

    # Dummy features from the FeatureLearner
    dummy_features = np.random.rand(1, 32)

    # Get a prediction
    prediction = predictor.predict(dummy_features)

    print("--- Risk-Aware Prediction Output ---")
    print(f"Expected Return: {prediction.expected_return:.4f}")
    print(f"Volatility: {prediction.volatility:.4f}")
    print(f"95% VaR: {prediction.var_95:.4f}")
    print(f"Predicted Sharpe: {prediction.sharpe_prediction:.2f}")
    print(f"Optimal Position Size (Kelly): {prediction.optimal_position:.2f}")
    print(f"Quantiles: {prediction.quantiles}")
