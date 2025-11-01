# src/feature_learner.py

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from torch.utils.data import DataLoader


class VariationalAutoencoder(nn.Module):
    """
    A Variational Autoencoder (VAE) for dimensionality reduction.
    """
    def __init__(self, input_dim: int, latent_dim: int):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_log_var = nn.Linear(16, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid() # Assuming input is normalized
        )

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var


class FeatureLearner:
    """
    Learns representations automatically from raw time-series data.
    """
    def __init__(self, config: Dict):
        self.config = config
        # The output of the TFT will be a set of quantiles, which we'll use as features.
        # The default is 7 quantiles for the QuantileLoss.
        self.tft_output_size = 7
        self.tft = self._init_tft()
        self.vae = VariationalAutoencoder(
            # The VAE's input dimension must match the TFT's output dimension.
            input_dim=self.tft_output_size,
            latent_dim=config['feature_learner']['vae']['latent_dim']
        )
        # Dummy dataset for inference
        self.training_dataset = self._create_dummy_dataset()

    def _init_tft(self) -> TemporalFusionTransformer:
        """
        Initializes the Temporal Fusion Transformer model.
        """
        dummy_dataset = self._create_dummy_dataset()

        return TemporalFusionTransformer.from_dataset(
            dummy_dataset,
            learning_rate=1e-3,
            hidden_size=self.config['feature_learner']['tft']['hidden_size'],
            attention_head_size=self.config['feature_learner']['tft']['attention_heads'],
            dropout=0.1,
            hidden_continuous_size=16,
            output_size=self.tft_output_size,
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4
        )

    def _create_dummy_dataset(self) -> TimeSeriesDataSet:
        """
        Creates a dummy dataset for model initialization and inference structure.
        """
        num_samples = 100
        num_features = self.config['feature_learner']['tft']['input_size']

        data = {
            f'feature_{i}': np.random.randn(num_samples) for i in range(num_features)
        }
        data['time_idx'] = np.arange(num_samples)
        data['group'] = 0
        data['target'] = np.random.randn(num_samples)

        df = pd.DataFrame(data)

        return TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target="target",
            group_ids=["group"],
            max_encoder_length=30,
            max_prediction_length=1,
            static_categoricals=[],
            static_reals=[],
            time_varying_known_categoricals=[],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=[f'feature_{i}' for i in range(num_features)],
            target_normalizer=GroupNormalizer(groups=["group"])
        )

    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """
        Transforms raw data to learned features using TFT and VAE.

        Args:
            data: A numpy array of shape (n_samples, n_features), where
                  n_features should match TFT input_size.

        Returns:
            A numpy array of shape (n_samples, latent_dim).
        """
        # 1. Convert numpy data to a pandas DataFrame in the required format
        num_samples, num_features = data.shape
        df_dict = {f'feature_{i}': data[:, i] for i in range(num_features)}
        df_dict['time_idx'] = np.arange(num_samples)
        df_dict['group'] = 0
        df_dict['target'] = 0 # Dummy target
        df = pd.DataFrame(df_dict)

        # 2. Create a DataLoader for inference. We set `predict=False` to process the entire
        # historical sequence, which will generate an embedding for each possible window.
        inference_dataset = TimeSeriesDataSet.from_dataset(self.training_dataset, df, predict=False, stop_randomization=True)
        dataloader = inference_dataset.to_dataloader(train=False, batch_size=64)

        # 3. Pass data through TFT to get embeddings
        all_embeddings = []
        with torch.no_grad():
            for x, _ in dataloader:
                out = self.tft(x)
                # The prediction tensor (quantiles) is used as the feature embedding.
                # Shape: (batch_size, prediction_length, output_size)
                prediction = out["prediction"]
                # Squeeze the prediction horizon dimension (which is 1).
                embedding = prediction.squeeze(1)
                all_embeddings.append(embedding)

        tft_embeddings = torch.cat(all_embeddings, dim=0)

        # 4. Pass embeddings through VAE to get final features
        with torch.no_grad():
            _, mu, _ = self.vae(tft_embeddings)

        return mu.numpy()


# Example usage (for testing)
if __name__ == '__main__':
    import yaml

    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    feature_learner = FeatureLearner(config)

    # Create dummy raw data
    dummy_raw_data = np.random.rand(40, config['feature_learner']['tft']['input_size'])

    # Extract features
    learned_features = feature_learner.extract_features(dummy_raw_data)

    print(f"Original data shape: {dummy_raw_data.shape}")
    print(f"Learned features shape: {learned_features.shape}")
    # With encoder length 30, 40 samples will produce 40 - 30 = 10 predictions
    assert learned_features.shape == (10, config['feature_learner']['vae']['latent_dim'])
