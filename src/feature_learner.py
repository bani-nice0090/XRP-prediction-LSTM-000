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
        self.tft = None
        self.vae = None
        self.training_dataset = None

    def train(self, historical_data: pd.DataFrame):
        """
        Trains the TFT and VAE models on historical data.
        """
        print("Training FeatureLearner...")
        # 1. Create TimeSeriesDataSet from historical data
        self.training_dataset = self._create_dataset(historical_data)

        # 2. Initialize and train TFT
        self.tft = self._init_and_train_tft(self.training_dataset)

        # 3. Get embeddings from TFT to train VAE
        tft_embeddings = self._get_tft_embeddings(self.training_dataset)

        # 4. Initialize and train VAE
        self.vae = self._init_and_train_vae(tft_embeddings)
        print("FeatureLearner training complete.")

    def _create_dataset(self, df: pd.DataFrame) -> TimeSeriesDataSet:
        """Creates a TimeSeriesDataSet from a pandas DataFrame."""
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
            time_varying_unknown_reals=[col for col in df.columns if col.startswith('feature_')],
            target_normalizer=GroupNormalizer(groups=["group"])
        )

    def _init_and_train_tft(self, dataset: TimeSeriesDataSet) -> TemporalFusionTransformer:
        """Initializes and trains the Temporal Fusion Transformer."""
        tft = TemporalFusionTransformer.from_dataset(
            dataset,
            learning_rate=1e-3,
            hidden_size=self.config['feature_learner']['tft']['hidden_size'],
            attention_head_size=self.config['feature_learner']['tft']['attention_heads'],
            output_size=self.config['feature_learner']['tft']['output_size'],
            loss=QuantileLoss()
        )

        dataloader = dataset.to_dataloader(train=True, batch_size=64)
        # Simple training loop for demonstration
        # In a real system, you'd use a proper trainer like PyTorch Lightning
        optimizer = torch.optim.Adam(tft.parameters(), lr=1e-3)
        for epoch in range(2): # Train for a few epochs
            for x, y in dataloader:
                optimizer.zero_grad()
                out = tft(x)
                target, _ = y # Unpack the tuple from the dataloader
                loss = tft.loss(out["prediction"], target)
                loss.backward()
                optimizer.step()
        return tft

    def _get_tft_embeddings(self, dataset: TimeSeriesDataSet) -> torch.Tensor:
        """Gets embeddings from the trained TFT model."""
        dataloader = dataset.to_dataloader(train=False, batch_size=64)
        all_embeddings = []
        with torch.no_grad():
            for x, _ in dataloader:
                out = self.tft(x)
                embedding = out["prediction"].squeeze(1)
                all_embeddings.append(embedding)
        return torch.cat(all_embeddings, dim=0)

    def _init_and_train_vae(self, embeddings: torch.Tensor) -> VariationalAutoencoder:
        """Initializes and trains the Variational Autoencoder."""
        input_dim = embeddings.shape[1]
        vae = VariationalAutoencoder(
            input_dim=input_dim,
            latent_dim=self.config['feature_learner']['vae']['latent_dim']
        )
        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
        for epoch in range(5): # Train for a few epochs
            recon, mu, log_var = vae(embeddings)
            # Simplified VAE loss
            recon_loss = nn.functional.mse_loss(recon, embeddings)
            kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + kld_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return vae

    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """
        Transforms raw data to learned features using the trained TFT and VAE.
        """
        if self.tft is None or self.vae is None:
            raise RuntimeError("FeatureLearner must be trained before extracting features.")

        # 1. Convert numpy data to DataFrame
        num_samples, num_features = data.shape
        df_dict = {f'feature_{i}': data[:, i] for i in range(num_features)}
        # Ensure time_idx is continuous from the training data for proper encoding
        last_time_idx = self.training_dataset.index.time.max()
        df_dict['time_idx'] = np.arange(last_time_idx + 1, last_time_idx + 1 + num_samples)
        df_dict['group'] = 0
        df_dict['target'] = 0  # Dummy target
        df = pd.DataFrame(df_dict)

        # 2. Create a dataset for inference
        # Create a dataset that creates sequences for feature extraction
        inference_dataset = TimeSeriesDataSet.from_dataset(self.training_dataset, df, stop_randomization=True)
        dataloader = inference_dataset.to_dataloader(train=False, batch_size=64)

        # 3. Use the dedicated predict method to get embeddings for all sequences
        with torch.no_grad():
            predictions = self.tft.predict(dataloader)
            # The output shape will be (n_sequences, n_prediction_steps, n_quantiles)
            # For feature extraction, we have 1 prediction step.
            tft_embeddings = predictions.squeeze(1)

        # 4. Get final features from VAE
        with torch.no_grad():
            _, mu, _ = self.vae(tft_embeddings)

        return mu.numpy()


# Example usage (for testing)
if __name__ == '__main__':
    import yaml

    with open("../config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    feature_learner = FeatureLearner(config)

    # Create dummy raw data
    dummy_raw_data = np.random.rand(40, config['feature_learner']['tft']['input_size'])

    # Create dummy training data
    train_df = pd.DataFrame({
        **{f'feature_{i}': np.random.randn(100) for i in range(config['feature_learner']['tft']['input_size'])},
        'time_idx': np.arange(100),
        'group': 0,
        'target': np.random.randn(100)
    })

    feature_learner.train(train_df)

    # Extract features
    learned_features = feature_learner.extract_features(dummy_raw_data)

    print(f"Original data shape: {dummy_raw_data.shape}")
    print(f"Learned features shape: {learned_features.shape}")
    # With encoder length 30, 40 samples will produce 40 - 30 = 10 predictions
    assert learned_features.shape == (10, config['feature_learner']['vae']['latent_dim'])
