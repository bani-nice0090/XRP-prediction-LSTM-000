# tests/test_all.py

import asyncio
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import yaml
import torch
import torch.nn as nn

# Add src to path to allow imports
import sys
sys.path.insert(0, '.')

from data_engine import IntelligentDataEngine, MarketData
from nas_engine import NASEngine
from meta_controller import MetaController
from feature_learner import FeatureLearner
from online_learner import OnlineLearner
from risk_predictor import RiskAwarePredictor
from experience_memory import ExperienceMemory, Experience


class TestEliteMLSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the test environment."""
        with open("config.yaml", 'r') as f:
            cls.config = yaml.safe_load(f)

    def test_data_engine_initialization(self):
        """Test that the IntelligentDataEngine initializes correctly."""
        data_engine = IntelligentDataEngine(self.config)
        self.assertIsNotNone(data_engine)
        self.assertIn("polygon", data_engine.sources)
        self.assertIsNotNone(data_engine.quality_model)

    def test_nas_engine_initialization(self):
        """Test that the NASEngine initializes correctly."""
        nas_engine = NASEngine(self.config)
        self.assertIsNotNone(nas_engine)
        self.assertIsNotNone(nas_engine.controller)
        self.assertGreater(len(nas_engine.search_space), 0)

    def test_meta_controller_initialization(self):
        """Test that the MetaController initializes correctly."""
        nas_engine = NASEngine(self.config)
        nas_engine.population = [(['lstm_256_True'], 2.1)]
        meta_controller = MetaController(nas_engine, self.config)
        self.assertIsNotNone(meta_controller)
        self.assertGreater(len(meta_controller.models), 0)
        self.assertIsNotNone(meta_controller.regime_detector)

    def test_feature_learner_extraction(self):
        """Test the feature extraction process."""
        feature_learner = FeatureLearner(self.config)
        # With an encoder length of 30, 40 samples will produce 10 predictions.
        raw_data = np.random.rand(40, self.config['feature_learner']['tft']['input_size'])
        features = feature_learner.extract_features(raw_data)
        self.assertEqual(features.shape, (10, self.config['feature_learner']['vae']['latent_dim']))

    def test_online_learner_update(self):
        """Test the online learning update mechanism."""
        model = nn.Sequential(nn.Linear(10, 1))
        config = self.config.copy()
        config['online_learner'] = {'replay_buffer_size': 100, 'ewc_lambda': 0.1}
        online_learner = OnlineLearner(model, config)

        X = np.random.rand(20, 10)
        y = np.random.rand(20)

        initial_params = [p.clone() for p in model.parameters()]
        online_learner.update((X, y))
        updated_params = [p.clone() for p in model.parameters()]

        # Check that parameters have been updated
        params_changed = any(not torch.equal(i, u) for i, u in zip(initial_params, updated_params))
        self.assertTrue(params_changed)

    def test_risk_predictor_output(self):
        """Test the output shape and types of the RiskAwarePredictor."""
        model = nn.Sequential(nn.Linear(32, 32))
        predictor = RiskAwarePredictor(model, self.config)
        features = np.random.rand(1, 32)
        prediction = predictor.predict(features)

        self.assertIsInstance(prediction.expected_return, float)
        self.assertIsInstance(prediction.volatility, float)
        self.assertIsInstance(prediction.sharpe_prediction, float)
        self.assertEqual(len(prediction.quantiles), len(self.config['risk_predictor']['quantiles']))

    def test_experience_memory_storage_and_recall(self):
        """Test storing and recalling experiences."""
        memory = ExperienceMemory(self.config)
        embedding_dim = self.config['experience_memory']['embedding_dim']

        # Create a dummy experience
        exp = Experience(
            market_state_embedding=np.random.rand(embedding_dim),
            prediction=MagicMock(),
            outcome=0.01,
            model_architecture=['test_arch'],
            regime='test_regime',
            timestamp=12345.0
        )

        memory.store(exp)
        self.assertEqual(len(memory), 1)

        recalled = memory.recall(np.random.rand(embedding_dim), k=1)
        self.assertEqual(len(recalled), 1)
        self.assertEqual(recalled[0].regime, 'test_regime')

if __name__ == '__main__':
    unittest.main()
