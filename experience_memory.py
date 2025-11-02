# src/experience_memory.py

from dataclasses import dataclass
from typing import List, Dict

import faiss
import numpy as np
from risk_predictor import PredictionOutput


@dataclass
class Experience:
    """
    Represents a single experience to be stored in memory.
    """
    market_state_embedding: np.ndarray
    prediction: PredictionOutput
    outcome: float
    model_architecture: List[str]
    regime: str
    timestamp: float

class ExperienceMemory:
    """
    An intelligent memory system using a vector database (FAISS)
    to find similar past situations.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.embedding_dim = config['experience_memory']['embedding_dim']

        # FAISS index for fast similarity search
        self.index = faiss.IndexFlatL2(self.embedding_dim)

        # We store the actual Experience objects separately
        self.experiences: List[Experience] = []

        self.max_size = config['experience_memory']['max_size']

    def store(self, experience: Experience):
        """
        Stores an experience in the memory.
        """
        if len(self.experiences) >= self.max_size:
            # Simple FIFO eviction strategy
            self.experiences.pop(0)
            # FAISS index needs to be rebuilt
            self._rebuild_index()

        # Add the new experience
        embedding = experience.market_state_embedding.astype('float32').reshape(1, -1)
        self.index.add(embedding)
        self.experiences.append(experience)

    def recall(self, current_state_embedding: np.ndarray, k: int = 5) -> List[Experience]:
        """
        Finds the K most similar past experiences.
        """
        if not self.experiences:
            return []

        query_embedding = current_state_embedding.astype('float32').reshape(1, -1)

        # Search the FAISS index
        distances, indices = self.index.search(query_embedding, k)

        # Retrieve the corresponding experiences
        recalled_experiences = [self.experiences[i] for i in indices[0] if i < len(self.experiences)]
        return recalled_experiences

    def _rebuild_index(self):
        """
        Rebuilds the FAISS index from the current list of experiences.
        """
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        if not self.experiences:
            return

        all_embeddings = np.array([exp.market_state_embedding for exp in self.experiences]).astype('float32')
        self.index.add(all_embeddings)

    def __len__(self) -> int:
        return len(self.experiences)


# Example usage (for testing)
if __name__ == '__main__':
    import yaml
    from src.risk_predictor import PredictionOutput

    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    memory = ExperienceMemory(config)

    # --- Store some dummy experiences ---
    for i in range(10):
        dummy_embedding = np.random.rand(config['experience_memory']['embedding_dim'])
        dummy_prediction = PredictionOutput(
            expected_return=0.001, volatility=0.01, quantiles={},
            var_95=-0.02, cvar_95=-0.03, sharpe_prediction=1.5, optimal_position=0.5
        )
        exp = Experience(
            market_state_embedding=dummy_embedding,
            prediction=dummy_prediction,
            outcome=0.0015,
            model_architecture=['lstm_256'],
            regime='volatile',
            timestamp=i
        )
        memory.store(exp)

    print(f"Memory size: {len(memory)}")

    # --- Recall similar experiences ---
    current_state = np.random.rand(config['experience_memory']['embedding_dim'])

    recalled = memory.recall(current_state, k=3)

    print(f"\nRecalled {len(recalled)} experiences for the current state.")
    if recalled:
        print("Example recalled experience:")
        print(f"  - Timestamp: {recalled[0].timestamp}")
        print(f"  - Regime: {recalled[0].regime}")
        print(f"  - Outcome: {recalled[0].outcome}")
