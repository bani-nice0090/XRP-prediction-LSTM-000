# src/online_learner.py

from collections import deque
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer:
    """
    A simple replay buffer to store experiences for EWC and retraining.
    """
    def __init__(self, max_size: int):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience: Tuple[np.ndarray, np.ndarray]):
        """Adds an experience to the buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Samples a batch of experiences from the buffer."""
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def __len__(self) -> int:
        return len(self.buffer)


class OnlineLearner:
    """
    Handles continuous learning without catastrophic forgetting using EWC.
    """
    def __init__(self, model: nn.Module, config: Dict):
        self.model = model
        self.config = config
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn = nn.MSELoss() # Assuming a regression task

        self.replay_buffer = ReplayBuffer(max_size=config['online_learner']['replay_buffer_size'])
        self.ewc_lambda = config['online_learner']['ewc_lambda']

        # EWC components
        self.fisher_matrix = None
        self.optimal_params = None

    def update(self, new_data: Tuple[np.ndarray, np.ndarray]):
        """
        Updates the model with a new batch of data using EWC.

        Args:
            new_data: A tuple containing (X, y) for the new task/data.
        """
        X_new, y_new = new_data
        X_new = torch.from_numpy(X_new).float()
        y_new = torch.from_numpy(y_new).float().unsqueeze(1)

        # 1. Standard task loss on the new data
        self.model.train()
        self.optimizer.zero_grad()

        predictions = self.model(X_new)
        task_loss = self.loss_fn(predictions, y_new)

        # 2. EWC penalty
        ewc_penalty = self._ewc_penalty()

        # 3. Combine losses
        total_loss = task_loss + self.ewc_lambda * ewc_penalty

        total_loss.backward()
        self.optimizer.step()

        # 4. Add new data to the replay buffer
        for i in range(len(X_new)):
            self.replay_buffer.add((new_data[0][i], new_data[1][i]))

    def _ewc_penalty(self) -> torch.Tensor:
        """
        Calculates the EWC penalty term.
        """
        if self.fisher_matrix is None or self.optimal_params is None:
            return torch.tensor(0.0)

        penalty = 0.0
        for (name, param), fisher_val, opt_param in zip(self.model.named_parameters(), self.fisher_matrix.values(), self.optimal_params.values()):
            if param.requires_grad:
                penalty += (fisher_val * (param - opt_param).pow(2)).sum()

        return penalty

    def compute_fisher_matrix(self):
        """
        Computes the Fisher Information Matrix for the current model.
        This should be called after a task is learned, to consolidate knowledge.
        """
        # 1. Store the current optimal parameters
        self.optimal_params = {name: param.clone().detach() for name, param in self.model.named_parameters() if param.requires_grad}

        # 2. Initialize Fisher matrix
        self.fisher_matrix = {name: torch.zeros_like(param) for name, param in self.model.named_parameters() if param.requires_grad}

        # 3. Use data from the replay buffer to compute Fisher
        if len(self.replay_buffer) == 0:
            return

        self.model.eval()
        dataset = self.replay_buffer.sample(batch_size=min(256, len(self.replay_buffer)))

        X_sample = torch.from_numpy(np.array([s[0] for s in dataset])).float()

        for i in range(len(X_sample)):
            self.optimizer.zero_grad()
            output = self.model(X_sample[i:i+1])

            # Use log-likelihood of the output. For MSE, this is proportional to the negative loss.
            # We are sampling from the model's predictive distribution.
            log_likelihood = -self.loss_fn(output, self.model(X_sample[i:i+1]))
            log_likelihood.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher_matrix[name] += param.grad.pow(2)

        # Average the Fisher matrix
        num_samples = len(X_sample)
        for name in self.fisher_matrix:
            self.fisher_matrix[name] /= num_samples


# Example usage (for testing)
if __name__ == '__main__':
    import yaml

    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Create a dummy model
    dummy_model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))

    online_learner = OnlineLearner(dummy_model, config)

    # --- Task 1: Learn to predict y = sum(x) ---
    X1 = np.random.rand(100, 10)
    y1 = np.sum(X1, axis=1)

    print("--- Training on Task 1 ---")
    for _ in range(10): # 10 epochs
        online_learner.update((X1, y1))

    # Test performance on Task 1
    with torch.no_grad():
        preds1 = dummy_model(torch.from_numpy(X1).float())
        loss1 = online_learner.loss_fn(preds1, torch.from_numpy(y1).float().unsqueeze(1))
        print(f"Loss on Task 1 after training: {loss1.item():.4f}")

    # Consolidate knowledge from Task 1
    print("\n--- Consolidating knowledge (computing Fisher matrix) ---")
    online_learner.compute_fisher_matrix()

    # --- Task 2: Learn to predict y = x[0] ---
    X2 = np.random.rand(100, 10)
    y2 = X2[:, 0]

    print("\n--- Training on Task 2 (with EWC) ---")
    for _ in range(10): # 10 epochs
        online_learner.update((X2, y2))

    # Test performance on Task 2
    with torch.no_grad():
        preds2 = dummy_model(torch.from_numpy(X2).float())
        loss2 = online_learner.loss_fn(preds2, torch.from_numpy(y2).float().unsqueeze(1))
        print(f"Loss on Task 2 after training: {loss2.item():.4f}")

    # Test performance on Task 1 AGAIN (to check for forgetting)
    with torch.no_grad():
        preds1_after = dummy_model(torch.from_numpy(X1).float())
        loss1_after = online_learner.loss_fn(preds1_after, torch.from_numpy(y1).float().unsqueeze(1))
        print(f"\nLoss on Task 1 after training on Task 2: {loss1_after.item():.4f}")

    assert loss1_after.item() < loss1.item() * 5 # Should not have forgotten too much
