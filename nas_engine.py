# src/nas_engine.py

import random
import time
from typing import Dict, List, Tuple

import numpy as np
import ray
import torch
import torch.nn as nn
import torch.optim as optim
from rich.console import Console
from rich.progress import Progress
from torch.distributions import Categorical
import torch.nn.functional as F


console = Console()

# --- Helper Layers ---

class AttentionLayer(nn.Module):
    def __init__(self, in_dim, heads, dim_head):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(in_dim, heads * dim_head * 3, bias=False)
        self.to_out = nn.Linear(heads * dim_head, in_dim)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(t.shape[0], t.shape[1], self.heads, self.dim_head).transpose(1, 2), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(out.shape[0], out.shape[2], -1)
        return self.to_out(out)


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) * dilation, dilation=dilation)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=(kernel_size - 1) * dilation, dilation=dilation)
        self.net = nn.Sequential(self.conv1, nn.BatchNorm1d(out_channels), self.relu, self.conv2, nn.BatchNorm1d(out_channels), nn.ReLU())
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

# --- NAS Engine ---

class ArchitectureController(nn.Module):
    """
    An LSTM-based controller that generates neural network architectures.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, search_space: Dict):
        super(ArchitectureController, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.search_space = search_space

        self.layer_types = list(search_space.keys())
        self.num_layer_types = len(self.layer_types)

        self.embedding = nn.Embedding(self.num_layer_types + 1, input_dim) # +1 for start token
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, self.num_layer_types)

    def forward(self, n: int, max_layers: int = 5) -> Tuple[List[List[str]], List[torch.Tensor]]:
        """Samples a batch of architectures."""
        architectures, log_probs_batch = [], []
        for _ in range(n):
            arch, log_probs = self._sample_architecture(max_layers)
            architectures.append(arch)
            log_probs_batch.append(log_probs)
        return architectures, log_probs_batch

    def _sample_architecture(self, max_layers: int) -> Tuple[List[str], torch.Tensor]:
        arch, log_probs_arch = [], []
        input_token = torch.tensor([[self.num_layer_types]], dtype=torch.long)
        h, c = (torch.zeros(self.num_layers, 1, self.hidden_dim), torch.zeros(self.num_layers, 1, self.hidden_dim))

        for _ in range(max_layers):
            embedded = self.embedding(input_token)
            output, (h, c) = self.lstm(embedded, (h, c))
            logits = self.fc(output.squeeze(1))

            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()

            log_probs_arch.append(dist.log_prob(action))
            layer_type = self.layer_types[action.item()]

            params = {p: random.choice(v) for p, v in self.search_space[layer_type].items()}
            arch.append((layer_type, params))

            input_token = torch.tensor([[action.item()]], dtype=torch.long)

        return arch, torch.stack(log_probs_arch)

    def update(self, log_probs_batch: List[torch.Tensor], rewards: List[float], optimizer: optim.Optimizer):
        """Updates the controller using the REINFORCE algorithm."""
        policy_loss = [-log_probs.sum() * R for log_probs, R in zip(log_probs_batch, rewards)]
        optimizer.zero_grad()
        loss = torch.stack(policy_loss).mean()
        # Add entropy regularization to encourage exploration
        entropy = -torch.stack([torch.exp(lp) * lp for lp_batch in log_probs_batch for lp in lp_batch]).mean()
        (loss - 0.01 * entropy).backward()
        optimizer.step()


class DynamicModel(nn.Module):
    def __init__(self, architecture: List[Tuple[str, Dict]], input_features: int, output_dim: int):
        super().__init__()
        self.layers = nn.ModuleList()
        current_dim = input_features

        for layer_type, params in architecture:
            if layer_type == 'attention':
                self.layers.append(AttentionLayer(current_dim, params['heads'], params['dim']))
            elif layer_type == 'conv1d':
                self.layers.append(nn.Conv1d(current_dim, params['filters'], params['kernel']))
                current_dim = params['filters']
            elif layer_type == 'tcn':
                self.layers.append(TCNBlock(current_dim, params['channels'], kernel_size=3, dilation=2))
                current_dim = params['channels']
            elif layer_type == 'lstm':
                self.layers.append(nn.LSTM(current_dim, params['units'], batch_first=True, bidirectional=params['bidirectional']))
                current_dim = params['units'] * (2 if params['bidirectional'] else 1)

        self.out = nn.Linear(current_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, (nn.LSTM, AttentionLayer)):
                if x.ndim == 2: x = x.unsqueeze(1) # Add sequence dim
            elif isinstance(layer, (nn.Conv1d, TCNBlock)):
                if x.ndim == 2: x = x.unsqueeze(-1) # Add channel dim
                if x.shape[1] != layer.conv1.in_channels:
                    x = x.transpose(1, 2)

            x, *_ = layer(x)
            if isinstance(layer, nn.LSTM) and x.ndim == 3: x = x[:, -1, :] # Take last output
            if isinstance(layer, (nn.Conv1d, TCNBlock)): x = x.transpose(1, 2)

        return self.out(x.squeeze())


class NASEngine:
    def __init__(self, config: Dict):
        self.config = config
        self.search_space = self._define_search_space()
        self.controller = ArchitectureController(128, 256, 2, self.search_space)
        self.controller_optimizer = optim.Adam(self.controller.parameters(), lr=1e-3)
        self.population = []

    def _define_search_space(self) -> Dict:
        return self.config['nas']['search_space']

    def search(self, data: np.ndarray, n_generations: int = 50):
        ray.init(ignore_reinit_error=True, num_cpus=4)
        data_ref = ray.put(data)

        with Progress() as progress:
            task = progress.add_task("[green]Running NAS...", total=n_generations)
            for gen in range(n_generations):
                architectures, log_probs = self.controller(n=20)
                rewards = self._parallel_evaluate(architectures, data_ref)
                self.controller.update(log_probs, rewards, self.controller_optimizer)
                self.population = self._evolve_population(architectures, rewards)
                self._display_progress(gen, rewards)
                progress.update(task, advance=1)
        ray.shutdown()

    def _parallel_evaluate(self, architectures: List[List[Tuple]], data_ref) -> List[float]:
        futures = [self._evaluate_architecture.remote(arch, data_ref) for arch in architectures]
        return ray.get(futures)

    @staticmethod
    @ray.remote
    def _evaluate_architecture(architecture: List[Tuple], data_ref) -> float:
        data = ray.get(data_ref)
        X, y = data[:, :-1], data[:, -1]
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()

        try:
            model = DynamicModel(architecture, X.shape[1], 1)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()

            # Train for a few epochs
            for _ in range(3):
                preds = model(X)
                loss = loss_fn(preds.squeeze(), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Simple reward: inverse of final loss
            return 1.0 / (loss.item() + 1e-6)
        except (RuntimeError, IndexError) as e:
            # Penalize architectures that fail to build or run
            return 0.01

    def _evolve_population(self, archs: List, rewards: List) -> List:
        combined = sorted(self.population + list(zip(archs, rewards)), key=lambda x: x[1], reverse=True)
        return combined[:self.config['nas']['population_size']]

    def _display_progress(self, generation: int, rewards: List[float]):
        console.log(f"Gen {generation+1:02d} | Best Reward: {max(rewards):.4f} | Avg Reward: {np.mean(rewards):.4f}")


# Example usage
if __name__ == '__main__':
    import yaml
    with open("config.yaml", 'r') as f: config = yaml.safe_load(f)
    nas = NASEngine(config)
    dummy_data = np.random.rand(100, 11) # 10 features, 1 target
    nas.search(dummy_data, n_generations=5)
    console.print("\n[bold green]Top Architectures:[/bold green]")
    for arch, reward in nas.population[:5]:
        arch_str = ' -> '.join([f"{l[0]}({l[1]})" for l in arch])
        console.print(f"  - Reward: {reward:.4f}, Arch: {arch_str}")
