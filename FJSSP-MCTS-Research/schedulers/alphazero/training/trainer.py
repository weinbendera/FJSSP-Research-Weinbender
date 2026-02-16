"""
Network trainer for AlphaZero.

Loss = cross_entropy(policy) + MSE(value)
Optimizer: Adam with weight decay.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

from schedulers.alphazero.model.gnn import FJSPNet
from schedulers.alphazero.training.replay_buffer import ReplayBuffer, TrainingExample
from torch_geometric.data import HeteroData


class Trainer:
    """Trains FJSPNet on self-play data."""

    def __init__(
        self,
        model: FJSPNet,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        value_loss_weight: float = 1.0,
        device: str = "cpu",
    ):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.value_loss_weight = value_loss_weight

    def train_batch(self, examples: List[TrainingExample]) -> Dict[str, float]:
        """
        Train on a batch of examples.

        Returns dict with loss metrics.
        """
        self.model.train()
        graphs, policy_targets, value_targets, legal_masks = (
            ReplayBuffer.examples_to_batch(examples)
        )

        total_policy_loss = 0.0
        total_value_loss = 0.0
        batch_size = len(graphs)

        # Process each example individually (heterogeneous graphs can't easily batch)
        self.optimizer.zero_grad()

        for i in range(batch_size):
            graph = graphs[i].to(self.device)
            legal_mask = legal_masks[i].to(self.device)
            policy_target = policy_targets[i].to(self.device)
            value_target = value_targets[i].to(self.device)

            policy, value = self.model(graph, legal_mask)

            # Policy loss: cross-entropy with target distribution
            # Avoid log(0) by clamping
            log_policy = torch.log(policy.clamp(min=1e-8))
            policy_loss = -torch.sum(policy_target * log_policy)

            # Value loss: MSE
            value_loss = F.mse_loss(value, value_target)

            loss = policy_loss + self.value_loss_weight * value_loss
            loss = loss / batch_size  # average over batch
            loss.backward()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        self.optimizer.step()

        return {
            "policy_loss": total_policy_loss / batch_size,
            "value_loss": total_value_loss / batch_size,
            "total_loss": (total_policy_loss + total_value_loss) / batch_size,
        }

    def train_epoch(
        self,
        buffer: ReplayBuffer,
        batch_size: int = 32,
        num_batches: int = 10,
    ) -> Dict[str, float]:
        """Train for multiple batches from replay buffer."""
        total_metrics = {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}

        for _ in range(num_batches):
            examples = buffer.sample(batch_size)
            metrics = self.train_batch(examples)
            for k in total_metrics:
                total_metrics[k] += metrics[k]

        for k in total_metrics:
            total_metrics[k] /= num_batches

        return total_metrics
