"""
Replay buffer for AlphaZero training.

Stores training examples as numpy arrays. Rebuilds PyG HeteroData during batch
construction to avoid storing heavy graph objects.
"""

import random
import numpy as np
import torch
from torch_geometric.data import HeteroData
from typing import List, Tuple

from schedulers.alphazero.training.self_play import TrainingExample


class ReplayBuffer:
    """Fixed-size replay buffer storing TrainingExample objects."""

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.buffer: List[TrainingExample] = []
        self.position = 0

    def add(self, examples: List[TrainingExample]) -> None:
        """Add a batch of training examples."""
        for ex in examples:
            if len(self.buffer) < self.capacity:
                self.buffer.append(ex)
            else:
                self.buffer[self.position] = ex
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[TrainingExample]:
        """Sample a random batch of examples."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)

    @staticmethod
    def examples_to_batch(
        examples: List[TrainingExample],
    ) -> Tuple[List[HeteroData], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert list of TrainingExamples to training batch.

        Returns:
            graphs: list of HeteroData (one per example)
            policy_targets: (batch_size, num_actions + 1) tensor
            value_targets: (batch_size,) tensor
            legal_masks: (batch_size, num_actions + 1) tensor
        """
        graphs = []
        policy_targets = []
        value_targets = []
        legal_masks = []

        for ex in examples:
            data = HeteroData()
            data["op"].x = torch.tensor(ex.op_features, dtype=torch.float32)
            data["machine"].x = torch.tensor(ex.machine_features, dtype=torch.float32)
            data.global_features = torch.tensor(
                ex.global_features, dtype=torch.float32
            ).unsqueeze(0)

            # Eligible edges
            if len(ex.eligible_edge_src) > 0:
                data["op", "eligible", "machine"].edge_index = torch.stack([
                    torch.tensor(ex.eligible_edge_src, dtype=torch.long),
                    torch.tensor(ex.eligible_edge_dst, dtype=torch.long),
                ])
            else:
                data["op", "eligible", "machine"].edge_index = torch.zeros(
                    (2, 0), dtype=torch.long
                )

            # Precedence edges
            data["op", "precedes", "op"].edge_index = torch.tensor(
                ex.precedes_edge_index, dtype=torch.long
            )

            graphs.append(data)
            policy_targets.append(torch.tensor(ex.policy_target, dtype=torch.float32))
            value_targets.append(ex.value_target)
            legal_masks.append(torch.tensor(ex.legal_mask, dtype=torch.bool))

        return (
            graphs,
            torch.stack(policy_targets),
            torch.tensor(value_targets, dtype=torch.float32),
            torch.stack(legal_masks),
        )
