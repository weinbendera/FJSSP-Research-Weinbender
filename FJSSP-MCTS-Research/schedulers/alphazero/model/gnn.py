"""
FJSPNet: Heterogeneous GNN for AlphaZero FJSP scheduling.

Architecture:
  - Input projection layers for op and machine features
  - 3 layers of HeteroConv(SAGEConv) with residual connections + LayerNorm
  - Policy head: scores (op, task_mode) pairs for current machine + idle
  - Value head: mean pool all embeddings + global features → MLP → tanh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv

from schedulers.alphazero.env.config_compiler import CompiledConfig


class FJSPNet(nn.Module):
    """
    GNN-based policy + value network for FJSP AlphaZero.

    Policy output: logits over all action triples + idle (size num_actions + 1).
    Value output: scalar in [-1, 1].
    """

    def __init__(
        self,
        config: CompiledConfig,
        op_feature_dim: int = 10,
        machine_feature_dim: int = 6,
        global_feature_dim: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 3,
        task_mode_feature_dim: int = 2,
    ):
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim
        self.num_actions = config.num_actions
        self.task_mode_feature_dim = task_mode_feature_dim

        # --- Input projections ---
        self.op_proj = nn.Linear(op_feature_dim, hidden_dim)
        self.machine_proj = nn.Linear(machine_feature_dim, hidden_dim)

        # --- HeteroConv layers with residual + LayerNorm ---
        self.convs = nn.ModuleList()
        self.norms_op = nn.ModuleList()
        self.norms_machine = nn.ModuleList()

        for _ in range(num_layers):
            conv = HeteroConv({
                ("op", "eligible", "machine"): SAGEConv(hidden_dim, hidden_dim),
                ("machine", "rev_eligible", "op"): SAGEConv(hidden_dim, hidden_dim),
                ("op", "precedes", "op"): SAGEConv(hidden_dim, hidden_dim),
            }, aggr="sum")
            self.convs.append(conv)
            self.norms_op.append(nn.LayerNorm(hidden_dim))
            self.norms_machine.append(nn.LayerNorm(hidden_dim))

        # --- Policy head ---
        # For each action: concat [op_emb, machine_emb, task_mode_features] → MLP → scalar
        policy_input_dim = hidden_dim * 2 + task_mode_feature_dim
        self.policy_mlp = nn.Sequential(
            nn.Linear(policy_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Learnable idle embedding
        self.idle_embedding = nn.Parameter(torch.randn(policy_input_dim))

        # --- Value head ---
        # mean pool op + machine embeddings + global features → MLP → tanh
        value_input_dim = hidden_dim * 2 + global_feature_dim
        self.value_mlp = nn.Sequential(
            nn.Linear(value_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

        # --- Precompute task mode features (static) ---
        # Normalized duration and total power for each task mode
        tm_dur = torch.tensor(config.task_mode_duration, dtype=torch.float32)
        tm_pow = torch.tensor(config.task_mode_total_power, dtype=torch.float32)
        max_dur = tm_dur.max().clamp(min=1.0)
        max_pow = tm_pow.max().clamp(min=1.0)
        self.register_buffer("tm_features", torch.stack([
            tm_dur / max_dur,
            tm_pow / max_pow,
        ], dim=-1))  # (num_task_modes, 2)

        # Precompute action → op/machine/task_mode indices as tensors
        self.register_buffer("action_op_idx",
            torch.tensor(config.action_op, dtype=torch.long))
        self.register_buffer("action_machine_idx",
            torch.tensor(config.action_machine, dtype=torch.long))
        self.register_buffer("action_tm_idx",
            torch.tensor(config.action_task_mode, dtype=torch.long))

    def forward(self, data: HeteroData, legal_mask: torch.Tensor) -> tuple:
        """
        Forward pass.

        Args:
            data: HeteroData with "op" and "machine" node features, edges.
            legal_mask: (num_actions + 1,) boolean mask of legal actions.

        Returns:
            policy: (num_actions + 1,) probability distribution (masked softmax).
            value: scalar in [-1, 1].
        """
        # --- Input projection ---
        x_op = F.relu(self.op_proj(data["op"].x))
        x_machine = F.relu(self.machine_proj(data["machine"].x))

        x_dict = {"op": x_op, "machine": x_machine}

        # Add reverse edges for message passing
        edge_index_dict = {}
        if ("op", "eligible", "machine") in data.edge_types:
            ei = data["op", "eligible", "machine"].edge_index
            edge_index_dict[("op", "eligible", "machine")] = ei
            edge_index_dict[("machine", "rev_eligible", "op")] = ei.flip(0)
        if ("op", "precedes", "op") in data.edge_types:
            edge_index_dict[("op", "precedes", "op")] = data["op", "precedes", "op"].edge_index

        # --- Message passing layers ---
        for i, conv in enumerate(self.convs):
            out_dict = conv(x_dict, edge_index_dict)
            # Residual + LayerNorm
            if "op" in out_dict:
                x_dict["op"] = self.norms_op[i](x_dict["op"] + out_dict["op"])
            if "machine" in out_dict:
                x_dict["machine"] = self.norms_machine[i](x_dict["machine"] + out_dict["machine"])

        op_emb = x_dict["op"]        # (num_ops, hidden_dim)
        machine_emb = x_dict["machine"]  # (num_machines, hidden_dim)

        # --- Policy head ---
        # Build input for each action: [op_emb[op_idx], machine_emb[machine_idx], tm_features[tm_idx]]
        action_op_emb = op_emb[self.action_op_idx]         # (num_actions, hidden_dim)
        action_machine_emb = machine_emb[self.action_machine_idx]  # (num_actions, hidden_dim)
        action_tm_feat = self.tm_features[self.action_tm_idx]      # (num_actions, 2)

        policy_input = torch.cat([action_op_emb, action_machine_emb, action_tm_feat], dim=-1)
        action_logits = self.policy_mlp(policy_input).squeeze(-1)  # (num_actions,)

        # Idle logit
        idle_logit = self.policy_mlp(self.idle_embedding.unsqueeze(0)).squeeze()  # scalar

        # Combine: (num_actions + 1,)
        all_logits = torch.cat([action_logits, idle_logit.unsqueeze(0)])

        # Apply legal mask: -inf for illegal actions
        all_logits = all_logits.masked_fill(~legal_mask, float("-inf"))
        policy = F.softmax(all_logits, dim=-1)

        # --- Value head ---
        op_pool = op_emb.mean(dim=0)        # (hidden_dim,)
        machine_pool = machine_emb.mean(dim=0)  # (hidden_dim,)
        global_feat = data.global_features.squeeze(0)  # (global_feature_dim,)

        value_input = torch.cat([op_pool, machine_pool, global_feat])
        value = self.value_mlp(value_input).squeeze()  # scalar

        return policy, value
