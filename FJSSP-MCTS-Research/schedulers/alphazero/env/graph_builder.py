"""
Graph Builder: Converts FJSPEnv state into PyG HeteroData for the GNN.

Node types:
  - "op": Operation nodes with features about status, progress, eligibility
  - "machine": Machine nodes with status, utilization, load

Edge types:
  - ("op", "eligible", "machine"): Dynamic edges from unscheduled ops to eligible machines
  - ("op", "precedes", "op"): Static precedence edges

Global features stored as graph-level attributes.
"""

import numpy as np
import torch
from torch_geometric.data import HeteroData

from schedulers.alphazero.env.config_compiler import CompiledConfig
from schedulers.alphazero.env.fjsp_env import FJSPEnv, UNSCHEDULED, IN_PROGRESS, COMPLETED


class GraphBuilder:
    """Builds PyG HeteroData graphs from FJSPEnv state."""

    def __init__(self, config: CompiledConfig):
        self.config = config

        # Precompute static features
        self._precedence_depth = self._compute_precedence_depth()
        self._num_eligible = self._compute_num_eligible()

        # Precompute static precedence edges
        if config.precedence_pairs.shape[0] > 0:
            self._prec_edge_index = torch.tensor(
                config.precedence_pairs.T, dtype=torch.long
            )  # (2, num_edges)
        else:
            self._prec_edge_index = torch.zeros((2, 0), dtype=torch.long)

        # Max duration for normalization
        self._max_duration = float(max(config.action_duration.max(), 1))
        # Max total power for normalization
        self._max_power = float(max(config.task_mode_total_power.max(), 1.0))
        # Max time for normalization (use last time leap or reasonable upper bound)
        self._max_time = float(config.time_leaps[-1]) if len(config.time_leaps) > 0 else 960.0

    def _compute_precedence_depth(self) -> np.ndarray:
        """Compute precedence depth for each operation (0 = no predecessors)."""
        cfg = self.config
        depth = np.zeros(cfg.num_ops, dtype=np.int16)
        changed = True
        while changed:
            changed = False
            for i in range(cfg.num_ops):
                preds = cfg.op_predecessors[i]
                if len(preds) > 0:
                    new_depth = int(np.max(depth[preds])) + 1
                    if new_depth > depth[i]:
                        depth[i] = new_depth
                        changed = True
        return depth

    def _compute_num_eligible(self) -> np.ndarray:
        """Number of eligible machines per operation."""
        cfg = self.config
        counts = np.zeros(cfg.num_ops, dtype=np.int16)
        for op_idx in range(cfg.num_ops):
            op_actions = np.where(cfg.action_op == op_idx)[0]
            if len(op_actions) > 0:
                counts[op_idx] = len(np.unique(cfg.action_machine[op_actions]))
        return counts

    def build(self, env: FJSPEnv) -> HeteroData:
        """Build HeteroData graph from current env state."""
        cfg = self.config
        data = HeteroData()

        # --- Operation node features (~10 dims) ---
        op_features = self._build_op_features(env)
        data["op"].x = torch.tensor(op_features, dtype=torch.float32)

        # --- Machine node features (~6 dims) ---
        machine_features = self._build_machine_features(env)
        data["machine"].x = torch.tensor(machine_features, dtype=torch.float32)

        # --- Eligible edges (dynamic): unscheduled ops → eligible machines ---
        elig_src, elig_dst = self._build_eligible_edges(env)
        data["op", "eligible", "machine"].edge_index = torch.stack([
            torch.tensor(elig_src, dtype=torch.long),
            torch.tensor(elig_dst, dtype=torch.long),
        ])

        # --- Precedence edges (static) ---
        data["op", "precedes", "op"].edge_index = self._prec_edge_index

        # --- Global features ---
        global_feat = self._build_global_features(env)
        data.global_features = torch.tensor(global_feat, dtype=torch.float32).unsqueeze(0)

        return data

    def _build_op_features(self, env: FJSPEnv) -> np.ndarray:
        """
        Operation features (num_ops, 10):
          0-2: status one-hot (unscheduled, in_progress, completed)
          3: duration_norm (mean duration of eligible task modes / max_duration)
          4: remaining_time_norm
          5: job_progress (fraction of job's ops completed)
          6: precedence_depth_norm
          7: num_eligible_norm
          8: is_ready (all preds done, not assigned)
          9: is_schedulable (ready + job not busy)
        """
        cfg = self.config
        n = cfg.num_ops
        feat = np.zeros((n, 10), dtype=np.float32)

        # Status one-hot
        feat[env.op_status == UNSCHEDULED, 0] = 1.0
        feat[env.op_status == IN_PROGRESS, 1] = 1.0
        feat[env.op_status == COMPLETED, 2] = 1.0

        # Duration norm: mean duration of all action triples for this op
        for op_idx in range(n):
            op_actions = np.where(cfg.action_op == op_idx)[0]
            if len(op_actions) > 0:
                feat[op_idx, 3] = np.mean(cfg.action_duration[op_actions]) / self._max_duration

        # Remaining time norm
        feat[:, 4] = env.op_remaining_time.astype(np.float32) / self._max_duration

        # Job progress
        for job_idx in range(cfg.num_jobs):
            job_ops = np.where(cfg.op_to_job == job_idx)[0]
            if len(job_ops) > 0:
                progress = float(np.sum(env.op_status[job_ops] == COMPLETED)) / len(job_ops)
                feat[job_ops, 5] = progress

        # Precedence depth norm
        max_depth = max(float(self._precedence_depth.max()), 1.0)
        feat[:, 6] = self._precedence_depth.astype(np.float32) / max_depth

        # Num eligible norm
        max_elig = max(float(self._num_eligible.max()), 1.0)
        feat[:, 7] = self._num_eligible.astype(np.float32) / max_elig

        # is_ready and is_schedulable
        for op_idx in range(n):
            if env.op_status[op_idx] != UNSCHEDULED:
                continue
            preds = cfg.op_predecessors[op_idx]
            if len(preds) == 0 or np.all(env.op_status[preds] == COMPLETED):
                feat[op_idx, 8] = 1.0  # is_ready
                job_idx = int(cfg.op_to_job[op_idx])
                if not env.job_being_processed[job_idx]:
                    feat[op_idx, 9] = 1.0  # is_schedulable

        return feat

    def _build_machine_features(self, env: FJSPEnv) -> np.ndarray:
        """
        Machine features (num_machines, 6):
          0-1: status one-hot (idle, busy)
          2: remaining_time_norm
          3: utilization (approximate)
          4: queue_load (number of schedulable ops for this machine, normalized)
          5: is_current (1 if this is the machine currently deciding)
        """
        n = self.config.num_machines
        feat = np.zeros((n, 6), dtype=np.float32)

        # Status one-hot
        feat[~env.machine_busy, 0] = 1.0
        feat[env.machine_busy, 1] = 1.0

        # Remaining time norm
        feat[:, 2] = env.machine_remaining.astype(np.float32) / self._max_duration

        # Utilization (approximate: busy_time / current_time)
        if env.current_time > 0:
            for m_idx in range(n):
                busy_steps = 0
                for op_idx in range(self.config.num_ops):
                    if env.op_assigned_machine[op_idx] == m_idx and env.op_status[op_idx] >= IN_PROGRESS:
                        start = int(env.op_start_step[op_idx])
                        op_actions = np.where(
                            (self.config.action_op == op_idx) &
                            (self.config.action_machine == m_idx) &
                            (self.config.action_task_mode == env.op_assigned_task_mode[op_idx])
                        )[0]
                        if len(op_actions) > 0:
                            dur = int(self.config.action_duration[op_actions[0]])
                            end = min(start + dur, env.current_time)
                            busy_steps += max(0, end - start)
                feat[m_idx, 3] = busy_steps / env.current_time

        # Queue load: count schedulable ops per machine
        for m_idx in range(n):
            count = 0
            m_actions = np.where(self.config.machine_action_mask[m_idx])[0]
            seen_ops = set()
            for a_idx in m_actions:
                op_idx = int(self.config.action_op[a_idx])
                if op_idx in seen_ops:
                    continue
                seen_ops.add(op_idx)
                if env.op_status[op_idx] == UNSCHEDULED:
                    preds = self.config.op_predecessors[op_idx]
                    if len(preds) == 0 or np.all(env.op_status[preds] == COMPLETED):
                        count += 1
            feat[m_idx, 4] = count / max(self.config.num_ops, 1)

        # Is current machine
        feat[env.current_machine_idx, 5] = 1.0

        return feat

    def _build_eligible_edges(self, env: FJSPEnv) -> tuple:
        """Build dynamic eligible edges: unscheduled ops → eligible machines."""
        cfg = self.config
        src_list = []
        dst_list = []

        for a_idx in range(cfg.num_actions):
            op_idx = int(cfg.action_op[a_idx])
            if env.op_status[op_idx] == UNSCHEDULED:
                m_idx = int(cfg.action_machine[a_idx])
                # Avoid duplicate edges
                if len(src_list) == 0 or not (src_list[-1] == op_idx and dst_list[-1] == m_idx):
                    src_list.append(op_idx)
                    dst_list.append(m_idx)

        if not src_list:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        # Deduplicate
        edges = set(zip(src_list, dst_list))
        src = np.array([e[0] for e in edges], dtype=np.int64)
        dst = np.array([e[1] for e in edges], dtype=np.int64)
        return src, dst

    def _build_global_features(self, env: FJSPEnv) -> np.ndarray:
        """
        Global features (5,):
          0: time_norm
          1: overall_progress
          2-4: current_machine one-hot (for 3 machines)
        """
        feat = np.zeros(2 + self.config.num_machines, dtype=np.float32)
        feat[0] = env.current_time / self._max_time
        feat[1] = env.get_progress()
        feat[2 + env.current_machine_idx] = 1.0
        return feat

    @property
    def op_feature_dim(self) -> int:
        return 10

    @property
    def machine_feature_dim(self) -> int:
        return 6

    @property
    def global_feature_dim(self) -> int:
        return 2 + self.config.num_machines
