"""
AlphaZero Online Scheduler: wraps trained GNN+MCTS model as an OnlineScheduler.

Integrates with the existing Factory simulation loop by translating between
string-keyed domain objects and the integer-indexed numpy environment.
"""

import torch
import numpy as np
from typing import Dict, List, Optional

from schedulers.scheduler import OnlineScheduler, ScheduledOperation, Schedule
from utils.factory_logic_loader import FactoryLogic
from utils.job_builder import JobBuilder, Job, Operation
from utils.input_schemas import EnergySource, ProductRequest

from schedulers.alphazero.env.config_compiler import CompiledConfig
from schedulers.alphazero.env.fjsp_env import FJSPEnv, UNSCHEDULED
from schedulers.alphazero.env.graph_builder import GraphBuilder
from schedulers.alphazero.model.gnn import FJSPNet
from schedulers.alphazero.mcts.mcts import MCTS, MCTSConfig


class AlphaZeroScheduler(OnlineScheduler):
    """
    OnlineScheduler that uses a trained AlphaZero model to make scheduling decisions.

    Can operate in two modes:
    - MCTS mode: full MCTS search (slower, better quality)
    - Policy mode: just use the policy network (fast inference)
    """

    def __init__(
        self,
        factory_logic: FactoryLogic,
        product_requests: List[ProductRequest],
        model_path: Optional[str] = None,
        use_mcts: bool = True,
        mcts_config: MCTSConfig = MCTSConfig(),
        hidden_dim: int = 64,
        num_gnn_layers: int = 3,
        device: str = "cpu",
    ):
        super().__init__()
        self.factory_logic = factory_logic
        self.device = device
        self.use_mcts = use_mcts
        self.mcts_config = mcts_config

        # Build jobs and compile config
        jobs = JobBuilder(factory_logic).build_jobs(product_requests)
        self.config = CompiledConfig.compile(factory_logic, jobs)

        # Build env and graph builder
        self.env = FJSPEnv(self.config)
        self.graph_builder = GraphBuilder(self.config)

        # Build model
        self.model = FJSPNet(
            config=self.config,
            op_feature_dim=self.graph_builder.op_feature_dim,
            machine_feature_dim=self.graph_builder.machine_feature_dim,
            global_feature_dim=self.graph_builder.global_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
        ).to(device)

        # Load trained weights
        if model_path is not None:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Track scheduling step for internal env sync
        self._step_count = 0

    def choose(
        self, jobs: List[Job], energy_sources: List[EnergySource]
    ) -> Dict[str, Optional[ScheduledOperation]]:
        """
        Choose actions for all idle machines.

        Translates string-keyed job state into the numpy env, runs MCTS or policy
        for each machine sequentially, then translates back to ScheduledOperations.
        """
        result: Dict[str, Optional[ScheduledOperation]] = {}

        for machine_id in self.config.machine_ids:
            m_idx = self.config.machine_id_to_idx[machine_id]

            # Sync env's current_machine_idx
            self.env.current_machine_idx = m_idx

            if self.env.machine_busy[m_idx]:
                result[machine_id] = None
                continue

            if self.use_mcts:
                action = self._mcts_action()
            else:
                action = self._policy_action()

            if action == self.env.num_actions:
                # Idle
                result[machine_id] = None
            else:
                # Translate action to ScheduledOperation
                op_idx = int(self.config.action_op[action])
                tm_idx = int(self.config.action_task_mode[action])
                duration = int(self.config.action_duration[action])

                result[machine_id] = ScheduledOperation(
                    operation_id=self.config.op_ids[op_idx],
                    job_id=self.config.job_ids[int(self.config.op_to_job[op_idx])],
                    task_id=self.config.op_task_ids[op_idx],
                    machine_id=machine_id,
                    task_mode_id=self.config.task_mode_ids[tm_idx],
                    start_step=self.env.current_time,
                    end_step=self.env.current_time + duration,
                )

            # Apply action to internal env
            self.env.step(action)

        self._step_count += 1
        return result

    def _mcts_action(self) -> int:
        """Get action via full MCTS search."""
        mcts = MCTS(
            self.env, self.graph_builder, self.model,
            self.mcts_config, self.device,
        )
        state = self.env.get_state()
        visit_counts = mcts.search(state, self._step_count)
        return mcts.select_action(visit_counts, self._step_count)

    def _policy_action(self) -> int:
        """Get action using just the policy network (no MCTS search)."""
        legal_mask = self.env.get_legal_actions()
        graph = self.graph_builder.build(self.env)
        legal_tensor = torch.tensor(legal_mask, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            graph = graph.to(self.device)
            policy, _ = self.model(graph, legal_tensor)
            policy = policy.cpu().numpy()

        return int(np.argmax(policy))

    def schedule(
        self, jobs: List[Job], energy_sources: List[EnergySource]
    ) -> Schedule:
        """
        Run the full schedule using the internal env (standalone mode).

        This bypasses the Factory class entirely - useful for evaluation.
        """
        self.env.reset()
        self._step_count = 0
        all_scheduled = []

        while not self.env.done:
            actions = self.choose(jobs, energy_sources)
            for machine_id, sched_op in actions.items():
                if sched_op is not None:
                    all_scheduled.append(sched_op)

        return Schedule(
            operations=all_scheduled,
            makespan=self.env.get_makespan(),
        )
