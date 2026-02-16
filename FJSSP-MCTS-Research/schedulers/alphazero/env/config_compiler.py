"""
Config Compiler: Converts FactoryLogic + Jobs into integer-indexed numpy arrays.

This is the bridge between the string-keyed domain objects (FactoryLogic, Job, Operation)
and the fast numpy environment. All lookups become integer indexing.

Usage:
    factory_logic = FactoryLogicLoader.load_from_file("data/Input_JSON_Schedule_Optimization.json")
    jobs = JobBuilder(factory_logic).build_jobs(product_requests)
    config = CompiledConfig.compile(factory_logic, jobs)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from utils.factory_logic_loader import FactoryLogic
from utils.job_builder import Job, Operation


@dataclass
class CompiledConfig:
    """Integer-indexed numpy arrays compiled from FactoryLogic + Jobs."""

    # --- Dimensions ---
    num_ops: int
    num_jobs: int
    num_machines: int
    num_task_modes: int
    num_tasks: int
    num_actions: int  # total valid (op, machine, task_mode) triples

    # --- Operation → Job mapping ---
    op_to_job: np.ndarray       # (num_ops,) int16 — job index for each operation
    op_to_task: np.ndarray      # (num_ops,) int16 — task index for each operation
    op_deadline: np.ndarray     # (num_ops,) int16 — deadline per op (-1 = no deadline)

    # --- Action table: all valid (op, machine, task_mode) triples ---
    action_op: np.ndarray       # (num_actions,) int16 — operation index
    action_machine: np.ndarray  # (num_actions,) int8  — machine index
    action_task_mode: np.ndarray  # (num_actions,) int16 — task mode index
    action_duration: np.ndarray   # (num_actions,) int16 — duration in steps

    # --- Per-machine action filtering ---
    # machine_action_mask[m] is a boolean mask over the action table for machine m
    machine_action_mask: np.ndarray  # (num_machines, num_actions) bool

    # --- Task mode features (for GNN policy head) ---
    task_mode_duration: np.ndarray     # (num_task_modes,) int16
    task_mode_total_power: np.ndarray  # (num_task_modes,) float32

    # --- Precedence constraints ---
    # Each row is (predecessor_op_idx, successor_op_idx)
    precedence_pairs: np.ndarray  # (num_precedence, 2) int16
    # op_predecessors[op] = list of predecessor op indices (ragged, stored as list of arrays)
    op_predecessors: List[np.ndarray]

    # --- Collision constraints ---
    # Pairs of task indices that cannot run simultaneously
    collision_pairs: np.ndarray  # (num_collisions, 2) int16

    # --- Time leaps ---
    time_leaps: np.ndarray  # (num_leaps,) int16

    # --- String ID lookups (for converting back to Schedule output) ---
    op_ids: List[str]           # op_index → operation string ID
    job_ids: List[str]          # job_index → job string ID
    machine_ids: List[str]      # machine_index → machine string ID
    task_mode_ids: List[str]    # task_mode_index → task mode string ID
    task_ids: List[str]         # task_index → task string ID
    op_task_ids: List[str]      # op_index → task string ID (for the operation)

    # --- Value normalization ---
    makespan_ub: float = 1.0  # upper bound for normalizing value targets to [-1, 0]

    # --- Reverse lookups ---
    op_id_to_idx: Dict[str, int] = field(default_factory=dict)
    job_id_to_idx: Dict[str, int] = field(default_factory=dict)
    machine_id_to_idx: Dict[str, int] = field(default_factory=dict)
    task_mode_id_to_idx: Dict[str, int] = field(default_factory=dict)
    task_id_to_idx: Dict[str, int] = field(default_factory=dict)

    @staticmethod
    def compile(factory_logic: FactoryLogic, jobs: List[Job]) -> "CompiledConfig":
        """Compile FactoryLogic + Jobs into integer-indexed numpy arrays."""

        # --- Build index mappings ---
        machine_ids = sorted(factory_logic.machines.keys())
        machine_id_to_idx = {mid: i for i, mid in enumerate(machine_ids)}
        num_machines = len(machine_ids)

        task_mode_ids = sorted(factory_logic.task_modes.keys())
        task_mode_id_to_idx = {tmid: i for i, tmid in enumerate(task_mode_ids)}
        num_task_modes = len(task_mode_ids)

        task_ids = sorted(factory_logic.tasks.keys())
        task_id_to_idx = {tid: i for i, tid in enumerate(task_ids)}
        num_tasks = len(task_ids)

        job_ids = [j.id for j in jobs]
        job_id_to_idx = {jid: i for i, jid in enumerate(job_ids)}
        num_jobs = len(jobs)

        # Flatten all operations across all jobs
        all_ops: List[Operation] = []
        for job in jobs:
            all_ops.extend(job.operations)
        num_ops = len(all_ops)

        op_ids = [op.id for op in all_ops]
        op_id_to_idx = {oid: i for i, oid in enumerate(op_ids)}

        # --- op_to_job, op_to_task, op_deadline ---
        op_to_job = np.zeros(num_ops, dtype=np.int16)
        op_to_task = np.zeros(num_ops, dtype=np.int16)
        op_deadline = np.full(num_ops, -1, dtype=np.int16)
        op_task_ids_list = []

        for i, op in enumerate(all_ops):
            op_to_job[i] = job_id_to_idx[op.job_id]
            op_to_task[i] = task_id_to_idx[op.task_id]
            op_task_ids_list.append(op.task_id)
            if op.deadline is not None:
                op_deadline[i] = op.deadline

        # --- Action table: enumerate all valid (op, machine, task_mode) triples ---
        action_ops = []
        action_machines = []
        action_task_modes = []
        action_durations = []

        for op_idx, op in enumerate(all_ops):
            for machine_id in sorted(op.eligible_machines):
                m_idx = machine_id_to_idx[machine_id]
                for tm_id in sorted(op.eligible_task_modes.get(machine_id, [])):
                    tm_idx = task_mode_id_to_idx[tm_id]
                    duration = len(factory_logic.task_modes[tm_id].power)
                    action_ops.append(op_idx)
                    action_machines.append(m_idx)
                    action_task_modes.append(tm_idx)
                    action_durations.append(duration)

        num_actions = len(action_ops)
        action_op = np.array(action_ops, dtype=np.int16)
        action_machine = np.array(action_machines, dtype=np.int8)
        action_task_mode = np.array(action_task_modes, dtype=np.int16)
        action_duration = np.array(action_durations, dtype=np.int16)

        # --- Per-machine action mask ---
        machine_action_mask = np.zeros((num_machines, num_actions), dtype=bool)
        for a_idx in range(num_actions):
            machine_action_mask[action_machine[a_idx], a_idx] = True

        # --- Task mode features ---
        tm_duration = np.zeros(num_task_modes, dtype=np.int16)
        tm_total_power = np.zeros(num_task_modes, dtype=np.float32)
        for tm_id, tm in factory_logic.task_modes.items():
            idx = task_mode_id_to_idx[tm_id]
            tm_duration[idx] = len(tm.power)
            tm_total_power[idx] = sum(tm.power)

        # --- Precedence constraints ---
        prec_pairs = []
        for job in jobs:
            for pred_id, succ_id in job.precedence_constraints:
                pred_idx = op_id_to_idx[pred_id]
                succ_idx = op_id_to_idx[succ_id]
                prec_pairs.append((pred_idx, succ_idx))

        if prec_pairs:
            precedence_pairs = np.array(prec_pairs, dtype=np.int16)
        else:
            precedence_pairs = np.zeros((0, 2), dtype=np.int16)

        # Build op_predecessors: for each op, list of predecessor op indices
        op_predecessors = [np.array([], dtype=np.int16) for _ in range(num_ops)]
        for pred_idx, succ_idx in prec_pairs:
            op_predecessors[succ_idx] = np.append(op_predecessors[succ_idx], pred_idx)

        # --- Collision constraints ---
        coll_pairs = []
        for cc in factory_logic.collision_constraints:
            if cc.task1 in task_id_to_idx and cc.task2 in task_id_to_idx:
                t1 = task_id_to_idx[cc.task1]
                t2 = task_id_to_idx[cc.task2]
                coll_pairs.append((t1, t2))

        if coll_pairs:
            collision_pairs = np.array(coll_pairs, dtype=np.int16)
        else:
            collision_pairs = np.zeros((0, 2), dtype=np.int16)

        # --- Time leaps ---
        time_leaps = np.array(factory_logic.time_leaps, dtype=np.int16)

        # --- Makespan upper bound (for value normalization) ---
        # Sum of max duration per op (worst case: all ops serial with slowest mode)
        max_dur_per_op = np.zeros(num_ops, dtype=np.float64)
        for a_idx in range(num_actions):
            op_idx = int(action_op[a_idx])
            max_dur_per_op[op_idx] = max(max_dur_per_op[op_idx], action_duration[a_idx])
        makespan_ub = float(max_dur_per_op.sum())

        return CompiledConfig(
            num_ops=num_ops,
            num_jobs=num_jobs,
            num_machines=num_machines,
            num_task_modes=num_task_modes,
            num_tasks=num_tasks,
            num_actions=num_actions,
            op_to_job=op_to_job,
            op_to_task=op_to_task,
            op_deadline=op_deadline,
            action_op=action_op,
            action_machine=action_machine,
            action_task_mode=action_task_mode,
            action_duration=action_duration,
            machine_action_mask=machine_action_mask,
            task_mode_duration=tm_duration,
            task_mode_total_power=tm_total_power,
            precedence_pairs=precedence_pairs,
            op_predecessors=op_predecessors,
            collision_pairs=collision_pairs,
            makespan_ub=makespan_ub,
            time_leaps=time_leaps,
            op_ids=op_ids,
            job_ids=job_ids,
            machine_ids=machine_ids,
            task_mode_ids=task_mode_ids,
            task_ids=task_ids,
            op_task_ids=op_task_ids_list,
            op_id_to_idx=op_id_to_idx,
            job_id_to_idx=job_id_to_idx,
            machine_id_to_idx=machine_id_to_idx,
            task_mode_id_to_idx=task_mode_id_to_idx,
            task_id_to_idx=task_id_to_idx,
        )

    def get_action_index(self, op_idx: int, machine_idx: int, task_mode_idx: int) -> Optional[int]:
        """Find the action index for a given (op, machine, task_mode) triple."""
        mask = (
            (self.action_op == op_idx) &
            (self.action_machine == machine_idx) &
            (self.action_task_mode == task_mode_idx)
        )
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return None
        return int(indices[0])

    def get_actions_for_op(self, op_idx: int) -> np.ndarray:
        """Get all action indices for a given operation."""
        return np.where(self.action_op == op_idx)[0]

    def get_actions_for_machine(self, machine_idx: int) -> np.ndarray:
        """Get all action indices for a given machine."""
        return np.where(self.machine_action_mask[machine_idx])[0]
