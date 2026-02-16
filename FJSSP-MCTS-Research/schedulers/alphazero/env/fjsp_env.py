"""
Fast Numpy FJSP Environment for AlphaZero MCTS.

Sequential machine decisions: at each step, machines M0→M1→M2 each choose an action.
After all machines decide, time advances, remaining times decrement, completed ops are freed.

State is pure numpy arrays (~2.5KB for 400 ops, 3 machines) enabling fast get_state/set_state
for MCTS tree search.

Action space: integer index into CompiledConfig.action_table, or num_actions = idle.
"""

import numpy as np
from typing import Tuple, Optional, NamedTuple
from schedulers.alphazero.env.config_compiler import CompiledConfig

# Operation status codes
UNSCHEDULED = 0
IN_PROGRESS = 1
COMPLETED = 2


class EnvState(NamedTuple):
    """Immutable snapshot of mutable env state for MCTS get_state/set_state."""
    op_status: np.ndarray
    op_remaining_time: np.ndarray
    op_assigned_machine: np.ndarray
    op_assigned_task_mode: np.ndarray
    op_start_step: np.ndarray
    op_duration: np.ndarray
    machine_busy: np.ndarray
    machine_current_op: np.ndarray
    machine_remaining: np.ndarray
    job_being_processed: np.ndarray
    current_time: int
    current_machine_idx: int
    done: bool


class FJSPEnv:
    """
    Fast numpy FJSP environment.

    Decision model:
        - At each "decision point", we cycle through machines M0→M1→M2.
        - Each machine picks an action (assign an op or idle).
        - After all machines have decided, time advances by 1 step.
        - Machines that finish their current op are freed.
        - Repeat until all ops are completed.
    """

    IDLE_ACTION = -1  # Sentinel for "do nothing"

    def __init__(self, config: CompiledConfig):
        self.config = config
        self.num_ops = config.num_ops
        self.num_machines = config.num_machines
        self.num_actions = config.num_actions  # valid triples count

        # Precompute collision lookup: task_idx → set of conflicting task_indices
        self._collision_map = {}
        for i in range(config.collision_pairs.shape[0]):
            t1, t2 = int(config.collision_pairs[i, 0]), int(config.collision_pairs[i, 1])
            self._collision_map.setdefault(t1, set()).add(t2)
            self._collision_map.setdefault(t2, set()).add(t1)

        # Precompute: for each op, the set of action indices
        self._op_action_indices = []
        for op_idx in range(self.num_ops):
            self._op_action_indices.append(
                np.where(config.action_op == op_idx)[0]
            )

        self.reset()

    def reset(self) -> None:
        """Reset environment to initial state."""
        self.op_status = np.zeros(self.num_ops, dtype=np.int8)
        self.op_remaining_time = np.zeros(self.num_ops, dtype=np.int16)
        self.op_assigned_machine = np.full(self.num_ops, -1, dtype=np.int8)
        self.op_assigned_task_mode = np.full(self.num_ops, -1, dtype=np.int16)
        self.op_start_step = np.full(self.num_ops, -1, dtype=np.int16)
        self.op_duration = np.zeros(self.num_ops, dtype=np.int16)

        self.machine_busy = np.zeros(self.num_machines, dtype=bool)
        self.machine_current_op = np.full(self.num_machines, -1, dtype=np.int16)
        self.machine_remaining = np.zeros(self.num_machines, dtype=np.int16)

        self.job_being_processed = np.zeros(self.config.num_jobs, dtype=bool)

        self.current_time = 0
        self.current_machine_idx = 0
        self._done = False

    def get_state(self) -> EnvState:
        """Snapshot current state (~2.5KB copy)."""
        return EnvState(
            op_status=self.op_status.copy(),
            op_remaining_time=self.op_remaining_time.copy(),
            op_assigned_machine=self.op_assigned_machine.copy(),
            op_assigned_task_mode=self.op_assigned_task_mode.copy(),
            op_start_step=self.op_start_step.copy(),
            op_duration=self.op_duration.copy(),
            machine_busy=self.machine_busy.copy(),
            machine_current_op=self.machine_current_op.copy(),
            machine_remaining=self.machine_remaining.copy(),
            job_being_processed=self.job_being_processed.copy(),
            current_time=self.current_time,
            current_machine_idx=self.current_machine_idx,
            done=self._done,
        )

    def set_state(self, state: EnvState) -> None:
        """Restore state from snapshot."""
        self.op_status = state.op_status.copy()
        self.op_remaining_time = state.op_remaining_time.copy()
        self.op_assigned_machine = state.op_assigned_machine.copy()
        self.op_assigned_task_mode = state.op_assigned_task_mode.copy()
        self.op_start_step = state.op_start_step.copy()
        self.op_duration = state.op_duration.copy()
        self.machine_busy = state.machine_busy.copy()
        self.machine_current_op = state.machine_current_op.copy()
        self.machine_remaining = state.machine_remaining.copy()
        self.job_being_processed = state.job_being_processed.copy()
        self.current_time = state.current_time
        self.current_machine_idx = state.current_machine_idx
        self._done = state.done

    @property
    def done(self) -> bool:
        return self._done

    def get_legal_actions(self) -> np.ndarray:
        """
        Returns boolean mask of shape (num_actions + 1,).
        Index num_actions = idle action.

        An action (op, machine, task_mode) is legal if:
        1. The action is for the current machine
        2. The operation is unscheduled
        3. All predecessors are completed
        4. The job is not currently being processed (one-op-per-job-at-a-time)
        5. No collision conflict with currently running ops
        6. The operation won't cross a time leap boundary
        7. Deadline not already passed (if applicable)
        """
        cfg = self.config
        m_idx = self.current_machine_idx
        mask = np.zeros(self.num_actions + 1, dtype=bool)

        # If machine is busy, only idle is legal
        if self.machine_busy[m_idx]:
            mask[self.num_actions] = True  # idle
            return mask

        # Get actions for this machine
        machine_actions = np.where(cfg.machine_action_mask[m_idx])[0]

        # Get currently running task types for collision check
        running_tasks = set()
        for op_idx in range(self.num_ops):
            if self.op_status[op_idx] == IN_PROGRESS:
                running_tasks.add(int(cfg.op_to_task[op_idx]))

        # Next time leap
        next_leap = self._get_next_time_leap()

        for a_idx in machine_actions:
            op_idx = int(cfg.action_op[a_idx])
            duration = int(cfg.action_duration[a_idx])

            # 1. Op must be unscheduled
            if self.op_status[op_idx] != UNSCHEDULED:
                continue

            # 2. All predecessors completed
            preds = cfg.op_predecessors[op_idx]
            if len(preds) > 0 and not np.all(self.op_status[preds] == COMPLETED):
                continue

            # 3. Job not being processed
            job_idx = int(cfg.op_to_job[op_idx])
            if self.job_being_processed[job_idx]:
                continue

            # 4. Collision check
            task_idx = int(cfg.op_to_task[op_idx])
            conflicting = self._collision_map.get(task_idx, set())
            if conflicting & running_tasks:
                continue

            # 5. Time leap check: op must finish before or at the next leap
            if next_leap is not None:
                end_time = self.current_time + duration
                if end_time > next_leap:
                    continue

            # 6. Deadline check
            deadline = int(cfg.op_deadline[op_idx])
            if deadline >= 0:
                end_time = self.current_time + duration
                if end_time > deadline:
                    continue

            mask[a_idx] = True

        # Idle is always legal
        mask[self.num_actions] = True
        return mask

    def step(self, action: int) -> Tuple[float, bool]:
        """
        Execute action for current machine, advance to next machine or next timestep.

        Args:
            action: index into action table, or self.num_actions for idle.

        Returns:
            (reward, done)
            reward: 0 for intermediate steps, -makespan at terminal.
        """
        if self._done:
            return 0.0, True

        cfg = self.config
        m_idx = self.current_machine_idx

        # Apply action
        if action != self.num_actions and not self.machine_busy[m_idx]:
            op_idx = int(cfg.action_op[action])
            duration = int(cfg.action_duration[action])
            tm_idx = int(cfg.action_task_mode[action])
            job_idx = int(cfg.op_to_job[op_idx])

            # Assign operation
            self.op_status[op_idx] = IN_PROGRESS
            self.op_remaining_time[op_idx] = duration
            self.op_assigned_machine[op_idx] = m_idx
            self.op_assigned_task_mode[op_idx] = tm_idx
            self.op_start_step[op_idx] = self.current_time
            self.op_duration[op_idx] = duration

            # Update machine
            self.machine_busy[m_idx] = True
            self.machine_current_op[m_idx] = op_idx
            self.machine_remaining[m_idx] = duration

            # Mark job as being processed
            self.job_being_processed[job_idx] = True

        # Advance to next machine
        self.current_machine_idx += 1

        # If all machines have decided, advance time
        if self.current_machine_idx >= self.num_machines:
            self._advance_time()
            self.current_machine_idx = 0

        return self._compute_reward()

    def _advance_time(self) -> None:
        """Advance time by 1 step, decrement remaining times, free completed machines."""
        self.current_time += 1

        # Decrement remaining time for all in-progress ops
        in_progress = self.op_status == IN_PROGRESS
        self.op_remaining_time[in_progress] -= 1

        # Check for completed ops
        completed_mask = in_progress & (self.op_remaining_time <= 0)
        newly_completed = np.where(completed_mask)[0]

        for op_idx in newly_completed:
            self.op_status[op_idx] = COMPLETED
            job_idx = int(self.config.op_to_job[op_idx])
            m_idx = int(self.op_assigned_machine[op_idx])

            # Free machine
            if self.machine_current_op[m_idx] == op_idx:
                self.machine_busy[m_idx] = False
                self.machine_current_op[m_idx] = -1
                self.machine_remaining[m_idx] = 0

            # Check if job has no more in-progress ops
            job_ops = np.where(self.config.op_to_job == job_idx)[0]
            if not np.any(self.op_status[job_ops] == IN_PROGRESS):
                self.job_being_processed[job_idx] = False

        # Update machine remaining times
        for m_idx in range(self.num_machines):
            if self.machine_busy[m_idx]:
                self.machine_remaining[m_idx] = max(0, self.machine_remaining[m_idx] - 1)

        # Check terminal
        if np.all(self.op_status == COMPLETED):
            self._done = True

    def _compute_reward(self) -> Tuple[float, bool]:
        """Return (reward, done). -makespan at terminal, 0 otherwise."""
        if self._done:
            makespan = self._compute_makespan()
            return -float(makespan), True
        return 0.0, False

    def _compute_makespan(self) -> int:
        """Makespan = max(start_step + duration) across all assigned operations."""
        assigned = self.op_assigned_machine >= 0
        if not np.any(assigned):
            return 0
        end_times = self.op_start_step[assigned] + self.op_duration[assigned]
        return int(np.max(end_times))

    def _get_next_time_leap(self) -> Optional[int]:
        """Get next time leap boundary after current_time."""
        for leap in self.config.time_leaps:
            if leap > self.current_time:
                return int(leap)
        return None

    def get_makespan(self) -> int:
        """Get makespan of completed schedule."""
        if not self._done:
            raise ValueError("Schedule not complete")
        return self._compute_makespan()

    def get_progress(self) -> float:
        """Fraction of operations completed."""
        return float(np.sum(self.op_status == COMPLETED)) / self.num_ops

    def get_machine_utilization(self) -> np.ndarray:
        """Per-machine utilization (fraction of time busy so far)."""
        if self.current_time == 0:
            return np.zeros(self.num_machines, dtype=np.float32)
        busy_time = np.zeros(self.num_machines, dtype=np.float32)
        for op_idx in range(self.num_ops):
            if self.op_status[op_idx] >= IN_PROGRESS and self.op_assigned_machine[op_idx] >= 0:
                m_idx = int(self.op_assigned_machine[op_idx])
                start = int(self.op_start_step[op_idx])
                duration = int(self.op_duration[op_idx])
                end = min(start + duration, self.current_time)
                busy_time[m_idx] += max(0, end - start)
        return busy_time / self.current_time
