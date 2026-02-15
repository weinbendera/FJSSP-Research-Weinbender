from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from factory.factory_state import FactoryState
from factory.factory_schemas import Action, FeasibleAction, Job
from schedulers.scheduler import OnlineScheduler


class RuleBasedScheduler(OnlineScheduler):
    """
    Online scheduler that applies a small set of deterministic rules instead of a single greedy metric.

    Rules (in priority order):
    1. Jobs with overdue or near-due deadlines are executed first.
    2. Among jobs with similar urgency, finish jobs that are nearly complete before starting new ones.
    3. Break ties by preferring lower-power task modes to keep cost down (configurable).
    4. Fall back to deterministic ordering to keep behaviour reproducible.
    """

    def __init__(
        self,
        urgent_window: int = 48,
        caution_window: int = 96,
        prefer_low_power: bool = True,
    ) -> None:
        super().__init__()
        self.urgent_window = urgent_window
        self.caution_window = caution_window
        self.prefer_low_power = prefer_low_power

    def choose(self, factory_state: FactoryState) -> Dict[str, Optional[Action]]:
        """
        Build the next step schedule by ranking feasible actions per machine via rule-based scoring.
        Ensures that a job is claimed by at most one machine during the current tick.
        """
        if not factory_state.jobs:
            return {}

        job_lookup = {job.id: job for job in factory_state.jobs}

        feasible_machine_actions = {
            machine_id: list(factory_state.get_feasible_actions(machine_id))
            for machine_id in factory_state.machine_ids
        }

        chosen_actions: Dict[str, Optional[Action]] = {}
        jobs_claimed_this_step: set[str] = set()

        for machine_id in factory_state.machine_ids:
            available = [
                action
                for action in feasible_machine_actions[machine_id]
                if action.job_id not in jobs_claimed_this_step
            ]

            if not available:
                chosen_actions[machine_id] = None
                continue

            best_feasible = self._select_best_action(
                factory_state,
                job_lookup,
                available,
            )

            if best_feasible is None:
                chosen_actions[machine_id] = None
                continue

            task_mode = factory_state.factory_logic.get_task_mode(best_feasible.task_mode_id)
            action = Action(
                machine_id=machine_id,
                job_id=best_feasible.job_id,
                operation_id=best_feasible.operation_id,
                task_mode_id=best_feasible.task_mode_id,
                start_step=factory_state.current_step,
                end_step=factory_state.current_step + len(task_mode.power),
            )
            chosen_actions[machine_id] = action
            jobs_claimed_this_step.add(action.job_id)

        self.scheduled_actions.extend(chosen_actions.values())
        return chosen_actions

    def _select_best_action(
        self,
        factory_state: FactoryState,
        job_lookup: Dict[str, Job],
        feasible_actions: List[FeasibleAction],
    ) -> Optional[FeasibleAction]:
        if not feasible_actions:
            return None

        ranked = sorted(
            feasible_actions,
            key=lambda action: self._score_action(factory_state, job_lookup[action.job_id], action),
        )
        return ranked[0]

    def _score_action(
        self,
        factory_state: FactoryState,
        job: Job,
        feasible_action: FeasibleAction,
    ) -> Tuple[float, float, int, float, int, str]:
        """
        Produce a tuple suitable for lexicographic sorting.
        Lower tuples are preferred.
        """
        task_mode = factory_state.factory_logic.get_task_mode(feasible_action.task_mode_id)
        duration = len(task_mode.power)
        total_power = sum(task_mode.power)

        remaining_ops = sum(1 for op in job.operations if not op.done)

        slack = float("inf")
        urgency_bucket = 3  # default for undated jobs
        if job.deadline is not None:
            slack = job.deadline - (factory_state.current_step + duration)
            if slack <= 0:
                urgency_bucket = 0  # already late, run immediately
            elif slack <= self.urgent_window:
                urgency_bucket = 0
            elif slack <= self.caution_window:
                urgency_bucket = 1
            else:
                urgency_bucket = 2

        energy_score = total_power if self.prefer_low_power else -total_power

        return (
            urgency_bucket,
            slack,
            remaining_ops,
            energy_score,
            duration,
            feasible_action.operation_id,
        )