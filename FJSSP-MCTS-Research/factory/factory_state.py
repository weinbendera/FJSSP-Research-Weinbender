from __future__ import annotations
from typing import Dict, List, TYPE_CHECKING, Tuple
from copy import deepcopy
from factory.factory_schemas import Operation, Job, FeasibleAction
from factory.factory_logic_loader import FactoryLogic

if TYPE_CHECKING:
    # Only for type hints – avoids circular import at runtime
    from factory.factory import Factory


class FactoryState:
    """
    Immutable snapshot of the factory at a given time step.
    FactoryState is used to get a snapshot of the factory at a given time step.
    Used to give the schedulers the information they need to make decisions.

    FactoryState answers: “Given this point in time, what can each machine legally do?”
    """

    __slots__ = (
        "_current_step",
        "_steps_until_end_of_day",
        "_jobs",
        "_factory_logic",
        "_machine_busy",
        "_initialized",
    )

    def __init__(self, factory: "Factory") -> None:

        object.__setattr__(self, "_initialized", False)

        object.__setattr__(self, "_current_step", factory.current_step)
        object.__setattr__(self, "_steps_until_end_of_day", self.get_steps_until_end_of_day())
        object.__setattr__(self,"_jobs", tuple(deepcopy(job) for job in factory.jobs)) # deep copy the jobs so schedulers can't mutate live jobs
        object.__setattr__(self, "_factory_logic", factory.factory_logic)
        object.__setattr__(self, "_machine_busy", {m_id: rt.busy for m_id, rt in factory.machine_runtimes_map.items()})

        # lock the instance so no later setattr is allowed
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, key, value):
        # after __init__, block all writes
        if getattr(self, "_initialized", False):
            raise AttributeError("FactoryState is immutable and cannot be modified.")
        object.__setattr__(self, key, value)

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def steps_until_end_of_day(self) -> int:
        return self._steps_until_end_of_day

    @property
    def jobs(self) -> Tuple[Job, ...]:
        """Read-only tuple of Job snapshots."""
        return self._jobs

    @property
    def factory_logic(self) -> FactoryLogic:
        return self._factory_logic

    @property
    def machine_ids(self) -> List[str]:
        return list(self._machine_busy.keys())


    """
    SCHEDULER API FUNCTIONS
    """

    def is_machine_busy(self, machine_id: str) -> bool:
        return self._machine_busy.get(machine_id, False)

    def get_feasible_actions(self, machine_id: str) -> List[FeasibleAction]:
        """ 
        Creates a list of feasible actions for a given machine.
        Each action includes the job_id, operation_id, and the task_mode_ids that are feasible for the machine.

        - Only includes operations that are:
          * not started
          * not done
          * not past their deadline
          * executable by this machine (intersection of task + machine modes)
          * finishable before their deadline (if the deadline exists)
        """
        # If the machine is currently busy, it has no new operation to start
        if self.is_machine_busy(machine_id):
            return []

        feasible_actions = []

        for job in self._jobs:
            for operation in job.operations:
                feasible_action = self._get_feasible_actions_for_operation(machine_id, job, operation)
                if feasible_action:
                    feasible_actions.extend(feasible_action)

        return feasible_actions

    def _get_feasible_actions_for_operation(self, machine_id: str, job: Job, operation: Operation) -> List[FeasibleAction]:
        """
        Internal feasibility logic for a single operation on a single machine.
        """
        if job.being_processed or job.done: # job is already being processed or done
            return []
        if operation.done or operation.started: # operation is already done or started
            return []
        if operation.deadline is not None and self._current_step > operation.deadline: # operation deadline has passed
            return []

        # get the feasible task modes for the operation
        feasible_task_mode_ids = self._factory_logic.get_feasible_task_modes(machine_id, operation)
        if not feasible_task_mode_ids:
            return []

        # if there is a deadline, operation must be able to finish in time
        if operation.deadline is not None:
            feasible_task_mode_ids = [
                mode_id for mode_id in feasible_task_mode_ids
                if self._current_step + len(self._factory_logic.get_task_mode(mode_id).power)
                 <= operation.deadline
            ]

        # filter out task modes that are not finishable before the end of the day
        # BUT: allow tasks that can start today and finish tomorrow (if we're not at the very end)
        steps_until_end_of_day = self.get_steps_until_end_of_day()
        # Only apply this constraint if we're close to the end of the day (less than 50 steps remaining)
        # This allows longer tasks to start near the end of the day and finish the next day
        if steps_until_end_of_day < 50:
            feasible_task_mode_ids = [
                mode_id for mode_id in feasible_task_mode_ids
                if len(self._factory_logic.get_task_mode(mode_id).power) <= steps_until_end_of_day
            ]

        # TODO: add precedence constraints built into the factory logic

        feasible_actions = []

        for task_mode_id in feasible_task_mode_ids:
            feasible_actions.append(FeasibleAction(
                machine_id=machine_id,
                job_id=job.id,
                operation_id=operation.id,
                task_mode_id=task_mode_id
            ))
        return feasible_actions

    def get_steps_until_end_of_day(self) -> int:
        """
        Returns the steps until the end of the current day using the current step
        """
        day_length = 192 # steps per day, 7:00am to 11:00pm
        steps_today = self.current_step % day_length
        
        # Calculate remaining steps
        steps_until_end_of_day = day_length - steps_today
        return steps_until_end_of_day