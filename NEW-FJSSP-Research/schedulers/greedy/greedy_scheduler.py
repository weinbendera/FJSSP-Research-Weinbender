from schedulers.scheduler import OnlineScheduler
from typing import List, Dict, Optional
from factory.factory_schemas import Action, FeasibleAction
from factory.factory_state import FactoryState


class GreedyScheduler(OnlineScheduler):
    """
    Greedy scheduler that schedules jobs using a given greedy algorithm.
    This is a step-wise scheduler that schedules jobs one step at a time.
    """
    def __init__(self, greedy_type: str = "min_power"):
        super().__init__()
        # make sure the greedy type is valid
        if greedy_type not in ["min_power", "min_steps", "max_power", "max_steps"]:
            raise ValueError(f"Invalid greedy type: {greedy_type}")

        self.greedy_type = greedy_type # min_power, min_steps, max_power, max_steps

    def choose(self, factory_state: FactoryState) -> Dict[str, Optional[Action]]:
        """
        Schedule the next actions for each machine in the factory.
        Gets feasible actions for all machines first, then chooses actions for all machines simultaneously.
        Ensures no two machines schedule the same operation by removing scheduled operations from other machines' feasible actions.
        Returns a dictionary of machine ids to actions.
        """
        if not factory_state.jobs:
            return {}
        
        # Get feasible actions for all machines at once (make copies so we can modify them)
        feasible_machine_actions = {}
        for machine_id in factory_state.machine_ids:
            feasible_machine_actions[machine_id] = list(factory_state.get_feasible_actions(machine_id))
        
        local_actions: Dict[str, Optional[Action]] = {}
        used_jobs: set[str] = set()  # job_ids that already got an operation this step

        for machine_id in factory_state.machine_ids:
            # filter out any feasible actions whose job is already taken this step
            available_for_machine = [
                fa for fa in feasible_machine_actions[machine_id]
                if fa.job_id not in used_jobs
            ]

            if not available_for_machine:
                local_actions[machine_id] = None
                continue

            chosen_action = self._choose_greedy_action_for_machine(
                factory_state, machine_id, available_for_machine
            )
            if chosen_action is None:
                local_actions[machine_id] = None
                continue

            local_actions[machine_id] = chosen_action

            # mark this job as in use for this timestep
            used_jobs.add(chosen_action.job_id)

        self.scheduled_actions.extend(local_actions.values())
        return local_actions

    def _choose_greedy_action_for_machine(self, factory_state: FactoryState, machine_id: str, feasible_actions: List[FeasibleAction]) -> Optional[Action]:
        """
        Creates a list of actions (action space) from the given feasible actions for the given machine.
        Looks at the power sequence for each Action (each feasible action)

        Returns the action corresponding to the greedy type.
        """
        if not feasible_actions:
            return None
        
        if self.greedy_type == "min_power":
            best_action = min(feasible_actions, key=lambda a: sum(factory_state.factory_logic.get_task_mode(a.task_mode_id).power))
        elif self.greedy_type == "min_steps":
            best_action = min(feasible_actions, key=lambda a: len(factory_state.factory_logic.get_task_mode(a.task_mode_id).power))
        elif self.greedy_type == "max_power":
            best_action = max(feasible_actions, key=lambda a: sum(factory_state.factory_logic.get_task_mode(a.task_mode_id).power))
        elif self.greedy_type == "max_steps":
            best_action = max(feasible_actions, key=lambda a: len(factory_state.factory_logic.get_task_mode(a.task_mode_id).power))
        else:
            return None
        
        # Create and return the Action
        task_mode = factory_state.factory_logic.get_task_mode(best_action.task_mode_id)
        return Action(
            machine_id=machine_id,  
            job_id=best_action.job_id,
            operation_id=best_action.operation_id,
            task_mode_id=best_action.task_mode_id,
            start_step=factory_state.current_step,
            end_step=factory_state.current_step + len(task_mode.power)
        )

