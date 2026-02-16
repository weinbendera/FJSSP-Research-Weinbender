from factory.factory_schemas import Operation, Job, Action
from utils.input_schemas import EnergySource
from utils.factory_logic_loader import FactoryLogic
from utils.job_builder import JobBuilder
from factory.machine_runtime import MachineRuntime
from factory.factory_state import FactoryState
from typing import List, Dict, Optional, Tuple

class Factory():
    """
    This class represents the factory/environment

    The Factory is responsible for changing the state of the factory's machines and jobs.
    Only simulates the environment, does not make decisons for itself.
    
    """

    def __init__(self, factory_logic: FactoryLogic) -> None:
        self.factory_logic = factory_logic
        self.jobs = []
        self.current_step = 0

        self.machine_runtimes_map: Dict[str, MachineRuntime] = {
            machine_id: MachineRuntime(machine_id=machine_id)
            for machine_id in self.factory_logic.machines.keys()
        }   


    def add_jobs(self, jobs: List[Job]) -> None:
        """
        Adds jobs to the factory.
        """
        self.jobs.extend(jobs)

    def reset(self) -> None:
        """
        Resets the factory to the initial state.
        """
        self.jobs = []
        self.current_step = 0
        self.machine_runtimes_map = {machine_id: MachineRuntime(machine_id=machine_id) for machine_id in self.factory_logic.machines.keys()}

    def apply_actions(self, actions: Dict[str, Optional[Action]]) -> None:
        """
        Applies actions to the factory.
        actions is a dictionary of machine ids to actions.
        """
        for machine_id, action in actions.items():
            if action is not None:
                self.dispatch_operation(machine_id, action.job_id, action.operation_id, action.task_mode_id)

    def step(self) -> Tuple[Dict]:
        """
        Steps the factory forward by one step
        Updates the power consumed by the factory.
        Returns the cost of the power consumed by the factory's machines in this step.
        """
        # calculate the power consumed by the factory's machines in this step
        power_consumed = 0 # power consumed by the factory's machines in this step
        for machine_runtime in self.machine_runtimes_map.values():
            power_consumed += machine_runtime.step_power()

        # calculate the total power cost for the factory in this step
        step_power_cost = self.get_step_power_cost(power_consumed, self.current_step)

        # check for newly completed jobs, label as complete
        newly_completed_jobs = []
        for job in self.jobs:
            was_done = job.done
            job.check_completion()
            if job.done and not was_done:
                newly_completed_jobs.append(job.id)


        step_info = {
            'machines_power_consumed': power_consumed,
            'solar_available': self.get_solar_power_available(self.current_step),
            'grid_cost': self.get_grid_power_cost(self.current_step),
            'step_power_cost': step_power_cost,
            'newly_completed_jobs': newly_completed_jobs,
            'completed_jobs': [job.id for job in self.jobs if job.done],
            'all_jobs': [job.id for job in self.jobs]
        }
        self.current_step += 1
        return step_info

    def done(self) -> bool:
        """
        Returns True if the factory is done, False otherwise.
        """
        return all(job.done for job in self.jobs)

    def get_step_power_cost(self, power_consumed: float, step: int) -> float:
        """
        Returns the total cost of the power consumed by the factory's machines in this step.
        """
        # find the solar power available at this time step
        solar_power_available = self.get_solar_power_available(step)
        # find the cost of the grid power per kwh at this time step
        grid_power_cost = self.get_grid_power_cost(step)

        kwh_needed = power_consumed - solar_power_available
        if kwh_needed > 0:
            # we need to buy some grid power
            return kwh_needed * grid_power_cost
        else:
            # we have enough solar power available, so we don't need to buy any grid power
            return 0

    def get_solar_power_available(self, step: int) -> float:
        """
        Returns the solar power available to the factory in this step.
        """
        return self.factory_logic.get_solar_power_available(step)

    def get_grid_power_cost(self, step: int) -> float:
        """
        Returns the cost of the grid power per kwh at this time step.
        """
        return self.factory_logic.get_grid_power_cost(step)

    """
    FACTORY STATE FUNCTIONS
    """

    def get_factory_state(self):
        """
        Returns a factory state object.
        The FactoryState is immutable, so it should not be modified.
        """

        return FactoryState(factory=self)

    """
    MACHINE RELATED FUNCTIONS
    """

    def dispatch_operation(self, machine_id: str, job_id: str, operation_id: str, task_mode_id: str) -> None:
        """
        Dispatches an operation and task_mode to a MachineRuntime.
        Gets the power sequence for the task mode and starts the operation on the machine.
        Ensures that the task mode is available for the task, machine, and factory logic.
        """
        job = self.get_job_by_id(job_id)
        if job.being_processed:
            raise ValueError(f"Job {job_id} is already being processed")

        operation = self.get_operation_by_id(operation_id)        
        # ensure that the task mode is available for the task
        if task_mode_id not in self.factory_logic.get_task(operation.task_id).task_modes:
            raise ValueError(f"Task mode {task_mode_id} not available for task {operation.task_id}")

        # get the power sequence for the task mode
        power_sequence = self.factory_logic.get_task_mode(task_mode_id).power

        # ensure that the machine is in the factory logic 
        if machine_id not in self.machine_runtimes_map or machine_id not in self.factory_logic.machines:
            raise ValueError(f"Machine {machine_id} not found in the factory")
        
        machine_runtime = self.machine_runtimes_map[machine_id]

        if machine_runtime.busy:
            raise ValueError(f"Machine {machine_id} is already busy")
        
        if operation.done:
            raise ValueError(f"Operation {operation.id} is already done")

        if operation.started:
            raise ValueError(f"Operation {operation.id} is already started")

        machine_runtime.start_operation(job, operation, power_sequence)

    def get_job_by_id(self, job_id: str) -> Job:
        """
        Returns the job with the given id.
        """
        for job in self.jobs:
            if job.id == job_id:
                return job
        raise ValueError(f"Job {job_id} not found in the factory")

    def get_operation_by_id(self, operation_id: str) -> Operation:
        """
        Returns the operation with the given id.
        """
        for job in self.jobs:
            for operation in job.operations:
                if operation.id == operation_id:
                    return operation
        raise ValueError(f"Operation {operation_id} not found in the factory")