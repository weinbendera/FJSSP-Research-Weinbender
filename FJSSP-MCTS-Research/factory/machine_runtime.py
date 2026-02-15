from typing import List, Dict
from factory.factory_schemas import Operation, Job


class MachineRuntime:
    """
    This class represents the runtime of a machine is used to track the machine's current state and power consumption.
    """
    def __init__(self, machine_id: str):
        self.busy = False # is the machine currently running a operation
        self.machine_id = machine_id
        self.job: Job = None # which job the machine is working on
        self.operation: Operation = None  # which operation the machine is working on
        self.power_sequence: List[float] = None # the power sequence for the current operation
        self.remaining_operation_steps = 0 # remaining steps for the current operation
        self.operation_step = 0  # The "step"/index of the current operation

    def start_operation(self, job: Job, operation: Operation, power_sequence: List[float]) -> None:
        """
        Start a new operation on this machine.
        """
        self.busy = True
        self.job = job
        self.job.being_processed = True
        self.operation = operation
        self.operation.started = True
        self.power_sequence = power_sequence

        self.remaining_operation_steps = len(power_sequence)
        self.operation_step = 0 # indexes through the power sequence
        
    def step_power(self) -> float:
        """
        Return power consumption for this step.
        """
        if not self.busy or self.power_sequence is None: # Machine may be Idle
            return 0
        
        power = self.power_sequence[self.operation_step]  # get current step power
        
        self.operation_step += 1
        self.remaining_operation_steps -= 1
        if self.remaining_operation_steps == 0:
            # operation is complete
            self.job.being_processed = False
            self.operation.done = True # operation is now done
                        
            self._reset() # reset the machine runtime, removing the operation and power sequence
            return power
        return power
    
    def _reset(self):
        self.busy = False
        self.job = None
        self.operation = None
        self.power_sequence = None
        self.remaining_operation_steps = 0
        self.operation_step = 0
