from typing import List, Optional, Set, Dict, Tuple
from dataclasses import dataclass, field

from utils.factory_logic_loader import FactoryLogic
from utils.input_schemas import ProductRequest, Product


@dataclass
class Operation:
    """
    Operation that needs to be scheduled.
    Contains task information and eligibility for machines/task_modes.
    """
    # Identity
    id: str
    task_id: str
    job_id: str
    run_index: int
    deadline: Optional[int] = None
    
    # Eligibility from JobBuilder and FactoryLogic
    eligible_machines: Set[str] = field(default_factory=set) # {machine_ids}
    eligible_task_modes: Dict[str, List[str]] = field(default_factory=dict)  # {machine_id: [task_mode_ids]}
    
    # Assignment from scheduler
    assigned_machine: Optional[str] = None
    assigned_task_mode: Optional[str] = None
    start_step: Optional[int] = None
    end_step: Optional[int] = None
    
    # Execution state
    started: bool = False
    done: bool = False
    
    def is_assigned(self) -> bool:
        """Check if operation has been assigned to a machine"""
        return self.assigned_machine is not None
    
    @staticmethod
    def make_id(job_id: str, task_id: str, run_index: int) -> str:
        """
        Create operation ID.
        Format: J_ProductID_D500_0001#TASKID#0001
        """
        return f"{job_id}#{task_id.upper()}#{run_index:04d}"


@dataclass
class Job:
    """
    Job representing a product to be manufactured.
    Contains operations and job-specific precedence constraints.
    """
    # Identity
    id: str
    product_id: str
    operations: List[Operation]
    deadline: Optional[int] = None
    
    # operation order constraints
    precedence_constraints: List[Tuple[str, str]] = field(default_factory=list)  # (predecessor_id, successor_id)
    
    # Execution state
    done: bool = False
    
    def get_unassigned_operations(self) -> List[Operation]:
        """Get operations that haven't been assigned yet"""
        return [op for op in self.operations if not op.is_assigned()]
    
    def get_ready_operations(self) -> List[Operation]:
        """
        Get operations that are ready to be scheduled.
        An operation is ready if:
        - Not assigned yet
        - All predecessor operations are done
        """
        ready = []
        for op in self.operations:
            if op.is_assigned():
                continue
            
            # Check if all predecessors are done
            predecessors = [
                pred_id for pred_id, succ_id in self.precedence_constraints
                if succ_id == op.id
            ]
            
            all_preds_done = all(
                pred_op.done
                for pred_op in self.operations
                if pred_op.id in predecessors
            )
            
            if all_preds_done:
                ready.append(op)
        
        return ready
    
    def check_completion(self) -> bool:
        """Check if all operations are complete --> Job is done"""
        if self.done:
            return True
        if all(op.done for op in self.operations):
            self.done = True
            return True
        return False
    
    @staticmethod
    def make_id(product_id: str, deadline: Optional[int], index: int) -> str:
        """
        Create job ID.
        Format: J_ProductID_D500_0001 or J_ProductID_D∞_0001
        """
        deadline_str = f"D{deadline}" if deadline else "No Deadline"
        return f"J_{product_id}_{deadline_str}_{index:04d}"


class JobBuilder:
    """
    Builds jobs with operations from product requests.
    Uses FactoryLogic to create operations and precedence constraints.
    """
    
    def __init__(self, factory_logic: FactoryLogic):
        self.factory_logic = factory_logic
    
    def build_jobs(self, product_requests: List[ProductRequest]) -> List[Job]:
        """
        Build jobs from product requests.
        """
        jobs = []
        
        for product_request in product_requests:
            # Validate product exists
            if product_request.product not in self.factory_logic.products:
                available_products = list(self.factory_logic.products.keys())
                raise ValueError(
                    f"Product '{product_request.product}' not found in factory configuration.\n"
                    f"Available products: {', '.join(available_products)}"
                )
            
            # Create one job per unit requested
            for unit_index in range(1, product_request.amount + 1):
                job = self._build_job(product_request, unit_index)
                jobs.append(job)
        
        return jobs
    
    def _build_job(self, product_request: ProductRequest, unit_index: int) -> Job:
        """
        Build a single job with operations and precedence constraints.
        """
        job_id = Job.make_id(product_request.product, product_request.deadline, unit_index)
        
        product = self.factory_logic.get_product(product_request.product)
        
        # Build operations with eligibility info from FactoryLogic
        operations = []
        for product_task in product.tasks:
            # Validate task exists
            if product_task.task not in self.factory_logic.tasks:
                raise ValueError(
                    f"Task '{product_task.task}' required for product '{product_request.product}' "
                    f"not found in factory configuration"
                )
            
            # Create one operation per run of the task
            for run_index in range(1, product_task.runs + 1):
                operation = self._build_operation(
                    task_id=product_task.task,
                    job_id=job_id,
                    run_index=run_index,
                    deadline=product_request.deadline
                )
                operations.append(operation)
        
        precedence = self._build_precedence_constraints(operations)
        
        return Job(
            id=job_id,
            product_id=product_request.product,
            operations=operations,
            precedence_constraints=precedence,
            deadline=product_request.deadline
        )
    
    def _build_operation(self, task_id: str, job_id: str, run_index: int, deadline: Optional[int]) -> Operation:
        """Build operation with eligibility information from FactoryLogic."""
        op_id = Operation.make_id(job_id, task_id, run_index)
        
        # Get eligibility from FactoryLogic helpers
        eligible_machines = self.factory_logic.get_eligible_machines_for_task(task_id)
        eligible_task_modes = self.factory_logic.get_eligible_task_modes_by_machine(task_id)
        
        # Validate that at least one machine can execute this task
        if not eligible_machines:
            raise ValueError(
                f"No machines can execute task '{task_id}'.\n"
                f"Check that task modes are properly configured for machines."
            )
        
        return Operation(
            id=op_id,
            task_id=task_id,
            job_id=job_id,
            run_index=run_index,
            deadline=deadline,
            eligible_machines=eligible_machines,
            eligible_task_modes=eligible_task_modes
        )
     
    def _build_precedence_constraints(self, operations: List[Operation]) -> List[Tuple[str, str]]:
        """
        Build precedence constraints ONLY from explicit order constraints in FactoryLogic.
        
        Product task list does NOT imply any order - tasks can be done in any sequence
        unless explicitly constrained by FactoryLogic.order_constraints.
        """
        constraints = []
        
        # ONLY explicit order constraints from FactoryLogic
        task_ids = {op.task_id for op in operations}
        global_constraints = self.factory_logic.get_order_constraints_for_tasks(task_ids)
        
        for order_constraint in global_constraints:
            # Find all operations with these tasks in this job
            pred_ops = [op for op in operations if op.task_id == order_constraint.first_task]
            succ_ops = [op for op in operations if op.task_id == order_constraint.second_task]
            
            # Add constraint: ALL successors must wait for ALL predecessors
            for pred_op in pred_ops:
                for succ_op in succ_ops:
                    constraints.append((pred_op.id, succ_op.id))
        
        return constraints