from typing import Dict, List, Set, Optional, Tuple, Union
import json
from utils.input_schemas import TaskMode, Task, Product, Machine, Cell, OrderConstraint, CollisionConstraint

class FactoryLogic:
    """
    Used to take in the loaded data from the factory logic loader.
    Contains the cells, task modes, tasks, products, machines, order constraints, and collision constraints.

    Provides helper methods for the job builder to create operations.
    Passed into schedulers to look at eligibility of tasks and machines as well as constraints for a given task.
    """
    
    def __init__(
        self,
        cells: List[Cell],
        task_modes: List[TaskMode],
        tasks: List[Task],
        products: List[Product],
        machines: List[Machine],
        constraints: List[Union[OrderConstraint, CollisionConstraint]],
        time_leaps: List[int]
    ):

        self.cells: Dict[str, Cell] = {c.id: c for c in cells}
        self.task_modes: Dict[str, TaskMode] = {tm.id: tm for tm in task_modes}
        self.tasks: Dict[str, Task] = {t.id: t for t in tasks}
        self.products: Dict[str, Product] = {p.id: p for p in products}
        self.machines: Dict[str, Machine] = {m.id: m for m in machines}
        self.order_constraints: List[OrderConstraint] = [c for c in constraints if isinstance(c, OrderConstraint)]
        self.collision_constraints: List[CollisionConstraint] = [c for c in constraints if isinstance(c, CollisionConstraint)]
        self.time_leaps: List[int] = time_leaps
    

    def get_task_mode(self, task_mode_id: str) -> TaskMode:
        """Get task mode by ID"""
        if task_mode_id not in self.task_modes:
            raise KeyError(f"Task mode '{task_mode_id}' not found")
        return self.task_modes[task_mode_id]
    
    def get_task(self, task_id: str) -> Task:
        """Get task by ID"""
        if task_id not in self.tasks:
            raise KeyError(f"Task '{task_id}' not found")
        return self.tasks[task_id]
    
    def get_product(self, product_id: str) -> Product:
        """Get product by ID"""
        if product_id not in self.products:
            raise KeyError(f"Product '{product_id}' not found")
        return self.products[product_id]
    
    def get_machine(self, machine_id: str) -> Machine:
        """Get machine by ID"""
        if machine_id not in self.machines:
            raise KeyError(f"Machine '{machine_id}' not found")
        return self.machines[machine_id]
    
    """
    JOB BUILDER HELPERS
    Used by the job builder to create operations.
    """

    def get_eligible_machines_for_task(self, task_id: str) -> Set[str]:
        """
        Get all machines that can execute this task.
        A machine is eligible if it shares at least one task mode with the task.
        """
        task = self.get_task(task_id)
        task_modes_set = set(task.task_modes)
        
        eligible = set()
        for machine_id, machine in self.machines.items():
            machine_modes_set = set(machine.task_modes)
            if task_modes_set & machine_modes_set:  # Any overlap?
                eligible.add(machine_id)
        
        return eligible
    
    def get_eligible_task_modes_by_machine(self, task_id: str) -> Dict[str, List[str]]:
        """
        Get eligible task modes grouped by machine.
        Returns: {machine_id: [task_mode_ids that both the task and machine support]}
        """
        task = self.get_task(task_id)
        task_modes_set = set(task.task_modes)
        
        result = {}
        for machine_id, machine in self.machines.items():
            machine_modes_set = set(machine.task_modes)
            compatible = task_modes_set & machine_modes_set
            
            if compatible:
                result[machine_id] = list(compatible)
        
        return result
    
    """
    SCHEDULER HELPERS
    Used by the scheduler to check if any constraints would be violated when scheduling an operation.
    """

    # ORDER CONSTRAINT HELPERS
    def get_order_constraints_for_tasks(self, task_ids: Set[str]) -> List[OrderConstraint]:
        """
        Get order constraints that apply to a set of tasks.
        Used when building precedence constraints for a job.
        """
        return [
            c for c in self.order_constraints
            if c.first_task in task_ids or c.second_task in task_ids
        ]
    
    # COLLISION CONSTRAINT HELPERS
    def get_collision_constraints_for_task(self, task_id: str) -> List[CollisionConstraint]:
        """Get collision constraints involving this task"""
        return [
            c for c in self.collision_constraints
            if c.task1 == task_id or c.task2 == task_id
        ]
    
    def tasks_have_collision(self, task1_id: str, task2_id: str) -> bool:
        """Check if two tasks cannot run simultaneously"""
        return any(
            (c.task1 == task1_id and c.task2 == task2_id) or
            (c.task1 == task2_id and c.task2 == task1_id)
            for c in self.collision_constraints
        )

    # TIME CONSTRAINT HELPERS
    def get_next_time_leap(self, current_step: int) -> Optional[int]:
        """
        Get the next time leap after current_step.
        
        Returns: Next time leap step, or None if no more breaks
        
        Used by: Schedulers to check if an operation would cross a break
        """
        for leap in self.time_leaps:
            if leap > current_step:
                return leap
        return None



class FactoryLogicLoader:
    """Loads factory configuration from JSON file"""
    
    @staticmethod
    def load_from_file(filepath: str) -> FactoryLogic:
        """Load factory logic from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return FactoryLogicLoader.load_from_dict(data)
    
    @staticmethod
    def load_from_dict(data: Dict) -> FactoryLogic:
        """Load factory logic from dictionary"""
        # Validate required keys
        required_keys = ["task_modes", "tasks", "products", "machines"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key '{key}' in factory logic")
        
        # Convert to Pydantic models
        cells = [Cell(**c) for c in data.get("cells", [])]
        task_modes = [TaskMode(**tm) for tm in data["task_modes"]]
        tasks = [Task(**t) for t in data["tasks"]]
        products = [Product(**p) for p in data["products"]]
        machines = [Machine(**m) for m in data["machines"]]

        # Extract constraints from nested parameter structure
        constraints = []
        for c in data.get("constraints", []):
            if c["type"] == "Order":
                # Flatten: extract parameters from nested structure
                constraint_data = {
                    "type": "Order",
                    "first_task": c["parameter"]["first_task"],
                    "second_task": c["parameter"]["second_task"]
                }
                constraints.append(OrderConstraint(**constraint_data))
            elif c["type"] == "Collision":
                # Flatten: extract parameters from nested structure
                constraint_data = {
                    "type": "Collision",
                    "task1": c["parameter"]["task1"],
                    "task2": c["parameter"]["task2"]
                }
                constraints.append(CollisionConstraint(**constraint_data))
                
        time_leaps = [192, 384, 576, 768, 960] 

        return FactoryLogic(
            cells=cells,
            task_modes=task_modes,
            tasks=tasks,
            products=products,
            machines=machines,
            constraints=constraints,
            time_leaps=time_leaps
        )