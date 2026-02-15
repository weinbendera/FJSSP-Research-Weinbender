from pydantic import BaseModel
from typing import List, Optional
from typing import Literal, Union

"""
MODELS FOR THE USER'S INPUT
"""

class ProductRequest(BaseModel):
    product: str
    amount: int
    deadline: Optional[int] = None

class ScheduleRequest(BaseModel):
    product_requests: List[ProductRequest]
    max_steps: int
    time_leaps: List[int]

"""
PYDANTIC MODELS OF THE INPUT DATA (FACTORY LOGIC)
USED TO LOAD THE INPUT DATA FROM THE JSON FILE ONLY
"""

class Cell(BaseModel):
    """
    A cell is a factory or production line.
    It has a list of machines that are available in the cell.
    """
    id: str
    machines: List[str]

class TaskMode(BaseModel):
    id: str
    power: List[float] # list of power values for the task mode for each step
 
class Task(BaseModel):
    id: str
    task_modes: List[str] # list of task modes that are available for the task e.g. Ironing --> Ironing TM1, Ironing TM3, Ironing TM4

class Machine(BaseModel):
    """
    Used to map each machine id to its task modes.
    """
    id: str
    task_modes: List[str] # list of task modes that are available for the machine e.g. MAQ118 --> Ironing TM4, Harden[2] TM1, Harden[1.5] TM3

class ProductTask(BaseModel):
    """
    Each Product has multiple tasks that need to be performed.
    They may also have mutliple runs of the same task.
    This is a schema for the tasks of a product.
    """
    task: str # e.g. Ironing, Harden[1.5], Sublimation, Anti-Shrinkage
    runs: int # number of times the task needs to be performed for the product to be completed

class Product(BaseModel):
    """
    A Product has a list of tasks that need to be done to complete the product.
    """
    id: str # id of the product e.g. WOVEN LABEL OURELA FABRIC
    tasks: List[ProductTask] # list of tasks that need to be done to complete the product e.g. Ironing, Harden[1.5], Sublimation, Anti-Shrinkage

class OrderConstraint(BaseModel):
    """
    Order constraint: first_task must complete before second_task starts.
    Applies to operations within the same job.
    """
    type: Literal["Order"] = "Order"  # Use Literal instead of str
    first_task: str
    second_task: str

class CollisionConstraint(BaseModel):
    """
    Collision constraint: task1 and task2 cannot run simultaneously.
    Applies globally across all jobs in the factory.
    """
    type: Literal["Collision"] = "Collision"  # Use Literal instead of str
    task1: str
    task2: str

Constraint = Union[OrderConstraint, CollisionConstraint]

class EnergySource(BaseModel):
    """
    Used to map each energy source id to its price and availability.
    Solar has the availability for the factory at that time step.
    Socket Energy has the price per kwh of the grid power for the factory at that time step.
    """
    id: str
    price: Optional[List[float]] = None
    availability: Optional[List[float]] = None