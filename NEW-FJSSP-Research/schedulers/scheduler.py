from abc import ABC, abstractmethod 
from typing import List, Dict, Optional
from pydantic import BaseModel

from utils.input_schemas import EnergySource
from utils.job_builder import Job

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import io

class ScheduledOperation(BaseModel):
    """
    A single operation that has been scheduled.
    Contains the assignment details (machine, task mode, timing).
    """
    # identity
    operation_id: str
    job_id: str
    task_id: str
    
    # assignment by scheduler
    machine_id: str
    task_mode_id: str
    start_step: int
    end_step: int
    
    # make immutable
    class Config:
        frozen = True

class Schedule(BaseModel):
    """
    Complete schedule solution done by the scheduler.
    """
    # scheduled operations
    operations: List[ScheduledOperation]
    
    # metrics
    makespan: int = 0
    energy_cost: float = 0.0
    
    # feasibility
    is_feasible: bool = True
    unscheduled_operations: int = 0
    deadline_violations: int = 0

    def plot_gantt_by_task(self) -> Optional[Image.Image]:
        """
        Alternative Gantt chart colored by task type instead of job.
        """
        if not self.operations:
            print("No operations to plot.")
            return None
        
        records = []
        for op in self.operations:
            records.append({
                "machine": op.machine_id,
                "task_id": op.task_id,
                "start": op.start_step,
                "finish": op.end_step,
                "duration": op.end_step - op.start_step + 1,
                "task_mode_id": op.task_mode_id,
            })
        
        df = pd.DataFrame(records)
        df = df.sort_values(by=["machine", "start"])
        
        # Create color map for tasks
        task_ids = sorted(df["task_id"].unique())
        cmap = plt.get_cmap("tab20")
        color_map = {tid: cmap(i % 20) for i, tid in enumerate(task_ids)}
        
        # Create y-axis positions for machines
        machines = sorted(df["machine"].unique())
        y_positions = {m: i for i, m in enumerate(machines)}
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Plot each operation as a bar
        for _, row in df.iterrows():
            y = y_positions[row["machine"]]
            color = color_map[row["task_id"]]
            
            ax.barh(
                y=y,
                width=row["duration"],
                left=row["start"],
                color=color,
                edgecolor="black",
                alpha=0.9,
            )
            
            # Task mode id label (vertical)
            ax.text(
                row["start"] + row["duration"] / 2,
                y,
                row["task_mode_id"],
                ha='center',
                va='center',
                fontsize=4,
                rotation=90,
                color='black',
                weight='bold'
            )
        
        # Configure axes
        ax.set_yticks([y_positions[m] for m in machines])
        ax.set_yticklabels(machines)
        ax.set_xlabel("Time (steps)")
        ax.set_ylabel("Machine")
        ax.set_title("Gantt Chart - Colored by Task Type")
        
        ax.set_xlim(df["start"].min() - 1, df["finish"].max() + 1)
        
        # Add legend
        legend_patches = [
            mpatches.Patch(color=color_map[tid], label=str(tid))
            for tid in task_ids
        ]
        ax.legend(
            handles=legend_patches,
            title="Task Types",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
        )
        
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        img = Image.open(buf)
        plt.close(fig)
        return img

    def plot_cost_with_solar_and_grid_prices(self) -> Optional[Image.Image]:
        """
        Plots the cost of the schedule with solar and grid prices.
        """
        if not self.operations:
            print("No operations to plot.")
            return None
        
        pass # TODO: Implement this
        records = []
        for op in self.operations:
            records.append({
                "machine": op.machine_id,
                "task_id": op.task_id,
                "start": op.start_step,
                "finish": op.end_step,
                "duration": op.end_step - op.start_step + 1,
                "task_mode_id": op.task_mode_id,
            })
        df = pd.DataFrame(records)
        df = df.sort_values(by=["machine", "start"])
        
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(df["start"], df["cost"])
        ax.set_xlabel("Time (steps)")
        ax.set_ylabel("Cost")
        ax.set_title("Cost with Solar and Grid Prices")
        return plt.gcf()


class Scheduler(ABC):
    """
    Abstract base class (interface) for all schedulers.
    Schedulers are used to schedule the given jobs for all machines.
    """
    def __init__(self):
        pass

    @abstractmethod
    def schedule(self, jobs: List[Job], energy_sources: List[EnergySource]) -> Schedule:
        """
        Schedules the given jobs for all machines.
        Returns the scheduled operations.
        """
        raise NotImplementedError("Subclasses must implement this method")

class OnlineScheduler(Scheduler):
    """
    Abstract base class (interface) for all online schedulers.
    All concrete online scheduler implementations must implement the choose method.
    Online schedulers are used to choose the next actions for each ofthe factory's machines, one step at a time.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def choose(self, jobs: List[Job], energy_sources: List[EnergySource]) -> Dict[str, ScheduledOperation | None]:
        """
        Chooses the next actions for each of the factory's machines.
        Returns the scheduled operations to be taken by the scheduler for each machine, or None if no operations were scheduled.
        """
        raise NotImplementedError("Subclasses must implement this method")

class OfflineScheduler(Scheduler):
    """
    Abstract base class (interface) for all offline schedulers.
    All concrete offline scheduler implementations must implement the schedule method.
    Offline schedulers are used to schedule all of the given jobs at once.
    """
    def __init__(self):
        super().__init__()