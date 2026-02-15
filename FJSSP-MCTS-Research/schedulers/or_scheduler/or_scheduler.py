from typing import List, Dict, Tuple, Optional
from ortools.sat.python import cp_model
import time

from schedulers.scheduler import OfflineScheduler, Schedule, ScheduledOperation
from utils.job_builder import Job, Operation
from utils.factory_logic_loader import FactoryLogic
from utils.input_schemas import EnergySource


class ORToolsScheduler(OfflineScheduler):
    """
    OR-Tools CP-SAT scheduler for flexible job shop scheduling.
    
    Uses constraint programming to find optimal schedules respecting:
    - Precedence constraints from jobs
    - Collision constraints from factory
    - Machine availability
    - Deadline constraints
    - Time leaps (production breaks)
    
    Optimizes for:
    - Energy cost (considering solar power offset and time-varying grid prices)
    - Makespan (total schedule duration)
    """
    
    def __init__(self, factory_logic: FactoryLogic, max_solve_time_seconds: int = 300, 
    cost_weight: float = 1.0, makespan_weight: float = 0.1,
        num_workers: int = 8, use_energy_optimization: bool = True
    ):
        """
        Initialize OR-Tools scheduler.
        
        Args:
            factory_logic: Factory configuration
            max_solve_time_seconds: Maximum time to search for solution
            cost_weight: Weight for energy cost in objective
            makespan_weight: Weight for makespan in objective
            num_workers: Number of parallel workers for search
            use_energy_optimization: If True, optimize for energy cost with solar/grid prices
        """
        super().__init__()
        self.factory_logic = factory_logic
        self.max_solve_time_seconds = max_solve_time_seconds
        self.cost_weight = cost_weight
        self.makespan_weight = makespan_weight
        self.num_workers = num_workers
        self.use_energy_optimization = use_energy_optimization
        
        # Energy data (extracted during schedule call)
        self.solar_availability: List[float] = []
        self.grid_prices: List[float] = []
    
    def schedule(self, jobs: List[Job], energy_sources: List[EnergySource]) -> Schedule:
        """
        Create schedule using OR-Tools CP-SAT solver.
        
        Args:
            jobs: List of jobs to schedule
            energy_sources: Energy availability and pricing
            
        Returns:
            Optimal or best-found Schedule
        """
        if not jobs:
            return Schedule(
                operations=[],
                makespan=0,
                energy_cost=0.0,
                is_feasible=True,
                unscheduled_operations=0,
                deadline_violations=0
            )
        
        # Extract energy data
        self._extract_energy_data(energy_sources)
        
        # Create CP model
        model = cp_model.CpModel()
        
        # Estimate horizon (upper bound on makespan)
        horizon = self._estimate_horizon(jobs)
        
        # Create decision variables
        variables = self._create_variables(model, jobs, horizon)
        
        # Add constraints
        self._add_constraints(model, jobs, variables, horizon)
        
        # Set objective
        if self.use_energy_optimization and self.solar_availability and self.grid_prices:
            self._set_energy_aware_objective(model, jobs, variables, horizon)
        else:
            self._set_simple_objective(model, jobs, variables, horizon)
        
        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.max_solve_time_seconds
        solver.parameters.num_search_workers = self.num_workers
        solver.parameters.log_search_progress = True
        
        print(f"\nSolving with OR-Tools CP-SAT (max time: {self.max_solve_time_seconds}s)...")
        start_time = time.time()
        status = solver.Solve(model)
        solve_time = time.time() - start_time
        
        print(f"Solve time: {solve_time:.2f}s")
        print(f"Status: {solver.StatusName(status)}")
        
        # Extract solution
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return self._extract_solution(solver, jobs, variables, status == cp_model.OPTIMAL)
        else:
            print(f"No solution found. Status: {solver.StatusName(status)}")
            return Schedule(
                operations=[],
                makespan=0,
                energy_cost=0.0,
                is_feasible=False,
                unscheduled_operations=sum(len(job.operations) for job in jobs),
                deadline_violations=0
            )
    
    def _extract_energy_data(self, energy_sources: List[EnergySource]):
        """Extract solar availability and grid prices."""
        self.solar_availability = []
        self.grid_prices = []
        
        for energy_source in energy_sources:
            if energy_source.id == "Solar" and energy_source.availability:
                self.solar_availability = energy_source.availability
            elif energy_source.id == "Socket Energy" and energy_source.price:
                self.grid_prices = energy_source.price
    
    def _estimate_horizon(self, jobs: List[Job]) -> int:
        """
        Estimate upper bound on makespan.
        
        Uses sum of maximum operation durations as conservative estimate.
        """
        total_duration = 0
        
        for job in jobs:
            for operation in job.operations:
                # Find longest task mode for this operation
                max_duration = 0
                for machine_id in operation.eligible_machines:
                    task_modes = operation.eligible_task_modes.get(machine_id, [])
                    for task_mode_id in task_modes:
                        task_mode = self.factory_logic.get_task_mode(task_mode_id)
                        duration = len(task_mode.power)
                        max_duration = max(max_duration, duration)
                
                total_duration += max_duration
        
        # Add buffer for time leaps and machine conflicts
        horizon = int(total_duration * 1.5)
        
        # Consider deadlines
        for job in jobs:
            if job.deadline is not None:
                horizon = max(horizon, job.deadline + 100)
        
        # Consider energy data length
        if self.solar_availability:
            horizon = min(horizon, len(self.solar_availability))
        if self.grid_prices:
            horizon = min(horizon, len(self.grid_prices))
        
        return horizon
    
    def _create_variables(
        self,
        model: cp_model.CpModel,
        jobs: List[Job],
        horizon: int
    ) -> Dict:
        """
        Create decision variables for the CP model.
        
        Variables:
        - start_vars: Start time for each operation
        - end_vars: End time for each operation
        - interval_vars: Interval for each (operation, machine, task_mode) tuple
        - machine_chosen: Boolean for which machine is chosen for each operation
        - task_mode_chosen: Boolean for which task mode is chosen
        - makespan_var: Overall makespan variable
        - power_at_step: Power consumption at each timestep (for energy optimization)
        - grid_power_at_step: Grid power needed at each timestep (after solar offset)
        """
        variables = {
            'start_vars': {},      # operation_id -> IntVar
            'end_vars': {},        # operation_id -> IntVar
            'intervals': {},       # (op_id, machine_id, task_mode_id) -> IntervalVar
            'machine_chosen': {},  # (op_id, machine_id) -> BoolVar
            'task_mode_chosen': {}, # (op_id, machine_id, task_mode_id) -> BoolVar
            'machine_intervals': {}, # machine_id -> list of intervals
            'is_active': {},       # (op_id, step) -> BoolVar (operation active at step)
        }
        
        # Initialize machine intervals dict
        for machine_id in self.factory_logic.machines.keys():
            variables['machine_intervals'][machine_id] = []
        
        # Create variables for each operation
        for job in jobs:
            for operation in job.operations:
                op_id = operation.id
                
                # Start and end time variables
                start_var = model.NewIntVar(0, horizon, f'start_{op_id}')
                end_var = model.NewIntVar(0, horizon, f'end_{op_id}')
                
                variables['start_vars'][op_id] = start_var
                variables['end_vars'][op_id] = end_var
                
                # For each eligible machine
                for machine_id in operation.eligible_machines:
                    # Machine choice variable
                    machine_key = (op_id, machine_id)
                    machine_chosen = model.NewBoolVar(f'machine_{op_id}_{machine_id}')
                    variables['machine_chosen'][machine_key] = machine_chosen
                    
                    # For each eligible task mode on this machine
                    task_modes = operation.eligible_task_modes.get(machine_id, [])
                    for task_mode_id in task_modes:
                        task_mode = self.factory_logic.get_task_mode(task_mode_id)
                        duration = len(task_mode.power)
                        
                        # Task mode choice variable
                        tm_key = (op_id, machine_id, task_mode_id)
                        tm_chosen = model.NewBoolVar(f'tm_{op_id}_{machine_id}_{task_mode_id}')
                        variables['task_mode_chosen'][tm_key] = tm_chosen
                        
                        # Optional interval variable (active only if this task mode is chosen)
                        interval = model.NewOptionalIntervalVar(
                            start_var,
                            duration,
                            end_var,
                            tm_chosen,
                            f'interval_{op_id}_{machine_id}_{task_mode_id}'
                        )
                        variables['intervals'][tm_key] = interval
                        
                        # Add to machine's interval list
                        variables['machine_intervals'][machine_id].append(interval)
        
        # Makespan variable
        variables['makespan'] = model.NewIntVar(0, horizon, 'makespan')
        
        # Energy-related variables (only if using energy optimization)
        if self.use_energy_optimization and self.solar_availability and self.grid_prices:
            # Power usage at each timestep
            variables['power_at_step'] = {}
            variables['grid_power_at_step'] = {}
            
            for step in range(horizon):
                # Total power needed at this step
                power_var = model.NewIntVar(0, 100000, f'power_step_{step}')
                variables['power_at_step'][step] = power_var
                
                # Grid power (after solar offset) at this step
                grid_power_var = model.NewIntVar(0, 100000, f'grid_power_step_{step}')
                variables['grid_power_at_step'][step] = grid_power_var
        
        return variables
    
    def _add_constraints(self, model: cp_model.CpModel, jobs: List[Job], variables: Dict, horizon: int):
        """Add all scheduling constraints to the model."""
        
        # 1. Each operation must choose exactly one machine
        for job in jobs:
            for operation in job.operations:
                op_id = operation.id
                machine_bools = [
                    variables['machine_chosen'][(op_id, m_id)]
                    for m_id in operation.eligible_machines
                ]
                model.AddExactlyOne(machine_bools)
        
        # 2. Each operation must choose exactly one task mode
        for job in jobs:
            for operation in job.operations:
                op_id = operation.id
                tm_bools = []
                for machine_id in operation.eligible_machines:
                    task_modes = operation.eligible_task_modes.get(machine_id, [])
                    for task_mode_id in task_modes:
                        tm_key = (op_id, machine_id, task_mode_id)
                        tm_bools.append(variables['task_mode_chosen'][tm_key])
                
                model.AddExactlyOne(tm_bools)
        
        # 3. Task mode can only be chosen if machine is chosen
        for job in jobs:
            for operation in job.operations:
                op_id = operation.id
                for machine_id in operation.eligible_machines:
                    machine_chosen = variables['machine_chosen'][(op_id, machine_id)]
                    task_modes = operation.eligible_task_modes.get(machine_id, [])
                    for task_mode_id in task_modes:
                        tm_chosen = variables['task_mode_chosen'][(op_id, machine_id, task_mode_id)]
                        # tm_chosen => machine_chosen
                        model.AddImplication(tm_chosen, machine_chosen)
        
        # 4. No overlap on same machine (NoOverlap constraint)
        for machine_id, intervals in variables['machine_intervals'].items():
            if intervals:
                model.AddNoOverlap(intervals)
        
        # 5. Precedence constraints
        for job in jobs:
            for pred_id, succ_id in job.precedence_constraints:
                # Predecessor must end before successor starts
                model.Add(
                    variables['end_vars'][pred_id] <= variables['start_vars'][succ_id]
                )
        
        # 6. Collision constraints (global mutual exclusion)
        self._add_collision_constraints(model, jobs, variables)
        
        # 7. Deadline constraints
        for job in jobs:
            if job.deadline is not None:
                for operation in job.operations:
                    model.Add(variables['end_vars'][operation.id] <= job.deadline)
        
        # 8. Makespan constraint
        for job in jobs:
            for operation in job.operations:
                model.Add(variables['makespan'] >= variables['end_vars'][operation.id])
        
        # 9. Energy-related constraints (if using energy optimization)
        if self.use_energy_optimization and self.solar_availability and self.grid_prices:
            self._add_energy_constraints(model, jobs, variables, horizon)
    
    def _add_collision_constraints(self, model: cp_model.CpModel, jobs: List[Job], variables: Dict):
        """
        Add collision constraints (tasks that cannot run simultaneously).
        
        For each pair of operations with collision constraint,
        ensure their intervals don't overlap.
        """
        # Collect all operations by task
        operations_by_task = {}
        for job in jobs:
            for operation in job.operations:
                task_id = operation.task_id
                if task_id not in operations_by_task:
                    operations_by_task[task_id] = []
                operations_by_task[task_id].append(operation)
        
        # For each collision constraint
        for task_id in operations_by_task.keys():
            collision_constraints = self.factory_logic.get_collision_constraints_for_task(task_id)
            
            for constraint in collision_constraints:
                other_task = (
                    constraint.task2 if constraint.task1 == task_id
                    else constraint.task1
                )
                
                if other_task not in operations_by_task:
                    continue
                
                # For each pair of operations with these tasks
                for op1 in operations_by_task[task_id]:
                    for op2 in operations_by_task[other_task]:
                        if op1.id >= op2.id:  # Avoid duplicate constraints
                            continue
                        
                        # op1 ends before op2 starts OR op2 ends before op1 starts
                        # Create boolean for each ordering
                        op1_before_op2 = model.NewBoolVar(f'collision_{op1.id}_before_{op2.id}')
                        
                        # If op1_before_op2, then op1.end <= op2.start
                        model.Add(
                            variables['end_vars'][op1.id] <= variables['start_vars'][op2.id]
                        ).OnlyEnforceIf(op1_before_op2)
                        
                        # If not op1_before_op2, then op2.end <= op1.start
                        model.Add(
                            variables['end_vars'][op2.id] <= variables['start_vars'][op1.id]
                        ).OnlyEnforceIf(op1_before_op2.Not())
    

    def _add_energy_constraints(self, model: cp_model.CpModel, jobs: List[Job], variables: Dict, horizon: int):
        """
        Add constraints for energy cost calculation.
        
        For each timestep:
        1. Calculate total power needed
        2. Calculate grid power (after solar offset)
        """
        for step in range(horizon):
            # Calculate total power at this timestep
            power_contributions = []
            
            for job in jobs:
                for operation in job.operations:
                    op_id = operation.id
                    
                    for machine_id in operation.eligible_machines:
                        task_modes = operation.eligible_task_modes.get(machine_id, [])
                        for task_mode_id in task_modes:
                            task_mode = self.factory_logic.get_task_mode(task_mode_id)
                            tm_chosen = variables['task_mode_chosen'][(op_id, machine_id, task_mode_id)]
                            
                            # For each timestep in the operation's duration
                            for time_offset, power in enumerate(task_mode.power):
                                # Check if this operation is active at 'step'
                                # Active if: start <= step < start + duration
                                # Which means: step == start + time_offset
                                
                                # Create boolean: is this exact timestep?
                                is_at_this_step = model.NewBoolVar(
                                    f'op_{op_id}_tm_{task_mode_id}_at_step_{step}_offset_{time_offset}'
                                )
                                
                                # is_at_this_step <=> (start_var == step - time_offset AND tm_chosen)
                                target_start = step - time_offset
                                
                                if 0 <= target_start <= horizon:
                                    # is_at_this_step => start == target_start
                                    model.Add(variables['start_vars'][op_id] == target_start).OnlyEnforceIf(is_at_this_step)
                                    # is_at_this_step => tm_chosen
                                    model.AddImplication(is_at_this_step, tm_chosen)
                                    
                                    # If not at this step or tm not chosen, then not is_at_this_step
                                    start_not_match = model.NewBoolVar(f'start_not_{op_id}_{step}_{time_offset}')
                                    model.Add(variables['start_vars'][op_id] != target_start).OnlyEnforceIf(start_not_match)
                                    model.AddBoolOr([is_at_this_step.Not(), tm_chosen]).OnlyEnforceIf(start_not_match)
                                    model.AddBoolOr([is_at_this_step.Not(), tm_chosen.Not()]).OnlyEnforceIf(tm_chosen.Not())
                                    
                                    # Add power contribution
                                    power_int = int(power * 1000)  # Scale to int
                                    power_contributions.append(power_int * is_at_this_step)
                                else:
                                    # Can't be at this step
                                    model.Add(is_at_this_step == 0)
            
            # Total power at step = sum of all contributions
            if power_contributions:
                model.Add(variables['power_at_step'][step] == sum(power_contributions))
            else:
                model.Add(variables['power_at_step'][step] == 0)
            
            # Grid power = max(0, total_power - solar_available)
            solar_available_int = int(self.solar_availability[step] * 1000) if step < len(self.solar_availability) else 0
            
            # grid_power >= total_power - solar
            model.Add(variables['grid_power_at_step'][step] >= variables['power_at_step'][step] - solar_available_int)
            # grid_power >= 0
            model.Add(variables['grid_power_at_step'][step] >= 0)
            # grid_power <= total_power (can't be more than total)
            model.Add(variables['grid_power_at_step'][step] <= variables['power_at_step'][step])
    
    def _set_simple_objective(self, model: cp_model.CpModel, jobs: List[Job], variables: Dict, horizon: int):
        """
        Simple objective: minimize makespan + power consumption.
        Used when energy optimization is disabled or energy data is unavailable.
        """
        makespan_term = variables['makespan'] * int(self.makespan_weight * 1000)
        
        # Sum of operation power consumption
        cost_terms = []
        for job in jobs:
            for operation in job.operations:
                op_id = operation.id
                for machine_id in operation.eligible_machines:
                    task_modes = operation.eligible_task_modes.get(machine_id, [])
                    for task_mode_id in task_modes:
                        task_mode = self.factory_logic.get_task_mode(task_mode_id)
                        total_power = sum(task_mode.power)
                        tm_chosen = variables['task_mode_chosen'][(op_id, machine_id, task_mode_id)]
                        
                        cost_terms.append(int(total_power * self.cost_weight * 1000) * tm_chosen)
        
        total_cost = sum(cost_terms) if cost_terms else 0
        
        model.Minimize(makespan_term + total_cost)
        print("Using simple objective (makespan + total power)")
    
    def _set_energy_aware_objective(self, model: cp_model.CpModel, jobs: List[Job], variables: Dict, horizon: int):
        """
        Energy-aware objective: minimize weighted sum of actual energy cost and makespan.
        
        Energy cost = sum over timesteps of (grid_power * grid_price)
        where grid_power = max(0, total_power - solar_available)
        """
        makespan_term = variables['makespan'] * int(self.makespan_weight * 1000)
        
        # Calculate energy cost for each timestep
        energy_cost_terms = []
        
        for step in range(horizon):
            grid_power = variables['grid_power_at_step'][step]
            grid_price = self.grid_prices[step] if step < len(self.grid_prices) else 0.1
            
            # Cost at this step = grid_power * grid_price
            # Both are scaled by 1000, so scale price by another 1000 for precision
            cost_at_step = grid_power * int(grid_price * 1000 * self.cost_weight)
            energy_cost_terms.append(cost_at_step)
        
        total_energy_cost = sum(energy_cost_terms) if energy_cost_terms else 0
        
        model.Minimize(makespan_term + total_energy_cost)
        print("Using energy-aware objective (solar offset + time-varying grid prices)")
    
    def _extract_solution(self, solver: cp_model.CpSolver, jobs: List[Job], variables: Dict, is_optimal: bool) -> Schedule:
        """Extract solution from solved model and convert to Schedule."""
        scheduled_ops = []
        makespan = solver.Value(variables['makespan'])
        
        # Extract scheduled operations
        for job in jobs:
            for operation in job.operations:
                op_id = operation.id
                start_time = solver.Value(variables['start_vars'][op_id])
                end_time = solver.Value(variables['end_vars'][op_id])
                
                # Find which machine and task mode were chosen
                chosen_machine = None
                chosen_task_mode = None
                
                for machine_id in operation.eligible_machines:
                    if solver.Value(variables['machine_chosen'][(op_id, machine_id)]):
                        chosen_machine = machine_id
                        
                        task_modes = operation.eligible_task_modes.get(machine_id, [])
                        for task_mode_id in task_modes:
                            tm_key = (op_id, machine_id, task_mode_id)
                            if solver.Value(variables['task_mode_chosen'][tm_key]):
                                chosen_task_mode = task_mode_id
                                break
                        break
                
                if chosen_machine and chosen_task_mode:
                    scheduled_ops.append(
                        ScheduledOperation(
                            operation_id=op_id,
                            job_id=job.id,
                            task_id=operation.task_id,
                            machine_id=chosen_machine,
                            task_mode_id=chosen_task_mode,
                            start_step=start_time,
                            end_step=end_time
                        )
                    )
        
        # Calculate actual energy cost
        energy_cost = self._calculate_actual_energy_cost(scheduled_ops, makespan)
        
        # Check deadline violations
        deadline_violations = 0
        for job in jobs:
            if job.deadline is not None:
                job_end = max(
                    (op.end_step for op in scheduled_ops if op.job_id == job.id),
                    default=0
                )
                if job_end > job.deadline:
                    deadline_violations += 1
        
        print(f"\n{'OPTIMAL' if is_optimal else 'FEASIBLE'} solution found:")
        print(f"  Operations scheduled: {len(scheduled_ops)}")
        print(f"  Makespan: {makespan}")
        print(f"  Energy cost: ${energy_cost:.2f}")
        
        return Schedule(
            operations=scheduled_ops,
            makespan=makespan,
            energy_cost=energy_cost,
            is_feasible=True,
            unscheduled_operations=0,
            deadline_violations=deadline_violations
        )
    
    def _calculate_actual_energy_cost(self, scheduled_ops: List[ScheduledOperation],makespan: int) -> float:
        """Calculate actual energy cost with solar offset and grid prices."""
        total_cost = 0.0
        
        for step in range(makespan):
            power_needed = 0.0
            
            for op in scheduled_ops:
                if op.start_step <= step < op.end_step:
                    task_mode = self.factory_logic.get_task_mode(op.task_mode_id)
                    step_in_operation = step - op.start_step
                    
                    if step_in_operation < len(task_mode.power):
                        power_needed += task_mode.power[step_in_operation]
            
            # Get solar and price
            solar_available = (
                self.solar_availability[step]
                if step < len(self.solar_availability)
                else 0.0
            )
            grid_price = (
                self.grid_prices[step]
                if step < len(self.grid_prices)
                else 0.1
            )
            
            # Calculate net grid power and cost
            grid_power = max(0, power_needed - solar_available)
            total_cost += grid_power * grid_price
        
        return total_cost