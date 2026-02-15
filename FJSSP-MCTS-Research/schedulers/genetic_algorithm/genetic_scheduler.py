from typing import List, Dict, Tuple, Optional
import random
import copy
import time

from schedulers.scheduler import OfflineScheduler, Schedule, ScheduledOperation
from utils.job_builder import Job, Operation
from utils.factory_logic_loader import FactoryLogic
from utils.input_schemas import EnergySource


"""
What are the Genomes?
A dictionary mapping operation ids to priority values
{operation_id: priority}

What are the Individuals?
Holds the Genome, as well metrics of the schedule
The fitness is then assessed by decoding the genome into a schedule and calculating the cost and makespan
The fitness is then used to select the best individuals for the next generation

How is the Fitness calculated?

"""


class Individual:
    """
    Represents a scheduling solution (chromosome/genome).
    Genome is a dictionary mapping operation_id to priority value.
    
    This is a schedule represented as a dictionary of operation ids to priority values
    Higher priority operations are scheduled first (when constraints allow)
    """
    def __init__(self, genome: Dict[str, float], fitness: float = 0.0) -> None:
        self.genome = genome  # Dict[operation_id, priority]
        self.fitness = fitness
        self.total_cost = float('inf')
        self.makespan = float('inf')
        self.is_feasible = True
        self.missing_count = 0
        self.deadline_violations = []
        self.schedule: Optional[Schedule] = None

class GeneticScheduler(OfflineScheduler):
    """
    Genetic Algorithm scheduler for flexible job shop scheduling.
    
    Uses priority-based encoding where each operation has a priority value.
    Schedules operations in priority order while respecting:
    - Precedence constraints from jobs
    - Collision constraints from factory
    - Machine availability
    - Deadline constraints
    
    Optimizes for:
    - Energy cost (considering solar power offset and grid prices)
    - Makespan (total schedule duration)
    """
    
    def __init__(
        self,
        factory_logic: FactoryLogic,
        population_size: int = 100,
        generations: int = 50,
        mutation_rate: float = 0.1,
        elitism_count: int = 2,
        deadline_penalty: float = 1000.0,
        cost_weight: float = 1.0,
        makespan_weight: float = 0.1
    ):
        """
        Initialize genetic algorithm scheduler.
        
        """
        super().__init__()
        self.factory_logic = factory_logic
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        self.deadline_penalty = deadline_penalty
        self.cost_weight = cost_weight
        self.makespan_weight = makespan_weight
        
        # Energy data
        self.solar_availability: List[float] = []
        self.grid_prices: List[float] = []

    def schedule(self, jobs: List[Job], energy_sources: List[EnergySource]) -> Schedule:
        """
        Schedules jobs using a genetic algorithm.
        
        Args:
            jobs: List of jobs to schedule
            energy_sources: List of energy sources with availability and pricing
        
        Returns:
            Complete Schedule with all operations assigned
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
        
        # Extract energy data from energy sources
        self._extract_energy_data(energy_sources)
        
        # Initialize population
        population = self._initialize_population(jobs)
        
        # Evaluate initial population
        for individual in population:
            self._evaluate_fitness(jobs, individual)
        
        best_individual = max(population, key=lambda ind: ind.fitness)
        
        # Evolution loop
        for generation in range(self.generations):
            # Create new population
            new_population = []
            
            # Elitism: preserve best individuals
            population.sort(key=lambda ind: ind.fitness, reverse=True)
            for i in range(self.elitism_count):
                new_population.append(copy.deepcopy(population[i]))
            
            # Generate rest through selection, crossover, mutation
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                child = self._crossover(parent1, parent2)
                
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                
                self._evaluate_fitness(jobs, child)
                new_population.append(child)
            
            population = new_population
            
            # Track best solution
            current_best = max(population, key=lambda ind: ind.fitness)
            if current_best.fitness > best_individual.fitness:
                best_individual = current_best
                self._print_generation_status(generation, best_individual)
        
        # Convert best individual to Schedule
        return self._convert_to_schedule(best_individual)

    
    def _extract_energy_data(self, energy_sources: List[EnergySource]):
        """Extract solar availability and grid prices from energy sources."""
        self.solar_availability = []
        self.grid_prices = []
        
        for energy_source in energy_sources:
            if energy_source.id == "Solar" and energy_source.availability:
                self.solar_availability = energy_source.availability
            elif energy_source.id == "Socket Energy" and energy_source.price:
                self.grid_prices = energy_source.price

    
    def _initialize_population(self, jobs: List[Job]) -> List[Individual]:
        """
        Creates initial population with random priority assignments.
        
        Gets all operation ids from all jobs
        Creates a random individual for each operation id
        This is done by setting a random priority value for each operation id in range of [0,1]
        """
        population = []
        
        # Get all operations
        all_operation_ids = []
        for job in jobs:
            for operation in job.operations:
                all_operation_ids.append(operation.id)
        
        # Set the priority for each operation id
        for _ in range(self.population_size):
            genome = {op_id: random.random() for op_id in all_operation_ids}
            population.append(Individual(genome))
        
        return population

    
    def _evaluate_fitness(self, jobs: List[Job], individual: Individual) -> float:
        """
        Evaluates the fitness of an individual by decoding the genome into a schedule
        Cal
        Evaluates fitness by decoding genome into schedule and calculating cost.
        """
        schedule, is_feasible, missing_count, deadline_violations = self._decode_genome_to_schedule(jobs, individual.genome)
        
        if not schedule:
            individual.fitness = 0.0
            individual.total_cost = float('inf')
            individual.makespan = float('inf')
            individual.is_feasible = False
            individual.missing_count = len(individual.genome)
            return 0.0
        
        individual.schedule = schedule
        individual.is_feasible = is_feasible
        individual.missing_count = missing_count
        individual.deadline_violations = deadline_violations
        
        # Calculate makespan
        makespan = max(item['end_step'] for item in schedule) if schedule else 0
        individual.makespan = makespan
        
        # Calculate energy cost with solar offset and grid prices
        total_cost = self._calculate_energy_cost(schedule, makespan)
        individual.total_cost = total_cost
        
        # Fitness calculation
        if not is_feasible:
            penalty = (missing_count * 100) + (len(deadline_violations) * self.deadline_penalty)
            individual.fitness = 1.0 / (1.0 + total_cost * self.cost_weight + makespan * self.makespan_weight + penalty)
        else:


            individual.fitness = 1.0 / (1.0 + total_cost * self.cost_weight + makespan * self.makespan_weight)
        

        
        return individual.fitness
    
    def _calculate_energy_cost(self, schedule: List[Dict], makespan: int) -> float:
        """Calculate total energy cost considering solar power and grid prices."""
        total_cost = 0.0
        
        for step in range(makespan):
            # Calculate total power needed at this timestep
            power_needed = 0.0
            
            for item in schedule:
                if item['start_step'] <= step <= item['end_step']:
                    task_mode = self.factory_logic.get_task_mode(item['task_mode_id'])
                    step_in_operation = step - item['start_step']
                    
                    if step_in_operation < len(task_mode.power):
                        power_needed += task_mode.power[step_in_operation]
            
            # Get solar power available and grid price at this timestep
            solar_available = self._get_solar_at_step(step)
            grid_price = self._get_price_at_step(step)
            
            # Calculate net power from grid (after solar offset)
            grid_power = max(0, power_needed - solar_available)
            
            # Calculate cost for this timestep
            step_cost = grid_power * grid_price
            total_cost += step_cost
        
        return total_cost
    
    def _get_solar_at_step(self, step: int) -> float:
        """Get solar power available at timestep."""
        if 0 <= step < len(self.solar_availability):
            return self.solar_availability[step]
        return 0.0
    
    def _get_price_at_step(self, step: int) -> float:
        """Get grid electricity price at timestep."""
        if 0 <= step < len(self.grid_prices):
            return self.grid_prices[step]
        return 0.0

    
    def _decode_genome_to_schedule(self, jobs: List[Job], genome: Dict[str, float]) -> Tuple[List[Dict], bool, int, List]:
        """
        Decodes priority genome into schedule.
        
        Assigns operations to machines and task modes in priority order
        while respecting precedence constraints, collision constraints, and deadline constraints
        if an operation cannot be scheduled, it is skipped this round and the next operation is considered
        this process is repeated until all operations are scheduled or we have hit max iterations
        """
        schedule = []
        machine_busy_until = {m_id: 0 for m_id in self.factory_logic.machines.keys()}
        operation_completion = {}
        is_feasible = True
        deadline_violations = []
        
        # Build operation lookup
        operation_lookup = {}
        for job in jobs:
            for operation in job.operations:
                operation_lookup[operation.id] = (job, operation)
        
        # Create priority queue
        operations_with_priority = [(op_id, priority) for op_id, priority in genome.items()]
        operations_with_priority.sort(key=lambda x: x[1], reverse=True)
        
        scheduled_operations = set()
        unscheduled_operations = set(genome.keys())
        
        # Schedule operations iteratively
        max_iterations = len(genome) * 2
        iteration = 0
        
        while unscheduled_operations and iteration < max_iterations:
            iteration += 1
            operations_scheduled_this_round = 0
            
            for operation_id, priority in operations_with_priority:
                if operation_id in scheduled_operations:
                    continue
                
                if operation_id not in operation_lookup:
                    continue
                
                job, operation = operation_lookup[operation_id]
                
                # Check precedence constraints
                if not self._precedence_satisfied(operation, job, operation_completion):
                    continue
                
                # Find best machine and task mode
                best_assignment = self._find_best_assignment(
                    operation, job, machine_busy_until, schedule, operation_completion
                )
                
                if best_assignment is None:
                    continue
                
                best_machine_id, best_task_mode_id, best_start_time, best_end_time = best_assignment
                
                # Check deadline constraint
                if job.deadline is not None and best_end_time > job.deadline:
                    is_feasible = False
                    deadline_violations.append((job.id, best_end_time, job.deadline))
                
                # Create schedule entry
                schedule_entry = {
                    'machine_id': best_machine_id,
                    'job_id': job.id,
                    'operation_id': operation_id,
                    'task_id': operation.task_id,
                    'task_mode_id': best_task_mode_id,
                    'start_step': best_start_time,
                    'end_step': best_end_time 
                }
                schedule.append(schedule_entry)
                
                # Update state
                machine_busy_until[best_machine_id] = best_end_time + 1 # inclusive end
                operation_completion[operation_id] = best_end_time + 1 # inclusive end
                scheduled_operations.add(operation_id)
                unscheduled_operations.remove(operation_id)
                operations_scheduled_this_round += 1
            
            if operations_scheduled_this_round == 0:
                break
        
        # Check if all operations were scheduled
        missing_count = len(genome) - len(scheduled_operations)
        if missing_count > 0:
            is_feasible = False
        
        return schedule, is_feasible, missing_count, deadline_violations
    
    def _find_best_assignment(self, operation: Operation, job: Job, machine_busy_until: Dict[str, int], schedule: List[Dict], operation_completion: Dict[str, int]) -> Tuple[str, str, int, int] | None:
        """
        Finds the best machine and task mode assignment for an operation
        The operations are given priority --> this method is called for each operation in priority order
        """
        best_machine_id = None
        best_task_mode_id = None
        best_start_time = float('inf')
        best_end_time = float('inf')
        
        for machine_id in operation.eligible_machines:
            task_modes = operation.eligible_task_modes.get(machine_id, [])
            
            for task_mode_id in task_modes:
                task_mode = self.factory_logic.get_task_mode(task_mode_id)
                duration = len(task_mode.power)
                
                # Get earliest possible start time considering BOTH machine AND predecessors
                earliest_start = machine_busy_until[machine_id]
                
                # Also consider when predecessor operations finish
                predecessors = [pred_id for pred_id, succ_id in job.precedence_constraints if succ_id == operation.id]
                for pred_id in predecessors:
                    if pred_id in operation_completion:
                        earliest_start = max(earliest_start, operation_completion[pred_id] + 1) # inclusive end
                
                # Adjust for time leaps - find valid start time after any breaks
                start_time = self._get_valid_start_time(earliest_start, duration)
                end_time = start_time + duration - 1 # inclusive end
                
                # Check collision constraints
                if not self._collision_satisfied(operation, schedule, start_time, end_time):
                    continue
                
                if end_time < best_end_time:
                    best_machine_id = machine_id
                    best_task_mode_id = task_mode_id
                    best_start_time = start_time
                    best_end_time = end_time
        
        if best_machine_id is None:
            return None
        
        return (best_machine_id, best_task_mode_id, best_start_time, best_end_time)

    def _get_valid_start_time(self, earliest_start: int, duration: int) -> int:
        """
        Find valid start time that doesn't conflict with time leaps.
        
        An operation can only start if:
        1. It doesn't start during a time leap
        2. It doesn't span across a time leap
        """
        current_start = earliest_start
        
        # Keep checking until we find a valid window
        max_iterations = 100  # Safety limit
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            proposed_end = current_start + duration
            
            # Check if this time window conflicts with any time leap
            next_leap = self.factory_logic.get_next_time_leap(current_start)
            
            if next_leap is None:
                return current_start
            
            if next_leap >= proposed_end:
                return current_start
            
            current_start = next_leap + 1
        
        return earliest_start
    
    def _precedence_satisfied(self, operation: Operation, job: Job, operation_completion: Dict[str, int]) -> bool:
        """Check if operation's precedence constraints are satisfied."""
        predecessors = [pred_id for pred_id, succ_id in job.precedence_constraints if succ_id == operation.id]
        
        for pred_id in predecessors:
            if pred_id not in operation_completion:
                return False
        
        return True
    
    def _collision_satisfied(self, operation: Operation, schedule: List[Dict], proposed_start: int, proposed_end: int) -> bool:
        """Check if operation can run without violating collision constraints."""
        collision_constraints = self.factory_logic.get_collision_constraints_for_task(
            operation.task_id
        )
        
        if not collision_constraints:
            return True
        
        for constraint in collision_constraints:
            other_task = (
                constraint.task2 if constraint.task1 == operation.task_id
                else constraint.task1
            )
            
            # Check if any scheduled operation with other_task overlaps in time
            for scheduled_item in schedule:
                if scheduled_item['task_id'] == other_task:
                    scheduled_start = scheduled_item['start_step']
                    scheduled_end = scheduled_item['end_step']
                    
                    # Check time overlap
                    if proposed_start < scheduled_end and proposed_end > scheduled_start:
                        return False
        
        return True

    
    def _tournament_selection(self, population: List[Individual], tournament_size: int = 3) -> Individual:
        """Tournament selection."""

        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda ind: ind.fitness)
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Uniform crossover on priority values."""

        child_genome = {}
        
        for operation_id in parent1.genome.keys():
            if random.random() < 0.5:
                child_genome[operation_id] = parent1.genome[operation_id]
            else:
                child_genome[operation_id] = parent2.genome.get(
                    operation_id,
                    parent1.genome[operation_id]
                )
        
        return Individual(child_genome)
    
    def _mutate(self, individual: Individual, mutation_strength: float = 0.2) -> Individual:
        """Mutates priority values by adding random noise."""

        for operation_id in individual.genome.keys():
            if random.random() < self.mutation_rate:
                noise = random.gauss(0, mutation_strength)
                individual.genome[operation_id] += noise
                individual.genome[operation_id] = max(0.0, min(1.0, individual.genome[operation_id]))
        return individual

    
    def _convert_to_schedule(self, individual: Individual) -> Schedule:
        """Convert Individual to Schedule object."""

        # Convert internal schedule dicts to ScheduledOperation objects
        scheduled_ops: List[ScheduledOperation] = [
            ScheduledOperation(
                operation_id=item['operation_id'],
                job_id=item['job_id'],
                task_id=item['task_id'],
                machine_id=item['machine_id'],
                task_mode_id=item['task_mode_id'],
                start_step=item['start_step'],
                end_step=item['end_step']
            )
            for item in individual.schedule
        ]
        
        return Schedule(
            operations=scheduled_ops,
            makespan=individual.makespan,
            energy_cost=individual.total_cost,
            is_feasible=individual.is_feasible,
            unscheduled_operations=individual.missing_count,
            deadline_violations=len(individual.deadline_violations)
        )
    
    def _print_generation_status(self, generation: int, individual: Individual):
        """Print status of best individual in current generation."""
        feasibility = "FEASIBLE" if individual.is_feasible else "INFEASIBLE"
        status_msg = (
            f"Gen {generation}: {feasibility} - "
            f"Fitness={individual.fitness:.4f}, "
            f"Cost=${individual.total_cost:.2f}, "
            f"Makespan={individual.makespan}"
        )
        
        if not individual.is_feasible:
            if individual.missing_count > 0:
                status_msg += f" | Missing: {individual.missing_count}"
            if individual.deadline_violations:
                status_msg += f" | Violations: {len(individual.deadline_violations)}"
        
        print(status_msg)