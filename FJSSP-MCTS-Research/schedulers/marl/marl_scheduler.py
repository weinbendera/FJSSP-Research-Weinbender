from schedulers.scheduler import OnlineScheduler
from factory.factory_schemas import Machine, Job, Action
from typing import List, Dict, Optional, Any
from factory.factory import Factory
from factory.factory_state import FactoryState

import numpy as np
from stable_baselines3 import PPO
import os

class MARLScheduler(OnlineScheduler):
    """
    Scheduler that uses trained MARL agents to make decisions.
    """
    
    def __init__(self, agents: Dict, max_feasible_actions: int, max_steps: int = 1152):
        """
        Args:
            agents: Dict of {machine_id: trained_model}
            max_feasible_actions: Must match training value
            max_steps: Must match training value
        """
        super().__init__()
        self.agents = agents
        self.max_feasible_actions = max_feasible_actions + 1
        self.max_steps = max_steps
    
    def choose(self, factory_state: FactoryState) -> Dict[str, Optional[Action]]:
        """
        Use trained agents to choose actions for each machine.
        This matches your existing Scheduler interface!
        """
        actions: Dict[str, Optional[Action]] = {}

        used_jobs: set[str] = set()
        for machine_id in factory_state.machine_ids:
            # Raw feasible actions for this machine
            feasible_actions = factory_state.get_feasible_actions(machine_id)
            feasible_actions = [
                feasible_action for feasible_action in feasible_actions
                if feasible_action.job_id not in used_jobs
            ]

            # Build observation using the filtered feasible actions
            observation = self._get_observation(
                machine_id=machine_id,
                factory_state=factory_state,
                feasible_actions=feasible_actions
            )
            
            # Agent predicts action index
            if machine_id in self.agents:
                action_idx, _ = self.agents[machine_id].predict(observation, deterministic=True)
                action_idx = int(action_idx)
            else:
                # Fallback: idle if agent not trained
                action_idx = 0
            
            # Convert action index to Action object using SAME filtered feasible_actions
            action, _ = self._index_to_action(
                machine_id=machine_id,
                action_idx=action_idx,
                factory_state=factory_state,
                feasible_actions=feasible_actions
            )
            actions[machine_id] = action
            
            if action:
                self.scheduled_actions.append(action)
                used_jobs.add(action.job_id)
        
        return actions
    
    def _get_observation(self, machine_id: str, factory_state: FactoryState, feasible_actions: List[Action]) -> np.ndarray:
        """
        Encode observation for one machine/agent

        Observation includes:
        - Machine-specific state (busy/idle)
        - Global factory state (time, jobs, energy)
        - Information about feasible actions
        """

        # Machine/Agent-specific features
        machine_is_busy = 1.0 if factory_state.is_machine_busy(machine_id) else 0.0
        
        # Global features (all agents see this)
        all_machines_busy = [
            1.0 if factory_state.is_machine_busy(m_id) else 0.0 
            for m_id in sorted(factory_state.machine_ids)  # Sort for consistency
        ]
        time_normalized = min(factory_state.current_step / self.max_steps, 1.0)
 
        incomplete_jobs = [j for j in factory_state.jobs if not j.done]
        jobs_remaining_normalized = len(incomplete_jobs) / max(len(factory_state.jobs), 1)

        solar = factory_state.factory_logic.get_solar_power_available(factory_state.current_step)
        solar_normalized = min(solar / 20.0, 1.0)  # Assume max 20 kWh solar
        
        grid_cost = factory_state.factory_logic.get_grid_power_cost(factory_state.current_step)
        grid_cost_normalized = min(grid_cost / 0.5, 1.0)  # Assume max $0.5/kWh
        
        num_feasible_actions_normalized = min(len(feasible_actions) / max(self.max_feasible_actions, 1), 1.0)
        
        # TODO: Add more features to the observation
        # TODO: Add deadlines to the observation
        
        observation = np.array([
            machine_is_busy,
            *all_machines_busy,
            time_normalized,
            jobs_remaining_normalized,
            num_feasible_actions_normalized,
            solar_normalized,
            grid_cost_normalized,
        ])
        
        return observation
    
    def _index_to_action(self, machine_id: str, action_idx: int, factory_state: FactoryState, feasible_actions: List[Action]) -> tuple[Optional[Action], Dict]:
        """
        Convert action index to Action object.
        Returns:
            Action object or None if no action is taken
            Details of the action
        """
        # Action 0 = do nothing
        if action_idx == 0:
            return None, {'type': 'idle', 'reason': 'chose_idle'}
        
        if not feasible_actions:
            return None, {'type': 'idle', 'reason': 'no_feasible_actions'}
        
        # Map action index to feasible action
        feasible_idx = action_idx - 1  # Offset by 1 (0 is idle)
        
        if feasible_idx >= len(feasible_actions):
            return None, {'type': 'idle', 'reason': 'invalid_action_index'}
        
        # Get the feasible action
        feasible_action = feasible_actions[feasible_idx]
        
        # Create Action object
        task_mode = factory_state.factory_logic.get_task_mode(feasible_action.task_mode_id)
        action = Action(
            machine_id=machine_id,
            job_id=feasible_action.job_id,
            operation_id=feasible_action.operation_id,
            task_mode_id=feasible_action.task_mode_id,
            start_step=factory_state.current_step,
            end_step=factory_state.current_step + len(task_mode.power)
        )
        
        details = {
            'type': 'dispatch',
            'job_id': feasible_action.job_id,
            'operation_id': feasible_action.operation_id,
            'task_mode_id': feasible_action.task_mode_id,
            'task_steps': len(task_mode.power),
            'task_energy': sum(task_mode.power),
        }
        
        return action, details

        
    @classmethod
    def load_from_directory(cls, model_dir: str, max_feasible_actions: int, max_steps: int = 1152):
        """
        Load trained agents from directory.
        
        Usage:
            scheduler = MARLScheduler.load_agents_from_directory("models/marl_independent")
        """
        
        agents = {}
        for filename in os.listdir(model_dir):
            if filename.endswith("_final.zip"):
                machine_id = filename.replace("_final.zip", "")
                model_path = os.path.join(model_dir, filename)
                agents[machine_id] = PPO.load(model_path)
                print(f"Loaded agent for {machine_id}")
        
        return cls(agents, max_feasible_actions, max_steps)