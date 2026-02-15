import gymnasium as gym
from pettingzoo import ParallelEnv
import numpy as np
from typing import Dict, Optional
from copy import deepcopy

from factory.factory import Factory
from factory.factory_state import FactoryState
from factory.factory_schemas import Action


class FactoryMultiAgentEnv(ParallelEnv):
    """
    Factory Gym Environment for Multi-Agent Reinforcement Learning
    Wraps the Factory class to make it compatible with the PettingZoo ParallelEnv interface.
    Each machine is an independent agent
    """
    
    def __init__(self, factory_logic, initial_jobs, max_steps:int = 1152):
        super().__init__()
        # set up the factory
        self.factory_logic = factory_logic
        self.factory = Factory(factory_logic)
        self.initial_jobs = initial_jobs
        self.factory.add_jobs(deepcopy(self.initial_jobs))
        
        self.current_episode_step = 0
        self.max_episode_steps = max_steps
        self.total_episode_power_cost = 0

        self.max_feasible_actions = self._calculate_max_feasible_actions()

        # set up the agents
        self.possible_agents = list(factory_logic.machines.keys())  # Required by PettingZoo
        self.agents = self.possible_agents.copy()
        # Define action and observation spaces per agent
        self._setup_spaces()

    def _setup_spaces(self):
        """
        Define the observation and action spaces for each agent.
        """
        observation_size = self._get_observation_size()
        self.observation_spaces = {
            agent: gym.spaces.Box(low=0, high=1, shape=(observation_size,))
            for agent in self.agents
        }
        self.action_spaces = {
            agent: gym.spaces.Discrete(self.max_feasible_actions + 1)  # Max feasible actions + 1 (do nothing)
            for agent in self.agents
        }

    
    def reset(self, seed=None, options=None):
        """
        Resets the env for a new episode for training.
        """
        if seed is not None:
            np.random.seed(seed)

        if options is not None:
            self.max_episode_steps = options.get('max_episode_steps', self.max_episode_steps)

        # reset the factory
        self.factory.reset()
        self.factory.add_jobs(deepcopy(self.initial_jobs))

        self.current_episode_step = 0
        self.total_episode_power_cost = 0

        # Calculate max feasible actions for this episode (for normalization)
        self.max_feasible_actions = self._calculate_max_feasible_actions()

        # reset the agents' observations
        observations = {
            agent: self._get_observation(agent)
            for agent in self.agents
        }

        infos = {
            agent: {
                'current_episode_step': self.current_episode_step,
            }
            for agent in self.agents
        }

        return observations, infos
    
    def step(self, actions: Dict[str, int]):
        """

        """
        factory_state = self.factory.get_factory_state()

        # Get the action and details for each agent
        tentative_actions = {}
        action_details = {}
        for agent, action_idx in actions.items():
            action, details = self._index_to_action(agent, action_idx, factory_state)
            tentative_actions[agent] = action # Action object
            action_details[agent] = details # Details of the action
            # print(f"DEBUG: Agent: {agent}, Action: {action}, Details: {details}")
        
        # Filter out any actions whose job is already taken this step
        used_jobs: set[str] = set()
        factory_actions: Dict[str, Optional[Action]] = {}

        for machine_id in factory_state.machine_ids:
            action = tentative_actions.get(machine_id)

            if action is None:
                factory_actions[machine_id] = None
                continue

            if action.job_id in used_jobs:
                # remove the action from the tentative actions
                factory_actions[machine_id] = None
                continue

            factory_actions[machine_id] = action
            used_jobs.add(action.job_id)

        # Apply to factory
        self.factory.apply_actions(factory_actions)

        # Step the factory step
        step_info = self.factory.step()
        
        # Calculate rewards per agent
        rewards = self._calculate_rewards(step_info, action_details)
        
        # Get new agent observations
        observations = {
            agent: self._get_observation(agent)
            for agent in self.agents
        }
        
        # Check termination
        terminated = self.factory.done() # True if all jobs are done
        truncated = self.current_episode_step >= self.max_episode_steps and not terminated
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        
        infos = {
            agent: {
                'step_info': step_info,
                'action_details': action_details[agent],
                'current_episode_step': self.current_episode_step,
            }
            for agent in self.agents
        }
        self.current_episode_step += 1

        return observations, rewards, terminations, truncations, infos
    
    def _get_observation(self, machine_id: str) -> np.ndarray:
        """
        Encode observation for one machine/agent

        Observation includes:
        - Machine-specific state (busy/idle)
        - Global factory state (time, jobs, energy)
        - Information about feasible actions
        """
        factory_state = self.factory.get_factory_state()
        
        # Machine/Agent-specific features
        # TODO : is this needed?
        machine_is_busy = 1.0 if factory_state.is_machine_busy(machine_id) else 0.0
        
        # Global features (all agents see this)
        all_machines_busy = [
            1.0 if factory_state.is_machine_busy(m_id) else 0.0 
            for m_id in sorted(factory_state.machine_ids)  # Sort for consistency
        ]
        time_normalized = min(factory_state.current_step / self.max_episode_steps, 1.0)
 
        incomplete_jobs = [j for j in factory_state.jobs if not j.done]
        jobs_remaining_normalized = len(incomplete_jobs) / max(len(factory_state.jobs), 1)

        solar = factory_state.factory_logic.get_solar_power_available(factory_state.current_step)
        solar_normalized = min(solar / 20.0, 1.0)  # Assume max 20 kWh solar
        
        grid_cost = factory_state.factory_logic.get_grid_power_cost(factory_state.current_step)
        grid_cost_normalized = min(grid_cost / 0.5, 1.0)  # Assume max $0.5/kWh
        
        feasible_actions = factory_state.get_feasible_actions(machine_id)
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

    def _get_observation_size(self) -> int:
        """
        Calculate the observation vector size.
        Must be fixed for all agents and all timesteps.
        """
        num_machines = len(self.possible_agents)
        
        size = (
            1 +              # machine_is_busy
            num_machines +   # all machines busy status (fixed size)
            1 +              # time_normalized
            1 +              # jobs_remaining_normalized
            1 +              # num_feasible_actions_normalized
            1 +              # solar_normalized
            1               # grid_cost_normalized
        )
        
        return size
    
    def _index_to_action(self, machine_id: str, action_idx: int, factory_state: FactoryState) -> tuple[Optional[Action], Dict]:
        """
        Convert action index to Action object.
        Returns:
            Action object or None if no action is taken
            Details of the action
        """
        # Action 0 = do nothing
        if action_idx == 0:
            return None, {'type': 'idle', 'reason': 'chose_idle'}
        
        feasible_actions = factory_state.get_feasible_actions(machine_id)

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
    
    def _calculate_rewards(self, step_info: Dict, action_details: Dict[str, Dict]) -> Dict[str, float]:
        """
        Calculate rewards for each agent
        Returns:
            Dictionary of agent ids to rewards
        """
        # shared rewards
        step_power_cost = -step_info['step_power_cost']
        job_completion_bonus = len(step_info['newly_completed_jobs']) * 10.0

        shared_reward = step_power_cost + job_completion_bonus

        # calculate reward for each agent
        rewards = {}
        for agent in self.agents:
            agent_reward = shared_reward
            
            # penalty for being idle when feasible actions available
            if action_details[agent]['type'] == 'idle':
                if action_details[agent]['reason'] == 'chose_idle':
                    agent_reward -= 0.1 
            
            rewards[agent] = agent_reward
        
        return rewards

    def _calculate_max_feasible_actions(self) -> int:
        """
        Calculate the maximum number of feasible actions any machine could have.
        
        This is called AFTER jobs are added to the factory.
        Returns an integer representing the action space size.
        """
        factory_state = self.factory.get_factory_state()
        
        # Get max across all machines
        max_feasible_actions = max(
            len(factory_state.get_feasible_actions(machine_id)) 
            for machine_id in factory_state.machine_ids
        )
        
        # Add 20% safety buffer
        max_feasible_actions = int(max_feasible_actions * 1.2)
        
        # Ensure minimum of 10
        max_feasible_actions = max(10, max_feasible_actions)
        
        return max_feasible_actions

    @property
    def observation_space(self) -> gym.Space:
        return self.observation_spaces[self.agents[0]]
    
    @property
    def action_space(self) -> gym.Space:
        return self.action_spaces[self.agents[0]]