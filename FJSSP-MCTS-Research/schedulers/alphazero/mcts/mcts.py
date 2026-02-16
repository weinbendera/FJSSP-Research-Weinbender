"""
AlphaZero MCTS for FJSP scheduling.

Features:
  - PUCT selection (c_puct configurable)
  - Lazy expansion: priors stored on expand, child nodes created on-demand
  - Value network evaluation (no rollouts)
  - Dirichlet noise at root
  - Temperature-based action selection
  - Running min/max value normalization
"""

import math
import numpy as np
import torch
from typing import Optional, Dict, Tuple
from dataclasses import dataclass, field

from schedulers.alphazero.env.fjsp_env import FJSPEnv, EnvState
from schedulers.alphazero.env.graph_builder import GraphBuilder
from schedulers.alphazero.model.gnn import FJSPNet


@dataclass
class MCTSConfig:
    num_simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature_threshold: float = 0.3  # fraction of total steps after which temp drops
    temperature_high: float = 1.0
    temperature_low: float = 0.1


class Node:
    """MCTS tree node with lazy expansion."""

    __slots__ = [
        "state", "parent", "action", "prior",
        "visit_count", "value_sum",
        "children", "legal_mask", "child_priors",
        "is_expanded", "is_terminal", "terminal_value",
    ]

    def __init__(
        self,
        state: Optional[EnvState] = None,
        parent: Optional["Node"] = None,
        action: int = -1,
        prior: float = 0.0,
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior

        self.visit_count = 0
        self.value_sum = 0.0

        self.children: Dict[int, "Node"] = {}
        self.legal_mask: Optional[np.ndarray] = None
        self.child_priors: Optional[np.ndarray] = None

        self.is_expanded = False
        self.is_terminal = False
        self.terminal_value = 0.0

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    """AlphaZero MCTS search."""

    def __init__(
        self,
        env: FJSPEnv,
        graph_builder: GraphBuilder,
        model: FJSPNet,
        config: MCTSConfig = MCTSConfig(),
        device: str = "cpu",
    ):
        self.env = env
        self.graph_builder = graph_builder
        self.model = model
        self.config = config
        self.device = device

        # Running min/max for value normalization
        self.value_min = float("inf")
        self.value_max = float("-inf")

    def search(self, root_state: EnvState, total_steps: int = 0, max_steps: int = 1) -> np.ndarray:
        """
        Run MCTS from root_state, return action visit counts.

        Args:
            root_state: current environment state
            total_steps: how many steps have been taken (for temperature)
            max_steps: estimated total steps (for temperature threshold)

        Returns:
            visit_counts: (num_actions + 1,) array of visit counts
        """
        root = Node(state=root_state)
        self._expand(root)

        # Add Dirichlet noise at root
        if root.child_priors is not None:
            noise = np.random.dirichlet(
                [self.config.dirichlet_alpha] * len(root.child_priors)
            )
            eps = self.config.dirichlet_epsilon
            legal = root.legal_mask
            noisy_priors = root.child_priors.copy()
            noisy_priors[legal] = (
                (1 - eps) * root.child_priors[legal] + eps * noise[:int(legal.sum())]
            )
            # Re-normalize over legal actions
            legal_sum = noisy_priors[legal].sum()
            if legal_sum > 0:
                noisy_priors[legal] /= legal_sum
            root.child_priors = noisy_priors

        for _ in range(self.config.num_simulations):
            node = root
            self.env.set_state(root_state)

            # --- Selection ---
            while node.is_expanded and not node.is_terminal:
                action, node = self._select_child(node)
                # Apply action to env
                if node.state is not None:
                    self.env.set_state(node.state)
                else:
                    # Lazy: apply action from parent state
                    self.env.set_state(node.parent.state)
                    reward, done = self.env.step(action)
                    node.state = self.env.get_state()
                    if done:
                        node.is_terminal = True
                        # Normalize terminal value to [-1, 0] like training targets
                        node.terminal_value = reward / self.env.config.makespan_ub

            # --- Expansion + Evaluation ---
            if node.is_terminal:
                value = node.terminal_value
            elif not node.is_expanded:
                value = self._expand(node)
            else:
                value = node.value

            # --- Backpropagation ---
            self._backpropagate(node, value)

        # Collect visit counts
        num_total = self.env.num_actions + 1
        visit_counts = np.zeros(num_total, dtype=np.float32)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count

        return visit_counts

    def select_action(
        self,
        visit_counts: np.ndarray,
        total_steps: int = 0,
        max_steps: int = 1,
    ) -> int:
        """Select action from visit counts using temperature."""
        fraction = total_steps / max(max_steps, 1)
        if fraction < self.config.temperature_threshold:
            temp = self.config.temperature_high
        else:
            temp = self.config.temperature_low

        if temp < 0.01:
            # Greedy
            return int(np.argmax(visit_counts))

        # Temperature-weighted sampling
        counts = visit_counts ** (1.0 / temp)
        total = counts.sum()
        if total == 0:
            # Fallback: uniform over legal
            return int(np.argmax(visit_counts))
        probs = counts / total
        return int(np.random.choice(len(probs), p=probs))

    def _expand(self, node: Node) -> float:
        """Expand node: evaluate with network, store priors. Returns value estimate."""
        self.env.set_state(node.state)

        legal_mask = self.env.get_legal_actions()
        node.legal_mask = legal_mask

        # Evaluate with network
        graph = self.graph_builder.build(self.env)
        legal_tensor = torch.tensor(legal_mask, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            graph = graph.to(self.device)
            policy, value = self.model(graph, legal_tensor)
            policy = policy.cpu().numpy()
            value = float(value.cpu())

        node.child_priors = policy
        node.is_expanded = True

        return value

    def _select_child(self, node: Node) -> Tuple[int, "Node"]:
        """Select best child using PUCT."""
        best_score = float("-inf")
        best_action = -1
        best_child = None

        legal_actions = np.where(node.legal_mask)[0]
        parent_visits = node.visit_count

        for action in legal_actions:
            action = int(action)

            if action in node.children:
                child = node.children[action]
                q = self._normalize_value(child.value)
                u = (self.config.c_puct * node.child_priors[action] *
                     math.sqrt(parent_visits) / (1 + child.visit_count))
                score = q + u
            else:
                # Unexpanded child: Q=0, high exploration bonus
                q = 0.0
                u = (self.config.c_puct * node.child_priors[action] *
                     math.sqrt(parent_visits))
                score = q + u

            if score > best_score:
                best_score = score
                best_action = action
                best_child = node.children.get(action)

        # Lazy expansion: create child node if not exists
        if best_child is None:
            best_child = Node(
                state=None,  # State computed lazily
                parent=node,
                action=best_action,
                prior=node.child_priors[best_action],
            )
            node.children[best_action] = best_child

        return best_action, best_child

    def _backpropagate(self, node: Node, value: float) -> None:
        """Backpropagate value up the tree."""
        # Update running min/max
        if value < self.value_min:
            self.value_min = value
        if value > self.value_max:
            self.value_max = value

        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += value
            current = current.parent

    def _normalize_value(self, value: float) -> float:
        """Normalize value to [0, 1] using running min/max."""
        if self.value_max > self.value_min:
            return (value - self.value_min) / (self.value_max - self.value_min)
        return 0.5
