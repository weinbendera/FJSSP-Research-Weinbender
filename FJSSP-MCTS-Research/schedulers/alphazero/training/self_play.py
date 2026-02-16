"""
Self-play episode generation for AlphaZero training.

Generates training data by playing complete episodes using MCTS,
collecting (state, policy_target, value_target) tuples.
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

from schedulers.alphazero.env.fjsp_env import FJSPEnv
from schedulers.alphazero.env.graph_builder import GraphBuilder
from schedulers.alphazero.env.config_compiler import CompiledConfig
from schedulers.alphazero.model.gnn import FJSPNet
from schedulers.alphazero.mcts.mcts import MCTS, MCTSConfig


@dataclass
class TrainingExample:
    """Single training example from self-play."""
    # Stored as numpy for compact storage; rebuilt as HeteroData during batching
    op_features: np.ndarray       # (num_ops, op_feat_dim)
    machine_features: np.ndarray  # (num_machines, machine_feat_dim)
    global_features: np.ndarray   # (global_feat_dim,)
    # Sparse edge storage
    eligible_edge_src: np.ndarray  # (num_elig_edges,)
    eligible_edge_dst: np.ndarray  # (num_elig_edges,)
    precedes_edge_index: np.ndarray  # (2, num_prec_edges)
    # Targets
    policy_target: np.ndarray  # (num_actions + 1,) normalized visit counts
    value_target: float        # terminal reward (applied retroactively)
    legal_mask: np.ndarray     # (num_actions + 1,) bool


def play_episode(
    env: FJSPEnv,
    graph_builder: GraphBuilder,
    model: FJSPNet,
    mcts_config: MCTSConfig = MCTSConfig(),
    device: str = "cpu",
) -> List[TrainingExample]:
    """
    Play one complete episode using MCTS, return training examples.

    Each decision point produces one training example.
    Value targets are set retroactively to the terminal reward.
    """
    env.reset()
    mcts = MCTS(env, graph_builder, model, mcts_config, device)

    examples = []
    step_count = 0
    # Rough estimate of max steps
    max_steps = env.num_ops * env.num_machines

    while not env.done:
        legal_mask = env.get_legal_actions()
        legal_indices = np.where(legal_mask)[0]

        # Skip MCTS for forced moves (only idle or single legal action)
        if len(legal_indices) == 1:
            action = legal_indices[0]
            env.step(action)
            step_count += 1
            continue

        state = env.get_state()

        # Run MCTS search (may mutate env internally, so restore state after)
        visit_counts = mcts.search(state, step_count, max_steps)
        env.set_state(state)  # restore after MCTS exploration

        # Build policy target from visit counts
        total_visits = visit_counts.sum()
        if total_visits > 0:
            policy_target = visit_counts / total_visits
        else:
            policy_target = np.zeros_like(visit_counts)
            policy_target[-1] = 1.0  # default to idle

        # Store graph features as numpy
        graph = graph_builder.build(env)

        eligible_ei = graph["op", "eligible", "machine"].edge_index.numpy()
        precedes_ei = graph["op", "precedes", "op"].edge_index.numpy()

        example = TrainingExample(
            op_features=graph["op"].x.numpy(),
            machine_features=graph["machine"].x.numpy(),
            global_features=graph.global_features.squeeze(0).numpy(),
            eligible_edge_src=eligible_ei[0] if eligible_ei.shape[1] > 0 else np.array([], dtype=np.int64),
            eligible_edge_dst=eligible_ei[1] if eligible_ei.shape[1] > 0 else np.array([], dtype=np.int64),
            precedes_edge_index=precedes_ei,
            policy_target=policy_target,
            value_target=0.0,  # filled retroactively
            legal_mask=legal_mask,
        )
        examples.append(example)

        # Select and apply action
        action = mcts.select_action(visit_counts, step_count, max_steps)
        reward, done = env.step(action)
        step_count += 1

    # Set value targets retroactively to normalized -makespan (in [-1, 0])
    if env.done:
        makespan = float(env.get_makespan())
        terminal_value = -makespan / env.config.makespan_ub
    else:
        terminal_value = 0.0
    for ex in examples:
        ex.value_target = terminal_value

    return examples
