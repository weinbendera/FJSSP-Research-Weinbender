# AlphaZero for Flexible Job Shop Scheduling (FJSP) — Implementation Reference

## Project Overview

This document serves as a technical reference for implementing an AlphaZero-based scheduler for a Flexible Job Shop Scheduling Problem (FJSP) using real textile manufacturing data. The goal is to minimize makespan using a GNN-based policy-value network guided by Monte Carlo Tree Search (MCTS).

This is a stepping stone toward a future MuZero implementation. Get AlphaZero working end-to-end first.

---

## 1. Problem Definition

### FJSP Setup

- **Domain**: Textile manufacturing
- **Machines**: 3 machines with flexible routing (operations can run on multiple eligible machines)
- **Jobs**: Configurable set of jobs, each decomposed into ordered operations
- **Precedence**: Few ordering constraints per job (some operations must precede others)
- **Objective**: Minimize makespan (total schedule completion time)
- **Time discretization**: 5-minute steps; all operation durations are multiples of 5 minutes

### Key Terminology

| Term                  | Definition                                                                       |
| --------------------- | -------------------------------------------------------------------------------- |
| Job                   | A complete unit of work (e.g., a textile order) composed of multiple operations  |
| Operation             | An atomic task within a job with a fixed duration and a set of eligible machines |
| Makespan              | Time from schedule start to completion of the last operation                     |
| Eligible machines     | The subset of machines that can perform a given operation                        |
| Precedence constraint | An ordering requirement: operation A must complete before operation B starts     |

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   Training Loop                      │
│                                                      │
│   Self-Play ──► Replay Buffer ──► Network Training   │
│       │                               │              │
│       └───────────────────────────────┘              │
└─────────────────────────────────────────────────────┘

Self-Play Detail:
┌──────────────────────────────────────────────────────┐
│  For each 5-min step:                                │
│    For each machine (sequential):                    │
│      1. Build bipartite graph from current state     │
│      2. GNN encodes graph → policy prior + value     │
│      3. MCTS uses prior + value to search            │
│      4. Select action from MCTS visit counts         │
│      5. Update state (assign operation to machine)   │
│    Advance environment by one 5-min step             │
└──────────────────────────────────────────────────────┘
```

---

## 3. Environment Design

### 3.1 Framework

- **Library**: Gymnasium (gym)
- **Language**: Python + PyTorch

### 3.2 Sequential Decision Model

At each 5-minute timestep, machines are queried **sequentially** in a fixed order (Machine 0 → Machine 1 → Machine 2). Each machine either:

- **Is busy** (mid-operation): No decision needed, automatically skip.
- **Is idle**: Must choose from eligible operations or remain idle.

Each sub-decision (one per idle machine) constitutes a separate MCTS search. The state updates between sub-decisions within the same timestep, so later machines see earlier machines' choices.

```python
class FJSPEnv(gym.Env):
    """
    Flexible Job Shop Scheduling Environment

    Observation: Bipartite graph (operation nodes <-> machine nodes)
    Action: Index into the current machine's eligible operations (+ idle action)
    Reward: Negative makespan at episode end (0 for intermediate steps)
    """

    def __init__(self, jobs_config, num_machines=3, step_duration=5):
        super().__init__()
        self.num_machines = num_machines
        self.step_duration = step_duration  # minutes per timestep
        self.jobs_config = jobs_config

        # Track which machine is currently deciding
        self.current_machine_idx = 0

    def reset(self):
        """
        Initialize all jobs/operations. No operations assigned yet.
        Returns: initial observation (bipartite graph), info dict
        """
        # Initialize job/operation data structures
        # Set current_time = 0
        # Set current_machine_idx = 0
        # Return initial graph observation
        pass

    def step(self, action):
        """
        Assign the chosen operation to the current machine, or idle.

        Args:
            action: int - index into current machine's eligible operations
                    (last index = idle/do nothing)

        Returns:
            observation: bipartite graph for NEXT machine's decision
            reward: 0 for intermediate steps, -makespan at episode end
            terminated: True when all operations are scheduled and complete
            truncated: False (no truncation)
            info: dict with debugging info
        """
        # 1. Apply action to current machine
        # 2. Advance current_machine_idx
        # 3. If all machines have decided this timestep:
        #      - Advance time by step_duration
        #      - Decrement remaining processing times
        #      - Free machines whose operations completed
        #      - Reset current_machine_idx = 0
        #      - Skip to next machine that needs a decision
        # 4. If no more decisions needed and all ops done: terminated = True
        # 5. Build and return new observation
        pass

    def get_legal_actions(self):
        """
        Returns a binary mask over the action space for the current machine.
        An operation is legal if:
          - It has not been assigned/completed
          - Its precedence constraints are satisfied
          - The current machine is in its eligible machine set
          - It is not currently assigned to another machine
        The idle action is always legal.
        """
        pass

    def get_state(self):
        """
        Returns a hashable/copyable state for MCTS tree nodes.
        Must capture full environment state for perfect simulation.
        """
        pass

    def set_state(self, state):
        """
        Restores environment to a previously captured state.
        Required for MCTS backtracking.
        """
        pass
```

### 3.3 State Representation (Internal)

The environment must track:

```python
@dataclass
class EnvironmentState:
    current_time: int                    # Current timestep (in 5-min increments)
    current_machine_idx: int             # Which machine is currently deciding

    # Per-operation state
    op_status: np.ndarray                # 0=unscheduled, 1=in_progress, 2=completed
    op_remaining_time: np.ndarray        # Steps remaining for in-progress ops
    op_assigned_machine: np.ndarray      # -1 if unassigned, else machine index

    # Per-machine state
    machine_status: np.ndarray           # 0=idle, 1=busy
    machine_current_op: np.ndarray       # -1 if idle, else operation index
    machine_remaining_steps: np.ndarray  # Steps until current op completes

    # Static data (does not change during episode)
    op_durations: np.ndarray             # Duration of each op in steps (per machine)
    op_eligible_machines: List[List[int]]# Which machines can do each operation
    precedence_edges: List[Tuple[int,int]]  # (predecessor_op, successor_op)
    op_to_job: np.ndarray                # Maps operation index to job index
```

### 3.4 Reward Design

```python
def compute_reward(self):
    if self.is_terminal():
        # Negative makespan so that minimizing makespan = maximizing reward
        makespan = self.current_time  # or track actual completion time
        return -makespan
    return 0.0  # No intermediate reward
```

> **Design note**: Sparse reward (only at episode end) is standard for AlphaZero since MCTS handles credit assignment through full-game simulation. Do not add shaping rewards — they bias the value function.

### 3.5 Episode Termination

An episode ends when **all operations** have status `completed`. This means:

- Every operation has been assigned to a machine
- Every operation has finished processing (remaining_time reached 0)

---

## 4. Graph Neural Network (GNN)

### 4.1 Bipartite Graph Structure

The state is represented as a **heterogeneous bipartite graph** with two node types and one edge type:

```
Node Types:
  - Operation nodes (one per operation in the problem)
  - Machine nodes (one per machine, i.e., 3 nodes)

Edge Type:
  - "eligible": connects operation i to machine j if machine j can process operation i
  - Edges are UNDIRECTED (message passing flows both ways)

Dynamic behavior:
  - Edges are REMOVED when an operation is assigned/completed
    (it no longer needs to be scheduled)
  - Node features UPDATE every step to reflect current state
```

### 4.2 Node Features

#### Operation Node Features

```python
operation_features = {
    "status":              # one-hot [unscheduled, in_progress, completed] → 3 dims
    "duration_normalized": # processing time / max_processing_time → 1 dim
    "remaining_time_norm": # remaining steps / duration (0 if not started) → 1 dim
    "job_progress":        # fraction of job's operations completed → 1 dim
    "precedence_depth":    # max chain length from this op to job end → 1 dim
    "num_eligible":        # number of eligible machines (normalized) → 1 dim
    "is_ready":            # 1 if all predecessors completed, 0 otherwise → 1 dim
    "is_schedulable":      # 1 if ready AND unscheduled → 1 dim
}
# Total: ~10 dimensions per operation node
```

#### Machine Node Features

```python
machine_features = {
    "status":              # one-hot [idle, busy] → 2 dims
    "remaining_time_norm": # steps until free / max_op_duration → 1 dim
    "utilization":         # fraction of elapsed time spent busy → 1 dim
    "queue_load":          # number of eligible unscheduled ops (normalized) → 1 dim
    "is_current":          # 1 if this is the machine currently deciding → 1 dim
}
# Total: ~6 dimensions per machine node
```

#### Global Features (appended after pooling or as a separate context)

```python
global_features = {
    "time_normalized":     # current_time / estimated_max_time → 1 dim
    "overall_progress":    # fraction of all operations completed → 1 dim
    "current_machine_idx": # one-hot encoding of deciding machine → 3 dims
}
# Total: ~5 dimensions
```

### 4.3 GNN Architecture

Use PyTorch Geometric (PyG) for the heterogeneous GNN.

```
Input: HeteroData graph with operation & machine nodes + eligible edges
    │
    ▼
Heterogeneous Message Passing (2-3 layers)
    │  Each layer:
    │    op → machine: aggregate info from eligible operations
    │    machine → op: aggregate info from eligible machines
    │    + residual connections + LayerNorm
    │
    ▼
Node Embeddings: op_embeddings (N_ops × D), machine_embeddings (3 × D)
    │
    ▼
Global Pooling: mean(all node embeddings) + global_features → context vector
    │
    ├──────────────────────┐
    ▼                      ▼
Policy Head            Value Head
```

```python
import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool

class FJSPNet(nn.Module):
    """
    GNN-based policy-value network for FJSP AlphaZero.
    """
    def __init__(self, op_feature_dim, machine_feature_dim, global_feature_dim,
                 hidden_dim=128, num_gnn_layers=3):
        super().__init__()

        # Input projections
        self.op_encoder = nn.Linear(op_feature_dim, hidden_dim)
        self.machine_encoder = nn.Linear(machine_feature_dim, hidden_dim)

        # Heterogeneous GNN layers
        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for _ in range(num_gnn_layers):
            conv = HeteroConv({
                ('operation', 'eligible', 'machine'): SAGEConv(hidden_dim, hidden_dim),
                ('machine', 'rev_eligible', 'operation'): SAGEConv(hidden_dim, hidden_dim),
            }, aggr='mean')
            self.gnn_layers.append(conv)
            self.layer_norms.append(nn.ModuleDict({
                'operation': nn.LayerNorm(hidden_dim),
                'machine': nn.LayerNorm(hidden_dim),
            }))

        # Policy head: scores eligible operations for the current machine
        # Input: concatenation of [op_embedding, current_machine_embedding]
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # scalar score per operation
        )

        # Idle action embedding (learnable)
        self.idle_embedding = nn.Parameter(torch.randn(hidden_dim))

        # Value head: predicts schedule quality from global state
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim + global_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # normalized value in [-1, 1]
        )

    def forward(self, hetero_data, current_machine_idx, legal_action_mask):
        """
        Args:
            hetero_data: PyG HeteroData with operation and machine nodes
            current_machine_idx: int, which machine is deciding
            legal_action_mask: boolean tensor, True for legal actions

        Returns:
            policy: probability distribution over legal actions (including idle)
            value: scalar state value estimate
        """
        # 1. Encode input features
        x_dict = {
            'operation': self.op_encoder(hetero_data['operation'].x),
            'machine': self.machine_encoder(hetero_data['machine'].x),
        }

        # 2. GNN message passing with residual connections
        for conv, norms in zip(self.gnn_layers, self.layer_norms):
            x_update = conv(x_dict, hetero_data.edge_index_dict)
            for node_type in x_dict:
                x_dict[node_type] = norms[node_type](
                    x_dict[node_type] + x_update[node_type]
                )

        # 3. Policy: score each eligible operation for the current machine
        current_machine_emb = x_dict['machine'][current_machine_idx]  # (D,)
        op_embeddings = x_dict['operation']  # (N_ops, D)

        # Concatenate each op embedding with current machine embedding
        machine_expanded = current_machine_emb.unsqueeze(0).expand(op_embeddings.size(0), -1)
        policy_input = torch.cat([op_embeddings, machine_expanded], dim=-1)  # (N_ops, 2D)
        op_scores = self.policy_head(policy_input).squeeze(-1)  # (N_ops,)

        # Add idle action score
        idle_input = torch.cat([self.idle_embedding, current_machine_emb])
        idle_score = self.policy_head(idle_input.unsqueeze(0)).squeeze()
        all_scores = torch.cat([op_scores, idle_score.unsqueeze(0)])  # (N_ops + 1,)

        # Apply legal action mask and softmax
        all_scores[~legal_action_mask] = float('-inf')
        policy = torch.softmax(all_scores, dim=0)

        # 4. Value: global pooling + value head
        all_embeddings = torch.cat([x_dict['operation'], x_dict['machine']], dim=0)
        global_emb = all_embeddings.mean(dim=0)
        global_input = torch.cat([global_emb, hetero_data.global_features])
        value = self.value_head(global_input)

        return policy, value
```

### 4.4 Key Implementation Notes for GNN

- **Use `torch_geometric`** (PyG) for heterogeneous graph support
- **Install**: `pip install torch-geometric`
- **SAGEConv** is a good starting point; can upgrade to GATConv (attention) later
- **Residual connections + LayerNorm** are critical for training stability with 3+ layers
- **The graph structure changes each step** — rebuild `HeteroData` from environment state each time
- **Batch graphs** across MCTS simulations for GPU efficiency during self-play

---

## 5. Monte Carlo Tree Search (MCTS)

### 5.1 AlphaZero MCTS Overview

MCTS in AlphaZero uses the neural network to **guide search** rather than random rollouts. At each node:

1. **Select**: Traverse tree using PUCT formula until reaching a leaf
2. **Expand**: Use the neural network to get policy prior and value estimate for the leaf
3. **Backpropagate**: Update visit counts and value estimates up the tree

There are **no rollouts** — the value network replaces them.

### 5.2 PUCT Selection Formula

```
PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))

Where:
  Q(s, a)     = mean value of action a from state s (average of backed-up values)
  P(s, a)     = prior probability from the policy network
  N(s)        = total visit count of state s
  N(s, a)     = visit count of action a from state s
  c_puct      = exploration constant (start with 1.5, tune later)
```

### 5.3 MCTS Implementation

```python
class MCTSNode:
    def __init__(self, state, parent=None, action=None, prior=0.0):
        self.state = state           # Environment state (copyable)
        self.parent = parent
        self.action = action         # Action that led to this node
        self.prior = prior           # P(s, a) from policy network

        self.children = {}           # action -> MCTSNode
        self.visit_count = 0         # N(s, a)
        self.value_sum = 0.0         # Total backed-up value
        self.is_expanded = False

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    def __init__(self, network, env, num_simulations=100, c_puct=1.5):
        self.network = network
        self.env = env
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def search(self, root_state):
        """
        Run MCTS from the given state. Returns action probabilities
        based on visit counts.
        """
        root = MCTSNode(state=root_state)
        self._expand(root)

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            # SELECT: traverse tree using PUCT
            while node.is_expanded and node.children:
                action, node = self._select_child(node)
                search_path.append(node)

            # EXPAND + EVALUATE
            value = self._expand(node)

            # BACKPROPAGATE
            self._backpropagate(search_path, value)

        # Return visit count distribution as action probabilities
        return self._get_action_probs(root)

    def _select_child(self, node):
        """Select child with highest PUCT score."""
        best_score = -float('inf')
        best_action = None
        best_child = None

        for action, child in node.children.items():
            puct_score = (
                child.q_value +
                self.c_puct * child.prior *
                math.sqrt(node.visit_count) / (1 + child.visit_count)
            )
            if puct_score > best_score:
                best_score = puct_score
                best_action = action
                best_child = child

        return best_action, best_child

    def _expand(self, node):
        """
        Expand a leaf node:
        1. Set environment to node's state
        2. Get legal actions
        3. Run GNN to get policy prior and value
        4. Create child nodes for each legal action
        """
        self.env.set_state(node.state)

        if self.env.is_terminal():
            # Terminal node: return actual normalized reward
            return self._normalize_value(self.env.get_makespan())

        legal_mask = self.env.get_legal_actions()
        graph = self.env.build_graph()  # Build PyG HeteroData
        current_machine = self.env.current_machine_idx

        with torch.no_grad():
            policy, value = self.network(graph, current_machine, legal_mask)

        # Create children for each legal action
        legal_actions = torch.where(legal_mask)[0]
        for action in legal_actions:
            action_idx = action.item()
            child_env = self.env.copy()
            child_env.set_state(node.state)
            child_env.step(action_idx)

            child = MCTSNode(
                state=child_env.get_state(),
                parent=node,
                action=action_idx,
                prior=policy[action_idx].item()
            )
            node.children[action_idx] = child

        node.is_expanded = True
        return value.item()

    def _backpropagate(self, search_path, value):
        """Update visit counts and values up the search path."""
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            # No sign flip — this is single-agent, not adversarial

    def _get_action_probs(self, root, temperature=1.0):
        """
        Convert visit counts to action probabilities.
        temperature=1.0 for exploration, temperature→0 for exploitation.
        """
        actions = list(root.children.keys())
        visits = np.array([root.children[a].visit_count for a in actions])

        if temperature == 0:
            # Deterministic: pick most-visited
            probs = np.zeros_like(visits, dtype=np.float32)
            probs[np.argmax(visits)] = 1.0
        else:
            visits_temp = visits ** (1.0 / temperature)
            probs = visits_temp / visits_temp.sum()

        return dict(zip(actions, probs))

    def _normalize_value(self, makespan):
        """
        Normalize makespan to [-1, 1] range for the value head.
        Use running min/max or fixed bounds from problem analysis.
        """
        # Option 1: Fixed normalization based on problem bounds
        min_possible = self.env.get_lower_bound()  # e.g., critical path length
        max_possible = self.env.get_upper_bound()   # e.g., sum of all durations
        # Normalized so that better (shorter) makespan → higher value
        return -2.0 * (makespan - min_possible) / (max_possible - min_possible) + 1.0
```

### 5.4 Critical MCTS Design Decisions

| Decision                | Choice                                | Rationale                                                   |
| ----------------------- | ------------------------------------- | ----------------------------------------------------------- |
| Rollout policy          | None (value network only)             | Standard AlphaZero; no random rollouts                      |
| Number of simulations   | Start with 100-200                    | 3 machines × few ops = small branching factor              |
| c_puct                  | 1.5                                   | Standard starting point; tune based on exploration behavior |
| Temperature             | 1.0 for first 30% of steps, 0.1 after | Explore early, exploit late in each episode                 |
| Dirichlet noise at root | α=0.3, ε=0.25                       | Standard AlphaZero exploration noise                        |
| Value normalization     | Map makespan to [-1, 1]               | Required for Tanh value head                                |

### 5.5 State Copy Performance

MCTS requires copying and restoring environment states thousands of times per move. This **must be fast**.

```python
import copy

class FJSPEnv:
    def get_state(self):
        """Return a lightweight, copyable state snapshot."""
        return EnvironmentState(
            current_time=self.current_time,
            current_machine_idx=self.current_machine_idx,
            op_status=self.op_status.copy(),
            op_remaining_time=self.op_remaining_time.copy(),
            op_assigned_machine=self.op_assigned_machine.copy(),
            machine_status=self.machine_status.copy(),
            machine_current_op=self.machine_current_op.copy(),
            machine_remaining_steps=self.machine_remaining_steps.copy(),
            # Static data: reference only (don't copy)
            op_durations=self.op_durations,
            op_eligible_machines=self.op_eligible_machines,
            precedence_edges=self.precedence_edges,
            op_to_job=self.op_to_job,
        )

    def set_state(self, state):
        """Restore from snapshot. Only copy dynamic arrays."""
        self.current_time = state.current_time
        self.current_machine_idx = state.current_machine_idx
        self.op_status = state.op_status.copy()
        self.op_remaining_time = state.op_remaining_time.copy()
        self.op_assigned_machine = state.op_assigned_machine.copy()
        self.machine_status = state.machine_status.copy()
        self.machine_current_op = state.machine_current_op.copy()
        self.machine_remaining_steps = state.machine_remaining_steps.copy()
```

> **Performance tip**: Profile `get_state`/`set_state` early. With 100+ MCTS simulations per move and 3 machines per step, you'll call these thousands of times per episode. Use numpy array copies, not deepcopy.

---

## 6. Training Loop

### 6.1 Self-Play Data Generation

```python
def self_play_episode(env, network, mcts_config):
    """
    Play one full episode using MCTS to generate training data.

    Returns:
        training_examples: list of (graph, policy_target, value_target)
    """
    training_examples = []
    state = env.reset()
    mcts = MCTS(network, env, **mcts_config)
    step_count = 0

    while not env.is_terminal():
        # Determine temperature
        temperature = 1.0 if step_count < total_steps * 0.3 else 0.1

        # Run MCTS from current state
        current_state = env.get_state()
        action_probs = mcts.search(current_state)

        # Store training example (graph, policy target, placeholder value)
        graph = env.build_graph()
        legal_mask = env.get_legal_actions()
        machine_idx = env.current_machine_idx

        # Convert action_probs dict to full-size policy vector
        policy_target = np.zeros(env.action_space_size)
        for action, prob in action_probs.items():
            policy_target[action] = prob

        training_examples.append({
            'graph': graph,
            'machine_idx': machine_idx,
            'legal_mask': legal_mask,
            'policy_target': policy_target,
            'value_target': None  # filled in after episode
        })

        # Sample action from MCTS probabilities
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        action = np.random.choice(actions, p=probs)

        env.step(action)
        step_count += 1

        # Add Dirichlet noise at root for next search
        # (handled inside MCTS.search)

    # Fill in value targets: all positions get the final outcome
    final_value = mcts._normalize_value(env.get_makespan())
    for example in training_examples:
        example['value_target'] = final_value

    return training_examples
```

### 6.2 Network Training

```python
def train_network(network, optimizer, replay_buffer, batch_size=256, epochs=10):
    """
    Train the policy-value network on self-play data.
    """
    for epoch in range(epochs):
        batch = replay_buffer.sample(batch_size)

        # Forward pass
        policies, values = [], []
        for example in batch:
            policy, value = network(
                example['graph'],
                example['machine_idx'],
                example['legal_mask']
            )
            policies.append(policy)
            values.append(value)

        policies = torch.stack(policies)
        values = torch.stack(values).squeeze()

        # Targets
        policy_targets = torch.stack([ex['policy_target'] for ex in batch])
        value_targets = torch.tensor([ex['value_target'] for ex in batch])

        # Loss = cross-entropy(policy) + MSE(value)
        policy_loss = -torch.sum(policy_targets * torch.log(policies + 1e-8), dim=1).mean()
        value_loss = nn.functional.mse_loss(values, value_targets)
        total_loss = policy_loss + value_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return policy_loss.item(), value_loss.item()
```

### 6.3 Full Training Pipeline

```python
def train_alphazero(env_config, num_iterations=100, games_per_iteration=50,
                    num_simulations=100, batch_size=256):
    """
    Main AlphaZero training loop.
    """
    env = FJSPEnv(**env_config)
    network = FJSPNet(
        op_feature_dim=10,
        machine_feature_dim=6,
        global_feature_dim=5,
        hidden_dim=128,
        num_gnn_layers=3
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-4)
    replay_buffer = ReplayBuffer(max_size=100_000)

    for iteration in range(num_iterations):
        # 1. Self-play: generate training data
        network.eval()
        for game in range(games_per_iteration):
            examples = self_play_episode(env, network, {
                'num_simulations': num_simulations,
                'c_puct': 1.5
            })
            replay_buffer.extend(examples)

        # 2. Train network on collected data
        network.train()
        policy_loss, value_loss = train_network(
            network, optimizer, replay_buffer, batch_size
        )

        # 3. Evaluate (optional: compare against previous best)
        if iteration % 10 == 0:
            makespan = evaluate(env, network, num_games=20)
            print(f"Iter {iteration}: makespan={makespan:.1f}, "
                  f"policy_loss={policy_loss:.4f}, value_loss={value_loss:.4f}")

        # 4. Learning rate schedule (optional)
        if iteration in [50, 80]:
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.1
```

### 6.4 Replay Buffer

```python
class ReplayBuffer:
    def __init__(self, max_size=100_000):
        self.buffer = deque(maxlen=max_size)

    def extend(self, examples):
        self.buffer.extend(examples)

    def sample(self, batch_size):
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)
```

---

## 7. Hyperparameters

### Recommended Starting Configuration

```yaml
# Environment
num_machines: 3
step_duration: 5  # minutes

# GNN
hidden_dim: 128
num_gnn_layers: 3
gnn_type: SAGEConv  # upgrade to GATConv later if needed

# MCTS
num_simulations: 100       # increase to 200-400 if compute allows
c_puct: 1.5
dirichlet_alpha: 0.3       # noise parameter
dirichlet_epsilon: 0.25    # fraction of noise mixed into prior
temperature_threshold: 0.3 # fraction of episode steps before switching to low temp
temperature_high: 1.0
temperature_low: 0.1

# Training
learning_rate: 1e-3
weight_decay: 1e-4
batch_size: 256
games_per_iteration: 50    # self-play games before training
training_epochs: 10        # passes over sampled data per iteration
replay_buffer_size: 100000
num_iterations: 100        # outer loop iterations

# Loss weights
policy_loss_weight: 1.0
value_loss_weight: 1.0
```

---

## 8. Build Order & Milestones

Follow this order strictly. Each milestone should be tested independently before moving on.

### Phase 1: Environment (Week 1)

- [ ] Define job/operation data structures from textile data
- [ ] Implement `FJSPEnv` with reset, step, is_terminal
- [ ] Implement sequential machine decision logic
- [ ] Implement `get_legal_actions` with precedence checking
- [ ] Implement `get_state` / `set_state` (fast copy)
- [ ] **Test**: Run environment with random actions, verify:
  - All operations get scheduled and completed
  - Precedence constraints are never violated
  - Makespan calculation is correct
  - `get_state`/`set_state` round-trips perfectly

### Phase 2: Graph Construction (Week 1-2)

- [ ] Implement `build_graph()` that converts env state to PyG HeteroData
- [ ] Define operation node features (10 dims)
- [ ] Define machine node features (6 dims)
- [ ] Define global features (5 dims)
- [ ] Build edge_index for eligible edges
- [ ] **Test**: Visualize graphs at different timesteps, verify features make sense

### Phase 3: GNN + Policy/Value Heads (Week 2)

- [ ] Implement `FJSPNet` with heterogeneous message passing
- [ ] Implement policy head with legal action masking
- [ ] Implement value head
- [ ] **Test**: Forward pass with random graph data, verify output shapes
- [ ] **Test**: Policy sums to 1.0, only legal actions have nonzero probability

### Phase 4: MCTS (Week 2-3)

- [ ] Implement MCTSNode and MCTS search
- [ ] Implement PUCT selection
- [ ] Implement expansion with neural network evaluation
- [ ] Implement backpropagation
- [ ] Implement temperature-based action selection
- [ ] Add Dirichlet noise at root
- [ ] **Test**: Run MCTS with random network, verify:
  - Visit counts accumulate correctly
  - Higher-prior actions get more visits initially
  - Action probabilities are valid distributions
  - Search produces deterministic results with fixed seed

### Phase 5: Self-Play + Training (Week 3-4)

- [ ] Implement self-play episode generation
- [ ] Implement replay buffer
- [ ] Implement training loop with policy + value loss
- [ ] Wire up full pipeline: self-play → buffer → train → repeat
- [ ] **Test**: Verify loss decreases over iterations
- [ ] **Test**: Compare learned policy against random baseline

### Phase 6: Evaluation & Baselines (Week 4+)

- [ ] Implement dispatching rule baselines (SPT, LPT, FIFO, random)
- [ ] Track makespan over training iterations
- [ ] Visualize Gantt charts of produced schedules
- [ ] Profile and optimize bottlenecks (likely state copy and GNN forward pass)

---

## 9. Multi-GPU Training Notes

With 8-14 GPUs available on the school supercomputer:

**Self-play parallelism** is the easiest win. Each GPU runs independent self-play games, feeding examples into a shared replay buffer.

```
GPU 0:     Network Training (main)
GPU 1-7:   Self-play workers (each generates games independently)

Flow:
  1. Broadcast current network weights to all workers
  2. Workers generate self-play games in parallel
  3. Collect all examples into shared replay buffer
  4. Train on GPU 0
  5. Repeat
```

Use `torch.distributed` or `torch.multiprocessing` for this. Ray is also a clean option for managing workers.

---

## 10. Common Pitfalls & Debugging

| Issue                         | Symptom                                       | Fix                                                                                               |
| ----------------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Value head not learning       | Value loss stays flat                         | Check normalization range matches Tanh output; verify terminal rewards are correct                |
| Policy ignores priors         | MCTS visit counts don't correlate with priors | Check c_puct isn't too low; verify prior probabilities are valid                                  |
| Illegal actions selected      | Environment crashes                           | Verify legal_mask is correct; ensure mask is applied BEFORE softmax                               |
| Slow MCTS                     | Training is bottlenecked on search            | Profile get_state/set_state; batch GNN forward passes; reduce num_simulations initially           |
| Schedule never completes      | Episode runs forever                          | Check idle action logic; ensure time always advances; add max_steps safety limit                  |
| Precedence violated           | Invalid schedules produced                    | Unit test precedence checking independently; verify it's checked in get_legal_actions             |
| GNN outputs NaN               | Training crashes                              | Add gradient clipping; check for log(0) in policy loss; verify LayerNorm placement                |
| All actions equal probability | Policy not learning                           | Check that policy targets from MCTS are not uniform; verify loss gradients flow through correctly |

---

## 11. File Structure

```
alphazero_fjsp/
├── README.md
├── requirements.txt
├── config/
│   ├── default.yaml           # Hyperparameters
│   └── textile_jobs.yaml      # Job/operation definitions from textile data
├── env/
│   ├── __init__.py
│   ├── fjsp_env.py            # FJSPEnv gymnasium environment
│   ├── state.py               # EnvironmentState dataclass
│   └── graph_builder.py       # Converts env state to PyG HeteroData
├── model/
│   ├── __init__.py
│   ├── gnn.py                 # FJSPNet (GNN + policy/value heads)
│   └── utils.py               # Network utilities
├── mcts/
│   ├── __init__.py
│   └── mcts.py                # MCTS implementation
├── training/
│   ├── __init__.py
│   ├── self_play.py           # Self-play episode generation
│   ├── trainer.py             # Network training loop
│   ├── replay_buffer.py       # Experience replay
│   └── pipeline.py            # Full AlphaZero training pipeline
├── evaluation/
│   ├── __init__.py
│   ├── baselines.py           # Dispatching rule baselines (SPT, LPT, etc.)
│   ├── evaluate.py            # Evaluation harness
│   └── visualization.py       # Gantt chart plotting
├── scripts/
│   ├── train.py               # Entry point for training
│   └── evaluate.py            # Entry point for evaluation
└── tests/
    ├── test_env.py            # Environment unit tests
    ├── test_graph.py          # Graph construction tests
    ├── test_mcts.py           # MCTS correctness tests
    └── test_network.py        # Network shape/output tests
```

---

## 12. Dependencies

```
# requirements.txt
torch>=2.0
torch-geometric>=2.4
gymnasium>=0.29
numpy>=1.24
pyyaml>=6.0
matplotlib>=3.7        # Gantt charts
tqdm>=4.65
tensorboard>=2.14      # Training logging (optional)
```

---

## 13. Key References

- Silver et al., "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" (AlphaZero, 2017)
- Song et al., "Flexible Job-Shop Scheduling via Graph Neural Network and Deep Reinforcement Learning" (2022)
- Park et al., "Learning to Schedule Job-Shop Problems: Representation and Policy Learning Using Graph Neural Network and Reinforcement Learning" (2021)
- Scholl et al., "A Survey on Neural Combinatorial Optimization" (2022)

---

## Appendix A: Quick Sanity Checks

Run these before any training to catch bugs early:

```python
# 1. Environment determinism
env = FJSPEnv(config)
env.reset(seed=42)
state1 = env.get_state()
env.step(0)
env.set_state(state1)
env.step(0)
state2 = env.get_state()
assert states_equal(state1_after_step, state2)  # Must be identical

# 2. Legal actions correctness
env.reset()
while not env.is_terminal():
    legal = env.get_legal_actions()
    assert legal.any(), "No legal actions but not terminal!"
    action = np.random.choice(np.where(legal)[0])
    env.step(action)

# 3. Graph construction shapes
graph = env.build_graph()
assert graph['operation'].x.shape[1] == OP_FEATURE_DIM
assert graph['machine'].x.shape[1] == MACHINE_FEATURE_DIM
assert graph['operation', 'eligible', 'machine'].edge_index.shape[0] == 2

# 4. Network output validity
policy, value = network(graph, machine_idx=0, legal_mask=legal)
assert torch.allclose(policy.sum(), torch.tensor(1.0), atol=1e-5)
assert -1 <= value <= 1
assert (policy[~legal] == 0).all()

# 5. MCTS produces valid distributions
action_probs = mcts.search(env.get_state())
assert abs(sum(action_probs.values()) - 1.0) < 1e-5
for action in action_probs:
    assert legal[action], f"MCTS selected illegal action {action}"
```
