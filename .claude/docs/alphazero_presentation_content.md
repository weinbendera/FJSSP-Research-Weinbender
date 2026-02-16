# AlphaZero for Flexible Job Shop Scheduling — Presentation Content

This document contains the content for a slide deck explaining the AlphaZero-based FJSP scheduler.

---

## Slide: Problem Overview

**Flexible Job Shop Scheduling Problem (FJSP)**

- We schedule operations across machines in a textile manufacturing facility
- Each job consists of one or more operations
- Each operation can run on a subset of eligible machines, each with multiple task modes (different speed/energy tradeoffs)
- The scheduler must choose: which operation to assign to which machine, using which task mode
- **Objective**: Minimize makespan (time from start to completion of all operations)

**Constraints:**

- Precedence: Some operations must finish before others can start
- Collision: Certain task types cannot run simultaneously on any machine
- Job exclusivity: Only one operation per job can be in-progress at a time
- Time leaps: Operations cannot cross day boundaries
- Deadlines: Some operations have hard deadlines

---



## Slide: Sequential Decision Model

At each 5-minute timestep, machines are queried sequentially:

1. **Machine 0** observes the current state → chooses an action (assign an operation or idle)
2. **Machine 1** sees Machine 0's choice → makes its own decision
3. **Machine 2** sees both previous choices → makes its decision
4. **Time advances** by one step (5 minutes)
5. Machines that finish their operations are freed
6. Repeat until all operations are completed

Each machine decision is a separate MCTS search — later machines see earlier machines' choices within the same timestep.

---

## Slide: State Representation — Heterogeneous Graph

The scheduling state is represented as a **heterogeneous graph** with two node types:

**Operation Nodes** — one per operation in the problem
**Machine Nodes** — one per machine (3 in our factory)

**Edge Types:**

- **Eligible edges** (operation → machine): Connect each unscheduled operation to the machines that can process it. These edges are **dynamic** — they disappear as operations are completed.
- **Precedence edges** (operation → operation): Encode ordering constraints between operations. These are **static** throughout the episode.

This graph structure naturally captures the relational information in the scheduling problem — which operations compete for which machines, and which must happen in order.

---

## Slide: Operation Node Features (10 dimensions)

Each operation node carries a feature vector encoding its current state:

| Index | Feature                     | Description                                                                  |
| ----- | --------------------------- | ---------------------------------------------------------------------------- |
| 0     | Is unscheduled              | 1 if the operation has not been assigned yet                                 |
| 1     | Is in-progress              | 1 if the operation is currently running on a machine                         |
| 2     | Is completed                | 1 if the operation has finished                                              |
| 3     | Duration                    | Mean processing time across eligible task modes (normalized by max duration) |
| 4     | Remaining time              | Time steps left until completion (normalized, 0 if not started)              |
| 5     | Job progress                | Fraction of this operation's job that is already completed                   |
| 6     | Precedence depth            | How deep this operation is in the precedence chain (normalized by max depth) |
| 7     | Number of eligible machines | How many machines can process this operation (normalized by max)             |
| 8     | Is ready                    | 1 if all predecessor operations are completed and operation is unscheduled   |
| 9     | Is schedulable              | 1 if ready AND the job has no other operation currently running              |

---

## Slide: Machine Node Features (6 dimensions)

Each machine node carries a feature vector encoding its current state:

| Index | Feature        | Description                                                                 |
| ----- | -------------- | --------------------------------------------------------------------------- |
| 0     | Is idle        | 1 if the machine is not currently processing an operation                   |
| 1     | Is busy        | 1 if the machine is currently processing an operation                       |
| 2     | Remaining time | Time steps until current operation completes (normalized by max duration)   |
| 3     | Utilization    | Fraction of elapsed time this machine has been busy                         |
| 4     | Queue load     | Number of schedulable operations for this machine (normalized by total ops) |
| 5     | Is current     | 1 if this is the machine currently making a decision                        |

---

## Slide: Global Features (5 dimensions)

Graph-level features providing overall scheduling context:

| Index | Feature               | Description                                       |
| ----- | --------------------- | ------------------------------------------------- |
| 0     | Time progress         | Current timestep normalized by total time horizon |
| 1     | Overall progress      | Fraction of all operations completed              |
| 2     | Is Machine 0 deciding | 1 if Machine 0 is the current decision maker      |
| 3     | Is Machine 1 deciding | 1 if Machine 1 is the current decision maker      |
| 4     | Is Machine 2 deciding | 1 if Machine 2 is the current decision maker      |

---

## Slide: GNN Architecture

The graph neural network processes the heterogeneous graph to produce operation and machine embeddings:

**Input Projection** — Separate linear layers project raw features into a shared hidden dimension for each node type

**Message Passing (3 layers)** — Each layer performs three simultaneous message-passing operations:

- Operations send information to their eligible machines ("what work is available?")
- Machines send information back to eligible operations ("how busy am I?")
- Operations send information along precedence edges ("what depends on me?")

Each layer uses **residual connections** (add the input back to the output) and **layer normalization** for training stability.

After 3 rounds of message passing, each node's embedding captures information from its multi-hop neighborhood in the graph.

---

## Slide: Policy Head — Action Scoring

The policy head scores every possible action to produce a probability distribution:

For each valid **(operation, machine, task mode)** triple:

1. Look up the **operation embedding** from the GNN
2. Look up the **machine embedding** from the GNN
3. Look up the **task mode features** (normalized duration and power consumption)
4. Concatenate all three → feed through a small MLP → produces a scalar score

Additionally, a **learnable idle embedding** is scored the same way, representing the "do nothing" action.

A **legal action mask** sets illegal actions to negative infinity, then **softmax** converts scores to a probability distribution over legal actions.

---

## Slide: Value Head — State Evaluation

The value head estimates how good the current scheduling state is:

1. **Mean-pool** all operation embeddings → single vector
2. **Mean-pool** all machine embeddings → single vector
3. **Concatenate** with global features
4. Feed through MLP with **tanh** activation → scalar in [-1, 1]

The value represents the expected quality of the final schedule from this state, where values closer to 0 are better (lower makespan) and values closer to -1 are worse.

---

## Slide: Monte Carlo Tree Search (MCTS)

MCTS is the planning algorithm that uses the neural network to search ahead from the current state.

**Key insight**: The neural network provides an initial "guess" (policy prior), and MCTS refines it by simulating future decisions.

**No random rollouts** — unlike classical MCTS, AlphaZero replaces rollouts with the value network's estimate. This is faster and more accurate once the network is trained.

---

## Slide: MCTS Search Process

Each search consists of many **simulations** (typically 100). Each simulation has 4 phases:

**1. Selection** — Starting from the root, traverse the tree by picking the child with the highest PUCT score until reaching an unexpanded node.

**2. Expansion** — Run the GNN on the new state to get policy priors (what actions look promising) and a value estimate (how good is this state).

**3. Backpropagation** — Send the value estimate back up the tree, updating visit counts and average values for every node along the path.

**4. Action Selection** — After all simulations, choose an action proportional to how many times each child was visited.

---

## Slide: PUCT Selection Formula

During selection, the algorithm balances **exploitation** (actions that have performed well) with **exploration** (actions the network thinks are promising but haven't been tried much):

$$
\text{PUCT}(s, a) = Q(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}
$$

Where:

- **Q(s, a)** = average value of taking action *a* from state *s* (exploitation)
- **P(s, a)** = neural network's prior probability for action *a* (guides exploration)
- **N(s)** = total visit count of state *s*
- **N(s, a)** = visit count of action *a* from state *s*
- **c_puct** = exploration constant (1.5) — controls the exploitation/exploration balance

Early in search, the prior P(s,a) dominates (network-guided exploration). As visits accumulate, Q(s,a) dominates (empirical quality).

---

## Slide: Lazy Expansion

A key optimization: child nodes are created **on-demand**, not all at once.

When a node is expanded, we store the network's policy priors but **don't create child nodes yet**. Child nodes and their states are computed only when first selected by PUCT.

**Why this matters**: With 50+ legal actions per state, eagerly creating all children would waste memory and computation on branches that MCTS never explores.

---

## Slide: Exploration Noise

To ensure diverse training data, **Dirichlet noise** is added to the root node's priors:

$$
P'(s, a) = (1 - \varepsilon) \cdot P(s, a) + \varepsilon \cdot \eta_a
$$

Where:

- **ε = 0.25** — fraction of noise mixed in
- **η ~ Dir(α)** with **α = 0.3** — random noise from a Dirichlet distribution

This prevents the search from always following the same path, encouraging the discovery of new strategies.

---

## Slide: Temperature-Based Action Selection

After MCTS completes, actions are selected from the visit count distribution:

$$
\pi(a) = \frac{N(s, a)^{1/\tau}}{\sum_b N(s, b)^{1/\tau}}
$$

- **High temperature (τ = 1.0)** in the first 30% of the episode → more exploration, diverse training data
- **Low temperature (τ = 0.1)** in the remaining 70% → nearly greedy, exploiting the best found actions

This mirrors AlphaZero's original approach: explore early, exploit late.

---

## Slide: Value Normalization

**Challenge**: The value head outputs values in [-1, 1] (tanh), but raw makespans can be large numbers (e.g., 300-1000 time steps).

**Solution**: Normalize value targets by dividing by an upper bound:

$$
v_{\text{target}} = \frac{-\text{makespan}}{\text{makespan}_{\text{ub}}}
$$

Where makespan_ub is the worst-case makespan (sum of the longest duration for every operation, as if everything ran serially). This maps targets to roughly [-1, 0].

Within MCTS, Q-values are also normalized to [0, 1] using a **running min/max** across all values seen during search.

---

## Slide: Reward Design

- **Intermediate steps**: Reward = 0 (no shaping rewards)
- **Terminal state**: Reward = -makespan

**Why sparse rewards?** MCTS handles credit assignment through look-ahead search. Adding intermediate shaping rewards would bias the value function and interfere with MCTS's ability to evaluate states objectively.

The negative sign means minimizing makespan = maximizing reward, which aligns with standard reinforcement learning conventions.

---

## Slide: Self-Play

Self-play is how AlphaZero generates its own training data

**One episode of self-play:**

1. Start with an empty schedule
2. At each decision point, run MCTS to get a visit count distribution (the "improved policy")
3. Record the current state features, the visit distribution, and the legal action mask
4. Sample an action from the visit distribution and apply it
5. Continue until all operations are scheduled
6. **Retroactively** label all recorded states with the final schedule quality (-makespan / upper_bound)

**Forced-move optimization**: When only one action is legal (e.g., machine is busy → must idle), skip MCTS entirely. This saves 60-70% of computation since most decision points are forced.

---

## Slide: Training Loop

Each training **iteration** consists of two phases:

**Phase 1: Self-Play (data generation)**

- Play N complete scheduling episodes using the current network + MCTS
- Each episode produces training examples: (graph features, MCTS policy, final schedule quality)
- Store all examples in a replay buffer

**Phase 2: Network Training (learning)**

- Sample random batches from the replay buffer
- For each example, compute:
  - **Policy loss**: Cross-entropy between network's output and MCTS visit distribution
  - **Value loss**: Mean squared error between network's value prediction and actual schedule quality
- Update network weights via gradient descent

$$
\mathcal{L} = \underbrace{-\sum_a \pi_{\text{MCTS}}(a) \log \pi_{\text{network}}(a)}_{\text{policy loss}} + \underbrace{(v_{\text{network}} - v_{\text{target}})^2}_{\text{value loss}}
$$

---

## Slide: The Self-Improvement Cycle

The key insight of AlphaZero is the **virtuous cycle** between the network and MCTS:

1. **MCTS improves the policy** — By searching ahead, MCTS finds better actions than the raw network would suggest. The visit distribution is a "corrected" policy.
2. **The network learns from MCTS** — Training on MCTS visit distributions teaches the network to directly output better policies, making future MCTS searches start from a stronger baseline.
3. **Better network → better MCTS → better training data → better network → ...**

Over many iterations, the network internalizes the strategic knowledge that MCTS discovers through search, while MCTS continues to find improvements beyond what the network alone can predict.

---

## Slide: Replay Buffer

Training examples are stored in a **circular replay buffer** (capacity: 100,000 examples).

**Why a replay buffer?**

- **Decorrelation**: Random sampling breaks the temporal correlation between consecutive states in an episode
- **Data efficiency**: Each example can be used for multiple training updates
- **Stability**: The network trains on a mix of recent and slightly older data, preventing overfitting to the latest games

Examples are stored as compact numpy arrays and reconstructed into graph format only during training.

---
