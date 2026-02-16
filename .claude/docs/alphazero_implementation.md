# AlphaZero for Flexible Job Shop Scheduling (FJSP) — Implementation Reference

## Project Overview

This document describes the implemented AlphaZero-based scheduler for a Flexible Job Shop Scheduling Problem (FJSP) using real textile manufacturing data. The goal is to minimize makespan using a GNN-based policy-value network guided by Monte Carlo Tree Search (MCTS).

This is a stepping stone toward a future MuZero implementation.

---

## 1. Problem Definition

### FJSP Setup

- **Domain**: Textile manufacturing
- **Machines**: 3 machines with flexible routing (operations can run on multiple eligible machines)
- **Jobs**: Configurable set of jobs, each decomposed into ordered operations
- **Precedence**: Few ordering constraints per job (some operations must precede others)
- **Collision constraints**: Certain task types cannot run simultaneously across any machines
- **Time leaps**: Day boundaries that operations cannot cross
- **Objective**: Minimize makespan (total schedule completion time)
- **Time discretization**: 5-minute steps; all operation durations are multiples of 5 minutes

### Key Terminology

| Term | Definition |
|------|------------|
| Job | A complete unit of work (e.g., a textile order) composed of one or more operations |
| Operation | An atomic task within a job with eligible machines and task modes |
| Task mode | A specific way to perform an operation on a machine (determines duration and power) |
| Action triple | A valid (operation, machine, task_mode) combination |
| Makespan | Time from schedule start to completion of the last operation |
| Eligible machines | The subset of machines that can perform a given operation |
| Precedence constraint | An ordering requirement: operation A must complete before B starts |
| Collision constraint | Two task types that cannot run simultaneously on any machines |
| Time leap | Day boundary that operations cannot cross |

---

## 2. Architecture Overview

```
Training Loop:
  Self-Play ──► Replay Buffer ──► Network Training ──► (repeat)
      │                                │
      └────────────────────────────────┘

Self-Play Detail (per episode):
  env.reset()
  while not done:
    For each machine M0 → M1 → M2 (sequential):
      1. If only 1 legal action → skip MCTS (forced move)
      2. Otherwise:
         a. Build HeteroData graph from current state
         b. GNN encodes graph → policy prior + value estimate
         c. MCTS uses prior + value to search (100 simulations)
         d. Record (graph features, visit counts, legal mask) as training example
         e. Select action from MCTS visit counts (temperature-based)
      3. Apply action, update state
    After all 3 machines decide → advance time by 1 step
  Set all value targets retroactively to -makespan / makespan_ub
```

---

## 3. Environment Design

### 3.1 Config Compiler (`env/config_compiler.py`)

Bridges between string-keyed domain objects (FactoryLogic, Job, Operation) and fast numpy arrays. All lookups become integer indexing.

**Key compiled arrays:**
- `action_op`, `action_machine`, `action_task_mode`, `action_duration` — all valid (op, machine, task_mode) triples
- `machine_action_mask` — (num_machines, num_actions) boolean for per-machine action filtering
- `op_to_job`, `op_to_task` — maps operations to their job and task type
- `op_predecessors` — ragged list of predecessor op indices per operation
- `precedence_pairs` — (num_precedence, 2) array of (pred, succ) indices
- `collision_pairs` — task type pairs that cannot run simultaneously
- `time_leaps` — day boundary timesteps
- `makespan_ub` — upper bound (sum of max duration per op) for value normalization
- String ID lookups for converting back to Schedule output

**Usage:**
```python
factory_logic = FactoryLogicLoader.load_from_file(data_path)
jobs = JobBuilder(factory_logic).build_jobs(product_requests)
config = CompiledConfig.compile(factory_logic, jobs)
```

### 3.2 Sequential Decision Model (`env/fjsp_env.py`)

Pure numpy environment — no Gymnasium dependency.

At each timestep, machines are queried **sequentially** in fixed order (M0 → M1 → M2). Each machine either:
- **Is busy**: Only idle is legal (forced move, skipped by MCTS in self-play)
- **Is idle**: Chooses from eligible (op, machine, task_mode) actions or idle

After all 3 machines decide, `_advance_time()` runs: decrements remaining times, frees completed machines, checks terminal condition.

**Action space**: Integer index into the compiled action table (0 to num_actions-1 = action triples, num_actions = idle).

### 3.3 Mutable State (~2.5KB for 60 ops, 3 machines)

```python
class EnvState(NamedTuple):
    op_status: np.ndarray           # (num_ops,) int8 — 0=unscheduled, 1=in_progress, 2=completed
    op_remaining_time: np.ndarray   # (num_ops,) int16
    op_assigned_machine: np.ndarray # (num_ops,) int8 — -1 if unassigned
    op_assigned_task_mode: np.ndarray # (num_ops,) int16
    op_start_step: np.ndarray       # (num_ops,) int16
    op_duration: np.ndarray         # (num_ops,) int16
    machine_busy: np.ndarray        # (num_machines,) bool
    machine_current_op: np.ndarray  # (num_machines,) int16
    machine_remaining: np.ndarray   # (num_machines,) int16
    job_being_processed: np.ndarray # (num_jobs,) bool
    current_time: int
    current_machine_idx: int
    done: bool
```

`get_state()` / `set_state()` use numpy `.copy()` — no deepcopy. Called thousands of times per MCTS search.

### 3.4 Legal Action Constraints

An action (op, machine, task_mode) is legal if:
1. The action is for the **current machine**
2. The operation is **unscheduled**
3. All **predecessors** are completed
4. The **job** is not currently being processed (one-op-per-job-at-a-time)
5. No **collision conflict** with currently running task types
6. The operation won't cross a **time leap** boundary
7. **Deadline** not already passed (if applicable)

Idle is always legal.

### 3.5 Reward Design

- Intermediate steps: reward = 0
- Terminal: reward = -makespan (raw negative makespan)
- Value targets for training are normalized: `-makespan / makespan_ub` (maps to roughly [-1, 0])

---

## 4. Graph Neural Network

### 4.1 Graph Structure (`env/graph_builder.py`)

Heterogeneous graph (PyG `HeteroData`) with:

**Node types:**
- `"op"`: One per operation (10 features)
- `"machine"`: One per machine (6 features)

**Edge types:**
- `("op", "eligible", "machine")`: Dynamic — unscheduled ops to their eligible machines. Shrinks as ops complete.
- `("op", "precedes", "op")`: Static — precedence constraints between operations.

**Global features:** Stored as `data.global_features` (5 dims for 3 machines).

### 4.2 Node Features

**Operation features (10 dims):**
| Index | Feature | Description |
|-------|---------|-------------|
| 0-2 | status one-hot | unscheduled / in_progress / completed |
| 3 | duration_norm | mean duration of eligible task modes / max_duration |
| 4 | remaining_time_norm | remaining steps / max_duration |
| 5 | job_progress | fraction of job's ops completed |
| 6 | precedence_depth_norm | max chain length from root / max_depth |
| 7 | num_eligible_norm | number of eligible machines / max_eligible |
| 8 | is_ready | 1 if all predecessors completed and unscheduled |
| 9 | is_schedulable | 1 if ready AND job not currently being processed |

**Machine features (6 dims):**
| Index | Feature | Description |
|-------|---------|-------------|
| 0-1 | status one-hot | idle / busy |
| 2 | remaining_time_norm | steps until free / max_duration |
| 3 | utilization | fraction of elapsed time spent busy |
| 4 | queue_load | schedulable ops for this machine / num_ops |
| 5 | is_current | 1 if this is the machine currently deciding |

**Global features (2 + num_machines dims):**
| Index | Feature | Description |
|-------|---------|-------------|
| 0 | time_norm | current_time / max_time |
| 1 | overall_progress | fraction of all ops completed |
| 2-4 | current_machine one-hot | which machine is deciding |

### 4.3 GNN Architecture (`model/gnn.py`)

```
Input Features
    │
    ▼
Input Projection (Linear + ReLU per node type)
    │
    ▼
3x HeteroConv layers:
    SAGEConv: op → machine (via "eligible" edges)
    SAGEConv: machine → op (via "rev_eligible" edges, flipped)
    SAGEConv: op → op (via "precedes" edges)
    + Residual connections + LayerNorm per node type
    │
    ├─────────────────────────────────────┐
    ▼                                     ▼
Policy Head                          Value Head
    │                                     │
For each action triple:              mean(op_emb) + mean(machine_emb)
  concat [op_emb, machine_emb,       + global_features
          task_mode_features]             │
    → MLP → scalar logit             MLP → tanh → scalar [-1, 1]
    │
+ learnable idle embedding → idle logit
    │
Apply legal mask (-inf for illegal)
    → softmax → probability distribution
```

**Key details:**
- Hidden dim: 64 (configurable)
- Policy scores ALL action triples (not just current machine's), then legal mask zeros out irrelevant ones
- Task mode features: [duration_normalized, total_power_normalized] (2 dims, precomputed as buffer)
- Action → op/machine/task_mode index mappings stored as registered buffers for efficient GPU lookup
- Aggregation: `"sum"` in HeteroConv

---

## 5. Monte Carlo Tree Search (`mcts/mcts.py`)

### 5.1 Key Design: Lazy Expansion

Child nodes are created **on-demand** during selection, not all at once during expansion. This avoids creating 50+ unused child states per node.

On expand: store the policy priors and legal mask on the parent node. Child `Node` objects are created only when first selected via PUCT.

Child state is computed lazily: when a child is first visited, `env.set_state(parent.state)` → `env.step(action)` → `node.state = env.get_state()`.

### 5.2 MCTS Configuration

```python
@dataclass
class MCTSConfig:
    num_simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature_threshold: float = 0.3  # fraction of steps before temp drops
    temperature_high: float = 1.0
    temperature_low: float = 0.1
```

### 5.3 Search Loop

```
For each simulation:
  1. Start at root, set env to root state
  2. SELECT: traverse expanded nodes using PUCT until reaching unexpanded or terminal
     - PUCT(a) = Q(a) + c_puct * P(a) * sqrt(N_parent) / (1 + N_child)
     - Q values normalized to [0,1] using running min/max
     - Lazy child creation: create Node on first visit
     - Lazy state computation: apply action from parent state on first visit
  3. EXPAND: run GNN on leaf state → get priors and value
  4. BACKPROPAGATE: update visit_count and value_sum up to root
```

**Dirichlet noise** added at root only: `(1-eps) * prior + eps * noise` over legal actions, then renormalized.

**Terminal values** normalized: `reward / makespan_ub` (maps raw -makespan to roughly [-1, 0]).

**Action selection** from visit counts uses temperature-based sampling (high temp early, low temp late in episode).

---

## 6. Training Pipeline

### 6.1 Self-Play (`training/self_play.py`)

Each episode generates training examples:

```python
def play_episode(env, graph_builder, model, mcts_config, device) -> List[TrainingExample]:
    # 1. Reset env
    # 2. For each decision point:
    #    - Skip MCTS if only 1 legal action (forced move optimization)
    #    - Run MCTS search → get visit counts
    #    - Restore env state (MCTS mutates it internally)
    #    - Store (graph_features, visit_count_policy, legal_mask) as numpy
    #    - Sample action from visit counts, apply to env
    # 3. Set all value_targets retroactively to -makespan / makespan_ub
```

**TrainingExample** stores numpy arrays (not PyG objects) for compact storage:
- `op_features`, `machine_features`, `global_features`
- Sparse edges: `eligible_edge_src/dst`, `precedes_edge_index`
- `policy_target` (normalized visit counts), `value_target`, `legal_mask`

### 6.2 Replay Buffer (`training/replay_buffer.py`)

- Fixed-size circular buffer of `TrainingExample` objects
- `sample(batch_size)` returns random batch
- `examples_to_batch()` reconstructs PyG `HeteroData` from stored numpy arrays during training

### 6.3 Trainer (`training/trainer.py`)

- Loss = cross_entropy(policy) + MSE(value)
- Optimizer: Adam with weight decay
- Processes examples individually (heterogeneous graphs with different sizes can't easily batch in PyG)
- Accumulates gradients over the batch, then steps once

```python
# Policy loss: cross-entropy with MCTS visit distribution
log_policy = torch.log(policy.clamp(min=1e-8))
policy_loss = -torch.sum(policy_target * log_policy)

# Value loss: MSE between predicted value and normalized -makespan
value_loss = F.mse_loss(value, value_target)

total_loss = policy_loss + value_loss
```

### 6.4 Pipeline (`training/pipeline.py`)

Orchestrates: self-play → replay buffer → training → repeat.

**Key features:**
- Serial self-play (single device) or parallel via persistent worker pool
- Persistent worker pool: workers spawned once via `mp.Process(spawn)`, receive `(model_state_dict, num_games)` via queues
- Dead worker detection with fallback to serial
- Checkpoint saving/loading (model + optimizer + history)
- tqdm progress bars

**PipelineConfig:**
```python
@dataclass
class PipelineConfig:
    num_iterations: int = 100       # outer training loop
    games_per_iteration: int = 50   # self-play episodes per iteration
    num_workers: int = 1            # parallel self-play workers
    mcts_simulations: int = 100     # MCTS sims per decision point
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    batch_size: int = 32
    batches_per_iteration: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    replay_buffer_size: int = 100_000
    hidden_dim: int = 64
    num_gnn_layers: int = 3
    checkpoint_interval: int = 10
    checkpoint_dir: str = "checkpoints"
```

---

## 7. AlphaZero Scheduler (`alphazero_scheduler.py`)

Wraps the trained model as an `OnlineScheduler` for integration with the existing Factory simulation loop.

**Two modes:**
- **MCTS mode**: Full MCTS search per decision (slower, better quality)
- **Policy mode**: Just the policy network argmax (fast inference)

**Standalone evaluation** via `schedule()` method — runs the full schedule using the internal numpy env, bypassing the Factory class entirely.

---

## 8. File Structure (Actual)

```
FJSSP-MCTS-Research/schedulers/alphazero/
├── __init__.py
├── alphazero_scheduler.py           # OnlineScheduler wrapper (MCTS or policy-only)
├── env/
│   ├── __init__.py
│   ├── config_compiler.py           # FactoryLogic + Jobs → numpy lookup tables
│   ├── fjsp_env.py                  # Fast numpy env with get_state/set_state
│   └── graph_builder.py             # Env state → PyG HeteroData
├── model/
│   ├── __init__.py
│   └── gnn.py                       # FJSPNet: HeteroConv GNN + policy/value heads
├── mcts/
│   ├── __init__.py
│   └── mcts.py                      # MCTS with PUCT, lazy expansion, value normalization
├── training/
│   ├── __init__.py
│   ├── self_play.py                 # Episode generation with forced-move skip
│   ├── replay_buffer.py             # Circular buffer, numpy storage, PyG reconstruction
│   ├── trainer.py                   # Policy + value loss, Adam optimizer
│   └── pipeline.py                  # Full orchestration with optional parallel workers
├── train_alphazero.ipynb            # Training notebook (supercomputer)
├── test_alphazero.ipynb             # Validation notebook
└── alphazero_visualizations.ipynb   # Graph and MCTS visualizations
```

---

## 9. Training Metrics & Interpretation

| Metric | Meaning | Healthy range |
|--------|---------|---------------|
| policy_loss | Cross-entropy between MCTS visits and network prior | ~4.0 = random, ~2.0 = learning, <1.0 = strong |
| value_loss | MSE between predicted and actual normalized makespan | Should decrease; 0.05-0.1 is good |
| mean_reward | Average -makespan/makespan_ub across games | [-1, 0]; closer to 0 = better |
| num_examples | Training examples generated (non-forced decision points) | Varies per episode; increases as model improves |

**Note on num_examples variability:** With the forced-move skip, the number of MCTS decision points per episode depends on the model's scheduling decisions. A random model keeps machines busy (many forced idles), while a trained model may create more states with multiple legal actions, increasing decision points and training time.

---

## 10. Common Issues Encountered

| Issue | Symptom | Fix |
|-------|---------|-----|
| Value loss explosion (~1M) | Raw -makespan (~-1002) vs tanh [-1,1] | Normalize: -makespan / makespan_ub |
| Training very slow | MCTS called on every decision point | Skip MCTS when only 1 legal action (forced move) |
| MCTS not improving reward | 20 simulations spread over 50+ actions | Need 100+ simulations for meaningful signal |
| Workers hanging (multiprocessing) | Spawned processes lack sys.path | Pass project_root, add try/except in worker loop |
| UTF-8 encoding error on Windows | `GORGORÃO` misread as `GORGORÃƒO` | Add `encoding='utf-8'` to `open()` calls |
| Iterations getting slower | More MCTS decision points as model improves | Reduce mcts_simulations or games_per_iteration |
| Cached modules after file update | Old code runs despite file changes | `importlib.reload()` or kernel restart |

---

## 11. Dependencies

```
torch>=2.0
torch-geometric>=2.4
numpy>=1.24
matplotlib>=3.7
tqdm>=4.65
networkx  # for visualizations
```

---

## 12. Key References

- Silver et al., "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" (AlphaZero, 2017)
- Song et al., "Flexible Job-Shop Scheduling via Graph Neural Network and Deep Reinforcement Learning" (2022)
- Park et al., "Learning to Schedule Job-Shop Problems: Representation and Policy Learning Using Graph Neural Network and Reinforcement Learning" (2021)
