# FJSSP-MCTS-Research Project Directory

## Overview

AlphaZero-based scheduler for Flexible Job Shop Scheduling Problems (FJSP) using real textile manufacturing data (3 machines, 5-minute time steps). Part of a research project comparing multiple scheduling approaches.

## Data Flow

```
Input JSON → FactoryLogicLoader → FactoryLogic (string-keyed config)
                                      ↓
ProductRequests → JobBuilder → Jobs/Operations (string-keyed)
                                      ↓
         FactoryLogic + Jobs → CompiledConfig (numpy integer-indexed)
                                      ↓
                              FJSPEnv (fast numpy state)
                                      ↓
                              GraphBuilder → PyG HeteroData
                                      ↓
                              FJSPNet (GNN policy+value)
                                      ↓
                              MCTS → action selection
```

## File Structure

### `data/`
- `Input_JSON_Schedule_Optimization.json` — Real textile factory data: 3 machines (MAQ118/119/120), 7 tasks, 13 task modes, 14 products, 132 units. 5-minute steps, 5 working days (960 steps).

### `utils/` — Shared utilities (used by all schedulers)
- `input_schemas.py` — Pydantic models: ProductRequest, Cell, TaskMode, Task, Machine, Product, OrderConstraint, CollisionConstraint, EnergySource
- `factory_logic_loader.py` — FactoryLogic (static config with helper methods) + FactoryLogicLoader (JSON → FactoryLogic)
- `job_builder.py` — Job/Operation dataclasses + JobBuilder (ProductRequests → Jobs with operations, eligibility, precedence)

### `factory/` — Simulation environment (used by greedy/rule-based/GA/MARL schedulers)
- `factory.py` — Factory class: manages simulation loop, applies actions, advances time
- `factory_state.py` — FactoryState: immutable snapshot with feasible action computation
- `machine_runtime.py` — MachineRuntime: per-machine execution tracking

**Note:** Factory classes use deepcopy and string lookups — too slow for MCTS. The AlphaZero scheduler uses its own numpy-based environment instead.

### `schedulers/` — All scheduler implementations
- `scheduler.py` — Base classes: Scheduler, OnlineScheduler, OfflineScheduler, Schedule, ScheduledOperation

#### `schedulers/greedy/` — Greedy heuristic
#### `schedulers/rulebased/` — Rule-based scheduler
#### `schedulers/genetic_algorithm/` — GA with priority-based encoding
#### `schedulers/marl/` — Multi-agent PPO (PettingZoo)
#### `schedulers/or_scheduler/` — OR-Tools constraint solver

#### `schedulers/alphazero/` — AlphaZero GNN+MCTS scheduler
- `alphazero_scheduler.py` — OnlineScheduler wrapper (MCTS or policy-only modes)
- `env/config_compiler.py` — Compiles FactoryLogic+Jobs → numpy lookup tables
- `env/fjsp_env.py` — Fast numpy environment with get_state/set_state (~2.5KB copies)
- `env/graph_builder.py` — Converts env state → PyG HeteroData (op/machine nodes, eligible/precedes edges)
- `model/gnn.py` — FJSPNet: HeteroConv GNN with policy head (action scoring) + value head
- `mcts/mcts.py` — MCTS with PUCT, lazy expansion, Dirichlet noise, value normalization
- `training/self_play.py` — Episode generation for training data
- `training/replay_buffer.py` — Experience storage (numpy-backed)
- `training/trainer.py` — Network training (cross-entropy + MSE loss)
- `training/pipeline.py` — Full training orchestration
- `config/default.yaml` — Hyperparameters
- `train_alphazero.ipynb` — Supercomputer training notebook
- `test_alphazero.ipynb` — Validation notebook

## Key Design Decisions

1. **Sequential machine decisions**: At each timestep, machines decide in order M0→M1→M2. After all decide, time advances by 1 step.
2. **Action space**: All valid (operation, machine, task_mode) triples + idle. Legal mask filters per-machine and per-constraint.
3. **State representation**: Pure numpy arrays (~2.5KB) for fast MCTS get_state/set_state.
4. **GNN architecture**: Heterogeneous graph with SAGEConv message passing between op and machine nodes.
5. **Constraints**: Precedence (op ordering within jobs), collision (task types can't run simultaneously), job exclusivity (one op per job at a time), time leaps (ops can't cross day boundaries).
