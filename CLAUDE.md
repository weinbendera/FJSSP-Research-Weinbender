# FJSSP Research Project

## Project Context

Read `.claude/docs/alphazero_implementation.md` for the full research plan and design decisions.
Read `.claude/project_directory.md` for the full project directory and architecture.

## Overview

AlphaZero-based scheduler for Flexible Job Shop Scheduling Problems (FJSP) using real textile manufacturing data (3 machines, 5-minute time steps). Goal is makespan minimization with eventual energy optimization and MuZero progression.

## Key Architecture Decisions

- GNN-based state representation
- MCTS + learned policy/value networks
- Hybrid approach: known dynamics (scheduling constraints) + learned components (energy forecasting)

## Conventions

- Python, PyTorch
- Testing is done in Jupyter notebooks, training is done using those notebooks on a school supercomputer

## File Structure

- `FJSSP-MCTS-Research/data/` — Input data (textile factory JSON)
- `FJSSP-MCTS-Research/utils/` — Shared: input schemas, factory logic loader, job builder
- `FJSSP-MCTS-Research/factory/` — Factory simulation (OOP, used by greedy/GA/MARL)
- `FJSSP-MCTS-Research/schedulers/` — All schedulers (greedy, rule-based, GA, MARL, OR, alphazero)
- `FJSSP-MCTS-Research/schedulers/alphazero/` — AlphaZero GNN+MCTS scheduler (env, model, mcts, training)
