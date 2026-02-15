# FJSSP Research Project

## Project Context

Read `.claude/docs/alphazero_implementation.md` for the full research plan and design decisions.

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

- [briefly describe where key files live once your repo takes shape]
