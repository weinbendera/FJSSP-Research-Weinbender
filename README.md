# FJSSP Research

AlphaZero-based scheduler for Flexible Job Shop Scheduling Problems (FJSP) using real textile manufacturing data.

## Setup

1. **Clone the repository**

   ```bash
   git clone <repo-url>
   cd "FJSSP Research"
   ```
2. **Create a virtual environment**

   ```bash
   python -m venv .venv
   ```
3. **Activate the virtual environment**

   ```bash
   # Windows (Git Bash)
   source .venv/Scripts/activate

   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1

   # macOS/Linux
   source .venv/bin/activate
   ```
4. **Install the project (editable mode)**

   ```bash
   pip install -e FJSSP-MCTS-Research
   ```

   This installs all dependencies and makes the project packages (`factory`, `schedulers`, `utils`) importable from anywhere.

## Running Notebooks

1. Open the project in VS Code
2. Select the `.venv` Python interpreter as your Jupyter kernel
3. Import project modules directly:
   ```python
   from factory.factory import Factory
   from utils.input_schemas import ProductRequest
   from schedulers.genetic_algorithm.genetic_scheduler import GeneticScheduler
   ```

## Project Structure

```
FJSSP-MCTS-Research/
├── data/                  # Input/output JSON scheduling data
├── factory/               # Factory simulation (machines, state, runtime)
├── schedulers/            # Scheduling algorithms
│   ├── genetic_algorithm/ # Genetic algorithm scheduler
│   ├── greedy/            # Greedy scheduler
│   ├── marl/              # Multi-agent RL scheduler
│   ├── or_scheduler/      # Operations research scheduler
│   └── rulebased/         # Rule-based scheduler
├── utils/                 # Data loading, schemas, job building
└── pyproject.toml         # Project config and dependencies
```
