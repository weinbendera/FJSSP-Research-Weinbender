"""
Full AlphaZero training pipeline.

Orchestrates: self-play → replay buffer → training → repeat.
Supports parallel self-play via persistent worker pool across multiple GPUs.
"""

import os
import time
import torch
import torch.multiprocessing as mp
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from utils.factory_logic_loader import FactoryLogic, FactoryLogicLoader
from utils.job_builder import JobBuilder
from utils.input_schemas import ProductRequest

from schedulers.alphazero.env.config_compiler import CompiledConfig
from schedulers.alphazero.env.fjsp_env import FJSPEnv
from schedulers.alphazero.env.graph_builder import GraphBuilder
from schedulers.alphazero.model.gnn import FJSPNet
from schedulers.alphazero.mcts.mcts import MCTSConfig
from schedulers.alphazero.training.self_play import play_episode
from schedulers.alphazero.training.replay_buffer import ReplayBuffer
from schedulers.alphazero.training.trainer import Trainer


@dataclass
class PipelineConfig:
    """Configuration for the full training pipeline."""
    # Self-play
    num_iterations: int = 100
    games_per_iteration: int = 50
    num_workers: int = 1  # number of parallel self-play workers (1 per GPU)
    # MCTS
    mcts_simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    # Training
    batch_size: int = 32
    batches_per_iteration: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    replay_buffer_size: int = 100_000
    # Model
    hidden_dim: int = 64
    num_gnn_layers: int = 3
    # Checkpointing
    checkpoint_interval: int = 10
    checkpoint_dir: str = "checkpoints"


def _worker_loop(
    worker_id: int,
    compiled_config: CompiledConfig,
    mcts_config: MCTSConfig,
    op_feature_dim: int,
    machine_feature_dim: int,
    global_feature_dim: int,
    hidden_dim: int,
    num_gnn_layers: int,
    device: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    project_root: str = None,
):
    """Persistent worker loop. Stays alive across iterations."""
    try:
        import sys
        if project_root and project_root not in sys.path:
            sys.path.insert(0, project_root)

        from schedulers.alphazero.env.fjsp_env import FJSPEnv
        from schedulers.alphazero.env.graph_builder import GraphBuilder
        from schedulers.alphazero.model.gnn import FJSPNet
        from schedulers.alphazero.training.self_play import play_episode

        env = FJSPEnv(compiled_config)
        graph_builder = GraphBuilder(compiled_config)
        model = FJSPNet(
            config=compiled_config,
            op_feature_dim=op_feature_dim,
            machine_feature_dim=machine_feature_dim,
            global_feature_dim=global_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
        ).to(device)
        model.eval()

        while True:
            task = task_queue.get()

            # Shutdown signal
            if task is None:
                break

            model_state_dict, num_games = task
            model.load_state_dict(model_state_dict)
            model.eval()

            all_examples = []
            rewards = []
            for _ in range(num_games):
                examples = play_episode(
                    env=env,
                    graph_builder=graph_builder,
                    model=model,
                    mcts_config=mcts_config,
                    device=device,
                )
                all_examples.extend(examples)
                if examples:
                    rewards.append(examples[0].value_target)

            result_queue.put((all_examples, rewards))

    except Exception as e:
        import traceback
        traceback.print_exc()
        # Keep sending empty results so parent doesn't hang
        while True:
            try:
                task = task_queue.get(timeout=1)
                if task is None:
                    break
                result_queue.put(([], []))
            except Exception:
                break


class AlphaZeroPipeline:
    """Full AlphaZero training loop with optional multi-GPU self-play."""

    def __init__(
        self,
        factory_logic: FactoryLogic,
        product_requests: List[ProductRequest],
        pipeline_config: PipelineConfig = PipelineConfig(),
        device: str = "cpu",
    ):
        self.pipeline_config = pipeline_config
        self.device = device

        # Detect project root for worker processes
        self._project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )

        # Build compiled config
        jobs = JobBuilder(factory_logic).build_jobs(product_requests)
        self.compiled_config = CompiledConfig.compile(factory_logic, jobs)

        # Build environment and graph builder
        self.env = FJSPEnv(self.compiled_config)
        self.graph_builder = GraphBuilder(self.compiled_config)

        # Build model
        self.model = FJSPNet(
            config=self.compiled_config,
            op_feature_dim=self.graph_builder.op_feature_dim,
            machine_feature_dim=self.graph_builder.machine_feature_dim,
            global_feature_dim=self.graph_builder.global_feature_dim,
            hidden_dim=pipeline_config.hidden_dim,
            num_layers=pipeline_config.num_gnn_layers,
        ).to(device)

        # Build trainer and replay buffer
        self.trainer = Trainer(
            model=self.model,
            lr=pipeline_config.learning_rate,
            weight_decay=pipeline_config.weight_decay,
            device=device,
        )
        self.replay_buffer = ReplayBuffer(pipeline_config.replay_buffer_size)

        # MCTS config
        self.mcts_config = MCTSConfig(
            num_simulations=pipeline_config.mcts_simulations,
            c_puct=pipeline_config.c_puct,
            dirichlet_alpha=pipeline_config.dirichlet_alpha,
            dirichlet_epsilon=pipeline_config.dirichlet_epsilon,
        )

        # Metrics history
        self.history: List[Dict] = []

        # Worker pool (initialized lazily)
        self._workers = []
        self._task_queues = []
        self._result_queue = None

    def _start_worker_pool(self):
        """Start persistent worker processes (once)."""
        cfg = self.pipeline_config
        num_workers = cfg.num_workers
        num_gpus = torch.cuda.device_count()

        mp.set_start_method("spawn", force=True)

        self._result_queue = mp.Queue()
        self._task_queues = []
        self._workers = []

        for worker_id in range(num_workers):
            if num_gpus > 0:
                worker_device = f"cuda:{worker_id % num_gpus}"
            else:
                worker_device = "cpu"

            task_queue = mp.Queue()
            p = mp.Process(
                target=_worker_loop,
                args=(
                    worker_id,
                    self.compiled_config,
                    self.mcts_config,
                    self.graph_builder.op_feature_dim,
                    self.graph_builder.machine_feature_dim,
                    self.graph_builder.global_feature_dim,
                    cfg.hidden_dim,
                    cfg.num_gnn_layers,
                    worker_device,
                    task_queue,
                    self._result_queue,
                    self._project_root,
                ),
            )
            p.daemon = True
            p.start()
            self._task_queues.append(task_queue)
            self._workers.append(p)

    def _stop_worker_pool(self):
        """Shut down worker processes."""
        for tq in self._task_queues:
            tq.put(None)  # shutdown signal
        for p in self._workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        self._workers = []
        self._task_queues = []
        self._result_queue = None

    def _run_self_play_serial(self, num_games: int) -> tuple:
        """Run self-play games sequentially (single device)."""
        from tqdm.auto import tqdm
        all_examples = []
        rewards = []
        for game in tqdm(range(num_games), desc="Self-play", leave=False):
            examples = play_episode(
                env=self.env,
                graph_builder=self.graph_builder,
                model=self.model,
                mcts_config=self.mcts_config,
                device=self.device,
            )
            all_examples.extend(examples)
            if examples:
                rewards.append(examples[0].value_target)
        return all_examples, rewards

    def _run_self_play_parallel(self, num_games: int) -> tuple:
        """Run self-play games across persistent worker pool."""
        cfg = self.pipeline_config
        num_workers = min(cfg.num_workers, num_games)

        # Check workers are alive — fall back to serial if any died
        dead_workers = [i for i, p in enumerate(self._workers) if not p.is_alive()]
        if dead_workers:
            print(f"WARNING: Workers {dead_workers} died. Falling back to serial.")
            self._stop_worker_pool()
            return self._run_self_play_serial(num_games)

        # Divide games across workers
        games_per_worker = [num_games // num_workers] * num_workers
        for i in range(num_games % num_workers):
            games_per_worker[i] += 1

        # Send current model weights + game count to each worker
        model_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        for worker_id in range(num_workers):
            self._task_queues[worker_id].put((model_state_dict, games_per_worker[worker_id]))

        # Collect results
        all_examples = []
        all_rewards = []
        for _ in range(num_workers):
            examples, rewards = self._result_queue.get()
            all_examples.extend(examples)
            all_rewards.extend(rewards)

        return all_examples, all_rewards

    def run(self, verbose: bool = True) -> List[Dict]:
        """Run the full training pipeline."""
        cfg = self.pipeline_config
        use_parallel = cfg.num_workers > 1 and torch.cuda.device_count() > 0

        if use_parallel:
            self._start_worker_pool()

        try:
            from tqdm.auto import tqdm
            for iteration in tqdm(range(1, cfg.num_iterations + 1), desc="Training iterations"):
                iter_start = time.time()

                # --- Self-play ---
                self.model.eval()

                if use_parallel:
                    all_examples, episode_rewards = self._run_self_play_parallel(
                        cfg.games_per_iteration
                    )
                else:
                    all_examples, episode_rewards = self._run_self_play_serial(
                        cfg.games_per_iteration
                    )

                self.replay_buffer.add(all_examples)
                total_examples = len(all_examples)

                # --- Training ---
                self.model.train()
                if len(self.replay_buffer) >= cfg.batch_size:
                    train_metrics = self.trainer.train_epoch(
                        buffer=self.replay_buffer,
                        batch_size=cfg.batch_size,
                        num_batches=cfg.batches_per_iteration,
                    )
                else:
                    train_metrics = {"policy_loss": 0, "value_loss": 0, "total_loss": 0}

                iter_time = time.time() - iter_start

                # Metrics
                metrics = {
                    "iteration": iteration,
                    "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0,
                    "best_reward": float(np.max(episode_rewards)) if episode_rewards else 0,
                    "num_examples": total_examples,
                    "buffer_size": len(self.replay_buffer),
                    **train_metrics,
                    "time": iter_time,
                }
                self.history.append(metrics)

                if verbose:
                    print(
                        f"Iter {iteration:3d} | "
                        f"Reward: {metrics['mean_reward']:.1f} (best: {metrics['best_reward']:.1f}) | "
                        f"Loss: {metrics['total_loss']:.4f} "
                        f"(p={metrics['policy_loss']:.4f}, v={metrics['value_loss']:.4f}) | "
                        f"Examples: {total_examples} | "
                        f"Buffer: {metrics['buffer_size']} | "
                        f"Time: {iter_time:.1f}s"
                    )

                # --- Checkpoint ---
                if iteration % cfg.checkpoint_interval == 0:
                    self.save_checkpoint(iteration)

        finally:
            if use_parallel:
                self._stop_worker_pool()

        return self.history

    def save_checkpoint(self, iteration: int) -> str:
        """Save model checkpoint."""
        os.makedirs(self.pipeline_config.checkpoint_dir, exist_ok=True)
        path = os.path.join(
            self.pipeline_config.checkpoint_dir,
            f"alphazero_iter{iteration:04d}.pt"
        )
        torch.save({
            "iteration": iteration,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.trainer.optimizer.state_dict(),
            "history": self.history,
        }, path)
        return path

    def load_checkpoint(self, path: str) -> int:
        """Load model checkpoint. Returns iteration number."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", [])
        return checkpoint["iteration"]
