import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Literal, TypeVar

import numpy as np
import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, record_function

from .dataset import (
    DataLoaderSplitCfg,
    DataLoaderWrapper,
    DatasetCfg,
    get_data_loader_train,
    to_device,
)
from .ddp import gather_to_rank_zero, get_local_rank, is_rank_zero, rank_zero_only
from .logging import Heartbeat
from .model import ModelCfg, get_model
from .model.model import ConfigurableModel, TrainStepOutput
from .optimizer import OptimizerCfg, build_optimizer
from .preemption import PreemptionException, PreemptionManager


@dataclass(frozen=True)
class ProfilingCfg:
    wait: int
    warmup: int
    active: int
    skip_first: int


@dataclass(frozen=True)
class CheckpointingCfg:
    save_interval_steps: int
    load_dataset_state: bool
    load: Path | None
    most_recent_to_keep: int


@dataclass(frozen=True)
class TrainingCfg:
    optimizer: OptimizerCfg
    warmup_steps: int
    num_steps: int
    vis_interval_steps: int
    checkpointing: CheckpointingCfg
    profiling: ProfilingCfg
    heartbeat_seconds: float
    find_unused_parameters: bool


@dataclass
class WandbCfg:
    entity: str | None
    project: str | None
    mode: Literal["online", "offline", "disabled"]
    notes: str | None


@dataclass(frozen=True)
class DataLoaderCfg:
    train: DataLoaderSplitCfg


@dataclass(frozen=True)
class TrainerCfg:
    training: TrainingCfg
    dataset: DatasetCfg
    data_loader: DataLoaderCfg
    model: ModelCfg
    on_existing_workspace: Literal["restore", "overwrite", "throw"]
    wandb: WandbCfg


C = TypeVar("C")  # model config type
TrainBatch = TypeVar("TrainBatch", bound=dict)
VisBatch = TypeVar("VisBatch", bound=dict)
TestBatch = TypeVar("TestBatch", bound=dict)


class Trainer(Generic[C, TrainBatch, VisBatch, TestBatch]):
    cfg: TrainingCfg
    workspace: Path
    train_step: DDP
    train_loader: DataLoaderWrapper
    step: int
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler

    def __init__(
        self,
        cfg: TrainerCfg,
        workspace: Path,
    ) -> None:
        super().__init__()
        self.cfg = cfg.training
        self.workspace = workspace
        model = get_model(cfg.model).to(self.device)
        self.train_step = DDP(
            model,
            device_ids=[get_local_rank()],
            output_device=get_local_rank(),
            broadcast_buffers=False,
            find_unused_parameters=cfg.training.find_unused_parameters,
        )
        self.train_loader = get_data_loader_train(cfg.dataset, cfg.data_loader.train)
        self.step = 0

        self.optimizer = build_optimizer(model, self.cfg.optimizer)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(step / self.cfg.warmup_steps, 1.0),
        )
        if self.cfg.checkpointing.load is not None:
            self.load_checkpoint(
                self.cfg.checkpointing.load,
                self.cfg.checkpointing.load_dataset_state,
            )

    def train(self) -> None:
        logging.info("Starting training.")
        self.handle_existing_workspace()
        preemption_manager = PreemptionManager()

        # Main training loop.
        try:
            heartbeat = Heartbeat(self.cfg.heartbeat_seconds)
            logging.info("Entering training loop.")
            with self.profiler as profiler:
                while self.step < self.cfg.num_steps:
                    # Handle logging a heartbeat.
                    if is_rank_zero():
                        heartbeat.step(self.step)
                    profiler.step()

                    with record_function("data loading"):
                        batch = next(self.train_loader)
                        batch = to_device(batch, self.device)

                    self.optimizer.zero_grad()

                    train_metrics = {}
                    with torch.autocast(
                        device_type="cuda",
                        enabled=True,
                        dtype=torch.bfloat16,
                    ):
                        with record_function("forward pass"):
                            train_step_output: TrainStepOutput = self.train_step(batch)
                            if isinstance(train_step_output, tuple):
                                loss, metrics = train_step_output
                                train_metrics |= gather_to_rank_zero(metrics)
                            else:
                                loss = train_step_output
                            train_metrics["loss"] = loss.item()
                            del train_step_output

                    with record_function("backward pass"):
                        loss.backward()

                    # Run the optimizer step.
                    with record_function("optimizer step"):
                        # Delay preemption during the optimizer step, since we don't
                        # want the model to be left in a half-updated state. Preemption
                        # is fine everywhere else.
                        with preemption_manager.atomic():
                            self.optimizer.step()
                            self.scheduler.step()
                            self.step += 1

                    if self.should_run(self.cfg.checkpointing.save_interval_steps):
                        self.save_checkpoint_to_workspace()

                    # Log gradient statistics.
                    with torch.no_grad():
                        grad_norm = 0
                        grad_num_nan = 0
                        grad_num_inf = 0
                        for p in self.model.parameters():
                            if p.grad is not None:
                                grad_norm += p.grad.norm(2).item() ** 2
                                grad_num_nan += p.grad.isnan().sum().item()
                                grad_num_inf += p.grad.isinf().sum().item()
                        train_metrics |= {
                            "grad_norm": grad_norm**0.5,
                            "grad_num_nan": float(grad_num_nan),
                            "grad_num_inf": float(grad_num_inf),
                            "learning_rate": self.scheduler.get_last_lr()[0],
                        }

                    # Write the train metrics.
                    if is_rank_zero():
                        train_metrics = {
                            f"train/{k}": np.mean(v) for k, v in train_metrics.items()
                        }
                        wandb.log(train_metrics, self.step)

                    # Save a checkpoint for visualization, which runs on a separate job.
                    if self.should_run(self.cfg.vis_interval_steps):
                        self.save_visualization_checkpoint()

        except PreemptionException:
            # If your cluster gives you enough time to save a checkpoint before your job
            # is killed, un-comment this:
            # logging.info("Detected preemption. Attempting to save checkpoint.")
            # self.save_checkpoint_to_workspace()

            if is_rank_zero():
                logging.info("Attempting to run wandb.finish")
                wandb.finish(99)

            # Exit gracefully.
            logging.info("Exiting due to preemption.")
            return

        # Ensure that the checkpoint at step == num_steps isn't dropped.
        self.save_checkpoint_to_workspace()

    #################
    # Checkpointing #
    #################

    @property
    def checkpoint_path(self) -> Path:
        return self.workspace / "checkpoints"

    @rank_zero_only
    def save_checkpoint(self, path: Path) -> None:
        logging.info(f"Saving checkpoint to {path}")
        path.parent.mkdir(exist_ok=True, parents=True)
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "train_loader": self.train_loader.state_dict(),
            "step": self.step,
        }
        torch.save(state_dict, path)
        logging.info(f"Done saving checkpoint to {path}")

    @rank_zero_only
    def save_visualization_checkpoint(self) -> None:
        # First save to a temp path, then rename, in order to prevent the visualizer
        # from reading partially saved checkpoints.
        path = self.workspace / "vis_checkpoints" / f"{self.step}.ckpt"
        temp_path = path.with_suffix(".tmp")
        self.save_checkpoint(temp_path)
        temp_path.rename(path)

    @rank_zero_only
    def save_checkpoint_to_workspace(self) -> None:
        self.save_checkpoint(self.checkpoint_path_for_step(self.step))

        # Figure out which checkpoints have currently been saved.
        checkpoint_steps = []
        for path in self.checkpoint_path.iterdir():
            if path.suffix != ".ckpt":
                continue
            checkpoint_steps.append(int(path.stem))
        checkpoint_steps = sorted(checkpoint_steps)

        # Figure out which checkpoints to keep.
        c = self.cfg.checkpointing
        keep_after = self.step - c.most_recent_to_keep * c.save_interval_steps
        kept_steps = self.get_kept_steps(keep_after, c.save_interval_steps)

        # Delete the checkpoints that need to go.
        for step in checkpoint_steps:
            if step >= keep_after or step in kept_steps:
                continue
            path = self.checkpoint_path_for_step(step)
            logging.info(f"Deleting checkpoint at {path}")
            if os.access(path, os.W_OK):
                path.unlink()

    def load_checkpoint(self, path: Path, load_dataset_state: bool = True) -> None:
        logging.info(f"Loading checkpoint at {path}")
        state_dict = torch.load(
            path,
            weights_only=True,
            map_location={"cuda:0": str(self.device)},
        )
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.step = state_dict["step"]
        if load_dataset_state:
            self.train_loader.load_state_dict(state_dict["train_loader"])
        logging.info(f"Done loading checkpoint at {path}")

    def handle_existing_workspace(self):
        # Find the latest checkpoint.
        logging.info("Looking for checkpoints.")
        latest_step = None
        try:
            for path in self.checkpoint_path.iterdir():
                try:
                    step = int(path.stem)
                    latest_step = max(latest_step or 0, step)
                except ValueError:
                    pass
        except FileNotFoundError:
            pass

        # If checkpoints exist, load the latest one.
        if latest_step is None:
            logging.info("No existing checkpoints found.")
        else:
            self.load_checkpoint(
                self.checkpoint_path_for_step(latest_step),
                self.cfg.checkpointing.load_dataset_state,
            )

    def get_kept_steps(self, step: int, base: int) -> list[int]:
        # Ensure that one-off checkpoints (e.g., for preemption) are handled correctly.
        step -= step % base

        # Keep exponentially fewer checkpoints in the distant past.
        result = []
        remaining = step
        power = base
        while remaining >= 0:
            result.append(remaining)
            remaining -= power
            if remaining % (2 * power) == 0:
                power *= 2
        return result

    def checkpoint_path_for_step(self, step: int) -> Path:
        step = str(step).zfill(len(str(self.cfg.num_steps)))
        return self.checkpoint_path / f"{step}.ckpt"

    ###########
    # Logging #
    ###########

    def should_run(self, interval: int | None) -> bool:
        if interval is None or self.step == 0:
            return False
        return self.step % interval == 0

    @property
    def profiler(self) -> profile:
        return profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=self.cfg.profiling.wait,
                warmup=self.cfg.profiling.warmup,
                active=self.cfg.profiling.active,
                skip_first=self.cfg.profiling.skip_first,
                skip_first_wait=True,
                repeat=0,
            ),
            on_trace_ready=self.on_trace_ready,
            profile_memory=False,
            with_stack=True,
            record_shapes=False,
        )

    @property
    def profile_dir(self) -> Path:
        return self.workspace / "profiles"

    def on_trace_ready(self, profiler: profile) -> None:
        logging.info(f"Saving profile at step {self.step}")
        path = self.profile_dir / str(self.step)
        path.mkdir(exist_ok=True, parents=True)
        profiler.export_chrome_trace(str(path / f"trace_{dist.get_rank()}.json.gz"))

    #######
    # DDP #
    #######

    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{get_local_rank()}")

    @property
    def model(self) -> ConfigurableModel[C, TrainBatch, VisBatch, TestBatch]:
        return self.train_step.module
