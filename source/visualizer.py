import logging
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Generic, TypeVar

import torch
import wandb

from .dataset import (
    DataLoaderSplitCfg,
    DataLoaderWrapper,
    DatasetCfg,
    get_data_loader_vis,
    to_device,
)
from .image_io import prep_image
from .model import ModelCfg, get_model
from .model.model import ConfigurableModel
from .trainer import WandbCfg


@dataclass(frozen=True)
class DataLoaderCfg:
    vis: DataLoaderSplitCfg


@dataclass(frozen=True)
class VisualizerCfg:
    dataset: DatasetCfg
    data_loader: DataLoaderCfg
    model: ModelCfg
    wandb: WandbCfg


C = TypeVar("C")  # model config type
TrainBatch = TypeVar("TrainBatch", bound=dict)
VisBatch = TypeVar("VisBatch", bound=dict)
TestBatch = TypeVar("TestBatch", bound=dict)


class Visualizer(Generic[C, TrainBatch, VisBatch, TestBatch]):
    cfg: VisualizerCfg
    workspace: Path
    model: ConfigurableModel[C, TrainBatch, VisBatch, TestBatch]
    vis_loader: DataLoaderWrapper
    step: int | None

    def __init__(
        self,
        cfg: VisualizerCfg,
        workspace: Path,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.workspace = workspace
        self.model = get_model(cfg.model).to(self.device).eval()
        self.vis_loader = get_data_loader_vis(cfg.dataset, cfg.data_loader.vis)
        self.step = None

    ###############################
    # Main Visualization Function #
    ###############################

    @torch.no_grad()
    def vis_step(self, batch: VisBatch) -> None:
        logging.info(f"Running visualization at step {self.step}.")
        output = self.model.vis_step(batch)

        logging.info("Passing visualizations to wandb.")
        metrics = {
            f"vis/{k}": v.mean().detach().cpu().float().numpy()
            for k, v in output.metrics.items()
        }
        images = {
            f"vis/{k}": [wandb.Image(prep_image(im)) for im in v]
            for k, v in output.images.items()
        }
        wandb_videos = {}
        for key, videos in output.videos.items():
            for index, video in enumerate(videos):
                wandb_videos[f"vis/{key}/{index}"] = wandb.Video(
                    BytesIO(video),
                    format="webm",
                )
        wandb.log({**metrics, **images, **wandb_videos}, self.step, commit=True)
        logging.info("Done passing visualizations to wandb.")

    ####################
    # Helper Functions #
    ####################

    def run_visualization_loop(self):
        logging.info("Loading visualization batch.")
        batch = next(self.vis_loader)
        batch = to_device(batch, self.device)

        while True:
            checkpoint_dir = self.workspace / "vis_checkpoints"
            if not checkpoint_dir.exists():
                time.sleep(10)
                continue

            # Find the latest checkpoint.
            checkpoints: list[tuple[int, Path]] = []
            for path in checkpoint_dir.iterdir():
                if not (path.is_file() and path.suffix == ".ckpt"):
                    continue
                checkpoints.append((int(path.stem), path))
            if not checkpoints:
                time.sleep(10)
                continue
            _, path = sorted(checkpoints, reverse=True)[0]

            # Load the checkpoint.
            self.load_checkpoint(path)

            # Run visualization for the step.
            with torch.autocast(
                device_type="cuda",
                enabled=True,
                dtype=torch.bfloat16,
            ):
                self.vis_step(batch)

            # Delete the checkpoint once we're done visualizing it.
            path.unlink()

    def load_checkpoint(self, path: Path) -> None:
        logging.info(f"Loading checkpoint at {path}")
        state_dict = torch.load(
            path,
            weights_only=True,
            map_location={"cuda:0": str(self.device)},
        )
        self.model.load_state_dict(state_dict["model"])
        self.step = state_dict["step"]
        logging.info(f"Done loading checkpoint at {path}")

    @property
    def device(self) -> torch.device:
        return torch.device("cuda:0")
