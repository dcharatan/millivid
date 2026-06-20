import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from ..components.action_encoder import ActionEncoder
from ..components.adaptivity import LevelsCfg
from .dataset import ConfigurableDataset, Split
from .format_pyramid import load_pyramid
from .video_index import VideoIndex, split_index


@dataclass(frozen=True)
class DatasetLatentsDenseCfg(LevelsCfg):
    name: Literal["latents_dense"]
    path: tuple[Path, ...]
    train_num_frames: int
    test_num_frames: int
    test_fraction: float


class DatasetLatentsDense(ConfigurableDataset[DatasetLatentsDenseCfg]):
    def __init__(self, cfg: DatasetLatentsDenseCfg, split: Split):
        super().__init__(cfg, split)
        with Path(self.path / "index.json").open("r") as f:
            index = json.load(f)
        self.index = index
        self.video_index = VideoIndex(
            split_index(index, split, test_fraction=cfg.test_fraction),
            self.num_frames,
        )

    def __len__(self) -> int:
        return len(self.video_index)

    def __getitem__(self, index: int):
        key, start = self.video_index[index]
        (latents,), actions = load_pyramid(
            (self.path / key).with_suffix(".pyramid"),
            self.cfg.available_num_levels,
            self.index[key],
            (self.cfg.latent_channels, *self.cfg.latent_shape),
            (ActionEncoder.num_action_channels(),),
            0,
            1,
            start,
            start + self.num_frames,
            torch.device("cpu"),
        )
        return {"latents": latents, "actions": actions, "key": key}

    @property
    def num_frames(self) -> int:
        return (
            self.cfg.train_num_frames
            if self.split == "train"
            else self.cfg.test_num_frames
        )

    @property
    def path(self) -> Path:
        for path in self.cfg.path:
            if path.exists():
                return path
        raise FileNotFoundError("Could not find a valid dataset path!")
