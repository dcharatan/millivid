import json
from dataclasses import dataclass
from pathlib import Path
from random import random
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat
from jaxtyping import Float, Int
from torch import Tensor

from ..components.action_encoder import ActionEncoder
from ..components.adaptivity import LevelsCfg
from ..strategy import UNSEEN, Strategy, StrategyCfg, get_strategy
from .dataset import ConfigurableDataset, Split
from .format_pyramid import load_pyramid
from .video_index import VideoIndex, split_index


@dataclass(frozen=True)
class DatasetLatentsUniversalCfg(LevelsCfg):
    name: Literal["latents_universal"]
    path: tuple[Path, ...]
    strategy: StrategyCfg
    test_num_frames: int
    first_denoised_frame: int
    test_fraction: float


type Tokens = Float[Tensor, "token token_channel"]
type Indices = Int[Tensor, "token flhw=4"]
type Actions = Float[Tensor, "token action_channel"]


class DatasetLatentsUniversal(ConfigurableDataset[DatasetLatentsUniversalCfg]):
    strategy: Strategy
    index: dict[str, int]
    video_index: VideoIndex

    def __init__(self, cfg: DatasetLatentsUniversalCfg, split: Split):
        super().__init__(cfg, split)
        self.strategy = get_strategy(cfg.strategy, cfg.base_num_tokens)
        with Path(self.path / "index.json").open("r") as f:
            index = json.load(f)
        self.index = index
        self.video_index = VideoIndex(
            split_index(index, split, test_fraction=cfg.test_fraction),
            (self.strategy.chunk_num_frames if split == "train" else None),
        )

    def __len__(self) -> int:
        return len(self.video_index)

    def __getitem__(self, index: int):
        f = self.get_train_example if self.split == "train" else self.get_test_example
        return f(index)

    def load_pyramid(
        self,
        index: int,
        level_start: int,
        level_end: int,
        frame_start: int,
        frame_end: int,
    ) -> tuple[Tokens, Indices, Actions, Float[Tensor, "frame action_channel"]]:
        key, frame_offset = self.video_index[index]
        h, w = self.cfg.latent_shape

        # Read the pyramid (custom binary format that minimizes number of syscalls).
        levels, per_frame_actions = load_pyramid(
            (self.path / key).with_suffix(".pyramid"),
            self.cfg.available_num_levels,
            self.index[key],
            (self.cfg.latent_channels, *self.cfg.latent_shape),
            (ActionEncoder.num_action_channels(),),
            level_start,
            level_end,
            frame_offset + frame_start,
            min(frame_offset + frame_end, self.index[key]),
            torch.device("cpu"),
        )

        # Define the indices and repeat the actions.
        tokens = []
        indices = []
        actions = []
        for index, level_tokens in enumerate(levels):
            # Define the indices.
            f, _, h, w = level_tokens.shape
            index_h = torch.arange(h, dtype=torch.int32)
            index_h = repeat(index_h, "h -> (f h w)", f=f, w=w)
            index_w = torch.arange(w, dtype=torch.int32)
            index_w = repeat(index_w, "w -> (f h w)", f=f, h=h)
            index_f = torch.arange(f, dtype=torch.int32)
            index_f = repeat(index_f, "f -> (f h w)", h=h, w=w) + frame_start
            index_l = torch.full_like(index_f, index + level_start)
            indices.append(torch.stack((index_f, index_l, index_h, index_w), dim=-1))

            # Repeat the actions for each token.
            actions.append(repeat(per_frame_actions, "f c -> (f h w) c", f=f, h=h, w=w))

            # Reshape the tokens.
            tokens.append(repeat(level_tokens, "f c h w -> (f h w) c"))

        return (
            torch.cat(tokens),
            torch.cat(indices),
            torch.cat(actions),
            per_frame_actions,
        )

    def get_train_example(self, index: int):
        # Randomly pick a step to train on.
        step_index = np.searchsorted(self.strategy.cdf, random())
        step_index = min(step_index, len(self.strategy.steps) - 1)

        # Figure out the smallest possible single read (this could almost certainly be
        # optimized further, but seems fine for the models we're testing).
        mask = self.strategy.selections[step_index] != UNSEEN
        frame_start = np.where(mask.any(axis=0))[0].min().item()
        frame_end = np.where(mask.any(axis=0))[0].max().item() + 1
        level_start = np.where(mask.any(axis=1))[0].min().item()
        level_end = np.where(mask.any(axis=1))[0].max().item() + 1
        tokens, indices, actions, _ = self.load_pyramid(
            index,
            level_start,
            level_end,
            frame_start,
            frame_end,
        )

        # Throw away everything that's not actually needed.
        index_f, index_l, _, _ = indices.unbind(dim=-1)
        mask = (self.strategy.selections[step_index] != UNSEEN)[index_l, index_f]
        tokens = tokens[mask]
        indices = indices[mask]
        actions = actions[mask]
        roles = self.strategy.selections[step_index][index_l, index_f][mask]
        assert not (roles == UNSEEN).any()

        # Pad to the maximum number of tokens for the current strategy.
        s, _ = tokens.shape
        assert s == self.strategy.num_tokens[step_index]
        padding = max(self.strategy.max_num_tokens - s, 0)
        tokens = F.pad(tokens, (0, 0, 0, padding))
        actions = F.pad(actions, (*([0] * (actions.ndim * 2 - 2)), 0, padding))
        indices = F.pad(indices, (0, 0, 0, padding))
        roles = F.pad(torch.tensor(roles), (0, padding))

        return {
            "tokens": tokens,
            "indices": indices,
            "actions": actions,
            "roles": roles,
        }

    def get_test_example(self, index: int):
        key, _ = self.video_index[index]

        # Determine how many frames need to be loaded in order to create enough buffer
        # for test_num_frames to be generated/denoised.
        _, _, limit = self.strategy.generate(
            self.cfg.first_denoised_frame,
            end=self.cfg.test_num_frames,
        )

        # Load that number of frames (or fewer, if the video is too short).
        tokens, indices, actions, frame_actions = self.load_pyramid(
            index,
            0,
            self.strategy.num_levels,
            0,
            limit,
        )

        # If the video is too short, pad it with empty frames and fake actions.
        index_f, _, _, _ = indices.unbind(dim=-1)
        frame_end = index_f.max().item() + 1
        if frame_end < limit:
            frame_actions = ActionEncoder.extend_actions(frame_actions, limit)

            # Define the indices and repeat the actions.
            tokens_extra = []
            indices_extra = []
            actions_extra = []
            for level in range(self.strategy.num_levels):
                h, w = (dim // 2**level for dim in self.cfg.latent_shape)
                f = limit - frame_end
                _, c = tokens.shape

                index_h = torch.arange(h, dtype=torch.int32)
                index_h = repeat(index_h, "h -> (f h w)", f=f, w=w)
                index_w = torch.arange(w, dtype=torch.int32)
                index_w = repeat(index_w, "w -> (f h w)", f=f, h=h)
                index_f = torch.arange(f, dtype=torch.int32)
                index_f = repeat(index_f, "f -> (f h w)", h=h, w=w) + frame_end
                index_l = torch.full_like(index_f, level)
                indices_extra.append(
                    torch.stack((index_f, index_l, index_h, index_w), dim=-1)
                )

                # Repeat the actions for each token.
                actions_extra.append(
                    repeat(frame_actions[-f:], "f c -> (f h w) c", f=f, h=h, w=w)
                )

                # Reshape the tokens.
                tokens_extra.append(torch.zeros((f * h * w, c), dtype=torch.float32))

            tokens = torch.cat((tokens, *tokens_extra))
            indices = torch.cat((indices, *indices_extra))
            actions = torch.cat((actions, *actions_extra))

        return {
            "key": key,
            "tokens": tokens,
            "indices": indices,
            "actions": actions,
        }

    @property
    def path(self) -> Path:
        for path in self.cfg.path:
            if path.exists():
                return path
        raise FileNotFoundError("Could not find a valid dataset path!")
