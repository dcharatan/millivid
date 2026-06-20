from dataclasses import dataclass
from typing import Generator, Literal

import numpy as np

from .strategy import CONTEXT, DENOISED, Step, Strategy


@dataclass(frozen=True, kw_only=True)
class StrategyFramePackCfg:
    name: Literal["framepack"] = "framepack"
    num_denoised_frames: int
    num_context_frames: tuple[int, ...]


class StrategyFramePack(Strategy[StrategyFramePackCfg]):
    def run(self) -> Generator[Step, None, None]:
        # Determine the selection's shape.
        num_levels = len(self.cfg.num_context_frames)
        num_frames = self.cfg.num_denoised_frames + sum(self.cfg.num_context_frames)
        selection = np.zeros((num_levels, num_frames), dtype=np.int32)

        # Mark the context frames.
        start = 0
        for level, num_frames in reversed(list(enumerate(self.cfg.num_context_frames))):
            selection[level, start : start + num_frames] = CONTEXT
            start += num_frames

        # Mark the denoised frames.
        selection[:, start:] = DENOISED

        yield Step(selection, None)
