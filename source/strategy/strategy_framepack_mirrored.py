from dataclasses import dataclass
from typing import Generator, Literal

import numpy as np

from .strategy import CONTEXT, DENOISED, Step, Strategy


@dataclass(frozen=True, kw_only=True)
class StrategyFramePackMirroredCfg:
    name: Literal["framepack_mirrored"] = "framepack_mirrored"
    num_denoised_frames: tuple[int, ...]
    num_context_frames: tuple[int, ...]


class StrategyFramePackMirrored(Strategy[StrategyFramePackMirroredCfg]):
    def run(self) -> Generator[Step, None, None]:
        # Determine the selection's shape.
        num_levels = len(self.cfg.num_context_frames)
        assert num_levels == len(self.cfg.num_denoised_frames)

        num_context_frames = sum(self.cfg.num_context_frames)
        num_denoised_frames = sum(self.cfg.num_denoised_frames)
        num_frames = num_denoised_frames + num_context_frames
        selection = np.zeros((num_levels, num_frames), dtype=np.int32)

        # Mark the context frames.
        start = 0
        for level, num_frames in reversed(list(enumerate(self.cfg.num_context_frames))):
            selection[level, start : start + num_frames] = CONTEXT
            start += num_frames

        # Mark the denoised frames.
        for level, num_frames in enumerate(self.cfg.num_denoised_frames):
            level_slice = slice(None) if level == 0 else level
            selection[level_slice, start : start + num_frames] = DENOISED
            start += num_frames

        yield Step(selection, None)

    @property
    def num_clipped_frames(self) -> int:
        return sum(self.cfg.num_denoised_frames) - self.cfg.num_denoised_frames[0]
