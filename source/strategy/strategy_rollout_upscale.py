from dataclasses import dataclass
from typing import Generator, Literal

import numpy as np

from .strategy import CONTEXT, DENOISED, Step, Strategy


@dataclass(frozen=True, kw_only=True)
class StrategyRolloutUpscaleCfg:
    name: Literal["rollout_upscale"] = "rollout_upscale"
    base_num_frames: int
    num_levels: int


class StrategyRolloutUpscale(Strategy[StrategyRolloutUpscaleCfg]):
    def run(self) -> Generator[Step, None, None]:
        # Compute the chunk widths for each level.
        widths = [self.cfg.base_num_frames * 4**i for i in range(self.cfg.num_levels)]

        # Since the coarsest level does not include context from a coarser level, it
        # holds 9/8 times as many frames as it otherwise would.
        widths[-1] = widths[-1] * 9 // 8

        # Roll out for each level.
        total = widths[-1] * 3 // 2
        for level in reversed(range(self.cfg.num_levels)):
            start = (widths[-1] - widths[level]) // 2
            while start + widths[level] <= total:
                # Define some helpful shorthands relative to the current step.
                middle = start + widths[level] // 2
                end = start + widths[level]

                # Define and yield the current step.
                selection = np.zeros((self.cfg.num_levels, total), dtype=np.int32)
                selection[level, start:middle] = CONTEXT
                selection[level, middle:end] = DENOISED
                if level != self.cfg.num_levels - 1:
                    selection[level + 1, middle:end] = CONTEXT
                yield Step(selection, level)

                # Move to the next step.
                start += widths[level] // 2
