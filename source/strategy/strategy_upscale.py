from dataclasses import dataclass
from typing import Generator, Literal

import numpy as np

from .strategy import CONTEXT, DENOISED, Step, Strategy


@dataclass(frozen=True, kw_only=True)
class StrategyUpscaleCfg:
    name: Literal["upscale"] = "upscale"
    base_num_frames: int
    num_levels: int


class StrategyUpscale(Strategy[StrategyUpscaleCfg]):
    def run(self) -> Generator[Step, None, None]:
        # Compute the chunk widths for each level.
        widths = [self.cfg.base_num_frames * 4**i for i in range(self.cfg.num_levels)]

        # Since the coarsest level does not include context from a coarser level, it
        # holds 5/4 times as many frames as it otherwise would.
        widths[-1] = widths[-1] * 5 // 4

        # Roll out at the coarsest level.
        total = widths[-1] * 3 // 2
        start = 0
        while start + widths[-1] <= total:
            # Define some helpful shorthands relative to the current step.
            middle = start + widths[-1] // 2
            end = start + widths[-1]

            # Define and yield the current step.
            selection = np.zeros((self.cfg.num_levels, total), dtype=np.int32)
            selection[-1, start:middle] = CONTEXT
            selection[-1, middle:end] = DENOISED
            yield Step(selection, self.cfg.num_levels - 1)

            # Move to the next step.
            start += widths[-1] // 2

        # Upscale at the other levels.
        for level in reversed(range(self.cfg.num_levels - 1)):
            start = widths[-1] // 2
            while start + widths[level] <= total:
                # Define some helpful shorthands relative to the current step.
                end = start + widths[level]

                # Define and yield the current step.
                selection = np.zeros((self.cfg.num_levels, total), dtype=np.int32)
                selection[level, start:end] = DENOISED
                selection[level + 1, start:end] = CONTEXT
                yield Step(selection, level)

                # Move to the next step.
                start += widths[level]
