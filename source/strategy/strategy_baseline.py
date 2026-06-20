from dataclasses import dataclass
from typing import Generator, Literal

import numpy as np

from .strategy import CONTEXT, DENOISED, Step, Strategy


@dataclass(frozen=True, kw_only=True)
class StrategyBaselineCfg:
    name: Literal["baseline"] = "baseline"
    num_frames: int


class StrategyBaseline(Strategy[StrategyBaselineCfg]):
    def run(self) -> Generator[Step, None, None]:
        # The baseline strategy only uses the finest level.
        selection = np.empty((1, self.cfg.num_frames), dtype=np.int32)

        # The first half is context; the second half is denoised.
        selection[:, : self.cfg.num_frames // 2] = CONTEXT
        selection[:, self.cfg.num_frames // 2 :] = DENOISED

        yield Step(selection, None)
