from dataclasses import dataclass
from typing import Generator, Literal

import numpy as np

from .strategy import CONTEXT, DENOISED, Step, Strategy


@dataclass(frozen=True, kw_only=True)
class StrategyMilliVidCfg:
    name: Literal["millivid"] = "millivid"
    num_frames_per_level: tuple[int, ...]
    extra_budget: int

    @property
    def num_levels(self) -> int:
        return len(self.num_frames_per_level)


class StrategyMilliVid(Strategy[StrategyMilliVidCfg]):
    def run(self) -> Generator[Step, None, None]:
        windows = self.cfg.num_frames_per_level
        starts = [
            self.chunk_num_frames - windows[-1] - windows[level]
            for level in range(self.cfg.num_levels)
        ]
        target_level = -1

        def traverse(level: int):
            nonlocal target_level
            starts[level] += windows[level]
            target_level = level
            yield
            if level > 0:
                for _ in range(windows[level] // windows[level - 1]):
                    yield from traverse(level - 1)

        def read_step() -> Step:
            selection = np.zeros(
                (self.cfg.num_levels, self.chunk_num_frames),
                dtype=np.int32,
            )
            left = starts[0]
            right = starts[0]
            for level in range(self.cfg.num_levels):
                start = starts[level]
                window = windows[level]
                end = start + window

                if level == target_level:
                    selection[level, start:end] = DENOISED

                    extra = self.num_extra_frames[level]
                    left -= extra
                    selection[level, left : left + extra] = CONTEXT
                    continue

                earlier = self.overlap(start, end, None, left)
                overlap = self.overlap(start, end, left, right)
                later = self.overlap(start, end, right, None)
                assert earlier + overlap + later == window

                left = left - earlier - overlap
                right = right + later

                selection[level, left : left + earlier + overlap] = CONTEXT
                selection[level, right - later : right] = CONTEXT

            return Step(selection, target_level)

        for _ in traverse(self.cfg.num_levels - 1):
            yield read_step()

    @property
    def num_extra_frames(self) -> tuple[int, ...]:
        return tuple(
            self.cfg.extra_budget // (self.base_num_tokens // 4**level)
            for level in range(self.cfg.num_levels)
        )

    @property
    def chunk_num_frames(self) -> int:
        return sum(self.cfg.num_frames_per_level) + self.num_extra_frames[-1]

    @staticmethod
    def overlap(
        start_a: int,
        end_a: int,
        start_b: int | None,
        end_b: int | None,
    ) -> int:
        start = start_a if start_b is None else max(start_a, start_b)
        end = end_a if end_b is None else min(end_a, end_b)
        return end - start if start < end else 0
