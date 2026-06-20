from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from typing import Generator, Generic, TypeVar

import numpy as np
from jaxtyping import Float, Int
from tqdm import trange

from ..ddp import warm_cached_properties


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


C = TypeVar("C")  # config type

# The valid values for a chunk:
UNSEEN = 0
CONTEXT = 1
DENOISED = 2

type Selection = Int[np.ndarray, "level frame"]


@dataclass(frozen=True)
class Block:
    frame_start: int
    frame_end: int
    level_start: int
    level_end: int


@dataclass(frozen=True)
class Step:
    selection: Selection
    sampling_group: int | str | None


class Strategy(ABC, Generic[C]):
    cfg: C
    base_num_tokens: int

    def __init__(self, cfg: C, base_num_tokens: int) -> None:
        self.cfg = cfg
        self.base_num_tokens = base_num_tokens
        warm_cached_properties(self)

    #############
    # Interface #
    #############

    @abstractmethod
    def run(self) -> Generator[Step, None, None]:
        pass

    @property
    def num_clipped_frames(self) -> int:
        """Some strategies involve generating frames and then throwing them away. If
        this is the case for your strategy, override num_clipped_frames to be nonzero.
        For example, if you want basic rollout (baseline) where you generate four frames
        and then throw two away, override num_clipped_frames to be 2. Note that for more
        complex strategies, some of the clipping will naturally occur as a result of
        "overwriting" denoised tokens in previous steps. In this case, num_clipped_steps
        is only needed to clip the steps near the end of the chunk."""
        return 0

    #######################
    # Computed Properties #
    #######################

    @cached_property
    def steps(self) -> tuple[Step, ...]:
        steps = tuple(self.run())

        # Validate the steps.
        num_levels, num_frames = steps[0].selection.shape
        for step in steps:
            s = step.selection
            assert s.shape == (num_levels, num_frames)
            assert np.all((s == UNSEEN) | (s == CONTEXT) | (s == DENOISED))

        return steps

    @cached_property
    def selections(self) -> tuple[Selection, ...]:
        return tuple(step.selection for step in self.steps)

    @cached_property
    def cdf(self) -> Float[np.ndarray, " step"]:
        # Compute the CDF for sampling steps during training. Each sampling group gets
        # an equal probability weight. For example, if there are 2 steps in group A and
        # 4 steps in group B, the steps in group A will have P = 0.25 and the steps in
        # group B will have P = 0.125.
        counts = Counter()
        for step in self.steps:
            counts[step.sampling_group] += 1
        weights = {k: 1 / (v * len(counts)) for k, v in counts.items()}
        pdf = [weights[step.sampling_group] for step in self.steps]
        return np.cumsum(pdf)

    @cached_property
    def num_levels(self) -> int:
        num_levels, _ = self.steps[0].selection.shape
        return num_levels

    @cached_property
    def chunk_num_frames(self) -> int:
        _, num_frames = self.steps[0].selection.shape
        return num_frames

    @cached_property
    def chunk_denoising_start(self) -> int:
        denoising_start_frames = [
            np.where(step.selection == DENOISED)[1].min().item() for step in self.steps
        ]
        return min(denoising_start_frames)

    @cached_property
    def num_tokens(self) -> tuple[int, ...]:
        tokens_per_level = self.base_num_tokens // 4 ** np.arange(self.num_levels)
        num_tokens = []
        for step in self.steps:
            mask = (step.selection == DENOISED) | (step.selection == CONTEXT)
            num_tokens.append((mask * tokens_per_level[:, None]).sum().item())
        return tuple(num_tokens)

    @cached_property
    def max_num_tokens(self) -> int:
        return max(self.num_tokens)

    @cached_property
    def requires_attention_masking(self) -> bool:
        return not all([x == self.max_num_tokens for x in self.num_tokens])

    def generate(
        self,
        start: int,
        end: int | None = None,
        limit: int | None = None,
    ) -> tuple[
        Generator[tuple[Step, int], None, None],
        int,  # video end (exclusive)
        int,  # buffer end (exclusive)
    ]:
        # The user must specify either an end frame or a limit. If an end frame is
        # specified, the buffer of partially denoised frames may go further. If a limit
        # is specified, the buffer will not exceed the limit.
        assert (end is None) != (limit is None)

        # Figure out how many chunks will be needed.
        buffer = self.num_clipped_frames
        denoised_width = self.chunk_num_frames - self.chunk_denoising_start - buffer

        if limit is not None:
            # Compute the number of frames based on denoising_limit. This ensures that
            # the buffer never goes past the denoising limit.
            denoised_length = limit - start - buffer
            num_chunks = denoised_length // denoised_width
        if end is not None:
            # Compute the number of frames based on denoising_end. This ensures that
            # frames are generated at least until denoising_end.
            denoised_length = end - start
            num_chunks = ceildiv(denoised_length, denoised_width)

        end = start + num_chunks * denoised_width
        limit = end + buffer

        # Iterate through the chunks.
        def sequence():
            chunk_start = start - self.chunk_denoising_start
            progress = trange(num_chunks * len(self.steps), desc="Sampling")
            for _ in range(num_chunks):
                for step in self.steps:
                    yield step, chunk_start
                    progress.update()
                chunk_start += denoised_width

        return sequence(), end, limit
