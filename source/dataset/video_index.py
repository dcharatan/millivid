import random
from typing import NamedTuple

import numpy as np

from .dataset import Split


class VideoIndexSample(NamedTuple):
    key: str
    frame_index: int


class VideoIndex:
    keys: tuple[str, ...]
    boundaries: tuple[int, ...]
    reordering: tuple[int, ...]

    def __init__(
        self,
        index: list[tuple[str, int]],
        frames_per_clip: int | None,
        seed: int = 0,
    ) -> None:
        if frames_per_clip is None:
            # In this case, there's no limit to the number of frames per clip, so each
            # video is weighted equally. Note that this may cause jagged batches to be
            # created.
            keys = [key for key, _ in index]
            weights = [1] * len(index)
        else:
            # Extract every possible fixed-length clip from every video.
            keys = []
            weights = []
            for key, length in index:
                if length < frames_per_clip:
                    continue
                keys.append(key)
                weights.append(length - frames_per_clip + 1)
        self.keys = tuple(keys)
        self.boundaries = tuple(np.cumsum(weights).tolist())

        # Add a fixed, random reordering to the data.
        rng = np.random.default_rng(seed)
        self.reordering = tuple(rng.permutation(self.boundaries[-1]).tolist())

    def __len__(self) -> int:
        return self.boundaries[-1]

    def __getitem__(self, index: int) -> VideoIndexSample:
        # Take the reordering into account.
        index = self.reordering[index]

        # Figure out which video and frame to return.
        video_index = np.searchsorted(self.boundaries, index, side="right")
        video_start = self.boundaries[video_index - 1] if video_index else 0

        # Return the video ID and frame index.
        key = self.keys[video_index]
        frame_index = index - video_start
        return VideoIndexSample(key, frame_index)


def split_index(
    index: dict[str, int],
    split: Split,
    test_fraction: float = 0.01,
) -> list[tuple[str, int]]:
    index = list(sorted(index.items()))
    random.Random(0).shuffle(index)
    cutoff = int(len(index) * test_fraction)
    return {
        "train": lambda: index[cutoff:],
        "vis": lambda: index[:cutoff],
        "test": lambda: index[:cutoff],
        "all": lambda: index,
    }[split]()
