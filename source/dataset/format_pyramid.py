import os
import sys
from functools import cache
from io import BytesIO
from math import prod
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from jaxtyping import BFloat16, Float32
from torch import Tensor


def save_pyramid(
    path: Path,
    levels: tuple[BFloat16[Tensor, "frame channel height=_ width=_"], ...],
    actions: Float32[Tensor, " frame *action_shape"],
) -> None:
    assert sys.byteorder == "little"

    # Convert the levels into bytes.
    f, *_ = actions.shape
    blobs = []
    for level in levels:
        level = rearrange(level, "f c h w -> (f c h w)").contiguous()
        level = torch.tensor(level.untyped_storage(), dtype=torch.uint8)
        level = rearrange(level, "(f byte) -> f byte", f=f)
        blobs.append(level)

    # Convert the actions into bytes.
    actions = rearrange(actions, "f ... -> (f ...)").contiguous()
    actions = torch.tensor(actions.untyped_storage(), dtype=torch.uint8)
    actions = rearrange(actions, "(f byte) -> f byte", f=f)
    blobs.append(actions)

    # Write the pyramid to a BytesIO.
    bytes_io = BytesIO()
    num_levels = len(levels)
    for start_level in range(num_levels):
        blob = torch.concatenate(blobs[start_level:], dim=-1)
        blob = rearrange(blob, "f byte -> (f byte)").contiguous()
        bytes_io.write(blob.cpu().numpy().tobytes())

    # Write the file to disk.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(bytes_io.getbuffer())


@cache
def compute_bytes_per_blob(
    num_levels: int,
    base_latent_shape: tuple[int, int, int],
    action_shape: tuple[int, ...],
) -> tuple[int, ...]:
    base_num_bytes = 2 * prod(base_latent_shape)  # bf16 = 2 bytes
    bytes_per_level = [(base_num_bytes // 4**level) for level in range(num_levels)]
    action_bytes = 4 * prod(action_shape)  # fp32 = 4 bytes
    return (*bytes_per_level, action_bytes)


def load_pyramid(
    path: Path,
    num_levels_in_pyramid: int,
    num_frames_in_pyramid: int,
    base_latent_shape: tuple[int, int, int],
    action_shape: tuple[int, ...],
    level_start: int,
    level_end: int | None,
    frame_start: int,
    frame_end: int,
    device: torch.device,
) -> tuple[
    tuple[BFloat16[Tensor, "frame channel height=_ width=_"], ...],
    Float32[Tensor, " frame *action_shape"],
]:
    # Compute the number of bytes per frame for each blob type.
    bytes_per_blob = compute_bytes_per_blob(
        num_levels_in_pyramid,
        base_latent_shape,
        action_shape,
    )
    bytes_per_frame = [
        sum(bytes_per_blob[start_level:])
        for start_level in range(num_levels_in_pyramid)
    ]
    bytes_per_frame = tuple(bytes_per_frame)

    # Compute the offset for previous levels.
    offset = 0
    for level in range(0, level_start):
        offset += num_frames_in_pyramid * bytes_per_frame[level]

    # Compute the offset for previous frames.
    offset += frame_start * bytes_per_frame[level_start]

    # Read the bytes for exactly what we need in the fewest possible syscalls.
    fd = os.open(path, os.O_RDONLY)
    f = frame_end - frame_start
    try:
        data = os.pread(fd, f * bytes_per_frame[level_start], offset)
    finally:
        os.close(fd)

    # Reinterpret the bytes as latents and actions.
    data = torch.tensor(np.frombuffer(data, dtype=np.uint8), device=device)
    data = rearrange(data, "(f byte) -> f byte", f=f)
    offset = 0
    levels = []
    c, h, w = base_latent_shape
    for level_index in range(level_start, num_levels_in_pyramid):
        num_bytes = bytes_per_blob[level_index]

        # Only read the level if it's not being disregarded.
        if level_end is None or level_index < level_end:
            level = data[:, offset : offset + num_bytes]

            # Reinterpret the blob as the correct shape and dtype.
            level = rearrange(level, "f byte -> (f byte)").contiguous()
            level = rearrange(
                level.view(torch.bfloat16),
                "(f c h w) -> f c h w",
                f=f,
                c=c,
                h=h // 2**level_index,
                w=w // 2**level_index,
            )
            levels.append(level)

        # Make sure the offset is correct for the next level.
        offset += num_bytes

    # Reinterpret the action blob.
    actions = data[:, offset:]
    actions = rearrange(actions, "f byte -> (f byte)").contiguous()
    actions = actions.view(torch.float32).view((f, *action_shape))

    return tuple(levels), actions


def load_pyramid_with_indices(
    path: Path,
    num_levels_in_pyramid: int,
    num_frames_in_pyramid: int,
    base_latent_shape: tuple[int, int, int],
    action_shape: tuple[int, ...],
    level_start: int,
    level_end: int | None,
    frame_indices: tuple[int, ...] | list[int],
    device: torch.device,
) -> tuple[
    tuple[BFloat16[Tensor, "frame channel height=_ width=_"], ...],
    Float32[Tensor, " frame *action_shape"],
]:
    # Compute the number of bytes per frame for each blob type.
    bytes_per_blob = compute_bytes_per_blob(
        num_levels_in_pyramid,
        base_latent_shape,
        action_shape,
    )
    bytes_per_frame = [
        sum(bytes_per_blob[start_level:])
        for start_level in range(num_levels_in_pyramid)
    ]
    bytes_per_frame = tuple(bytes_per_frame)

    # Compute the offset for previous levels.
    base_offset = 0
    for level in range(0, level_start):
        base_offset += num_frames_in_pyramid * bytes_per_frame[level]

    # Read exactly what we need.
    levels = [[] for _ in range(level_end)]
    actions = []
    fd = os.open(path, os.O_RDONLY)
    try:
        for frame_index in sorted(frame_indices):
            offset = base_offset + frame_index * bytes_per_frame[level_start]
            data = os.pread(fd, bytes_per_frame[level_start], offset)

            # Reinterpret the bytes as latents and actions.
            data = torch.tensor(np.frombuffer(data, dtype=np.uint8), device=device)
            offset = 0
            c, h, w = base_latent_shape
            for level_index in range(level_start, num_levels_in_pyramid):
                num_bytes = bytes_per_blob[level_index]

                # Only read the level if it's not being disregarded.
                if level_end is None or level_index < level_end:
                    frame = data[offset : offset + num_bytes]

                    # Reinterpret the blob as the correct shape and dtype.
                    frame = rearrange(
                        frame.view(torch.bfloat16),
                        "(c h w) -> c h w",
                        c=c,
                        h=h // 2**level_index,
                        w=w // 2**level_index,
                    )
                    levels[level_index].append(frame)

                # Make sure the offset is correct for the next level.
                offset += num_bytes

            # Reinterpret the action blob.
            action = data[offset:]
            action = action.view(torch.float32).view(action_shape)
            actions.append(action)
    finally:
        os.close(fd)

    return tuple(torch.stack(x) for x in levels if x), torch.stack(actions)
