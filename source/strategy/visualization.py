import numpy as np
import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ..image_io import VideoEncoder
from ..layout import add_border, add_label, vcat
from .strategy import CONTEXT, DENOISED, Selection, Strategy

COLORS = (
    (1.0, 0.85, 0.85),  # pending
    (0.133, 0.612, 1),  # context
    (1, 0, 0),  # denoising
    (0.25, 0.25, 0.25),  # complete
    (0.85, 0.85, 0.85),  # excluded
)


def draw_chunk(
    selection: Selection,
    ph: int = 16,
    pw: int = 16,
    frames_per_row: int = 128,
    label: str | None = None,
) -> Float[Tensor, "rgb=3 height width"]:
    selection = torch.tensor(selection)
    selection = repeat(selection, "l f -> (l ph) (f pw)", ph=ph + 1, pw=pw + 1)
    h, w = selection.shape

    # Color the squares.
    image = torch.zeros((h, w, 3), dtype=torch.float32, device=selection.device)
    for index, color in enumerate(COLORS):
        color = torch.tensor(color, dtype=torch.float32, device=selection.device)
        image[selection == index] = color

    # Add lines between the squares.
    image[:: ph + 1] = 1
    image[:, :: pw + 1] = 1

    # Split the image into rows.
    image = rearrange(image, "h w c -> c h w")
    image = image[:, :-1]
    image = image.split(frames_per_row * (pw + 1), dim=2)
    image = [x[:, :, :-1] for x in image]
    image = vcat(image, gap=16)

    # Add a label and encode the frame.
    _, h, w = image.shape
    if label is not None:
        image = add_label(image, label)
    return add_border(image)


def animate(
    strategy: Strategy,
    frames_per_row: int = 128,
    start: int = 256,
    end: int | None = None,
    limit: int | None = 768,
) -> bytes:
    video_encoder = VideoEncoder(fps=4)
    state = None

    sequence, end, limit = strategy.generate(start, end=end, limit=limit)
    state = np.zeros((strategy.num_levels, limit), dtype=np.int32)
    state[:, :start] = 3
    state[:, end:] = 4
    width = strategy.chunk_num_frames

    for i, (step, offset) in enumerate(sequence):
        drawn = state.copy()
        drawn[:, offset : offset + width][step.selection == CONTEXT] = CONTEXT
        drawn[:, offset : offset + width][step.selection == DENOISED] = DENOISED
        cost = strategy.num_tokens[i % len(strategy.steps)]
        image = draw_chunk(
            drawn,
            frames_per_row=frames_per_row,
            label=f"Cost: {cost} Group: {step.sampling_group}",
        )
        state[:, offset : offset + width][step.selection == DENOISED] = 3
        video_encoder.add_frames(image[None])
    return video_encoder.result()[0]


if __name__ == "__main__":

    from . import get_strategy
    from .strategy_baseline import StrategyBaselineCfg
    from .strategy_framepack import StrategyFramePackCfg
    from .strategy_framepack_mirrored import StrategyFramePackMirroredCfg
    from .strategy_millivid import StrategyMilliVidCfg
    from .strategy_rollout_upscale import StrategyRolloutUpscaleCfg
    from .strategy_upscale import StrategyUpscaleCfg

    cfgs = (
        StrategyBaselineCfg(num_frames=4),
        StrategyFramePackCfg(num_denoised_frames=1, num_context_frames=(1, 4, 16)),
        StrategyMilliVidCfg(num_frames_per_level=(1, 4, 16), extra_budget=256),
        StrategyFramePackMirroredCfg(
            num_context_frames=(1, 4, 16, 64),
            num_denoised_frames=(1, 4, 16, 64),
        ),
        StrategyRolloutUpscaleCfg(base_num_frames=4, num_levels=4),
        StrategyUpscaleCfg(base_num_frames=4, num_levels=4),
    )

    for cfg in cfgs:
        strategy = get_strategy(cfg, 16 * 16)
        with open(f"strategy_{cfg.name}_re10k_3_levels.mp4", "wb") as f:
            f.write(animate(strategy))
