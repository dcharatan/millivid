import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Generator, Literal, TypedDict

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from jaxtyping import Float, Int
from torch import Tensor, nn
from tqdm import trange

from ..components.adaptivity import DiTAdaptive, LevelsCfg
from ..components.dit_per_token_c import Unpatchify
from ..components.noise_schedule import NoiseSchedule
from ..dataset.dataset_images import DatasetImagesCfg
from ..evaluation.benchmarker import Benchmarker
from ..image_io import encode_videos
from ..layout import add_label, hcat
from .model import ConfigurableModel, TestStepMetadata, TrainStepOutput, VisStepOutput
from .model_autoencoder import Decoder, load_decoder


class Batch(TypedDict):
    latents: Float[Tensor, "batch frame channel height width"]
    actions: Int[Tensor, "batch frame"]
    level: Int[Tensor, " batch"]


@dataclass(frozen=True)
class Block:
    frame_start: int
    frame_end: int
    level_start: int
    level_end: int
    context: bool


@dataclass(frozen=True)
class ModelVideoFramePackCfg(LevelsCfg):
    name: Literal["video_framepack"]
    model_size: Literal["S", "B", "L", "XL"]
    decoder_path: tuple[Path, ...]
    image_dataset: DatasetImagesCfg
    bottleneck: int | None
    sampling_steps: int
    shift: float
    first_denoised_frame: int
    guidance_dropout_probability: float
    num_denoised_frames: int
    num_context_frames: tuple[int, ...]
    test_num_frames: int

    def __post_init__(self) -> None:
        assert len(self.num_context_frames) == self.available_num_levels

    @property
    def blocks(self) -> tuple[Block, ...]:
        offset = 0

        # Add the context frames.
        context = []
        for level, num_frames in reversed(list(enumerate(self.num_context_frames))):
            if num_frames == 0:
                continue
            context.append(Block(offset, offset + num_frames, level, level + 1, True))
            offset += num_frames

        # Add the denoised frames.
        denoised = Block(
            offset,
            offset + self.num_denoised_frames,
            0,
            self.available_num_levels,
            False,
        )

        return (*context, denoised)

    @property
    def num_frames(self) -> int:
        return self.num_denoised_frames + sum(self.num_context_frames)


def get_decoding_indices(
    x: Float[Tensor, "batch frame channel height width"],
    level: int,
    start_frame: int = 0,
) -> Int[Tensor, "batch frame*height*width flhw=4"]:
    b, f, _, h, w = x.shape
    index_f = torch.arange(f, dtype=torch.int32, device=x.device)
    index_f = repeat(index_f, "f -> f h w", h=h, w=w) + start_frame
    index_l = torch.full_like(index_f, level)
    index_h = torch.arange(h, dtype=torch.int32, device=x.device)
    index_h = repeat(index_h, "h -> f h w", f=f, w=w)
    index_w = torch.arange(w, dtype=torch.int32, device=x.device)
    index_w = repeat(index_w, "w -> f h w", f=f, h=h)
    indices = torch.stack((index_f, index_l, index_h, index_w), dim=-1)
    return repeat(indices, "f h w flhw -> b (f h w) flhw", b=b)


##############
# Main Model #
##############


class DiTWrapper(nn.Module):
    cfg: ModelVideoFramePackCfg
    patchify: nn.ModuleList
    dit: DiTAdaptive
    unpatchify: Unpatchify

    def __init__(self, cfg: ModelVideoFramePackCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.dit = DiTAdaptive(cfg.available_num_levels, cfg.model_size)
        patchify = [
            (
                nn.Linear(cfg.latent_channels * 4**level, self.dit.hidden_channels)
                if cfg.bottleneck is None
                else nn.Sequential(
                    nn.Linear(cfg.latent_channels * 4**level, cfg.bottleneck),
                    nn.Linear(cfg.bottleneck, self.dit.hidden_channels),
                )
            )
            for level, _ in enumerate(cfg.num_context_frames)
        ]
        self.patchify = nn.ModuleList(patchify)
        self.unpatchify = Unpatchify(self.dit.hidden_channels, cfg.latent_channels)

    def forward(
        self,
        x: Float[Tensor, "batch frame channel height width"],
        t: Float[Tensor, "batch frame"],
        actions: Float[Tensor, "batch frame action_channel"],
    ) -> Float[Tensor, "batch denoised_frame channel height width"]:
        x, t, actions, indices = self.make_tokens(x, t, actions)
        x, c = self.dit(x, indices, t, actions)

        # Extract the tokens that correspond to the denoised frames and unpatchify them.
        nd = self.cfg.base_num_tokens * self.cfg.num_denoised_frames
        x = self.unpatchify.forward(x[:, -nd:], c[:, -nd:])
        h, w = self.cfg.latent_shape
        return rearrange(x, "b (f h w) c -> b f c h w", h=h, w=w)

    def make_indices(
        self,
        bfhw: tuple[int, int, int, int],
        frame_offset: int,
        level: int,
        device: torch.device,
    ) -> Int[Tensor, "batch token flhw=4"]:
        b, f, h, w = bfhw
        index_h = torch.arange(h, dtype=torch.int32, device=device)
        index_h = repeat(index_h, "h -> b (f h w)", b=b, f=f, w=w)
        index_w = torch.arange(w, dtype=torch.int32, device=device)
        index_w = repeat(index_w, "w -> b (f h w)", b=b, f=f, h=h)
        index_f = torch.arange(f, dtype=torch.int32, device=device) + frame_offset
        index_f = repeat(index_f, "f -> b (f h w)", b=b, h=h, w=w)
        index_l = torch.full_like(index_f, level)
        return torch.stack((index_f, index_l, index_h, index_w), dim=-1)

    def make_tokens(
        self,
        x: Float[Tensor, "batch frame channel height width"],
        t: Float[Tensor, "batch frame"],
        actions: Float[Tensor, "batch frame action_channel"],
    ) -> tuple[
        Float[Tensor, "batch token hidden_channel"],  # x
        Float[Tensor, "batch token"],  # t
        Float[Tensor, "batch token action_channel"],  # actions
        Int[Tensor, "batch token flhw=4"],  # indices
    ]:
        offset_f = 0
        all_x = []
        all_t = []
        all_actions = []
        all_indices = []

        # Define the specification for patchification.
        specification = list(reversed(list(enumerate(self.cfg.num_context_frames))))
        specification.append((0, self.cfg.num_denoised_frames))
        assert sum([f for _, f in specification]) == x.shape[1]

        # Tokenize everything with varying patchify kernels.
        for level, num_frames in specification:
            x_level = x[:, offset_f : offset_f + num_frames]
            t_level = t[:, offset_f : offset_f + num_frames]
            actions_level = actions[:, offset_f : offset_f + num_frames]

            # Create tokens for the given level.
            x_level = rearrange(
                x_level,
                "b f c (h ph) (w pw) -> b f h w (c ph pw)",
                ph=2**level,
                pw=2**level,
            )
            x_level: Tensor = self.patchify[level](x_level)
            b, f, h, w, _ = x_level.shape
            x_level = rearrange(x_level, "b f h w c -> b (f h w) c")

            # Repeat noise levels and actions so that we get one per token.
            t_level = repeat(t_level, "b f -> b (f h w)", h=h, w=w)
            actions_level = repeat(actions_level, "b f c -> b (f h w) c", h=h, w=w)

            # Record the level's tokens.
            indices_level = self.make_indices((b, f, h, w), offset_f, level, x.device)
            all_x.append(x_level)
            all_t.append(t_level)
            all_actions.append(actions_level)
            all_indices.append(indices_level)
            offset_f += num_frames

        # Concatenate all of the inputs.
        all_x = torch.cat(all_x, dim=1)
        all_t = torch.cat(all_t, dim=1)
        all_actions = torch.cat(all_actions, dim=1)
        all_indices = torch.cat(all_indices, dim=1)
        return all_x, all_t, all_actions, all_indices


class ModelVideoFramePack(
    ConfigurableModel[ModelVideoFramePackCfg, Batch, Batch, Batch]
):
    dit: DiTWrapper
    decoder: Decoder
    noise_schedule: NoiseSchedule

    def __init__(self, cfg: ModelVideoFramePackCfg) -> None:
        super().__init__(cfg)
        self.dit = DiTWrapper(cfg)
        self.noise_schedule = NoiseSchedule(shift=cfg.shift)
        self.decoder = load_decoder(cfg, cfg.decoder_path)

    ############
    # Sampling #
    ############

    def get_sampling_noise_levels(
        self,
        x: Float[Tensor, "batch frame channel height width"],
        t: float,
    ) -> Float[Tensor, "batch frame"]:
        b, f, _, _, _ = x.shape
        t = torch.full((b, f), t, dtype=torch.float32, device=x.device)
        t[:, : -self.cfg.num_denoised_frames] = 0
        return t

    def denoise(
        self,
        x: Float[Tensor, "batch frame channel height width"],
        eps: Float[Tensor, "batch frame channel height width"],
        actions: Float[Tensor, "batch frame action_channel"],
        t: Float[Tensor, "batch frame"],
    ) -> tuple[
        Float[Tensor, "batch denoised_frame channel height width"],  # x_hat
        Float[Tensor, "batch denoised_frame channel height width"],  # eps_hat
    ]:
        # Define the input.
        alpha_t, sigma_t = self.noise_schedule.coefficients(t)
        alpha_t = alpha_t[:, :, None, None, None]
        sigma_t = sigma_t[:, :, None, None, None]
        x_t = alpha_t * x + sigma_t * eps

        # Get the model's prediction of V.
        v_hat: Tensor = self.dit(x_t, t, actions)

        # Trim everything else to only the denoised frames.
        nd = self.cfg.num_denoised_frames
        alpha_t = alpha_t[:, -nd:]
        sigma_t = sigma_t[:, -nd:]
        x_t = x_t[:, -nd:]

        # Convert the model's prediction of V to predictions of signal and noise.
        x_hat = alpha_t * x_t - sigma_t * v_hat
        eps_hat = alpha_t * v_hat + sigma_t * x_t

        return x_hat.type(x.dtype), eps_hat.type(eps.dtype)

    def generate(
        self,
        batch: Batch,
        seed: int = 0,
    ) -> Float[Tensor, "batch frame channel height width"]:
        # Figure out at which frame generation should start.
        for block in self.cfg.blocks:
            if block.context:
                continue
            required_context = block.frame_start
            break
        nd = self.cfg.num_denoised_frames
        start = self.cfg.first_denoised_frame
        num_rollout_steps = (self.cfg.test_num_frames - start) // nd
        assert start >= required_context

        # Create the initial sample by zeroing out everything that's not context.
        x = batch["latents"].clone()
        x[:, start:] = 0
        eps = torch.empty_like(x)
        eps.normal_(generator=torch.Generator(self.device).manual_seed(seed))

        # Figure out how many evaluations will be needed.
        total = self.cfg.sampling_steps * num_rollout_steps

        # Do the actual sampling.
        progress = trange(total, desc="Sampling")
        offset = start - required_context
        for chunk_start in range(offset, offset + nd * num_rollout_steps, nd):
            chunk_end = chunk_start + self.cfg.num_frames
            x_chunk = x[:, chunk_start:chunk_end]
            eps_chunk = eps[:, chunk_start:chunk_end]
            actions_chunk = batch["actions"][:, chunk_start:chunk_end]
            for t in torch.linspace(1, 0, self.cfg.sampling_steps):
                x_hat, eps_hat = self.denoise(
                    x_chunk,
                    eps_chunk,
                    actions_chunk,
                    self.get_sampling_noise_levels(x_chunk, t.item()),
                )
                x_chunk[:, -nd:] = x_hat
                eps_chunk[:, -nd:] = eps_hat
                progress.update()
            x[:, chunk_end - nd : chunk_end] = x_chunk[:, -nd:]
            eps[:, chunk_end - nd : chunk_end] = eps_chunk[:, -nd:]
        progress.close()

        return x

    ############
    # Training #
    ############

    def get_training_noise_levels(
        self,
        x: Float[Tensor, "batch frame channel height width"],
    ) -> Float[Tensor, "batch frame"]:
        b, f, _, _, _ = x.shape
        t = torch.rand((b,), dtype=torch.float32, device=x.device)
        t = repeat(t, "b -> b f", f=f).contiguous()
        p = self.cfg.guidance_dropout_probability
        dropout = torch.rand((b, 1), dtype=torch.float32, device=self.device) < p
        t[:, : -self.cfg.num_denoised_frames] = dropout.float()
        return t

    def train_step(self, batch: Batch) -> TrainStepOutput:
        x = batch["latents"]
        eps = torch.randn_like(x)

        # Defie the model's input.
        t = self.get_training_noise_levels(x)
        alpha_t, sigma_t = self.noise_schedule.coefficients(t)
        alpha_t = alpha_t[:, :, None, None, None]
        sigma_t = sigma_t[:, :, None, None, None]
        x_t = alpha_t * x + sigma_t * eps

        # Get the model's prediction of V.
        v_hat: Tensor = self.dit(x_t, t, batch["actions"])

        # Compute a loss on the predicted V.
        v = alpha_t * eps - sigma_t * x

        # Only compute a loss on the denoised tokens.
        return F.mse_loss(v_hat, v[:, -self.cfg.num_denoised_frames :])

    #################
    # Visualization #
    #################

    def vis_step(self, batch: Batch) -> VisStepOutput:
        images, videos, metrics = {}, {}, {}

        # Visualize generation with ground-truth actions.
        videos["sample"] = self.vis_generation(batch)

        return VisStepOutput(images=images, metrics=metrics, videos=videos)

    def vis_generation(self, batch: Batch) -> tuple[bytes, ...]:
        logging.info("Visualizing generation")

        x = batch["latents"]
        sample, start = self.generate(batch)

        sample = sample[:, start - 1 :]
        x = x[:, start - 1 :]

        def vis_frame_generator():
            # Set up generators for decoding the sample and ground truth.
            sample_generator = self.decoder.decode_video(
                rearrange(sample, "b f c h w -> b (f h w) c"),
                get_decoding_indices(sample, 0),
                level=0,
            )
            gt_generator = self.decoder.decode_video(
                rearrange(x, "b f c h w -> b (f h w) c"),
                get_decoding_indices(x, 0),
                level=0,
            )

            # Merge the decoded outputs into visualization frames, then yield them.
            for (sample_frame, _), (gt_frame, _) in zip(sample_generator, gt_generator):
                vis_sample = add_label(sample_frame, "Sample")
                vis_gt = add_label(gt_frame, "Ground Truth")
                yield hcat((vis_sample, vis_gt), gap=16, border=16)

        return encode_videos(vis_frame_generator())

    ###########
    # Testing #
    ###########

    @property
    def benchmarker(self) -> Benchmarker:
        if not hasattr(self, "_benchmarker"):
            self._benchmarker = Benchmarker(self.cfg.image_dataset)
        return self._benchmarker

    def test_step(
        self,
        batch: Batch,
        results: tuple[BytesIO, ...],
        metadata: TestStepMetadata,
    ) -> None:
        sample = self.generate(batch)

        # Skip past the context frames and decode the sample.
        sample = sample[:, self.cfg.first_denoised_frame :]
        sample = self.decoder.decode_video(
            rearrange(sample, "b f c h w -> b (f h w) c"),
            get_decoding_indices(sample, 0, self.cfg.first_denoised_frame),
            level=0,
        )

        # Measure the video's quality and save it to disk.
        self.benchmarker.benchmark(sample, tuple(batch["key"]), results, metadata)

    #############
    # Demo Code #
    #############

    def demo_step(
        self,
        batch: Batch,
    ) -> Generator[tuple[Float[Tensor, "batch rgb=3 height width"], int], None, None]:
        sample = self.generate(batch)
        return self.decoder.decode_video(
            rearrange(sample, "b f c h w -> b (f h w) c"),
            get_decoding_indices(sample, 0, 0),
            level=0,
        )
