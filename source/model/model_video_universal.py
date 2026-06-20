import logging
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Generator, Literal, NamedTuple, TypedDict

import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from ..components.adaptivity import DiTAdaptive, LevelsCfg, scatter_by_index_
from ..components.dit_per_token_c import Unpatchify
from ..components.noise_schedule import NoiseSchedule
from ..dataset.dataset_images import DatasetImagesCfg
from ..evaluation.benchmarker import Benchmarker
from ..image_io import encode_videos
from ..layout import add_label, hcat
from ..strategy import DENOISED, UNSEEN, Step, Strategy, StrategyCfg, get_strategy
from .model import ConfigurableModel, TestStepMetadata, TrainStepOutput, VisStepOutput
from .model_autoencoder import load_decoder


class Metadata(TypedDict):
    indices: Int[Tensor, "batch token flhw=4"]
    actions: Float[Tensor, "batch token action_channel"]
    roles: Int[Tensor, "batch token"]
    key: list[str]
    mask: Bool[Tensor, " batch"] | None = None


class Batch(Metadata):
    tokens: Float[Tensor, "batch token channel"]


class Predictions(NamedTuple):
    x: Float[Tensor, "batch token channel"]
    eps: Float[Tensor, "batch token channel"]


class Selection(NamedTuple):
    metadata: Metadata
    values: Float[Tensor, "batch token *shape"]


@dataclass(frozen=True)
class ModelVideoUniversalCfg(LevelsCfg):
    name: Literal["video_universal"]
    model_size: Literal["S", "B", "L", "XL"]
    decoder_path: tuple[Path, ...]
    image_dataset: DatasetImagesCfg
    sampling_steps: int
    shift: float
    strategy: StrategyCfg
    test_num_frames: int
    first_denoised_frame: int
    guidance_dropout_probability: float
    guidance_scales: tuple[float, ...]


##############
# Main Model #
##############


class DiTWrapper(nn.Module):
    def __init__(self, cfg: ModelVideoUniversalCfg, strategy: Strategy) -> None:
        super().__init__()
        self.dit = DiTAdaptive(cfg.available_num_levels, cfg.model_size)
        self.patchify = nn.Linear(cfg.latent_channels, self.dit.hidden_channels)
        self.unpatchify = Unpatchify(self.dit.hidden_channels, cfg.latent_channels)
        self.strategy = strategy

    def forward(
        self,
        x: Float[Tensor, "batch token channel"],
        t: Float[Tensor, "batch token"],
        metadata: Metadata,
        conditioning_mask: Bool[Tensor, " batch"],
    ) -> Float[Tensor, "batch token channel"]:
        x = self.patchify(x)
        x, c = self.dit(
            x,
            metadata["indices"],
            t,
            metadata["actions"],
            action_mask=conditioning_mask,
        )
        return self.unpatchify(x, c)


class ModelVideoUniversal(
    ConfigurableModel[ModelVideoUniversalCfg, Batch, Batch, Batch]
):
    def __init__(self, cfg: ModelVideoUniversalCfg) -> None:
        super().__init__(cfg)
        self.strategy = get_strategy(cfg.strategy, cfg.base_num_tokens)
        self.dit = DiTWrapper(cfg, self.strategy)
        self.noise_schedule = NoiseSchedule(shift=cfg.shift)
        self.decoder = load_decoder(cfg, cfg.decoder_path)

    ############
    # Training #
    ############

    @torch.compile(
        fullgraph=True,
        dynamic=False,
        mode="reduce-overhead",
        disable=not os.getenv("USE_TORCH_COMPILE", False),
    )
    def train_step(self, batch: Batch) -> TrainStepOutput:
        x = batch["tokens"]
        eps = torch.randn_like(x)

        # Get the noise levels used for training.
        b, _, _ = batch["indices"].shape
        u = torch.rand((b,), dtype=torch.float32, device=self.device)
        conditioning_mask = u > self.cfg.guidance_dropout_probability
        t = torch.where(
            batch["roles"] == DENOISED,
            torch.rand((b, 1), dtype=torch.float32, device=self.device),
            (~conditioning_mask[:, None]).float(),  # 0 is pure data; 1 is pure noise
        )

        # Define the model's input.
        alpha_t, sigma_t = self.noise_schedule.coefficients(t)
        alpha_t = alpha_t[:, :, None]
        sigma_t = sigma_t[:, :, None]
        x_t = alpha_t * x + sigma_t * eps

        # Get the model's prediction of V.
        v_hat: Tensor = self.dit(x_t, t, batch, conditioning_mask)

        # Compute a loss on the predicted V.
        v = alpha_t * eps - sigma_t * x
        loss = F.mse_loss(v_hat, v, reduction="none")

        # Only compute a loss on the denoised tokens.
        loss = torch.where(
            (batch["roles"] == DENOISED)[:, :, None],
            loss,
            torch.nan,
        )
        return loss.nanmean()

    ############
    # Sampling #
    ############

    def select_tokens(
        self,
        metadata: Metadata,
        values: Float[Tensor, "batch token channel"],
        step: Step,
        window_start: int,
    ) -> Selection:
        # Confirm that the indices match across the batch.
        assert (metadata["indices"] == metadata["indices"][:1]).all()
        assert window_start >= 0

        selection = torch.tensor(step.selection, device=values.device)
        _, selection_num_frames = selection.shape
        out_of_bounds = torch.full_like(selection[:, :1], UNSEEN)
        selection = torch.cat((selection, out_of_bounds), dim=-1)

        # Make out-of-bounds frame indices map to frame index -1, whose value is UNSEEN.
        index_f, index_l, _, _ = metadata["indices"][0].unbind(dim=-1)
        index_f = index_f - window_start
        index_f[index_f < 0] = -1
        index_f[index_f >= selection_num_frames] = -1
        assert index_f.max() == selection_num_frames - 1

        # Create a mask for tokens that are seen during this step.
        roles = selection[index_l, index_f]
        mask = roles != UNSEEN

        # Select the values and metadata, then pad them to the correct sequence length.
        values = values[:, mask]
        b, s, _ = values.shape
        padding = max(self.strategy.max_num_tokens - s, 0)
        assert padding == 0 or self.strategy.requires_attention_masking
        values = F.pad(values, (0, 0, 0, padding))
        action_padding = (*([0] * (metadata["actions"].ndim * 2 - 4)), 0, padding)
        metadata = {
            "indices": F.pad(metadata["indices"][:, mask], (0, 0, 0, padding)),
            "actions": F.pad(metadata["actions"][:, mask], action_padding),
            "roles": F.pad(repeat(roles[mask], "s -> b s", b=b), (0, padding)),
        }
        return Selection(metadata, values)

    def get_sampling_noise_levels(
        self,
        t: float,
        metadata: Metadata,
    ) -> Float[Tensor, "batch token"]:
        return torch.where(metadata["roles"] == DENOISED, t, 0)

    def denoise(
        self,
        predictions: Predictions,
        metadata: Metadata,
        t: Float[Tensor, "batch token"],
        guidance_scale: float = 1.0,
    ) -> Predictions:
        # Define the input.
        alpha_t, sigma_t = self.noise_schedule.coefficients(t)
        alpha_t = alpha_t[:, :, None]
        sigma_t = sigma_t[:, :, None]
        x_t = alpha_t * predictions.x + sigma_t * predictions.eps
        b, _, _ = x_t.shape

        # Get the model's prediction of V.
        v_hat: Tensor = self.dit.forward(
            x_t,
            t,
            metadata,
            torch.ones((b,), device=x_t.device, dtype=bool),
        )

        # Add guidance.
        if guidance_scale != 1.0:
            t_uc = torch.where(metadata["roles"] == DENOISED, t, 1)
            alpha_t_uc, sigma_t_uc = self.noise_schedule.coefficients(t_uc)
            alpha_t_uc = alpha_t_uc[:, :, None]
            sigma_t_uc = sigma_t_uc[:, :, None]
            x_t_uc = alpha_t_uc * predictions.x + sigma_t_uc * predictions.eps
            v_hat_uc: Tensor = self.dit(
                x_t_uc,
                t_uc,
                metadata,
                torch.zeros((b,), dtype=bool, device=self.device),
            )
            v_hat = v_hat_uc + guidance_scale * (v_hat - v_hat_uc)

        # Convert the model's prediction of V to predictions of signal and noise.
        x_hat = alpha_t * x_t - sigma_t * v_hat
        eps_hat = alpha_t * v_hat + sigma_t * x_t

        # Only update the denoised tokens (to avoid drift in context).
        mask = (metadata["roles"] == DENOISED)[:, :, None]
        return Predictions(
            torch.where(mask, x_hat.type(predictions.x.dtype), predictions.x),
            torch.where(mask, eps_hat.type(predictions.eps.dtype), predictions.eps),
        )

    def generate(
        self,
        batch: Batch,
        seed: int = 0,
        guidance_scale: float = 1.0,
    ) -> Float[Tensor, "batch out_token channel"]:
        generator = torch.Generator(batch["tokens"].device).manual_seed(seed)

        # Create the initial sample by zeroing out everything that's not context.
        x = batch["tokens"].clone()
        index_f, _, _, _ = batch["indices"].unbind(dim=-1)
        start = self.cfg.first_denoised_frame
        x[index_f >= start] = 0

        sequence, _, _ = self.strategy.generate(start, end=self.cfg.test_num_frames)
        for step, chunk_start in sequence:
            step_metadata, step_x = self.select_tokens(
                batch,
                x,
                step,
                chunk_start,
            )
            step_eps = torch.empty_like(step_x).normal_(generator=generator)
            predictions = Predictions(step_x, step_eps)
            for t in torch.linspace(1, 0, self.cfg.sampling_steps):
                predictions = self.denoise(
                    predictions,
                    step_metadata,
                    self.get_sampling_noise_levels(t.item(), step_metadata),
                    guidance_scale,
                )
            scatter_by_index_(
                predictions.x,
                step_metadata["indices"],
                x,
                batch["indices"],
                source_mask=step_metadata["roles"] == DENOISED,
            )

        return x

    ##############
    # Evaluation #
    ##############

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
        # Generate a video.
        sample = self.generate(batch)

        # Create a mask that only selects generated frames.
        start = self.cfg.first_denoised_frame
        index_f, _, _, _ = batch["indices"].unbind(dim=-1)
        assert (index_f == index_f[0]).all()
        mask = ((index_f >= start) & (index_f < self.cfg.test_num_frames))[0]

        # Save the generated latents.
        output_path = self.benchmarker.get_output_path(metadata)
        for key, tokens, indices in zip(batch["key"], sample, batch["indices"]):
            latent_path = output_path / f"{key}_latents.npz"
            latent_path.parent.mkdir(exist_ok=True, parents=True)
            np.savez(
                latent_path,
                tokens=tokens.view(torch.int16).detach().cpu().numpy(),
                indices=indices.detach().cpu().numpy(),
            )

        # Measure the video's quality and save it to disk.
        self.benchmarker.benchmark(
            self.decoder.decode_video(sample[:, mask], batch["indices"][:, mask], 0),
            tuple(batch["key"]),
            results,
            metadata,
        )

    #################
    # Visualization #
    #################

    def vis_step(self, batch: Batch) -> VisStepOutput:
        images, videos, metrics = {}, {}, {}
        for gs in self.cfg.guidance_scales:
            videos[f"sample_{gs}"] = self.vis_generation(batch, gs)
        return VisStepOutput(images=images, metrics=metrics, videos=videos)

    def vis_generation(self, batch: Batch, guidance_scale: float) -> tuple[bytes, ...]:
        logging.info("Visualizing generation.")

        sample = self.generate(batch, guidance_scale=guidance_scale)

        start = self.cfg.first_denoised_frame
        index_f, _, _, _ = batch["indices"].unbind(dim=-1)
        assert (index_f == index_f[0]).all()
        mask = (index_f[0] >= start - 1) & (index_f[0] < self.cfg.test_num_frames)

        def vis_frame_generator():
            # Set up generators for decoding the sample and ground truth.
            sample_generators = [
                self.decoder.decode_video(
                    sample[:, mask],
                    batch["indices"][:, mask],
                    level,
                )
                for level in range(self.strategy.num_levels)
            ]
            gt_generator = self.decoder.decode_video(
                batch["tokens"][:, mask],
                batch["indices"][:, mask],
                0,
            )

            # Merge the decoded outputs into visualization frames, then yield them.
            for gt, _ in gt_generator:
                kwargs = dict(font_size=16, align="center")
                gt = add_label(gt, "Ground Truth", **kwargs)
                sample_levels = [
                    add_label(next(gen)[0], f"Sample (Level {level})", **kwargs)
                    for level, gen in enumerate(sample_generators)
                ]
                yield hcat((gt, *sample_levels), gap=16, border=16)

        return encode_videos(vis_frame_generator())

    #############
    # Demo Code #
    #############

    def demo_step(
        self,
        batch: Batch,
    ) -> Generator[tuple[Float[Tensor, "batch rgb=3 height width"], int], None, None]:
        sample = self.generate(batch)
        return self.decoder.decode_video(sample, batch["indices"], 0)
