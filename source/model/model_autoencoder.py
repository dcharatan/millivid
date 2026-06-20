import logging
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Generator, Literal, TypedDict

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from jaxtyping import Float, Int
from lpips import LPIPS
from torch import Tensor, nn

from ..components.adaptivity import LevelsCfg, scatter_by_index_
from ..components.dit_global_c import DiT, DiT_B
from ..components.transformer import Transformer, Transformer_B
from ..dataset.format_pyramid import save_pyramid
from ..ddp import rank_zero_first
from ..image_io import encode_videos
from ..layout import add_label, hcat, vcat
from ..metric.metric_psnr import compute_psnr
from ..optimizer import convert_to_buffer
from .model import (
    ConfigurableModel,
    Metrics,
    TestStepMetadata,
    TrainStepOutput,
    VisStepOutput,
)


class Batch(TypedDict):
    frames: Float[Tensor, "batch frame channel height width"]
    key: list[str]


@dataclass(frozen=True)
class ModelAutoencoderCfg(LevelsCfg):
    name: Literal["autoencoder"]
    lpips_weight: float
    encode_in_path: Path
    encode_out_path: Path
    variant: Literal["adaptive", "cascaded"]


##################
# Patchification #
##################


def patchify(
    images: Float[Tensor, "batch channel height width"],
    cfg: ModelAutoencoderCfg,
) -> tuple[
    Float[Tensor, "batch token channel patch_height patch_width"],
    Int[Tensor, "batch token lhw=3"],
]:
    b, _, _, _ = images.shape
    ph, pw = cfg.patch_shape
    lh, lw = cfg.latent_shape
    patches = []
    indices = []

    for level in range(cfg.available_num_levels):
        # Go to the next level.
        if level > 0:
            images = reduce(images, "b c (h fh) (w fw) -> b c h w", "mean", fh=2, fw=2)
            lh //= 2
            lw //= 2

        # Make the (frame, height, width) indices for the current level.
        index_h = torch.arange(lh, device=images.device)
        index_h = repeat(index_h, "lh -> lh lw", lw=lw)
        index_w = torch.arange(lw, device=images.device)
        index_w = repeat(index_w, "lw -> lh lw", lh=lh)
        index_l = torch.full_like(index_h, level)
        level_indices = torch.stack((index_l, index_h, index_w), dim=-1)

        # Identically rearrange the patches and indices, then accumulate them.
        level_patches = rearrange(
            images,
            "b c (h ph) (w pw) -> b (h w) c ph pw",
            ph=ph,
            pw=pw,
        )
        level_indices = repeat(level_indices, "h w lhw -> b (h w) lhw", b=b)
        patches.append(level_patches)
        indices.append(level_indices)

    return torch.cat(patches, dim=1), torch.cat(indices, dim=1)


def unpatchify(
    tokens: Float[Tensor, "batch token channel patch_height patch_width"],
    cfg: ModelAutoencoderCfg,
    limit: int | None = None,
) -> tuple[Float[Tensor, "batch channel height=_ width=_"], ...]:
    # This assumes that the structure is the same as patchify's outputs.
    images = []
    lh, lw = cfg.latent_shape
    for _ in range(limit if limit is not None else cfg.available_num_levels):
        # Extract the current level's image.
        level_num_tokens = lh * lw
        level_image = rearrange(
            tokens[:, :level_num_tokens],
            "b (lh lw) c ph pw -> b c (lh ph) (lw pw)",
            lh=lh,
            lw=lw,
        )
        images.append(level_image)

        # Go to the next level.
        tokens = tokens[:, level_num_tokens:]
        lh //= 2
        lw //= 2
    return tuple(images)


######################
# Position Embedding #
######################


def posemb_sincos_2d(
    ij: Int[Tensor, "*shape ij=2"],
    dim: int,
    temperature: int = 10000,
    dtype: torch.dtype = torch.float32,
) -> Float[Tensor, "*shape dim"]:
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4, device=ij.device) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)
    y, x = ij.unbind(dim=-1)
    y = y[..., None] * omega
    x = x[..., None] * omega
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=-1)
    return pe.type(dtype)


###########
# Encoder #
###########


class Encoder(nn.Module):
    cfg: ModelAutoencoderCfg
    level_embedding: nn.Embedding
    transformer: Transformer
    patchify: nn.Linear
    bottleneck: nn.Linear

    def __init__(self, cfg: ModelAutoencoderCfg) -> None:
        super().__init__()
        ph, pw = cfg.patch_shape
        self.cfg = cfg
        self.transformer = Transformer_B()
        self.level_embedding = nn.Embedding(
            cfg.available_num_levels,
            self.hidden_channels,
        )
        self.patchify = nn.Linear(3 * ph * pw, self.hidden_channels)
        self.bottleneck = nn.Linear(self.hidden_channels, cfg.latent_channels)

    def forward(
        self,
        x: Float[Tensor, "batch token rgb=3 patch_height patch_width"],
        indices: Int[Tensor, "batch token lhw=3"],
    ) -> Float[Tensor, "batch token channel"]:
        x = rearrange(x, "b s c ph pw -> b s (c ph pw)")
        x = self.patchify(x)
        x = x + posemb_sincos_2d(indices[..., 1:], self.hidden_channels)
        x = x + self.level_embedding(indices[:, :, 0])
        x = self.transformer(x)
        x = torch.tanh(self.bottleneck(x))

        if self.cfg.variant == "cascaded":
            # Ablation: Instead of using adaptive latents, just mean-pool the finest
            # latent to create coarser latents.
            (current,) = unpatchify(x[:, :, :, None, None], self.cfg, 1)
            levels = [rearrange(current, "b c h w -> b (h w) c")]
            for _ in range(1, self.cfg.available_num_levels):
                current = reduce(
                    current,
                    "b c (h ph) (w pw) -> b c h w",
                    "mean",
                    ph=2,
                    pw=2,
                )

                # We detach the lower levels, since otherwise the encode and decoder can
                # conspire to create adaptive latents through pooling. We effectively
                # want a full-resolution encoder-decoder gradients, plus decoder-only
                # gradients for the coarser levels.
                levels.append(rearrange(current, "b c h w -> b (h w) c").detach())
            x = torch.cat(levels, dim=1)

        return x

    @property
    def hidden_channels(self) -> int:
        return self.transformer.hidden_channels


###########
# Decoder #
###########


class Decoder(nn.Module):
    cfg: ModelAutoencoderCfg
    level_embedding: nn.Embedding
    transformer: DiT
    unpatchify: nn.Linear
    unbottleneck: nn.Linear

    def __init__(self, cfg: ModelAutoencoderCfg) -> None:
        super().__init__()
        ph, pw = cfg.patch_shape
        self.cfg = cfg
        self.transformer = DiT_B()
        self.level_embedding = nn.Embedding(
            cfg.available_num_levels,
            self.hidden_channels,
        )
        self.unpatchify = nn.Linear(self.hidden_channels, 3 * ph * pw)
        self.unbottleneck = nn.Linear(cfg.latent_channels, self.hidden_channels)

    def forward(
        self,
        x: Float[Tensor, "batch token channel"],
        indices: Int[Tensor, "batch token lhw=3"],
        level: Int[Tensor, " batch"],
    ) -> Float[Tensor, "batch token rgb=3 patch_height patch_width"]:
        x = self.unbottleneck(x)
        x = x + posemb_sincos_2d(indices[..., 1:], self.hidden_channels)
        x = x + self.level_embedding(indices[:, :, 0])
        x = self.transformer(x, self.level_embedding(level))
        x = self.unpatchify(x)
        ph, pw = self.cfg.patch_shape
        return rearrange(x, "b s (c ph pw) -> b s c ph pw", ph=ph, pw=pw)

    def decode_video(
        self,
        x: Float[Tensor, "batch token channel"],
        indices: Int[Tensor, "batch token flhw=4"],
        level: int,
    ) -> Generator[tuple[Float[Tensor, "batch rgb=3 height width"], int], None, None]:
        logging.info(f"Decoding video at level {level}")
        device = x.device

        # Define a single frame's (l, h, w) indices.
        b, _, c = x.shape
        h, w = self.cfg.image_shape
        dummy = torch.zeros((b, 3, h, w), device=device, dtype=torch.bfloat16)
        _, frame_indices_lhw = patchify(dummy, self.cfg)
        _, s, _ = frame_indices_lhw.shape

        # Decode one frame at a time.
        index_f, index_l, _, _ = indices.unbind(dim=-1)
        for frame in range(index_f.min().item(), index_f.max().item() + 1):
            # Prepare this frame's indices and empty tokens for this frame.
            frame_index_f = torch.full_like(frame_indices_lhw[:, :, :1], frame)
            frame_indices = torch.cat((frame_index_f, frame_indices_lhw), dim=-1)
            frame_x = torch.zeros((b, s, c), dtype=x.dtype, device=device)

            # Scatter the relevant tokens into the frame's token array.
            scatter_by_index_(x, indices, frame_x, frame_indices, index_l == level)

            # Feed the tokens through the model.
            frame_level = torch.full((b,), level, dtype=torch.int32, device=device)
            frame_x = self(frame_x, frame_indices[:, :, 1:], frame_level)

            # Unpatchify the tokens. Since the tokens have been ordered according to
            # the patchify function's output, we can directly unpatchify here.
            yield unpatchify(frame_x, self.cfg, 1)[0].cpu(), frame

    @property
    def hidden_channels(self) -> int:
        return self.transformer.hidden_channels


def load_decoder(cfg: LevelsCfg, paths: tuple[Path, ...]) -> Decoder:
    # Create the decoder.
    decoder_cfg = ModelAutoencoderCfg(
        cfg.available_num_levels,
        cfg.image_shape,
        cfg.patch_shape,
        cfg.latent_channels,
        "autoencoder",
        0.0,
        Path(),
        Path(),
        "adaptive",
    )
    decoder = Decoder(decoder_cfg)

    # Load and freeze the decoder's weights.
    for path in paths:
        try:
            state_dict = torch.load(path, map_location="cpu")
            break
        except FileNotFoundError:
            continue
    else:
        raise FileNotFoundError("Could not find a valid decoder checkpoint path!")
    state_dict = {
        k[len("decoder.") :]: v
        for k, v in state_dict["model"].items()
        if k.startswith("decoder.")
    }
    incompatible = decoder.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        logging.warning(f"missing: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        logging.warning(f"unexpected: {incompatible.unexpected_keys}")
    convert_to_buffer(decoder, False)
    return decoder


###############
# Autoencoder #
###############


class ModelAutoencoder(ConfigurableModel[ModelAutoencoderCfg, Batch, Batch, Batch]):
    encoder: Encoder

    def __init__(self, cfg: ModelAutoencoderCfg) -> None:
        super().__init__(cfg)
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        with rank_zero_first():
            self.lpips = LPIPS(net="vgg")
        convert_to_buffer(self.lpips, False)

    ############
    # Training #
    ############

    def train_step(self, batch: Batch) -> TrainStepOutput:
        _, _, _, h, w = batch["frames"].shape
        assert (h, w) == self.cfg.image_shape
        (images,) = batch["frames"].unbind(dim=1)

        # Encode the image into multi-level latents.
        patches, indices = patchify(images, self.cfg)
        latents: Tensor = self.encoder(patches, indices)

        # Pick a random level to keep.
        b, _, _ = indices.shape
        level = torch.randint(
            0,
            self.cfg.available_num_levels,
            (b,),
            device=latents.device,
        )
        latents = torch.where(indices[..., :1] == level[:, None, None], latents, 0)

        # Decode using only the kept level.
        patches_hat = self.decoder(latents, indices, level)

        # Compute an MSE loss on all levels.
        mse = F.mse_loss(patches, patches_hat, reduction="none")
        mse = mse * 4 ** indices[:, :, 0, None, None, None]
        mse = mse.mean()

        # Compute LPIPS on the highest-resolution level.
        images_hat = unpatchify(patches_hat, self.cfg, limit=1)[0]
        lpips: Tensor = self.lpips(images, images_hat, normalize=True).mean()
        lpips = lpips * self.cfg.lpips_weight
        return mse + lpips, {"loss_mse": mse.detach(), "loss_lpips": lpips.detach()}

    #################
    # Visualization #
    #################

    def vis_grid(
        self,
        image: Float[Tensor, "batch rgb=3 height width"],
    ) -> tuple[Float[Tensor, "batch rgb=3 vis_height vis_width"], Metrics]:
        patches, indices = patchify(image, self.cfg)
        latents: Tensor = self.encoder(patches, indices)

        # Decode at every level.
        h, w = self.cfg.image_shape
        vis = []
        metrics = {}
        for level in range(self.cfg.available_num_levels):
            level_latents = torch.where(indices[..., :1] == level, latents, 0)
            b, _, _ = level_latents.shape
            patches_hat = self.decoder(
                level_latents,
                indices,
                torch.full((b,), level, dtype=torch.int32, device=latents.device),
            )
            images_hat = unpatchify(patches_hat, self.cfg)

            # Compute metrics on the highest-resolution level.
            metrics[f"level_{level}_psnr"] = compute_psnr(image, images_hat[0])
            metrics[f"level_{level}_lpips"] = self.lpips(
                image,
                images_hat[0],
                normalize=True,
            )

            # Visualize this level's decoded images (various resolutions) side-by-side.
            level_vis = [
                repeat(
                    images,
                    "b c h w -> b c (h ph) (w pw)",
                    ph=h // images.shape[-2],
                    pw=w // images.shape[-1],
                )
                for images in images_hat
            ]
            vis.append(add_label(hcat(level_vis), f"Level {level}"))

        vis = vcat(vis)
        vis = hcat((add_label(image, "G.T."), vis), border=8)
        return vis, metrics

    def vis_video(self, batch: Batch) -> tuple[bytes, ...]:
        def vis_frame_generator():
            b, _, _, _, _ = batch["frames"].shape
            for frame in batch["frames"].unbind(dim=1):
                patches, indices = patchify(frame, self.cfg)
                latents: Tensor = self.encoder(patches, indices)
                hat = self.decoder(
                    torch.where(indices[..., :1] == 0, latents, 0),
                    indices,
                    torch.zeros((b,), dtype=torch.int32, device=latents.device),
                )
                hat = unpatchify(hat, self.cfg)[0].cpu()
                hat = add_label(hat, "Autoencoded")
                gt = add_label(frame, "Ground Truth").cpu()
                yield hcat((gt, hat), border=8)

        return encode_videos(vis_frame_generator())

    def vis_step(self, batch: Batch) -> VisStepOutput:
        image, metrics = self.vis_grid(batch["frames"][:, 0])
        video = self.vis_video(batch)
        return VisStepOutput(
            images={"grid": image},
            metrics=metrics,
            videos={"reconstruction": video},
        )

    ###########
    # Testing #
    ###########

    def test_step(
        self,
        batch: Batch,
        results: tuple[BytesIO, ...],
        metadata: TestStepMetadata,
    ) -> None:
        """Encode images and save their latents to disk."""

        # Since entire videos may have different lengths, we only support batch size 1.
        (frames,) = batch["frames"]
        (key,) = batch["key"]

        # Encode the image into multi-level latents.
        latents = defaultdict(list)
        for frames_chunk in frames.split(1024, dim=0):
            patches, indices = patchify(frames_chunk, self.cfg)
            latents_chunk: Tensor = self.encoder(patches, indices)
            latents_chunk = unpatchify(latents_chunk[..., None, None], self.cfg)
            for level_index, level in enumerate(latents_chunk):
                latents[level_index].append(level)
        latents = [torch.cat(latents[level], dim=0) for _, level in enumerate(latents)]
        latents = tuple(latents)

        # Load the actions.
        actions = np.load((self.cfg.encode_in_path / key).with_suffix(".npy"))
        actions = torch.tensor(actions)

        tag = f"{metadata.run_id}_{metadata.step}"
        adaptive_path = self.cfg.encode_out_path / f"{tag}_adaptive/{key}.pyramid"
        save_pyramid(adaptive_path, latents, actions)

        # Save cascaded latents, which are just level 0 latents but downscaled.
        cascades = [
            reduce(
                latents[0].float(),
                "f c (h ph) (w pw) -> f c h w",
                "mean",
                ph=2**i,
                pw=2**i,
            ).bfloat16()
            for i, _ in enumerate(latents)
        ]
        cascaded_path = self.cfg.encode_out_path / f"{tag}_cascaded/{key}.pyramid"
        save_pyramid(cascaded_path, tuple(cascades), actions)
