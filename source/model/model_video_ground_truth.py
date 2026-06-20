import random
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Generator, Literal, TypedDict

import torch
from einops import rearrange, repeat
from jaxtyping import Float, Int
from torch import Tensor, nn

from ..components.action_encoder import ActionEncoder
from ..components.adaptivity import LevelsCfg
from ..dataset.dataset_images import DatasetImagesCfg
from ..dataset.dataset_latents_dense import DatasetLatentsDense, DatasetLatentsDenseCfg
from ..dataset.format_pyramid import load_pyramid
from ..evaluation.benchmarker import Benchmarker
from .model import ConfigurableModel, TestStepMetadata, TrainStepOutput, VisStepOutput
from .model_autoencoder import Decoder, load_decoder


class Batch(TypedDict):
    tokens: Float[Tensor, "batch token channel"]
    indices: Int[Tensor, "batch token flhw=4"]
    key: list[str]


@dataclass(frozen=True)
class ModelVideoGroundTruthCfg(LevelsCfg):
    name: Literal["video_ground_truth"]
    decoder_path: tuple[Path, ...]
    image_dataset: DatasetImagesCfg
    latent_dataset: DatasetLatentsDenseCfg
    test_num_frames: int
    first_denoised_frame: int
    mode: Literal[
        "ground_truth",
        "autoencoded",
        "random_ground_truth",
        "random_autoencoded",
    ]


class ModelVideoGroundTruth(
    ConfigurableModel[ModelVideoGroundTruthCfg, Batch, Batch, Batch]
):
    decoder: Decoder

    def __init__(self, cfg: ModelVideoGroundTruthCfg) -> None:
        super().__init__(cfg)
        self.decoder = load_decoder(cfg, cfg.decoder_path)
        self.dummy = nn.Parameter(torch.zeros((1,), dtype=torch.float32))

    ################
    # Unused Stuff #
    ################

    def train_step(self, batch: Batch) -> TrainStepOutput:
        raise NotImplementedError()

    def vis_step(self, batch: Batch) -> VisStepOutput:
        raise NotImplementedError()

    #######################
    # Ground-Truth Frames #
    #######################

    def generate_ground_truth(
        self,
        batch: Batch,
        start: int,
    ) -> Generator[tuple[Float[Tensor, "batch rgb=3 height width"], int], None, None]:
        frame_range = range(start, self.cfg.test_num_frames)
        for frame_index in frame_range:
            # We're technically loading identical frames twice (once here and once in
            # the benchmarker), but there's no point optimizing this.
            frames = [
                self.benchmarker.dataset.load_frames(key, frame_index, frame_index + 1)
                for key in batch["key"]
            ]
            yield torch.cat(frames), frame_index

    ######################
    # Autoencoded Frames #
    ######################

    def generate_autoencoded(
        self,
        batch: Batch,
        start: int,
    ) -> Generator[tuple[Float[Tensor, "batch rgb=3 height width"], int], None, None]:
        end = self.cfg.test_num_frames
        index_f, _, _, _ = batch["indices"].unbind(dim=-1)
        assert (index_f == index_f[0]).all()
        mask = (index_f[0] >= start) & (index_f[0] < end)

        return self.decoder.decode_video(
            batch["tokens"][:, mask],
            batch["indices"][:, mask],
            0,
        )

    ##############################
    # Random Ground-Truth Frames #
    ##############################

    def generate_random_ground_truth(
        self,
        batch: Batch,
        start: int,
    ) -> Generator[tuple[Float[Tensor, "batch rgb=3 height width"], int], None, None]:
        frame_range = range(start, self.cfg.test_num_frames)
        for frame_index in frame_range:
            frames = [
                self.benchmarker.dataset.load_frames(
                    random.choice(list(self.benchmarker.dataset.index)),  # random key
                    frame_index,
                    frame_index + 1,
                )
                for _ in batch["key"]
            ]
            yield torch.cat(frames), frame_index

    #############################
    # Random Autoencoded Frames #
    #############################

    @property
    def ld(self) -> DatasetLatentsDense:
        if not hasattr(self, "_latent_dataset"):
            self._latent_dataset = DatasetLatentsDense(self.cfg.latent_dataset, "all")
        return self._latent_dataset

    def make_indices(
        self,
        bhw: tuple[int, int, int],
        frame_index: int,
        device: torch.device,
    ) -> Int[Tensor, "batch token flhw=4"]:
        b, h, w = bhw
        index_h = torch.arange(h, dtype=torch.int32, device=device)
        index_h = repeat(index_h, "h -> b (h w)", b=b, w=w)
        index_w = torch.arange(w, dtype=torch.int32, device=device)
        index_w = repeat(index_w, "w -> b (h w)", b=b, h=h)
        index_f = torch.full_like(index_h, frame_index)
        index_l = torch.zeros_like(index_h)
        return torch.stack((index_f, index_l, index_h, index_w), dim=-1)

    def generate_random_autoencoded(
        self,
        batch: Batch,
        start: int,
    ) -> Generator[tuple[Float[Tensor, "batch rgb=3 height width"], int], None, None]:
        end = self.cfg.test_num_frames
        for frame_index in range(start, end):
            latents_batch = []
            for true_key in batch["key"]:
                key = true_key
                while key == true_key:
                    key = random.choice(list(self.benchmarker.dataset.index))
                (latents,), _ = load_pyramid(
                    (self.ld.path / key).with_suffix(".pyramid"),
                    self.cfg.available_num_levels,
                    self.ld.index[key],
                    (self.ld.cfg.latent_channels, *self.ld.cfg.latent_shape),
                    (ActionEncoder.num_action_channels(),),
                    0,
                    1,
                    frame_index,
                    frame_index + 1,
                    batch["tokens"].device,
                )
                latents_batch.append(latents)
            latents_batch = torch.cat(latents_batch)
            b, _, h, w = latents_batch.shape
            yield from self.decoder.decode_video(
                rearrange(latents_batch, "b c h w -> b (h w) c"),
                self.make_indices((b, h, w), frame_index, latents_batch.device),
                0,
            )

    ############################
    # Evaluation-Related Stuff #
    ############################

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
        self.benchmarker.benchmark(
            self.demo_step(batch, start=self.cfg.first_denoised_frame),
            tuple(batch["key"]),
            results,
            replace(metadata, run_id=self.cfg.mode),
        )

    #############
    # Demo Code #
    #############

    def demo_step(
        self,
        batch: Batch,
        start: int = 0,
    ) -> Generator[tuple[Float[Tensor, "batch rgb=3 height width"], int], None, None]:
        return {
            "ground_truth": self.generate_ground_truth,
            "autoencoded": self.generate_autoencoded,
            "random_ground_truth": self.generate_random_ground_truth,
            "random_autoencoded": self.generate_random_autoencoded,
        }[self.cfg.mode](batch, start=start)
