from collections import defaultdict
from io import BytesIO
from itertools import batched
from pathlib import Path
from pickle import dump
from typing import Generator, Iterable

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from PIL import Image
from torch import Tensor, nn
from tqdm import tqdm

from ..dataset.dataset_images import DatasetImages, DatasetImagesCfg
from ..metric.metric import Metric
from ..metric.metric_dino import MetricDINO
from ..metric.metric_fid import MetricFID
from ..metric.metric_fvd import MetricFVD
from ..metric.metric_keypoints import MetricKeypoints
from ..metric.metric_lpips import MetricLPIPS
from ..metric.metric_psnr import MetricPSNR
from ..metric.metric_ssim import MetricSSIM
from ..model.model import TestStepMetadata


class Benchmarker(nn.Module):
    def __init__(self, cfg: DatasetImagesCfg):
        super().__init__()
        self.dataset = DatasetImages(cfg, split="test")
        self._metrics = nn.ModuleList(
            [
                MetricDINO(),
                MetricFID(),
                MetricFVD(),
                MetricKeypoints(),
                MetricLPIPS(),
                MetricPSNR(),
                MetricSSIM(),
            ],
        )
        self.eval()
        self.to("cuda")

    @property
    def metrics(self) -> Iterable[Metric]:
        return self._metrics

    @torch.no_grad()
    def benchmark(
        self,
        video: Generator[
            tuple[Float[Tensor, "batch rgb=3 height width"], int], None, None
        ],
        keys: tuple[str, ...],
        results: tuple[BytesIO, ...],
        metadata: TestStepMetadata,
        chunk_size: int = 16,  # matches FVD
    ) -> None:
        # Store a per-key dictionary of output data.
        data = defaultdict(dict)  # key -> tag -> whatever
        metrics = defaultdict(lambda: defaultdict(list))  # key -> tag -> values

        for chunk in tqdm(batched(video, chunk_size), desc="Evaluating"):
            # Gather the generated frames.
            frames = torch.stack([frame for frame, _ in chunk], dim=1).cuda().float()
            frames = frames.clip(min=0, max=1)
            frame_idxs = tuple(index for _, index in chunk)
            assert (np.diff(frame_idxs) == 1).all()

            # Load the ground-truth frames.
            try:
                frames_gt = [
                    self.dataset.load_frames(key, frame_idxs[0], frame_idxs[-1] + 1)
                    for key in keys
                ]
                frames_gt = torch.stack(frames_gt).cuda().float()
            except ValueError:
                frames_gt = None

            # Compute metrics and embeddings.
            chunk_metrics = {}
            embeddings = {}
            with torch.autocast(device_type="cuda", enabled=False):
                for metric in self.metrics:
                    if frames_gt is not None:
                        chunk_metrics.update(metric.compute(frames, frames_gt))
                    embeddings.update(metric.embed(frames))

            # Collect the results.
            frames = rearrange(frames, "b f c h w -> b f h w c")
            frames = (frames * 255).type(torch.uint8).cpu()
            for batch_index, key in enumerate(keys):
                # Save the generated frames.
                for frame_index, frame in zip(frame_idxs, frames[batch_index]):
                    bytes_io = BytesIO()
                    frame = Image.fromarray(frame.numpy()).save(bytes_io, format="PNG")
                    frame = np.frombuffer(bytes_io.getvalue(), dtype=np.uint8)
                    data[key][f"frame_{frame_index}"] = frame
                    metrics[key]["frame_index"].append(frame_index)

                # Save the metrics.
                for tag, metric in chunk_metrics.items():
                    metrics[key][tag].extend(metric[batch_index].tolist())

                # Save the embeddings.
                for tag, embedding in embeddings.items():
                    for frame_index, emb in zip(frame_idxs, embedding[batch_index]):
                        data[key][f"{tag}_{frame_index}"] = emb.cpu().numpy()

        # Save the output data.
        output_path = self.get_output_path(metadata)
        for key, key_result in zip(keys, results):
            key_data = data[key]
            key_metrics = dict(metrics[key])

            # Save everything to disk in a single NumPy archive.
            data_path = output_path / f"{key}.npz"
            data_path.parent.mkdir(exist_ok=True, parents=True)
            np.savez(data_path, **{**key_data, "metrics": key_metrics})

            # Save the metrics to the database for quick access.
            dump(key_metrics, key_result)

    def get_output_path(self, metadata: TestStepMetadata) -> Path:
        name = f"eval_{metadata.run_id if metadata.step is None else metadata.step}"
        if metadata.tag is not None:
            name = f"{name}_{metadata.tag}"
        return metadata.workspace / name
