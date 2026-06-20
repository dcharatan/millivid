from io import BytesIO
from pathlib import Path
from typing import Iterable, NamedTuple

import av
import numpy as np
import torch
import torch.nn.functional as F
from av import VideoStream
from av.container.output import OutputContainer
from einops import rearrange, repeat
from jaxtyping import Float, Shaped
from PIL import Image
from torch import Tensor


def load_image(
    path: Path | str | BytesIO,
    device: torch.device = torch.device("cpu"),
) -> Float[Tensor, "channel height width"]:
    image = np.array(Image.open(path), dtype=np.float32) / 255
    image = rearrange(image, "h w c -> c h w")
    return torch.tensor(image, dtype=torch.float32, device=device)


type ImageTensor = (
    Shaped[Tensor | np.ndarray, "channel height width"]
    | Shaped[Tensor | np.ndarray, "height width"]
)


def prep_image(image: ImageTensor) -> Image.Image:
    if isinstance(image, Tensor):
        image = image.float().detach().cpu().numpy()
    image = image.astype(np.float32)
    image = np.clip(image, a_min=0, a_max=1)
    if image.ndim == 2:
        image = repeat(image, "h w -> h w c", c=3)
    else:
        image = rearrange(image, "c h w -> h w c")
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image)


def save_image(image: ImageTensor, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    prep_image(image).save(path)


class VideoEncoderBundle(NamedTuple):
    bytes_io: BytesIO
    container: OutputContainer
    stream: VideoStream


class VideoEncoder:
    fps: int
    bundles: list[VideoEncoderBundle]

    def __init__(self, fps: int = 20) -> None:
        self.fps = fps
        self.bundles = []

    def add_frames(self, frames: Float[Tensor, "batch rgb=3 height width"]) -> None:
        # Ensure that the height and width are divisible by 2.
        b, _, h, w = frames.shape
        frames = F.pad(frames, (0, w % 2, 0, h % 2, 0, 0, 0, 0), value=1)
        _, _, h, w = frames.shape

        # Lazily initialize the containers and streams.
        if not self.bundles:
            assert b > 0
            for _ in range(b):
                bytes_io = BytesIO()
                container = av.open(bytes_io, "w", "mp4")
                stream = container.add_stream("libx264", self.fps)
                stream.width = w
                stream.height = h
                stream.pix_fmt = "yuv420p"
                self.bundles.append(VideoEncoderBundle(bytes_io, container, stream))

        # Add the frames to their respective streams.
        frames = rearrange(frames.detach(), "b c h w -> b h w c")
        frames = (frames.clip(min=0, max=1) * 255).to(torch.uint8).cpu().numpy()
        for batch_index, frame in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in self.bundles[batch_index].stream.encode(frame):
                self.bundles[batch_index].container.mux(packet)

    def result(self) -> tuple[bytes, ...]:
        # Flush the streams.
        for bundle in self.bundles:
            for packet in bundle.stream.encode():
                bundle.container.mux(packet)
            bundle.container.close()

        return tuple(bundle.bytes_io.getvalue() for bundle in self.bundles)


def encode_videos(
    videos: Iterable[Float[Tensor, "batch rgb=3 height width"]],
    fps: int = 20,
) -> tuple[bytes, ...]:
    encoder = VideoEncoder(fps)
    for video in videos:
        encoder.add_frames(video)
    return encoder.result()
