import numpy as np
import torch
from cleanfid.features import build_feature_extractor
from cleanfid.resize import build_resizer
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from ..image_io import prep_image
from .metric import Metric


class MetricFID(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.resizer = build_resizer("clean")
        self.model = build_feature_extractor("clean", use_dataparallel=False)

    def embed(
        self,
        prediction: Float[Tensor, "batch frame rgb=3 height width"],
    ) -> dict[str, Float[Tensor, " batch #frame channel"]]:
        b, f, _, _, _ = prediction.shape
        device = prediction.device
        prediction = rearrange(prediction, "b f c h w -> (b f) c h w")

        batch = [self.resizer(np.array(prep_image(image))) for image in prediction]
        batch = np.stack(batch)
        batch = torch.tensor(batch, dtype=torch.float32, device=device)
        batch = rearrange(batch, "b h w c -> b c h w")
        with torch.autocast("cuda", enabled=False):
            embeddings = self.model(batch)

        return {"fid": rearrange(embeddings, "(b f) c -> b f c", b=b, f=f)}
