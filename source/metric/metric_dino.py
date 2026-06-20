import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from .metric import Metric


class MetricDINO(Metric):
    def __init__(self, model_name: str = "dinov2_vitl14") -> None:
        """Cosine-similarity metric in DINOv2 feature space."""
        super().__init__()

        # Load DINOv2 from the official Facebook Research repo via torch.hub.
        # Available models: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14.
        self.dino: torch.nn.Module = torch.hub.load(
            "facebookresearch/dinov2", model_name
        ).eval()

        # ImageNet normalization stats (DINOv2 expects ImageNet-normalized inputs).
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def encode(
        self,
        video: Float[Tensor, "batch frame rgb=3 height width"],
    ) -> Float[Tensor, "batch frame dim"]:
        """Encode each frame independently with DINOv2 (CLS token)."""
        b, f, _, h, w = video.shape
        assert h % 14 == 0 and w % 14 == 0, "DINOv2 ViT patch size is 14."

        frames = rearrange(video, "b f c h w -> (b f) c h w")
        frames = (frames - self.mean) / self.std
        features = self.dino(frames)  # CLS token: (b * f, dim)
        return rearrange(features, "(b f) c -> b f c", b=b, f=f)

    def compute(
        self,
        prediction: Float[Tensor, "batch frame rgb=3 height width"],
        ground_truth: Float[Tensor, "batch frame rgb=3 height width"],
    ) -> dict[str, Float[Tensor, "batch frame"]]:
        """Per-frame cosine similarity in DINOv2 feature space."""
        _, _, _, h, w = prediction.shape
        assert (h, w) == (256, 256)

        # Crop to a multiple of 14 (DINOv2 patch size). 256 -> 252 (= 18 * 14).
        prediction = prediction[:, :, :, 2:-2, 2:-2]
        ground_truth = ground_truth[:, :, :, 2:-2, 2:-2]

        prediction_features = self.encode(prediction)
        ground_truth_features = self.encode(ground_truth)
        similarity = F.cosine_similarity(
            prediction_features, ground_truth_features, dim=-1
        )

        return {"dinov2": similarity}
