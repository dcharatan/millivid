from einops import rearrange
from jaxtyping import Float
from torch import Tensor
from torchmetrics.functional.image import structural_similarity_index_measure

from .metric import Metric


class MetricSSIM(Metric):
    def compute(
        self,
        prediction: Float[Tensor, "batch frame rgb=3 height width"],
        ground_truth: Float[Tensor, "batch frame rgb=3 height width"],
    ) -> dict[str, Float[Tensor, " batch #frame"]]:
        b, f, _, _, _ = prediction.shape
        ssim = structural_similarity_index_measure(
            rearrange(prediction, "b f c h w -> (b f) c h w"),
            rearrange(ground_truth, "b f c h w -> (b f) c h w"),
            data_range=(0.0, 1.0),
            reduction="none",
        )
        return {"ssim": rearrange(ssim, "(b f) -> b f", b=b, f=f)}
