from einops import rearrange
from jaxtyping import Float
from lpips import LPIPS
from torch import Tensor

from .metric import Metric


class MetricLPIPS(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.lpips = LPIPS(net="alex").eval()

    def compute(
        self,
        prediction: Float[Tensor, "batch frame rgb=3 height width"],
        ground_truth: Float[Tensor, "batch frame rgb=3 height width"],
    ) -> dict[str, Float[Tensor, " batch #frame"]]:
        b, f, _, _, _ = prediction.shape
        lpips = self.lpips(
            rearrange(prediction, "b f c h w -> (b f) c h w"),
            rearrange(ground_truth, "b f c h w -> (b f) c h w"),
            normalize=True,
        )
        return {"lpips": rearrange(lpips, "(b f) 1 1 1 -> b f", b=b, f=f)}
