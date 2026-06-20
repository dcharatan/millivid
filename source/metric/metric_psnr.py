import torch
from einops import reduce
from jaxtyping import Float
from torch import Tensor

from .metric import Metric


@torch.no_grad()
def compute_psnr(
    prediction: Float[Tensor, "*batch channel height width"],
    ground_truth: Float[Tensor, "*batch channel height width"],
) -> Float[Tensor, "*batch"]:
    with torch.autocast("cuda", enabled=False):
        prediction = prediction.float().clip(min=0, max=1)
        ground_truth = ground_truth.float().clip(min=0, max=1)
        mse = reduce((ground_truth - prediction) ** 2, "... c h w -> ...", "mean")
        return -10 * mse.log10()


class MetricPSNR(Metric):
    def compute(
        self,
        prediction: Float[Tensor, "batch frame rgb=3 height width"],
        ground_truth: Float[Tensor, "batch frame rgb=3 height width"],
    ) -> dict[str, Float[Tensor, " batch #frame"]]:
        return {"psnr": compute_psnr(prediction, ground_truth)}
