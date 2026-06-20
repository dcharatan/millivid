from abc import ABC

from jaxtyping import Float
from torch import Tensor, nn


class Metric(ABC, nn.Module):
    def compute(
        self,
        prediction: Float[Tensor, "batch frame rgb=3 height width"],
        ground_truth: Float[Tensor, "batch frame rgb=3 height width"],
    ) -> dict[str, Float[Tensor, " batch #frame"]]:
        """Used for metrics like PSNR, where one can immediately compute a value."""
        return {}

    def embed(
        self,
        prediction: Float[Tensor, "batch frame rgb=3 height width"],
    ) -> dict[str, Float[Tensor, " batch #frame channel"]]:
        """Used for metrics like FID, where there's a later aggregation step."""
        return {}
