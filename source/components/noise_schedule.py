from math import atan, exp, log
from typing import NamedTuple

import torch
from jaxtyping import Float
from torch import Tensor, nn


class NoiseScheduleOutput(NamedTuple):
    alpha: Float[Tensor, "*shape"]
    sigma: Float[Tensor, "*shape"]


class NoiseSchedule(nn.Module):
    """
    Cosine noise schedule that can be shifted from base resolution to target resolution,
    proposed in Simple Diffusion (2023, https://arxiv.org/abs/2301.11093).
    Here, `shift` should be set to `base_resolution / target_resolution`.
    """

    def __init__(
        self,
        logsnr_min: float = -15.0,
        logsnr_max: float = 15.0,
        shift: float = 1.0,
    ) -> None:
        super().__init__()
        self.t_min = atan(exp(-0.5 * logsnr_max))
        self.t_max = atan(exp(-0.5 * logsnr_min))
        self.shift = 2 * log(shift)

    def forward(
        self,
        t: Float[Tensor, "*#shape"],
    ) -> Float[Tensor, "*shape"]:
        return (
            -2 * torch.log(torch.tan(self.t_min + t * (self.t_max - self.t_min)))
            + self.shift
        )

    @torch.no_grad
    def coefficients(self, t: Float[Tensor, "*#shape"]) -> NoiseScheduleOutput:
        log_snr = self(t)
        alpha = torch.sigmoid(log_snr).sqrt()
        sigma = torch.sigmoid(-log_snr).sqrt()
        return NoiseScheduleOutput(alpha, sigma)
