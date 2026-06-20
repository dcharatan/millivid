# Adapted from https://github.com/facebookresearch/DiT/blob/main/models.py
# The core DiT model, but modified to have per-token conditioning.


import math
from typing import Literal

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float
from timm.models.vision_transformer import Mlp
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention

#############
# Attention #
#############


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        proj_bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, **dd)
        self.q_norm = torch.nn.RMSNorm(self.head_dim, **dd)
        self.k_norm = torch.nn.RMSNorm(self.head_dim, **dd)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, **dd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = (
            rearrange(t, "b s (h c) -> b h s c", c=self.head_dim)
            for t in self.qkv(x).chunk(3, dim=-1)
        )
        with torch.autocast("cuda", enabled=False):
            q = self.q_norm(q.float()).bfloat16()
            k = self.k_norm(k.float()).bfloat16()

        x = scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b h s c -> b s (h c)")
        return self.proj(x)


##################
# Core DiT Model #
##################


def modulate(
    x: Float[Tensor, "batch token channel"],
    shift: Float[Tensor, "batch token channel"],
    scale: Float[Tensor, "batch token channel"],
) -> Float[Tensor, "batch token channel"]:
    return x * (1 + scale) + shift


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_channels: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_channels, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_channels, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_channels, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(
            in_features=hidden_channels,
            hidden_features=int(hidden_channels * mlp_ratio),
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_channels, 6 * hidden_channels, bias=True),
        )

        # Initialize the linear layers.
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Zero out the adaLN modulation layers.
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(
        self,
        x: Float[Tensor, "batch token channel"],
        c: Float[Tensor, "batch token channel"],
    ) -> Float[Tensor, "batch token channel"]:
        c = self.adaLN_modulation(c).chunk(6, dim=-1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = c
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        return x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))


class DiT(nn.Module):
    blocks: nn.ModuleList
    hidden_channels: int

    def __init__(
        self,
        hidden_channels: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()

        blocks = [DiTBlock(hidden_channels, num_heads, mlp_ratio) for _ in range(depth)]
        self.blocks = nn.ModuleList(blocks)
        self.hidden_channels = hidden_channels

    def forward(
        self,
        x: Float[Tensor, "batch token channel"],
        c: Float[Tensor, "batch token channel"],
    ) -> Float[Tensor, "batch token channel"]:
        for block in self.blocks:
            x = block(x, c)
        return x


##############
# Unpatchify #
##############


class Unpatchify(nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_channels,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.linear = nn.Linear(hidden_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_channels, 2 * hidden_channels, bias=True),
        )

        # Initialize as in DiT.
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(
        self,
        x: Float[Tensor, "batch token channel"],
        c: Float[Tensor, "batch token channel"],
    ) -> Float[Tensor, "batch token out_channel"]:
        # Apply the DiT's final layer.
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


######################
# Timestep Embedding #
######################


class TimestepEmbedder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        embedding_size: int = 256,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.embedding_size = embedding_size

        # Initialize as in DiT.
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def forward(
        self,
        t: Float[Tensor, "*shape"],
    ) -> Float[Tensor, "*shape hidden_channels"]:
        # Since our timesteps are in the range [0, 1] and the embedding below is
        # designed for timesteps in the range [0, 1000], we multiply our timesteps.
        t = t * 1000

        # This is a slightly cleaner version of DiT's time embedding.
        half = self.embedding_size // 2
        frequencies = torch.exp(
            -math.log(10000)
            * torch.linspace(0, 1, half, dtype=torch.float32, device=t.device)
        )
        args = t[..., None] * frequencies
        args = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(args)


######################
# DiT Configurations #
######################


def DiT_XL() -> DiT:
    return DiT(depth=28, hidden_channels=1152, num_heads=16)


def DiT_L() -> DiT:
    return DiT(depth=24, hidden_channels=1024, num_heads=16)


def DiT_B() -> DiT:
    return DiT(depth=12, hidden_channels=768, num_heads=12)


def DiT_S() -> DiT:
    return DiT(depth=12, hidden_channels=384, num_heads=6)


def get_dit(model_size: Literal["S", "B", "L", "XL"]) -> DiT:
    return {"S": DiT_S, "B": DiT_B, "L": DiT_L, "XL": DiT_XL}[model_size]()
