import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float
from timm.models.vision_transformer import Mlp
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.q_norm = torch.nn.RMSNorm(self.head_dim)
        self.k_norm = torch.nn.RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
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


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_heads: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_channels, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim=hidden_channels, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_channels, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(
            in_features=hidden_channels,
            hidden_features=int(hidden_channels * mlp_ratio),
            act_layer=lambda: nn.GELU(approximate="tanh"),
        )

        # Initialize the linear layers.
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def forward(
        self,
        x: Float[Tensor, "batch token channel"],
    ) -> Float[Tensor, "batch token channel"]:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Transformer(nn.Module):
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

        blocks = [
            TransformerBlock(hidden_channels, num_heads, mlp_ratio)
            for _ in range(depth)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.hidden_channels = hidden_channels

    def forward(
        self,
        x: Float[Tensor, "batch token channel"],
    ) -> Float[Tensor, "batch token channel"]:
        for block in self.blocks:
            x = block(x)
        return x


def Transformer_XL() -> Transformer:
    return Transformer(depth=28, hidden_channels=1152, num_heads=16)


def Transformer_L() -> Transformer:
    return Transformer(depth=24, hidden_channels=1024, num_heads=16)


def Transformer_B() -> Transformer:
    return Transformer(depth=12, hidden_channels=768, num_heads=12)


def Transformer_S() -> Transformer:
    return Transformer(depth=12, hidden_channels=384, num_heads=6)
