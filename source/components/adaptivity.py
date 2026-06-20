from dataclasses import dataclass
from typing import Literal

import torch
from einops import repeat
from jaxtyping import Bool, Float, Int, Int64
from torch import Tensor, nn

from .action_encoder import ActionEncoder
from .dit_per_token_c import DiT, TimestepEmbedder, get_dit

##################
# Configurations #
##################


@dataclass(frozen=True)
class LevelsCfg:
    available_num_levels: int
    image_shape: tuple[int, int]
    patch_shape: tuple[int, int]
    latent_channels: int

    @property
    def latent_shape(self) -> tuple[int, int]:
        ph, pw = self.patch_shape
        h, w = self.image_shape
        return h // ph, w // pw

    @property
    def base_num_tokens(self) -> int:
        h, w = self.latent_shape
        return h * w

    def num_tokens_at_level(self, level: int) -> int:
        assert 0 <= level < self.available_num_levels
        return self.base_num_tokens // 4**level


##########################
# Indexed Copy (Scatter) #
##########################


def indices_to_keys(
    indices: Int[Tensor, "*shape flhw=4"],
    max_level: int = 256,  # 8 bits
    max_height: int = 65536,  # 16 bits
    max_width: int = 65536,  # 16 bits
) -> Int64[Tensor, "*shape"]:
    index_f, index_l, index_h, index_w = indices.unbind(dim=-1)
    keys = index_f.clone().type(torch.int64)
    keys *= max_level
    keys += index_l
    keys *= max_height
    keys += index_h
    keys *= max_width
    keys += index_w
    return keys


@torch.no_grad
def scatter_by_index_(
    source_values: Float[Tensor, "batch source_token *shape"],
    source_indices: Int[Tensor, "batch source_token flhw=4"],
    target_values: Float[Tensor, "batch target_token *shape"],
    target_indices: Int[Tensor, "batch target_token flhw=4"],
    source_mask: Bool[Tensor, "batch source_token"] | None = None,
) -> None:
    """Transfer values from source_values to target_values wherever the corresponding
    index in source_indices is equal to the corresponding index in target_indices. This
    code assumes that every index in source_indices is present in target_indices.
    """

    # First, convert the flhw indices to 1D int64 keys that can be compared.
    source_keys = indices_to_keys(source_indices)
    target_keys = indices_to_keys(target_indices)

    # If a source mask is provided, prevent masked-out entries from being copied.
    if source_mask is not None:
        source_keys[~source_mask] = -1

    scatter_by_key_(source_values, source_keys, target_values, target_keys)


@torch.no_grad
def scatter_by_key_(
    source_values: Float[Tensor, "batch source_token *shape"],
    source_keys: Int[Tensor, "batch source_token"],
    target_values: Float[Tensor, "batch target_token *shape"],
    target_keys: Int[Tensor, "batch target_token"],
) -> None:
    """Transfer values from source_values to target_values wherever the corresponding
    key in source_keys is equal to the corresponding key in target_keys. This code
    assumes that every key in source_keys and target_keys is unique.
    """

    # For every key in source_keys, find the index of the matching key in target_keys.
    # Note that we do this separately for each batch element; in other words, we're only
    # doing this along the token dimension. This code is a bit confusing. We do the
    # following:
    # 1. Sort the target keys so that they're searchable with binary sort. Keep track of
    #    the reordering so that we can undo it later.
    # 2. Use binary search to find the index (in the sorted array) of each source key.
    #    If no exact match is found, this index will be incorrect.
    # 3. Clip the indices to s. This is necessary because if an exact match isn't found,
    #    the resulting index can be equal to s, which would cause a crash.
    # 4. Use the reordering from before (undo_sort) to convert indices within
    #    sorted_target_keys to indices within target_keys.
    # 5. Create a mask of successful matches by comparing the keys at the found indices.
    b, s_source = source_keys.shape
    _, s_target = target_keys.shape
    sorted_target_keys, undo_sort = torch.sort(target_keys, dim=-1)
    token_indices = torch.searchsorted(sorted_target_keys, source_keys)
    batch_indices = torch.arange(b, device=source_keys.device)
    batch_indices = repeat(batch_indices, "b -> b s", s=s_source)
    token_indices = token_indices.clip(max=s_target - 1)
    token_indices = undo_sort[batch_indices, token_indices]
    valid_mask = target_keys[batch_indices, token_indices] == source_keys

    # Scatter the source values to the target using the indexing from above, but only in
    # places where valid_mask is true.
    batch_indices = batch_indices[valid_mask]
    token_indices = token_indices[valid_mask]
    target_values[batch_indices, token_indices] = source_values[valid_mask]


###############
# DiTAdaptive #
###############


def posemb_flhw(
    indices: Int[Tensor, "*shape flhw=4"],
    dim: int,
    temperature: int = 10000,
    dtype: torch.dtype = torch.float32,
) -> Float[Tensor, "*shape dim"]:
    omega = torch.arange(dim // 6, device=indices.device) / (dim // 6 - 1)
    omega = 1.0 / (temperature**omega)
    index_f, _, index_h, index_w = indices.unbind(dim=-1)
    f = index_f[..., None] * omega
    h = index_h[..., None] * omega
    w = index_w[..., None] * omega
    pe = torch.cat((f.sin(), f.cos(), h.sin(), h.cos(), w.sin(), w.cos()), dim=-1)
    pe = torch.cat((pe, torch.zeros_like(pe[..., : dim - pe.shape[-1]])), dim=-1)
    return pe.type(dtype)


class DiTAdaptive(nn.Module):
    time_embedding: TimestepEmbedder
    level_embedding: nn.Embedding
    dit: DiT

    def __init__(
        self,
        num_levels: int,
        model_size: Literal["S", "B", "L", "XL"],
    ) -> None:
        super().__init__()
        self.dit = get_dit(model_size)
        self.time_embedding = TimestepEmbedder(self.dit.hidden_channels)
        self.level_embedding = nn.Embedding(num_levels, self.dit.hidden_channels)
        self.action_encoder = ActionEncoder(self.dit.hidden_channels)

    @staticmethod
    def make_indices_relative(
        indices: Int[Tensor, "batch token flhw=4"],
    ) -> Int[Tensor, "batch token flhw=4"]:
        """Return indices where the frame indices are always 0 to N."""
        index_f, index_l, index_h, index_w = indices.unbind(dim=-1)
        offset = index_f.min(dim=-1).values[:, None]
        return torch.stack((index_f - offset, index_l, index_h, index_w), dim=-1)

    def forward(
        self,
        x: Float[Tensor, "batch token channel"],
        indices: Int[Tensor, "batch token flhw=4"],
        t: Float[Tensor, "batch token"],
        actions: Float[Tensor, "batch token action_channel"],
        action_mask: Bool[Tensor, " batch"] | None = None,
    ) -> tuple[
        Float[Tensor, "batch token channel"],  # x
        Float[Tensor, "batch token channel"],  # c
    ]:
        indices = self.make_indices_relative(indices)

        # Define the input (x) and conditioning (c).
        x = x + posemb_flhw(indices, self.dit.hidden_channels)
        _, index_l, _, _ = indices.unbind(dim=-1)
        actions = self.action_encoder(actions)
        if action_mask is not None:
            actions = torch.where(action_mask[:, None, None], actions, 0)
        c = self.time_embedding(t) + self.level_embedding(index_l) + actions

        # Run the main DiT transformer.
        return self.dit(x, c), c

    @property
    def hidden_channels(self) -> int:
        return self.dit.hidden_channels
