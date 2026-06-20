import os
from contextlib import contextmanager
from functools import cache, cached_property, wraps
from typing import Any, Callable, ParamSpec, TypeVar

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from torch.utils._pytree import tree_map

P = ParamSpec("P")
R = TypeVar("R")


@cache
def get_local_rank() -> int:
    return int(os.environ["LOCAL_RANK"])


def is_rank_zero() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def rank_zero_only(func: Callable[P, R]) -> Callable[P, R | None]:
    """Execute function only on rank 0 in distributed training."""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | None:
        # Only execute the function on rank 0.
        if is_rank_zero():
            return func(*args, **kwargs)
        return None

    return wrapper


def gather_to_rank_zero(tree: Any) -> Any:
    def _gather_to_rank_zero(x: Any) -> np.ndarray:
        # Only gather tensors.
        if not isinstance(x, Tensor):
            return None

        # Make sure the tensor can be gathered.
        x = x.contiguous().to(torch.device(f"cuda:{get_local_rank()}"))
        if x.ndim == 0:
            x = x[None]

        # All-gather the tensor.
        gathered = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, x)
        return torch.cat(gathered).cpu().float().numpy()

    return tree_map(_gather_to_rank_zero, tree)


@contextmanager
def rank_zero_first():
    """A context manager that executes code on rank 0 first, then on all other ranks."""

    if not dist.is_initialized():
        yield
        return

    if not is_rank_zero():
        dist.barrier()
        torch.cuda.synchronize()
    yield
    if is_rank_zero():
        dist.barrier()
        torch.cuda.synchronize()


def warm_cached_properties(obj: object) -> None:
    """Evaluate every cached_property on obj so its value gets cached."""
    seen: set[str] = set()
    for klass in type(obj).__mro__:
        for name, attr in vars(klass).items():
            if isinstance(attr, cached_property) and name not in seen:
                seen.add(name)
                getattr(obj, name)
