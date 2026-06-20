from pathlib import Path
from typing import TypeVar

from dacite import Config, from_dict
from omegaconf import DictConfig, OmegaConf

T = TypeVar("T")


def get_typed_config(cfg: DictConfig, cls: type[T]) -> T:
    # Convert the configuration to a nested dataclass. The type hooks are necessary
    # because Dacite doesn't handle casting of union elements correctly.
    type_hooks = {
        tuple[int, int] | None: lambda x: None if x is None else tuple(x),
    }
    return from_dict(
        cls,
        OmegaConf.to_container(cfg, resolve=True),
        Config(cast=[Path, tuple], type_hooks=type_hooks),
    )
