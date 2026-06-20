from typing import Generic, Literal, TypeVar

from torch.utils.data import Dataset

Split = Literal["train", "vis", "test", "all"]

C = TypeVar("C")  # config type


class ConfigurableDataset(Dataset, Generic[C]):
    cfg: C
    split: Split

    def __init__(self, cfg: C, split: Split) -> None:
        super().__init__()
        self.cfg = cfg
        self.split = split
