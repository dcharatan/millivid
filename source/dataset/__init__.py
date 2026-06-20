import signal
import sys
from dataclasses import dataclass
from typing import Iterator, TypeVar

import torch
from torch import Tensor
from torch.utils._pytree import tree_map
from torch.utils.data import Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from .dataset import ConfigurableDataset, Split
from .dataset_images import DatasetImages, DatasetImagesCfg
from .dataset_latents_dense import DatasetLatentsDense, DatasetLatentsDenseCfg
from .dataset_latents_universal import (
    DatasetLatentsUniversal,
    DatasetLatentsUniversalCfg,
)

# This should be a union of all dataset configuration types.
DatasetCfg = DatasetImagesCfg | DatasetLatentsUniversalCfg | DatasetLatentsDenseCfg

DATASETS: dict[str, type[ConfigurableDataset]] = {
    "images": DatasetImages,
    "latents_dense": DatasetLatentsDense,
    "latents_universal": DatasetLatentsUniversal,
}


class SameBatchWrapper(Dataset):
    dataset: Dataset
    per_device_batch_size: int

    def __init__(self, dataset: Dataset, per_device_batch_size: int) -> None:
        assert not torch.distributed.is_initialized()
        self.dataset = dataset
        self.per_device_batch_size = per_device_batch_size

    def __len__(self) -> int:
        return self.per_device_batch_size

    def __getitem__(self, index: int):
        return self.dataset[index]


@dataclass(frozen=True)
class DataLoaderSplitCfg:
    per_device_batch_size: int
    num_workers: int
    seed: int


class MultipleEpochWrapper(Iterator):
    data_loader: StatefulDataLoader
    iterator: Iterator
    epoch: int

    def __init__(self, data_loader: StatefulDataLoader) -> None:
        super().__init__()
        self.epoch = 0
        self.data_loader = data_loader
        self.iterator = None

    def __iter__(self) -> "MultipleEpochWrapper":
        return self

    def __next__(self):
        if self.iterator is None:
            self.data_loader.sampler.set_epoch(self.epoch)
            self.iterator = iter(self.data_loader)
        try:
            return next(self.iterator)
        except StopIteration:
            self.epoch += 1
            self.data_loader.sampler.set_epoch(self.epoch)
            self.iterator = iter(self.data_loader)
            return next(self.iterator)

    def state_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "data_loader": self.data_loader.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.epoch = state_dict["epoch"]
        self.data_loader.load_state_dict(state_dict["data_loader"])


class SingleEpochWrapper(Iterator):
    data_loader: StatefulDataLoader

    def __init__(self, data_loader: StatefulDataLoader):
        super().__init__()
        self.data_loader = data_loader
        self.iterator = iter(self.data_loader)

    def __iter__(self) -> "SingleEpochWrapper":
        return self

    def __next__(self):
        return next(self.iterator)


type DataLoaderWrapper = SingleEpochWrapper | MultipleEpochWrapper


def exit_cleanly_on_preemption(sig, frame):
    sys.exit(0)


def worker_init_fn(worker_id):
    signal.signal(signal.SIGTERM, exit_cleanly_on_preemption)
    signal.signal(signal.SIGINT, exit_cleanly_on_preemption)


def get_dataset(dataset_cfg: DatasetCfg, split: Split) -> ConfigurableDataset:
    return DATASETS[dataset_cfg.name](dataset_cfg, split)


def get_data_loader_vis(
    dataset_cfg: DatasetCfg,
    data_loader_cfg: DataLoaderSplitCfg,
) -> DataLoaderWrapper:
    assert not torch.distributed.is_initialized()
    dataset = get_dataset(dataset_cfg, "vis")
    dataset = SameBatchWrapper(dataset, data_loader_cfg.per_device_batch_size)
    data_loader = StatefulDataLoader(
        dataset,
        batch_size=data_loader_cfg.per_device_batch_size,
        shuffle=False,
        num_workers=data_loader_cfg.num_workers,
        sampler=RandomSampler(dataset, generator=torch.Generator().manual_seed(0)),
        worker_init_fn=worker_init_fn,
        drop_last=False,
    )
    return SingleEpochWrapper(data_loader)


def get_data_loader_train(
    dataset_cfg: DatasetCfg,
    data_loader_cfg: DataLoaderSplitCfg,
) -> DataLoaderWrapper:
    dataset = get_dataset(dataset_cfg, "train")
    data_loader = StatefulDataLoader(
        dataset,
        batch_size=data_loader_cfg.per_device_batch_size,
        shuffle=False,
        num_workers=data_loader_cfg.num_workers,
        sampler=DistributedSampler(
            dataset,
            drop_last=True,
            seed=data_loader_cfg.seed,
            shuffle=True,
        ),
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )
    return MultipleEpochWrapper(data_loader)


T = TypeVar("T", bound=dict)


def to_device(batch: T, device: torch.device) -> T:
    return tree_map(lambda x: x.to(device) if isinstance(x, Tensor) else x, batch)
