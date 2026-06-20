from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Generic, TypeVar, final

import torch
from jaxtyping import Float
from torch import Tensor, nn

C = TypeVar("C")  # config type
TrainBatch = TypeVar("TrainBatch", bound=dict)
VisBatch = TypeVar("VisBatch", bound=dict)
TestBatch = TypeVar("TestBatch", bound=dict)


type Metrics = dict[str, Float[Tensor, "..."]]
type Images = dict[str, Float[Tensor, "batch channel=_ height=_ width=_"]]
type Videos = dict[str, tuple[bytes, ...]]


type Loss = Float[Tensor, ""]
type TrainStepOutput = Loss | tuple[Loss, Metrics]


@dataclass(frozen=True, kw_only=True)
class VisStepOutput:
    images: Images = field(default_factory=dict)
    metrics: Metrics = field(default_factory=dict)

    # These are encoded (e.g., mp4 or webm) videos as bytes. Use the encode_videos
    # function from source.image_io to encode videos to this format.
    videos: Videos = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class TestStepMetadata:
    run_id: str | None
    step: int | str | None  # can be str if used for tagging
    workspace: Path
    tag: str | None


class ConfigurableModel(nn.Module, Generic[C, TrainBatch, VisBatch, TestBatch], ABC):
    cfg: C

    def __init__(self, cfg: C) -> None:
        super().__init__()
        self.cfg = cfg

    @final
    def forward(self, batch: TrainBatch) -> TrainStepOutput:
        return self.train_step(batch)

    @abstractmethod
    def train_step(self, batch: TrainBatch) -> TrainStepOutput:
        pass

    @abstractmethod
    def vis_step(self, batch: VisBatch) -> VisStepOutput:
        pass

    @abstractmethod
    def test_step(
        self,
        batch: TestBatch,
        results: tuple[BytesIO, ...],
        metadata: TestStepMetadata,
    ) -> None:
        pass

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
