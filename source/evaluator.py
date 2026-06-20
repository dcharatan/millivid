from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Generic, TypeVar

import torch
from distribute import execute_tasks_batched
from torch.utils.data import default_collate

from .dataset import DataLoaderWrapper, DatasetCfg, Split, get_dataset, to_device
from .model import ModelCfg, get_model
from .model.model import ConfigurableModel, TestStepMetadata


@dataclass(frozen=True)
class EvaluatorCfg:
    checkpoint_step: int | str | None
    split: Split
    batch_size: int
    dataset: DatasetCfg
    model: ModelCfg
    tag: str | None


C = TypeVar("C")  # model config type
TrainBatch = TypeVar("TrainBatch", bound=dict)
VisBatch = TypeVar("VisBatch", bound=dict)
TestBatch = TypeVar("TestBatch", bound=dict)


class Evaluator(Generic[C, TrainBatch, VisBatch, TestBatch]):
    cfg: EvaluatorCfg
    workspace: Path
    test_loader: DataLoaderWrapper

    def __init__(
        self,
        cfg: EvaluatorCfg,
        workspace: Path,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.workspace = workspace
        self.dataset = get_dataset(cfg.dataset, cfg.split)

    @property
    def model(self) -> ConfigurableModel[C, TrainBatch, VisBatch, TestBatch]:
        # Lazily load the model
        if not hasattr(self, "_model"):
            self._model = get_model(self.cfg.model).to("cuda").eval()
            self.load_model()
        return self._model

    def load_model(self) -> None:
        checkpoint_step = self.cfg.checkpoint_step
        if checkpoint_step is None or isinstance(checkpoint_step, str):
            return
        for path in (self.workspace / "checkpoints").iterdir():
            if not path.is_file():
                continue
            step = int(path.stem)
            if step != checkpoint_step:
                continue
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint["model"])
            break
        else:
            raise ValueError(f"Checkpoint for step {checkpoint_step} not found.")

    @torch.no_grad()
    def evaluate_example(
        self,
        keys: tuple[str, ...],
        results: tuple[BytesIO, ...],
    ) -> None:
        indices = tuple(int(key[len("index_") :]) for key in keys)
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            batch = default_collate([self.dataset[index] for index in indices])
            batch = to_device(batch, self.model.device)
            metadata = TestStepMetadata(
                run_id=self.run_id,
                step=self.cfg.checkpoint_step,
                workspace=self.workspace,
                tag=self.cfg.tag,
            )
            self.model.test_step(batch, results, metadata)

    #################
    # Orchestration #
    #################

    @property
    def num_examples(self) -> int:
        return len(self.dataset)

    @property
    def run_id(self) -> str | None:
        try:
            with (self.workspace / "id.txt").open() as f:
                return f.read().strip()
        except FileNotFoundError:
            return None

    @property
    def job_name(self) -> str:
        run_id = self.workspace.stem if self.run_id is None else self.run_id
        tag = () if self.cfg.tag is None else (self.cfg.tag,)
        return "_".join(("eval", *tag, run_id, str(self.cfg.checkpoint_step)))

    def evaluate(self) -> None:
        execute_tasks_batched(self.job_name, self.evaluate_example, self.cfg.batch_size)

    def debug(self) -> None:
        self.evaluate_example(
            tuple(f"index_{i}" for i in range(self.cfg.batch_size)),
            tuple(BytesIO() for i in range(self.cfg.batch_size)),
        )
