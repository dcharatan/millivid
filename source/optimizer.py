from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.optim import AdamW, Muon, Optimizer


def convert_to_buffer(module: nn.Module, persistent: bool):
    # Recurse over child modules.
    for name, child in list(module.named_children()):
        convert_to_buffer(child, persistent)

    # Also re-save buffers to change persistence.
    for name, parameter_or_buffer in (
        *module.named_parameters(recurse=False),
        *module.named_buffers(recurse=False),
    ):
        value = parameter_or_buffer.detach().clone()
        delattr(module, name)
        module.register_buffer(name, value, persistent=persistent)


@dataclass(frozen=True)
class OptimizerCfg:
    variant: Literal["muon_adamw", "adamw"]
    learning_rate: float
    weight_decay: float


class MuonAdamW(Optimizer):
    muon: Muon
    adamw: AdamW

    def __init__(
        self,
        muon: Muon,
        adamw: AdamW,
    ) -> None:
        self.muon = muon
        self.adamw = adamw
        super().__init__(self.muon.param_groups + self.adamw.param_groups, {})

    @torch.no_grad()
    def step(self, closure: None = None) -> None:
        assert closure is None
        self.muon.step()
        self.adamw.step()

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])


def build_muon_adamw_optimizer(model: nn.Module, cfg: OptimizerCfg) -> MuonAdamW:
    # Split the model's parameters.
    muon_params = []
    other_params = []
    for module in model.modules():
        for p in module.parameters(recurse=False):
            # Skip frozen parameters.
            if not p.requires_grad:
                continue

            # Let Muon handle linear layer weight matrices.
            if isinstance(module, nn.Linear) and p.ndim == 2:
                muon_params.append(p)
            else:
                other_params.append(p)

    # Build the combined optimizer.
    return MuonAdamW(
        Muon(
            muon_params,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            adjust_lr_fn="match_rms_adamw",
        ),
        AdamW(other_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay),
    )


def build_adamw_optimizer(model: nn.Module, cfg: OptimizerCfg) -> AdamW:
    return AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )


def build_optimizer(model: nn.Module, cfg: OptimizerCfg) -> Optimizer:
    build_optimizer = {
        "muon_adamw": build_muon_adamw_optimizer,
        "adamw": build_adamw_optimizer,
    }[cfg.variant]
    return build_optimizer(model, cfg)
