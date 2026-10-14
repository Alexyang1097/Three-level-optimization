from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import TargetScales
from .model import gather_query_predictions


@dataclass(frozen=True)
class Losses:
    e1: torch.Tensor
    e2: torch.Tensor
    e3: torch.Tensor

    @property
    def f1(self) -> torch.Tensor:
        return self.e1

    @property
    def f2(self) -> torch.Tensor:
        return 0.5 * (self.e1 + self.e2)

    @property
    def f3(self) -> torch.Tensor:
        return (self.e1 + self.e2 + self.e3) / 3.0

    def objective(self, name: str) -> torch.Tensor:
        if name == "f1":
            return self.f1
        if name == "f2":
            return self.f2
        if name == "f3":
            return self.f3
        raise ValueError(f"Unknown objective {name!r}")

    def floats(self) -> dict[str, float]:
        return {
            "e1": self.e1.item(),
            "e2": self.e2.item(),
            "e3": self.e3.item(),
            "f1": self.f1.item(),
            "f2": self.f2.item(),
            "f3": self.f3.item(),
        }


def compute_losses(outputs: torch.Tensor, batch: dict[str, torch.Tensor], scales: TargetScales) -> Losses:
    p1, p2, p3 = gather_query_predictions(outputs, batch["pred_pos"].to(outputs.device))
    target1 = batch["target1"].to(outputs.device)
    target2 = batch["target2"].to(outputs.device)
    target3 = batch["target3"].to(outputs.device)
    a1, a2, a3 = scales.as_tensor(outputs.device)

    e1 = (p1 - target1).square().sum(dim=-1).mean() / a1
    e2 = (p2 - target2).square().sum(dim=-1).mean() / a2
    e3 = (p3 - target3).square().sum(dim=-1).mean() / a3
    return Losses(e1=e1, e2=e2, e3=e3)

