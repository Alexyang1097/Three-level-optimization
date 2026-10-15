from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from .data import SyntheticCoTDataset, TargetScales
from .evaluate import evaluate_teacher_forced
from .model import CausalCoTTransformer


@dataclass
class ComputeTracker:
    """Shared accounting for comparable optimization and evaluation cost."""

    backward_budget: int | None = None
    optimizer_updates: int = 0
    backward_calls: int = 0
    training_prompts: int = 0
    full_train_evaluations: int = 0
    full_train_prompts: int = 0
    threshold_evaluations: int = 0
    threshold_evaluation_prompts: int = 0
    scheduled_evaluations: int = 0
    condition_checks: int = 0
    condition_check_prompts: int = 0
    oracle_calls: int = 0
    training_data_generation_sec: float = 0.0
    model_setup_sec: float = 0.0
    elapsed_train_sec: float = 0.0
    full_train_evaluation_sec: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)

    def can_step(self) -> bool:
        return self.backward_budget is None or self.backward_calls < self.backward_budget

    @property
    def budget_exhausted(self) -> bool:
        return not self.can_step()

    def record_backward(self, prompts: int) -> None:
        if not self.can_step():
            raise RuntimeError("Backward budget exhausted")
        self.optimizer_updates += 1
        self.backward_calls += 1
        self.training_prompts += int(prompts)

    def record_full_evaluation(
        self,
        prompts: int,
        *,
        reason: str,
        elapsed_sec: float,
        threshold_evaluation: bool = False,
    ) -> None:
        self.full_train_evaluations += 1
        self.full_train_prompts += int(prompts)
        self.full_train_evaluation_sec += float(elapsed_sec)
        if threshold_evaluation:
            self.threshold_evaluations += 1
            self.threshold_evaluation_prompts += int(prompts)
        if "scheduled" in reason:
            self.scheduled_evaluations += 1

    def summary(self) -> dict[str, Any]:
        payload = {key: value for key, value in self.__dict__.items() if key != "history"}
        payload["total_processed_prompts"] = (
            self.training_prompts + self.full_train_prompts + self.condition_check_prompts
        )
        payload["budget_exhausted"] = self.budget_exhausted
        return payload


def tracked_teacher_forced_evaluation(
    model: CausalCoTTransformer,
    dataset: SyntheticCoTDataset,
    scales: TargetScales,
    *,
    tracker: ComputeTracker,
    phase: str,
    reason: str,
    batch_size: int,
    device: torch.device | str,
    threshold_evaluation: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, float]:
    started = synchronized_time(device)
    metrics = evaluate_teacher_forced(model, dataset, scales, batch_size=batch_size, device=device)
    elapsed_sec = synchronized_time(device) - started
    tracker.record_full_evaluation(
        len(dataset),
        reason=reason,
        elapsed_sec=elapsed_sec,
        threshold_evaluation=threshold_evaluation,
    )
    row: dict[str, Any] = {
        "backward_calls": tracker.backward_calls,
        "optimizer_updates": tracker.optimizer_updates,
        "training_prompts": tracker.training_prompts,
        "full_train_evaluations": tracker.full_train_evaluations,
        "evaluation_elapsed_sec": elapsed_sec,
        "phase": phase,
        "evaluation_reason": reason,
        **metrics,
    }
    if extra:
        row.update(extra)
    tracker.history.append(row)
    return metrics


def record_metric_snapshot(
    tracker: ComputeTracker,
    metrics: dict[str, float],
    *,
    phase: str,
    reason: str,
    extra: dict[str, Any] | None = None,
) -> None:
    row: dict[str, Any] = {
        "backward_calls": tracker.backward_calls,
        "optimizer_updates": tracker.optimizer_updates,
        "training_prompts": tracker.training_prompts,
        "full_train_evaluations": tracker.full_train_evaluations,
        "phase": phase,
        "evaluation_reason": reason,
        **metrics,
    }
    if extra:
        row.update(extra)
    tracker.history.append(row)


def synchronized_time(device: torch.device | str) -> float:
    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(resolved)
    elif resolved.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
    return time.perf_counter()
