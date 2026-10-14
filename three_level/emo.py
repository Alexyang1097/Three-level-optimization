from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader

from .data import SyntheticCoTDataset, TargetScales, collate_teacher_forcing
from .evaluate import evaluate_teacher_forced
from .model import CausalCoTTransformer
from .objectives import Losses, compute_losses


@dataclass(frozen=True)
class Constraint:
    objective: str
    threshold: float


@dataclass
class EpigraphResult:
    state: dict[str, torch.Tensor]
    metrics: dict[str, float]
    cap: float
    feasible: bool
    history: list[dict[str, Any]]


def minimize_single_objective(
    model: CausalCoTTransformer,
    train_set: SyntheticCoTDataset,
    scales: TargetScales,
    *,
    objective: str,
    updates: int,
    eval_interval: int,
    batch_size: int,
    lr: float,
    device: torch.device | str,
) -> EpigraphResult:
    """Stage-1 reference solve: minimize f1 and keep the best attained value."""

    loader = _cycle(_loader(train_set, batch_size))
    optimizer = _make_adam(model, lr)
    initial = evaluate_teacher_forced(model, train_set, scales, batch_size=batch_size, device=device)
    best = _checkpoint(model, initial)
    history: list[dict[str, Any]] = [{"update": 0, "active": objective, **initial}]

    for update in range(1, updates + 1):
        losses = _batch_losses(model, next(loader), scales, device)
        _adam_step(optimizer, losses.objective(objective))
        if update % eval_interval == 0 or update == updates:
            metrics = evaluate_teacher_forced(model, train_set, scales, batch_size=batch_size, device=device)
            history.append({"update": update, "active": objective, **metrics})
            if metrics[objective] < best["metrics"][objective]:
                best = _checkpoint(model, metrics)

    return EpigraphResult(
        state=best["state"],
        metrics=best["metrics"],
        cap=best["metrics"][objective],
        feasible=True,
        history=history,
    )


def solve_epigraph_level(
    model: CausalCoTTransformer,
    train_set: SyntheticCoTDataset,
    scales: TargetScales,
    *,
    start_state: dict[str, torch.Tensor],
    target_objective: str,
    constraints: list[Constraint],
    upper_cap: float,
    bisection_steps: int,
    inner_updates: int,
    eval_interval: int,
    batch_size: int,
    lr: float,
    device: torch.device | str,
) -> EpigraphResult:
    """Solve min cap s.t. min_theta max(residuals) <= 0 by bisection.

    For a fixed cap, the inner loop follows EMO's switching subgradient update:
    select the largest residual and update with the corresponding objective.
    """

    low = 0.0
    high = max(float(upper_cap), 1e-12)
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []

    for bisect_idx in range(bisection_steps):
        cap = 0.5 * (low + high)
        model.load_state_dict(start_state)
        optimizer = _make_adam(model, lr)
        loader = _cycle(_loader(train_set, batch_size))
        active_counts = {constraint.objective: 0 for constraint in constraints}
        active_counts[target_objective] = 0

        for update in range(1, inner_updates + 1):
            losses = _batch_losses(model, next(loader), scales, device)
            objective_tensor, active_name, residual_value = select_epigraph_objective(
                losses,
                constraints=constraints,
                target_objective=target_objective,
                cap=cap,
            )
            active_counts[active_name] += 1
            _adam_step(optimizer, objective_tensor)

            if update % eval_interval == 0 or update == inner_updates:
                metrics = evaluate_teacher_forced(model, train_set, scales, batch_size=batch_size, device=device)
                history.append(
                    {
                        "bisection": bisect_idx,
                        "cap": cap,
                        "inner_update": update,
                        "active": active_name,
                        "batch_max_residual": residual_value,
                        **metrics,
                    }
                )

        metrics = evaluate_teacher_forced(model, train_set, scales, batch_size=batch_size, device=device)
        residuals = full_residuals(metrics, constraints=constraints, target_objective=target_objective, cap=cap)
        feasible = max(residuals.values()) <= 0.0
        history.append(
            {
                "bisection": bisect_idx,
                "cap": cap,
                "feasible": feasible,
                "max_residual": max(residuals.values()),
                "active_counts": active_counts,
                **{f"residual_{key}": value for key, value in residuals.items()},
                **metrics,
            }
        )

        if feasible:
            high = cap
            best = _checkpoint(model, metrics)
        else:
            low = cap

    if best is None:
        model.load_state_dict(start_state)
        metrics = evaluate_teacher_forced(model, train_set, scales, batch_size=batch_size, device=device)
        best = _checkpoint(model, metrics)
        return EpigraphResult(best["state"], best["metrics"], high, False, history)

    return EpigraphResult(best["state"], best["metrics"], high, True, history)


def select_epigraph_objective(
    losses: Losses,
    *,
    constraints: list[Constraint],
    target_objective: str,
    cap: float,
) -> tuple[torch.Tensor, str, float]:
    residual_tensors: list[torch.Tensor] = []
    objective_tensors: list[torch.Tensor] = []
    names: list[str] = []

    for constraint in constraints:
        tensor = losses.objective(constraint.objective)
        residual_tensors.append(tensor - tensor.new_tensor(constraint.threshold))
        objective_tensors.append(tensor)
        names.append(constraint.objective)

    target_tensor = losses.objective(target_objective)
    residual_tensors.append(target_tensor - target_tensor.new_tensor(cap))
    objective_tensors.append(target_tensor)
    names.append(target_objective)

    residual_stack = torch.stack(residual_tensors)
    active_idx = int(torch.argmax(residual_stack.detach()).item())
    return objective_tensors[active_idx], names[active_idx], float(residual_stack[active_idx].detach().cpu().item())


def full_residuals(
    metrics: dict[str, float],
    *,
    constraints: list[Constraint],
    target_objective: str,
    cap: float,
) -> dict[str, float]:
    residuals = {constraint.objective: metrics[constraint.objective] - constraint.threshold for constraint in constraints}
    residuals[target_objective] = metrics[target_objective] - cap
    return residuals


def _loader(train_set: SyntheticCoTDataset, batch_size: int) -> DataLoader:
    return DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_teacher_forcing,
    )


def _cycle(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def _batch_losses(
    model: CausalCoTTransformer,
    batch: dict[str, torch.Tensor],
    scales: TargetScales,
    device: torch.device | str,
) -> Losses:
    batch = {key: value.to(device) for key, value in batch.items()}
    outputs = model(batch["tokens"], batch["types"], batch["padding_mask"])
    return compute_losses(outputs, batch, scales)


def _make_adam(model: CausalCoTTransformer, lr: float) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)


def _adam_step(optimizer: torch.optim.Optimizer, objective: torch.Tensor) -> None:
    optimizer.zero_grad(set_to_none=True)
    objective.backward()
    optimizer.step()


def _checkpoint(model: CausalCoTTransformer, metrics: dict[str, float]) -> dict[str, Any]:
    return {"state": _clone_state(model), "metrics": copy.deepcopy(metrics)}


def _clone_state(model: CausalCoTTransformer) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
