from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

from .data import SyntheticCoTDataset, TargetScales, collate_teacher_forcing
from .evaluate import evaluate_teacher_forced
from .model import CausalCoTTransformer
from .objectives import Losses, compute_losses


ResidualBuilder = Callable[[Losses], list[tuple[str, torch.Tensor]]]


@dataclass
class EMOResult:
    """Result of one epigraph oracle, including its best feasible state."""

    state: dict[str, torch.Tensor]
    metrics: dict[str, float]
    feasible: bool
    history: list[dict[str, Any]]
    last_state: dict[str, torch.Tensor]
    last_metrics: dict[str, float]


def minimize_level1(
    model: CausalCoTTransformer,
    train_set: SyntheticCoTDataset,
    scales: TargetScales,
    *,
    updates: int,
    eval_interval: int,
    batch_size: int,
    lr: float,
    device: torch.device | str,
    loader_seed: int,
) -> EMOResult:
    """Estimate E1 = min_theta f1(theta), as required by the paper."""

    initial = evaluate_teacher_forced(model, train_set, scales, batch_size=batch_size, device=device)
    best = _checkpoint(model, initial)
    history: list[dict[str, Any]] = [{"oracle": "level1", "update": 0, **initial}]
    optimizer = _make_adam(model, lr)
    batches = _cycle(_loader(train_set, batch_size, loader_seed))

    for update in range(1, updates + 1):
        losses = _batch_losses(model, next(batches), scales, device)
        _adam_step(optimizer, losses.f1)
        if _should_eval(update, eval_interval, updates):
            metrics = evaluate_teacher_forced(model, train_set, scales, batch_size=batch_size, device=device)
            history.append({"oracle": "level1", "update": update, **metrics})
            if metrics["f1"] < best["metrics"]["f1"]:
                best = _checkpoint(model, metrics)

    model.load_state_dict(best["state"])
    return EMOResult(
        state=best["state"],
        metrics=best["metrics"],
        feasible=True,
        history=history,
        last_state=_clone_state(model),
        last_metrics=best["metrics"],
    )


def solve_level2_value(
    model: CausalCoTTransformer,
    train_set: SyntheticCoTDataset,
    scales: TargetScales,
    *,
    start_state: dict[str, torch.Tensor],
    e1: float,
    epsilon: float,
    upper_hint: float,
    bisection_steps: int,
    inner_updates: int,
    oracle_eval_interval: int,
    condition_check_interval: int,
    batch_size: int,
    lr: float,
    device: torch.device | str,
    loader_seed: int,
) -> tuple[EMOResult, float]:
    """Recursively estimate E2 using the level-2 epigraph problem.

    This solves min z2 subject to min_theta max(g1(theta), f2(theta)-z2) <= 0,
    where g1 = f1 - E1 - epsilon.
    """

    def residuals(cap: float) -> ResidualBuilder:
        return lambda losses: [
            ("f1", losses.f1 - losses.f1.new_tensor(e1 + epsilon)),
            ("f2", losses.f2 - losses.f2.new_tensor(cap)),
        ]

    result, cap = _bisect_single_cap(
        model,
        train_set,
        scales,
        start_state=start_state,
        upper_hint=upper_hint,
        residuals_for_cap=residuals,
        bisection_steps=bisection_steps,
        inner_updates=inner_updates,
        oracle_eval_interval=oracle_eval_interval,
        condition_check_interval=condition_check_interval,
        batch_size=batch_size,
        lr=lr,
        device=device,
        loader_seed=loader_seed,
        oracle_name="level2",
    )
    if not result.feasible:
        raise RuntimeError(
            "EMO level-2 epigraph problem has no feasible z2 within the "
            "configured search range"
        )
    return result, cap


def solve_final_nested(
    model: CausalCoTTransformer,
    train_set: SyntheticCoTDataset,
    scales: TargetScales,
    *,
    start_state: dict[str, torch.Tensor],
    e1: float,
    e2: float,
    epsilon: float,
    upper_hint: float,
    bisection_steps: int,
    inner_updates: int,
    oracle_eval_interval: int,
    condition_check_interval: int,
    batch_size: int,
    lr: float,
    device: torch.device | str,
    loader_seed: int,
) -> tuple[EMOResult, float, float]:
    """Solve the K=3 final EMO problem with nested z1/z2 bisection.

    For fixed z1 and z2, the paper's switching oracle minimizes

        max(g1, g2 - z2, f3 - z1 - z2).

    The inner bisection chooses z2 for a fixed z1. The outer bisection then
    chooses the smallest feasible z1.
    """

    history: list[dict[str, Any]] = []
    outer_low = 0.0
    outer_high = max(float(upper_hint), 1e-8)
    best: EMOResult | None = None
    best_z1 = outer_high
    best_z2 = 0.0

    for expansion in range(12):
        inner_result, inner_z2 = _solve_inner_z2(
            model,
            train_set,
            scales,
            start_state=start_state,
            z1=outer_high,
            e1=e1,
            e2=e2,
            epsilon=epsilon,
            bisection_steps=bisection_steps,
            inner_updates=inner_updates,
            oracle_eval_interval=oracle_eval_interval,
            condition_check_interval=condition_check_interval,
            batch_size=batch_size,
            lr=lr,
            device=device,
            loader_seed=loader_seed,
            oracle_name=f"final_z1_upper_{expansion}",
        )
        history.extend(inner_result.history)
        if inner_result.feasible:
            best = inner_result
            best_z1 = outer_high
            best_z2 = inner_z2
            break
        outer_high *= 2.0
    else:
        raise RuntimeError("EMO could not find a feasible upper bracket for z1")

    for outer_step in range(bisection_steps):
        z1 = 0.5 * (outer_low + outer_high)
        inner_result, inner_z2 = _solve_inner_z2(
            model,
            train_set,
            scales,
            start_state=start_state,
            z1=z1,
            e1=e1,
            e2=e2,
            epsilon=epsilon,
            bisection_steps=bisection_steps,
            inner_updates=inner_updates,
            oracle_eval_interval=oracle_eval_interval,
            condition_check_interval=condition_check_interval,
            batch_size=batch_size,
            lr=lr,
            device=device,
            loader_seed=loader_seed,
            oracle_name=f"final_z1_{outer_step}",
        )
        history.extend(inner_result.history)
        history.append(
            {
                "oracle": "final_outer",
                "bisection": outer_step,
                "z1": z1,
                "z2": inner_z2,
                "feasible": inner_result.feasible,
                **inner_result.metrics,
            }
        )
        if inner_result.feasible:
            outer_high = z1
            best = inner_result
            best_z1 = z1
            best_z2 = inner_z2
        else:
            outer_low = z1

    assert best is not None
    model.load_state_dict(best.state)
    return (
        EMOResult(
            state=best.state,
            metrics=best.metrics,
            feasible=True,
            history=history,
            last_state=best.last_state,
            last_metrics=best.last_metrics,
        ),
        best_z1,
        best_z2,
    )


def _solve_inner_z2(
    model: CausalCoTTransformer,
    train_set: SyntheticCoTDataset,
    scales: TargetScales,
    *,
    start_state: dict[str, torch.Tensor],
    z1: float,
    e1: float,
    e2: float,
    epsilon: float,
    bisection_steps: int,
    inner_updates: int,
    oracle_eval_interval: int,
    condition_check_interval: int,
    batch_size: int,
    lr: float,
    device: torch.device | str,
    loader_seed: int,
    oracle_name: str,
) -> tuple[EMOResult, float]:
    def residuals(cap: float) -> ResidualBuilder:
        return lambda losses: [
            ("f1", losses.f1 - losses.f1.new_tensor(e1 + epsilon)),
            ("f2", losses.f2 - losses.f2.new_tensor(e2 + epsilon + cap)),
            ("f3", losses.f3 - losses.f3.new_tensor(z1 + cap)),
        ]

    start_metrics = evaluate_teacher_forced(model, train_set, scales, batch_size=batch_size, device=device)
    result, cap = _bisect_single_cap(
        model,
        train_set,
        scales,
        start_state=start_state,
        upper_hint=max(start_metrics["f2"] - e2, start_metrics["f3"] - z1, 1.0),
        residuals_for_cap=residuals,
        bisection_steps=bisection_steps,
        inner_updates=inner_updates,
        oracle_eval_interval=oracle_eval_interval,
        condition_check_interval=condition_check_interval,
        batch_size=batch_size,
        lr=lr,
        device=device,
        loader_seed=loader_seed,
        oracle_name=oracle_name,
    )
    return result, cap


def _bisect_single_cap(
    model: CausalCoTTransformer,
    train_set: SyntheticCoTDataset,
    scales: TargetScales,
    *,
    start_state: dict[str, torch.Tensor],
    upper_hint: float,
    residuals_for_cap: Callable[[float], ResidualBuilder],
    bisection_steps: int,
    inner_updates: int,
    oracle_eval_interval: int,
    condition_check_interval: int,
    batch_size: int,
    lr: float,
    device: torch.device | str,
    loader_seed: int,
    oracle_name: str,
) -> tuple[EMOResult, float]:
    low = 0.0
    high = max(float(upper_hint), 1e-8)
    best: EMOResult | None = None
    last_result: EMOResult | None = None

    for expansion in range(12):
        result = _run_switching_oracle(
            model,
            train_set,
            scales,
            start_state=start_state,
            residual_builder=residuals_for_cap(high),
            inner_updates=inner_updates,
            oracle_eval_interval=oracle_eval_interval,
            condition_check_interval=condition_check_interval,
            batch_size=batch_size,
            lr=lr,
            device=device,
            loader_seed=loader_seed,
            oracle_name=oracle_name,
            cap=high,
        )
        last_result = result
        if result.feasible:
            best = result
            break
        high *= 2.0

    if best is None:
        # A failed upper-bracket search is an inner infeasibility result for
        # the caller.  The final z1 solver uses it to increase its lower
        # bracket; level 2 turns it into a clear fatal error because E2 is
        # needed before the final nested problem can start.
        assert last_result is not None
        last_result.history.append(
            {
                "oracle": oracle_name,
                "bracket_failed": True,
                "upper_cap": high / 2.0,
                "feasible": False,
            }
        )
        return last_result, high / 2.0

    for step in range(bisection_steps):
        cap = 0.5 * (low + high)
        result = _run_switching_oracle(
            model,
            train_set,
            scales,
            start_state=start_state,
            residual_builder=residuals_for_cap(cap),
            inner_updates=inner_updates,
            oracle_eval_interval=oracle_eval_interval,
            condition_check_interval=condition_check_interval,
            batch_size=batch_size,
            lr=lr,
            device=device,
            loader_seed=loader_seed,
            oracle_name=oracle_name,
            cap=cap,
        )
        if result.feasible:
            high = cap
            best = result
        else:
            low = cap

    return best, high


def _run_switching_oracle(
    model: CausalCoTTransformer,
    train_set: SyntheticCoTDataset,
    scales: TargetScales,
    *,
    start_state: dict[str, torch.Tensor],
    residual_builder: ResidualBuilder,
    inner_updates: int,
    oracle_eval_interval: int,
    condition_check_interval: int,
    batch_size: int,
    lr: float,
    device: torch.device | str,
    loader_seed: int,
    oracle_name: str,
    cap: float,
) -> EMOResult:
    """Run Eq. (11): update with the objective behind the largest residual."""

    model.load_state_dict(start_state)
    optimizer = _make_adam(model, lr)
    batches = _cycle(_loader(train_set, batch_size, loader_seed))
    history: list[dict[str, Any]] = []

    if condition_check_interval <= 0:
        raise ValueError("condition_check_interval must be positive")

    stopped_early = False
    for update in range(1, inner_updates + 1):
        batch = next(batches)
        losses = _batch_losses(model, batch, scales, device)
        residuals = residual_builder(losses)
        values = torch.stack([value for _, value in residuals])
        active_idx = int(torch.argmax(values.detach()).item())
        active_name, _ = residuals[active_idx]
        _adam_step(optimizer, losses.objective(active_name))

        batch_condition = False
        if update % condition_check_interval == 0:
            post_losses = _batch_losses(model, batch, scales, device)
            post_residuals = residual_builder(post_losses)
            post_values = torch.stack([value for _, value in post_residuals])
            batch_condition = float(post_values.detach().max().cpu().item()) <= 0.0
        if batch_condition or (
            oracle_eval_interval > 0 and _should_eval(update, oracle_eval_interval, inner_updates)
        ):
            metrics = evaluate_teacher_forced(model, train_set, scales, batch_size=batch_size, device=device)
            full_residuals = _metrics_residuals(metrics, residual_builder)
            history.append(
                {
                    "oracle": oracle_name,
                    "update": update,
                    "cap": cap,
                    "active": active_name,
                    "max_residual": max(full_residuals.values()),
                    "feasible": max(full_residuals.values()) <= 0.0,
                    "batch_condition": batch_condition,
                    **metrics,
                }
            )
            # The full training-set condition is authoritative. The minibatch
            # check only decides whether an extra full evaluation is worthwhile.
            if max(full_residuals.values()) <= 0.0:
                stopped_early = True
                break

    metrics = evaluate_teacher_forced(model, train_set, scales, batch_size=batch_size, device=device)
    full_residuals = _metrics_residuals(metrics, residual_builder)
    feasible = max(full_residuals.values()) <= 0.0
    history.append(
        {
            "oracle": oracle_name,
            "update": inner_updates,
            "cap": cap,
            "final": True,
            "stopped_early": stopped_early,
            "max_residual": max(full_residuals.values()),
            "feasible": feasible,
            **{f"residual_{key}": value for key, value in full_residuals.items()},
            **metrics,
        }
    )
    last_state = _clone_state(model)
    return EMOResult(
        state=last_state,
        metrics=metrics,
        feasible=feasible,
        history=history,
        last_state=last_state,
        last_metrics=metrics,
    )


def _metrics_residuals(metrics: dict[str, float], residual_builder: ResidualBuilder) -> dict[str, float]:
    # Reuse the exact residual definition with detached scalar tensors so the
    # feasibility check cannot silently diverge from the minibatch selector.
    losses = Losses(
        e1=torch.tensor(metrics["e1"]),
        e2=torch.tensor(metrics["e2"]),
        e3=torch.tensor(metrics["e3"]),
    )
    return {name: float(value.detach().cpu().item()) for name, value in residual_builder(losses)}


def _loader(train_set: SyntheticCoTDataset, batch_size: int, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
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


def _should_eval(update: int, interval: int, total_updates: int) -> bool:
    return update % interval == 0 or update == total_updates


def _checkpoint(model: CausalCoTTransformer, metrics: dict[str, float]) -> dict[str, Any]:
    return {"state": _clone_state(model), "metrics": copy.deepcopy(metrics)}


def _clone_state(model: CausalCoTTransformer) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
