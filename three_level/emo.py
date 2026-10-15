from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

from .data import SyntheticCoTDataset, TargetScales, collate_teacher_forcing
from .model import CausalCoTTransformer
from .objectives import Losses, compute_losses
from .tracking import ComputeTracker, tracked_teacher_forced_evaluation


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


class EMOBudgetExhausted(RuntimeError):
    def __init__(
        self,
        *,
        state: dict[str, torch.Tensor],
        metrics: dict[str, float],
        phase: str,
    ) -> None:
        super().__init__(f"EMO backward budget exhausted during {phase}")
        self.state = state
        self.metrics = metrics
        self.phase = phase


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
    tracker: ComputeTracker,
) -> EMOResult:
    """Estimate E1 = min_theta f1(theta), as required by the paper."""

    initial = _tracked_eval(
        model,
        train_set,
        scales,
        tracker=tracker,
        phase="level1",
        reason="initial,scheduled,threshold_reference",
        batch_size=batch_size,
        device=device,
        threshold_evaluation=True,
    )
    best = _checkpoint(model, initial)
    history: list[dict[str, Any]] = [{"oracle": "level1", "update": 0, **initial}]
    optimizer = _make_adam(model, lr)
    batches = _cycle(_loader(train_set, batch_size, loader_seed))

    for update in range(1, updates + 1):
        _check_budget(
            model,
            train_set,
            scales,
            tracker=tracker,
            phase="level1",
            batch_size=batch_size,
            device=device,
        )
        batch = next(batches)
        losses = _batch_losses(model, batch, scales, device)
        _adam_step(optimizer, losses.f1, tracker, batch["tokens"].shape[0])
        if _should_eval(update, eval_interval, updates):
            metrics = _tracked_eval(
                model,
                train_set,
                scales,
                tracker=tracker,
                phase="level1",
                reason="scheduled,threshold_reference" + (",phase_final" if update == updates else ""),
                batch_size=batch_size,
                device=device,
                threshold_evaluation=True,
            )
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
    eval_interval: int,
    tracker: ComputeTracker,
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
        eval_interval=eval_interval,
        tracker=tracker,
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
    eval_interval: int,
    tracker: ComputeTracker,
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
            eval_interval=eval_interval,
            tracker=tracker,
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
            eval_interval=eval_interval,
            tracker=tracker,
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
    eval_interval: int,
    tracker: ComputeTracker,
) -> tuple[EMOResult, float]:
    def residuals(cap: float) -> ResidualBuilder:
        return lambda losses: [
            ("f1", losses.f1 - losses.f1.new_tensor(e1 + epsilon)),
            ("f2", losses.f2 - losses.f2.new_tensor(e2 + epsilon + cap)),
            ("f3", losses.f3 - losses.f3.new_tensor(z1 + cap)),
        ]

    model.load_state_dict(start_state)
    start_metrics = _tracked_eval(
        model,
        train_set,
        scales,
        tracker=tracker,
        phase="final",
        reason="oracle_setup",
        batch_size=batch_size,
        device=device,
        extra={"oracle": oracle_name, "z1": z1},
    )
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
        eval_interval=eval_interval,
        tracker=tracker,
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
    eval_interval: int,
    tracker: ComputeTracker,
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
            eval_interval=eval_interval,
            tracker=tracker,
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
            eval_interval=eval_interval,
            tracker=tracker,
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
    eval_interval: int,
    tracker: ComputeTracker,
) -> EMOResult:
    """Run Eq. (11): update with the objective behind the largest residual."""

    _check_budget(
        model,
        train_set,
        scales,
        tracker=tracker,
        phase=oracle_name,
        batch_size=batch_size,
        device=device,
    )
    model.load_state_dict(start_state)
    tracker.oracle_calls += 1
    optimizer = _make_adam(model, lr)
    batches = _cycle(_loader(train_set, batch_size, loader_seed))
    history: list[dict[str, Any]] = []

    if condition_check_interval <= 0:
        raise ValueError("condition_check_interval must be positive")

    stopped_early = False
    for update in range(1, inner_updates + 1):
        _check_budget(
            model,
            train_set,
            scales,
            tracker=tracker,
            phase=oracle_name,
            batch_size=batch_size,
            device=device,
        )
        batch = next(batches)
        losses = _batch_losses(model, batch, scales, device)
        residuals = residual_builder(losses)
        values = torch.stack([value for _, value in residuals])
        active_idx = int(torch.argmax(values.detach()).item())
        active_name, _ = residuals[active_idx]
        _adam_step(optimizer, losses.objective(active_name), tracker, batch["tokens"].shape[0])

        batch_condition = False
        if update % condition_check_interval == 0:
            tracker.condition_checks += 1
            tracker.condition_check_prompts += int(batch["tokens"].shape[0])
            post_losses = _batch_losses(model, batch, scales, device)
            post_residuals = residual_builder(post_losses)
            post_values = torch.stack([value for _, value in post_residuals])
            batch_condition = float(post_values.detach().max().cpu().item()) <= 0.0
        scheduled = tracker.backward_calls % eval_interval == 0
        oracle_scheduled = oracle_eval_interval > 0 and _should_eval(update, oracle_eval_interval, inner_updates)
        if batch_condition or scheduled or oracle_scheduled:
            reasons = []
            if scheduled:
                reasons.append("scheduled")
            if batch_condition:
                reasons.append("feasibility_check")
            if oracle_scheduled:
                reasons.append("oracle_interval")
            metrics = _tracked_eval(
                model,
                train_set,
                scales,
                tracker=tracker,
                phase=_oracle_phase(oracle_name),
                reason=",".join(reasons),
                batch_size=batch_size,
                device=device,
                threshold_evaluation=_oracle_phase(oracle_name) == "level2",
                extra={"oracle": oracle_name, "cap": cap, "active": active_name},
            )
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

    metrics = _metrics_at_current_call(tracker)
    if metrics is None:
        metrics = _tracked_eval(
            model,
            train_set,
            scales,
            tracker=tracker,
            phase=_oracle_phase(oracle_name),
            reason="oracle_final",
            batch_size=batch_size,
            device=device,
            threshold_evaluation=_oracle_phase(oracle_name) == "level2",
            extra={"oracle": oracle_name, "cap": cap},
        )
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


def _adam_step(
    optimizer: torch.optim.Optimizer,
    objective: torch.Tensor,
    tracker: ComputeTracker,
    prompts: int,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    objective.backward()
    optimizer.step()
    tracker.record_backward(prompts)


def _tracked_eval(
    model: CausalCoTTransformer,
    train_set: SyntheticCoTDataset,
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
    return tracked_teacher_forced_evaluation(
        model,
        train_set,
        scales,
        tracker=tracker,
        phase=phase,
        reason=reason,
        batch_size=batch_size,
        device=device,
        threshold_evaluation=threshold_evaluation,
        extra=extra,
    )


def _check_budget(
    model: CausalCoTTransformer,
    train_set: SyntheticCoTDataset,
    scales: TargetScales,
    *,
    tracker: ComputeTracker,
    phase: str,
    batch_size: int,
    device: torch.device | str,
) -> None:
    if tracker.can_step():
        return
    metrics = _metrics_at_current_call(tracker)
    if metrics is None:
        metrics = _tracked_eval(
            model,
            train_set,
            scales,
            tracker=tracker,
            phase=_oracle_phase(phase),
            reason="budget_exhausted",
            batch_size=batch_size,
            device=device,
            threshold_evaluation=_oracle_phase(phase) in ("level1", "level2"),
        )
    raise EMOBudgetExhausted(state=_clone_state(model), metrics=metrics, phase=phase)


def _metrics_at_current_call(tracker: ComputeTracker) -> dict[str, float] | None:
    if not tracker.history or tracker.history[-1].get("backward_calls") != tracker.backward_calls:
        return None
    row = tracker.history[-1]
    keys = ("e1", "e2", "e3", "f1", "f2", "f3")
    if not all(key in row for key in keys):
        return None
    return {key: float(row[key]) for key in keys}


def _oracle_phase(oracle_name: str) -> str:
    if oracle_name == "level1":
        return "level1"
    if oracle_name == "level2":
        return "level2"
    return "final"


def _should_eval(update: int, interval: int, total_updates: int) -> bool:
    return update % interval == 0 or update == total_updates


def _checkpoint(model: CausalCoTTransformer, metrics: dict[str, float]) -> dict[str, Any]:
    return {"state": _clone_state(model), "metrics": copy.deepcopy(metrics)}


def _clone_state(model: CausalCoTTransformer) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
