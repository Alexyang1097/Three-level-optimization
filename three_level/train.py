from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .data import SyntheticCoTDataset, TargetScales, collate_teacher_forcing, compute_target_scales
from .model import CausalCoTTransformer
from .objectives import OBJECTIVE_FAMILIES, Losses, compute_losses
from .tracking import ComputeTracker, record_metric_snapshot, synchronized_time, tracked_teacher_forced_evaluation


@dataclass
class ExperimentConfig:
    train_size: int = 8192
    val_size: int = 1024
    test_size: int = 256
    batch_size: int = 32
    lr: float = 1e-4
    seed: int = 11
    data_seed: int = 10_000
    test_seed: int = 30_000
    eval_interval: int = 250
    stage1_updates: int = 1000
    stage2_updates: int = 1500
    stage3_updates: int = 1500
    uniform_updates: int = 4000
    tolerance: float = 0.02
    stage2_margin: float = 0.002
    dual_nu: float = 0.02
    dual_omega: float = 0.05
    device: str = "cpu"
    objective_family: str = "cumulative"

    def __post_init__(self) -> None:
        if self.objective_family not in OBJECTIVE_FAMILIES:
            raise ValueError(f"Unknown objective family {self.objective_family!r}")


@dataclass
class TrainArtifacts:
    method: str
    selected_state: dict[str, torch.Tensor]
    last_state: dict[str, torch.Tensor]
    thresholds: dict[str, float]
    selected_metrics: dict[str, float]
    last_metrics: dict[str, float]
    history: list[dict[str, Any]]
    scales: TargetScales
    compute: dict[str, Any]
    status: str = "complete"
    dual_history: list[dict[str, Any]] = field(default_factory=list)


def run_method(method: str, cfg: ExperimentConfig) -> TrainArtifacts:
    _set_seed(cfg.seed)
    device = torch.device(cfg.device)
    tracker = ComputeTracker()
    data_started = synchronized_time(device)
    train_set = SyntheticCoTDataset(cfg.train_size, seed=cfg.data_seed, sample_demo_count=True)
    scales = compute_target_scales(train_set)
    tracker.training_data_generation_sec = synchronized_time(device) - data_started
    model_started = synchronized_time(device)
    model = CausalCoTTransformer().to(device)
    tracker.model_setup_sec = synchronized_time(device) - model_started
    started = synchronized_time(device)

    if method == "uniform":
        artifacts = train_uniform(model, train_set, scales, cfg, tracker)
    elif method == "staged":
        artifacts = train_staged_unconstrained(model, train_set, scales, cfg, tracker)
    elif method == "constrained":
        artifacts = train_staged_constrained(model, train_set, scales, cfg, tracker)
    else:
        raise ValueError(f"Unknown method {method!r}")

    tracker.elapsed_train_sec = synchronized_time(device) - started
    artifacts.compute = tracker.summary()
    return artifacts


def train_uniform(
    model: CausalCoTTransformer,
    train_set: SyntheticCoTDataset,
    scales: TargetScales,
    cfg: ExperimentConfig,
    tracker: ComputeTracker,
) -> TrainArtifacts:
    state = _new_train_state(model, train_set, cfg)
    initial = _tracked_eval(model, train_set, scales, cfg, tracker, phase="uniform", reason="initial,scheduled")
    best = _checkpoint(model, initial)
    last_metrics = initial

    optimizer = _make_adam(model, cfg)
    for update in range(1, cfg.uniform_updates + 1):
        batch = next(state["batches"])
        losses = _batch_losses(model, batch, scales, cfg.device, cfg.objective_family)
        _adam_step(optimizer, losses.joint, tracker, batch["tokens"].shape[0])
        if _should_eval(update, cfg.eval_interval, cfg.uniform_updates):
            last_metrics = _tracked_eval(
                model,
                train_set,
                scales,
                cfg,
                tracker,
                phase="uniform",
                reason=_eval_reason(update, cfg.uniform_updates),
            )
            if _uniform_score(last_metrics, cfg) < _uniform_score(best["metrics"], cfg):
                best = _checkpoint(model, last_metrics)

    return TrainArtifacts(
        "uniform",
        best["state"],
        _clone_state(model),
        {},
        best["metrics"],
        last_metrics,
        tracker.history,
        scales,
        {},
    )


def train_staged_unconstrained(
    model: CausalCoTTransformer,
    train_set: SyntheticCoTDataset,
    scales: TargetScales,
    cfg: ExperimentConfig,
    tracker: ComputeTracker,
) -> TrainArtifacts:
    state = _new_train_state(model, train_set, cfg)
    initial = _tracked_eval(model, train_set, scales, cfg, tracker, phase="stage1", reason="initial,scheduled")
    best = _checkpoint(model, initial)
    last_metrics = initial

    for stage_name, objective_name, updates in [
        ("stage1", "f1", cfg.stage1_updates),
        ("stage2", "f2", cfg.stage2_updates),
        ("stage3", "f3", cfg.stage3_updates),
    ]:
        optimizer = _make_adam(model, cfg)
        for local_update in range(1, updates + 1):
            batch = next(state["batches"])
            losses = _batch_losses(model, batch, scales, cfg.device, cfg.objective_family)
            _adam_step(optimizer, losses.objective(objective_name), tracker, batch["tokens"].shape[0])
            if _should_eval(local_update, cfg.eval_interval, updates):
                last_metrics = _tracked_eval(
                    model,
                    train_set,
                    scales,
                    cfg,
                    tracker,
                    phase=stage_name,
                    reason=_eval_reason(local_update, updates),
                )
                if last_metrics["f3"] < best["metrics"]["f3"]:
                    best = _checkpoint(model, last_metrics)

    return TrainArtifacts(
        "staged",
        best["state"],
        _clone_state(model),
        {},
        best["metrics"],
        last_metrics,
        tracker.history,
        scales,
        {},
    )


def train_staged_constrained(
    model: CausalCoTTransformer,
    train_set: SyntheticCoTDataset,
    scales: TargetScales,
    cfg: ExperimentConfig,
    tracker: ComputeTracker,
) -> TrainArtifacts:
    state = _new_train_state(model, train_set, cfg)
    dual_history: list[dict[str, Any]] = []

    initial_metrics = _tracked_eval(
        model,
        train_set,
        scales,
        cfg,
        tracker,
        phase="stage1",
        reason="initial,scheduled,threshold_reference",
        threshold_evaluation=True,
    )
    theta1 = _checkpoint(model, initial_metrics)
    last_metrics = initial_metrics

    optimizer = _make_adam(model, cfg)
    for local_update in range(1, cfg.stage1_updates + 1):
        batch = next(state["batches"])
        losses = _batch_losses(model, batch, scales, cfg.device, cfg.objective_family)
        _adam_step(optimizer, losses.f1, tracker, batch["tokens"].shape[0])
        if _should_eval(local_update, cfg.eval_interval, cfg.stage1_updates):
            last_metrics = _tracked_eval(
                model,
                train_set,
                scales,
                cfg,
                tracker,
                phase="stage1",
                reason=_eval_reason(local_update, cfg.stage1_updates, threshold_reference=True),
                threshold_evaluation=True,
            )
            if last_metrics["f1"] < theta1["metrics"]["f1"]:
                theta1 = _checkpoint(model, last_metrics)

    model.load_state_dict(theta1["state"])
    b1 = theta1["metrics"]["f1"] + cfg.tolerance
    record_metric_snapshot(
        tracker,
        theta1["metrics"],
        phase="stage2",
        reason="stage_transition",
        extra={"b1": b1},
    )

    theta2 = _checkpoint(model, theta1["metrics"])
    lambda_stage2 = torch.zeros(1, device=cfg.device)
    optimizer = _make_adam(model, cfg)
    for local_update in range(1, cfg.stage2_updates + 1):
        batch = next(state["batches"])
        losses = _batch_losses(model, batch, scales, cfg.device, cfg.objective_family)
        lambda_before = lambda_stage2
        lambda_stage2 = _dual_update(lambda_stage2, [losses.f1], [b1], cfg)
        _record_dual_feedback(
            dual_history, lambda_before, lambda_stage2, [losses.f1], [b1], cfg,
            tracker=tracker, phase="stage2", local_update=local_update,
            batch_size=batch["tokens"].shape[0],
        )
        objective = losses.f2 + lambda_stage2[0].detach() * losses.f1
        _adam_step(optimizer, objective, tracker, batch["tokens"].shape[0])
        if _should_eval(local_update, cfg.eval_interval, cfg.stage2_updates):
            last_metrics = _tracked_eval(
                model,
                train_set,
                scales,
                cfg,
                tracker,
                phase="stage2",
                reason=_eval_reason(local_update, cfg.stage2_updates, threshold_reference=True),
                threshold_evaluation=True,
                extra={"lambda1": lambda_stage2.item(), "b1": b1},
            )
            if last_metrics["f1"] <= b1 - cfg.stage2_margin and last_metrics["f2"] < theta2["metrics"]["f2"]:
                theta2 = _checkpoint(model, last_metrics)

    model.load_state_dict(theta2["state"])
    b2 = theta2["metrics"]["f2"] + cfg.tolerance
    record_metric_snapshot(
        tracker,
        theta2["metrics"],
        phase="stage3",
        reason="stage_transition",
        extra={"b1": b1, "b2": b2},
    )

    selected = _checkpoint(model, theta2["metrics"])
    lambda_stage3 = torch.zeros(2, device=cfg.device)
    optimizer = _make_adam(model, cfg)
    for local_update in range(1, cfg.stage3_updates + 1):
        batch = next(state["batches"])
        losses = _batch_losses(model, batch, scales, cfg.device, cfg.objective_family)
        lambda_before = lambda_stage3
        lambda_stage3 = _dual_update(lambda_stage3, [losses.f1, losses.f2], [b1, b2], cfg)
        _record_dual_feedback(
            dual_history, lambda_before, lambda_stage3, [losses.f1, losses.f2], [b1, b2], cfg,
            tracker=tracker, phase="stage3", local_update=local_update,
            batch_size=batch["tokens"].shape[0],
        )
        objective = losses.f3 + lambda_stage3[0].detach() * losses.f1 + lambda_stage3[1].detach() * losses.f2
        _adam_step(optimizer, objective, tracker, batch["tokens"].shape[0])
        if _should_eval(local_update, cfg.eval_interval, cfg.stage3_updates):
            last_metrics = _tracked_eval(
                model,
                train_set,
                scales,
                cfg,
                tracker,
                phase="stage3",
                reason=_eval_reason(local_update, cfg.stage3_updates),
                extra={
                    "lambda1": lambda_stage3[0].item(),
                    "lambda2": lambda_stage3[1].item(),
                    "b1": b1,
                    "b2": b2,
                },
            )
            if (
                last_metrics["f1"] <= b1
                and last_metrics["f2"] <= b2
                and last_metrics["f3"] < selected["metrics"]["f3"]
            ):
                selected = _checkpoint(model, last_metrics)

    return TrainArtifacts(
        "constrained",
        selected["state"],
        _clone_state(model),
        {"b1": b1, "b2": b2},
        selected["metrics"],
        last_metrics,
        tracker.history,
        scales,
        {},
        dual_history=dual_history,
    )


def save_artifacts(path: Path, artifacts: TrainArtifacts, cfg: ExperimentConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": artifacts.method,
        "selected_state": artifacts.selected_state,
        "last_state": artifacts.last_state,
        "thresholds": artifacts.thresholds,
        "selected_metrics": artifacts.selected_metrics,
        "last_metrics": artifacts.last_metrics,
        "history": artifacts.history,
        "scales": asdict(artifacts.scales),
        "compute": artifacts.compute,
        "status": artifacts.status,
        "config": asdict(cfg),
        "selection_objective": "mean(e1,e2,e3)" if artifacts.method == "uniform" else "f3",
    }
    if artifacts.method == "constrained":
        payload["dual_history"] = artifacts.dual_history
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _new_train_state(model: CausalCoTTransformer, train_set: SyntheticCoTDataset, cfg: ExperimentConfig) -> dict[str, Any]:
    loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_teacher_forcing,
    )
    return {"batches": _cycle(loader), "model": model}


def _cycle(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def _batch_losses(
    model: CausalCoTTransformer,
    batch: dict[str, torch.Tensor],
    scales: TargetScales,
    device: torch.device | str,
    objective_family: str = "cumulative",
) -> Losses:
    batch = {key: value.to(device) for key, value in batch.items()}
    outputs = model(batch["tokens"], batch["types"], batch["padding_mask"])
    return compute_losses(outputs, batch, scales, objective_family=objective_family)


def _uniform_score(metrics: dict[str, float], cfg: ExperimentConfig) -> float:
    if cfg.objective_family == "cumulative":
        return metrics["f3"]
    return (metrics["e1"] + metrics["e2"] + metrics["e3"]) / 3.0


def _dual_update(
    lambdas: torch.Tensor,
    constraints: list[torch.Tensor],
    thresholds: list[float],
    cfg: ExperimentConfig,
) -> torch.Tensor:
    if cfg.dual_nu <= 0:
        raise ValueError("dual_nu must be positive")
    values = torch.stack(
        [
            constraint.detach() - constraint.new_tensor(threshold)
            for constraint, threshold in zip(constraints, thresholds)
        ]
    )
    return torch.relu(
        (1.0 - cfg.dual_omega) * lambdas
        + (cfg.dual_omega / cfg.dual_nu) * values
    )


@torch.no_grad()
def _record_dual_feedback(
    history: list[dict[str, Any]],
    before: torch.Tensor,
    after: torch.Tensor,
    constraints: list[torch.Tensor],
    thresholds: list[float],
    cfg: ExperimentConfig,
    *,
    tracker: ComputeTracker,
    phase: str,
    local_update: int,
    batch_size: int,
) -> None:
    """Observe existing minibatch signals before Adam, without model/oracle work."""
    f_hat = torch.stack([value.detach() for value in constraints])
    c_hat = f_hat - f_hat.new_tensor(thresholds)
    q = c_hat - cfg.dual_nu * before
    unprojected = (1.0 - cfg.dual_omega) * before + (cfg.dual_omega / cfg.dual_nu) * c_hat
    # One device transfer per update, with no retained autograd graph.
    values = torch.stack((f_hat, c_hat, before, q, after, after - before, unprojected)).cpu().tolist()
    row: dict[str, Any] = {
        "backward_calls": tracker.backward_calls + 1,
        "backward_calls_before": tracker.backward_calls,
        "optimizer_updates": tracker.optimizer_updates + 1,
        "phase": phase,
        "local_update": local_update,
        "dual_reset": local_update == 1,
        "batch_size": int(batch_size),
        "signal_scope": "minibatch",
        "objective_family": cfg.objective_family,
        "signal_timing": "before_primal_update",
        "dual_nu": cfg.dual_nu,
        "dual_omega": cfg.dual_omega,
    }
    names = ("f_hat", "c_hat", "lambda_before", "q", "lambda_after", "delta_lambda", "lambda_unprojected")
    for index, threshold in enumerate(thresholds):
        suffix = index + 1
        row[f"b{suffix}"] = threshold
        for name, vector in zip(names, values):
            row[f"{name}{suffix}"] = vector[index]
        row[f"projection_active{suffix}"] = values[-1][index] < 0.0
    history.append(row)


def _make_adam(model: CausalCoTTransformer, cfg: ExperimentConfig) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)


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
    cfg: ExperimentConfig,
    tracker: ComputeTracker,
    *,
    phase: str,
    reason: str,
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
        batch_size=cfg.batch_size,
        device=cfg.device,
        threshold_evaluation=threshold_evaluation,
        extra=extra,
        objective_family=cfg.objective_family,
    )


def _eval_reason(local_update: int, stage_updates: int, *, threshold_reference: bool = False) -> str:
    reasons = ["scheduled"]
    if local_update == stage_updates:
        reasons.append("stage_final")
    if threshold_reference:
        reasons.append("threshold_reference")
    return ",".join(reasons)


def _should_eval(local_update: int, interval: int, stage_updates: int) -> bool:
    return local_update % interval == 0 or local_update == stage_updates


def _checkpoint(model: CausalCoTTransformer, metrics: dict[str, float]) -> dict[str, Any]:
    return {"state": _clone_state(model), "metrics": copy.deepcopy(metrics)}


def _clone_state(model: CausalCoTTransformer) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Section 7 three-level synthetic CoT methods.")
    parser.add_argument("--method", choices=["uniform", "staged", "constrained"], default="constrained")
    parser.add_argument("--objective-family", choices=OBJECTIVE_FAMILIES, default="cumulative")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--data-seed", type=int, default=10_000)
    parser.add_argument("--test-seed", type=int, default=30_000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("runs/three_level.pt"))
    parser.add_argument("--train-size", type=int, default=8192)
    parser.add_argument("--val-size", type=int, default=1024)
    parser.add_argument("--test-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--stage1-updates", type=int, default=1000)
    parser.add_argument("--stage2-updates", type=int, default=1500)
    parser.add_argument("--stage3-updates", type=int, default=1500)
    parser.add_argument("--uniform-updates", type=int, default=4000)
    parser.add_argument("--quick", action="store_true", help="Use tiny sizes and update counts for smoke testing.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = ExperimentConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        seed=args.seed,
        data_seed=args.data_seed,
        test_seed=args.test_seed,
        eval_interval=args.eval_interval,
        stage1_updates=args.stage1_updates,
        stage2_updates=args.stage2_updates,
        stage3_updates=args.stage3_updates,
        uniform_updates=args.uniform_updates,
        device=args.device,
        objective_family=args.objective_family,
    )
    if args.quick:
        cfg.train_size = 96
        cfg.val_size = 32
        cfg.test_size = 32
        cfg.batch_size = min(cfg.batch_size, 16)
        cfg.eval_interval = 5
        cfg.stage1_updates = 5
        cfg.stage2_updates = 5
        cfg.stage3_updates = 5
        cfg.uniform_updates = 15

    start = time.time()
    artifacts = run_method(args.method, cfg)
    save_artifacts(args.output, artifacts, cfg)
    elapsed = time.time() - start
    summary = {
        "method": artifacts.method,
        "objective_family": cfg.objective_family,
        "status": artifacts.status,
        "selection_objective": "mean(e1,e2,e3)" if artifacts.method == "uniform" else "f3",
        "thresholds": artifacts.thresholds,
        "selected_metrics": artifacts.selected_metrics,
        "last_metrics": artifacts.last_metrics,
        "compute": artifacts.compute,
        "elapsed_sec": elapsed,
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
