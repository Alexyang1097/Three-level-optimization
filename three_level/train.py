from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .data import SyntheticCoTDataset, TargetScales, collate_teacher_forcing, compute_target_scales
from .evaluate import evaluate_teacher_forced
from .model import CausalCoTTransformer
from .objectives import Losses, compute_losses


@dataclass
class ExperimentConfig:
    train_size: int = 8192
    val_size: int = 1024
    test_size: int = 256
    batch_size: int = 32
    lr: float = 1e-4
    seed: int = 11
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


def run_method(method: str, cfg: ExperimentConfig) -> TrainArtifacts:
    _set_seed(cfg.seed)
    device = torch.device(cfg.device)
    train_set = SyntheticCoTDataset(cfg.train_size, seed=10_000 + cfg.seed, sample_demo_count=True)
    scales = compute_target_scales(train_set)
    model = CausalCoTTransformer().to(device)

    if method == "uniform":
        return train_uniform(model, train_set, scales, cfg)
    if method == "staged":
        return train_staged_unconstrained(model, train_set, scales, cfg)
    if method == "constrained":
        return train_staged_constrained(model, train_set, scales, cfg)
    raise ValueError(f"Unknown method {method!r}")


def train_uniform(
    model: CausalCoTTransformer,
    train_set: SyntheticCoTDataset,
    scales: TargetScales,
    cfg: ExperimentConfig,
) -> TrainArtifacts:
    state = _new_train_state(model, train_set, cfg)
    history: list[dict[str, Any]] = []
    best = _checkpoint(model, evaluate_teacher_forced(model, train_set, scales, batch_size=cfg.batch_size, device=cfg.device))
    history.append({"update": 0, "stage": "uniform", **best["metrics"]})

    optimizer = _make_adam(model, cfg)
    for update in range(1, cfg.uniform_updates + 1):
        batch = next(state["batches"])
        losses = _batch_losses(model, batch, scales, cfg.device)
        _adam_step(optimizer, losses.f3)
        if _should_eval(update, cfg.eval_interval, cfg.uniform_updates):
            metrics = evaluate_teacher_forced(model, train_set, scales, batch_size=cfg.batch_size, device=cfg.device)
            history.append({"update": update, "stage": "uniform", **metrics})
            if metrics["f3"] < best["metrics"]["f3"]:
                best = _checkpoint(model, metrics)

    last_metrics = evaluate_teacher_forced(model, train_set, scales, batch_size=cfg.batch_size, device=cfg.device)
    return TrainArtifacts("uniform", best["state"], _clone_state(model), {}, best["metrics"], last_metrics, history, scales)


def train_staged_unconstrained(
    model: CausalCoTTransformer,
    train_set: SyntheticCoTDataset,
    scales: TargetScales,
    cfg: ExperimentConfig,
) -> TrainArtifacts:
    state = _new_train_state(model, train_set, cfg)
    history: list[dict[str, Any]] = []
    best = _checkpoint(model, evaluate_teacher_forced(model, train_set, scales, batch_size=cfg.batch_size, device=cfg.device))
    history.append({"update": 0, "stage": "init", **best["metrics"]})
    total_update = 0

    for stage_name, objective_name, updates in [
        ("stage1", "f1", cfg.stage1_updates),
        ("stage2", "f2", cfg.stage2_updates),
        ("stage3", "f3", cfg.stage3_updates),
    ]:
        optimizer = _make_adam(model, cfg)
        for local_update in range(1, updates + 1):
            total_update += 1
            batch = next(state["batches"])
            losses = _batch_losses(model, batch, scales, cfg.device)
            _adam_step(optimizer, losses.objective(objective_name))
            if _should_eval(local_update, cfg.eval_interval, updates):
                metrics = evaluate_teacher_forced(model, train_set, scales, batch_size=cfg.batch_size, device=cfg.device)
                history.append({"update": total_update, "stage": stage_name, **metrics})
                if metrics["f3"] < best["metrics"]["f3"]:
                    best = _checkpoint(model, metrics)

    last_metrics = evaluate_teacher_forced(model, train_set, scales, batch_size=cfg.batch_size, device=cfg.device)
    return TrainArtifacts("staged", best["state"], _clone_state(model), {}, best["metrics"], last_metrics, history, scales)


def train_staged_constrained(
    model: CausalCoTTransformer,
    train_set: SyntheticCoTDataset,
    scales: TargetScales,
    cfg: ExperimentConfig,
) -> TrainArtifacts:
    state = _new_train_state(model, train_set, cfg)
    history: list[dict[str, Any]] = []
    total_update = 0

    initial_metrics = evaluate_teacher_forced(model, train_set, scales, batch_size=cfg.batch_size, device=cfg.device)
    theta1 = _checkpoint(model, initial_metrics)
    history.append({"update": 0, "stage": "init", **initial_metrics})

    optimizer = _make_adam(model, cfg)
    for local_update in range(1, cfg.stage1_updates + 1):
        total_update += 1
        batch = next(state["batches"])
        losses = _batch_losses(model, batch, scales, cfg.device)
        _adam_step(optimizer, losses.f1)
        if _should_eval(local_update, cfg.eval_interval, cfg.stage1_updates):
            metrics = evaluate_teacher_forced(model, train_set, scales, batch_size=cfg.batch_size, device=cfg.device)
            history.append({"update": total_update, "stage": "stage1", **metrics})
            if metrics["f1"] < theta1["metrics"]["f1"]:
                theta1 = _checkpoint(model, metrics)

    model.load_state_dict(theta1["state"])
    b1 = theta1["metrics"]["f1"] + cfg.tolerance

    theta2 = _checkpoint(model, theta1["metrics"])
    lambda_stage2 = torch.zeros(1, device=cfg.device)
    optimizer = _make_adam(model, cfg)
    for local_update in range(1, cfg.stage2_updates + 1):
        total_update += 1
        batch = next(state["batches"])
        losses = _batch_losses(model, batch, scales, cfg.device)
        lambda_stage2 = _dual_update(lambda_stage2, [losses.f1], [b1], cfg)
        objective = losses.f2 + lambda_stage2[0].detach() * losses.f1
        _adam_step(optimizer, objective)
        if _should_eval(local_update, cfg.eval_interval, cfg.stage2_updates):
            metrics = evaluate_teacher_forced(model, train_set, scales, batch_size=cfg.batch_size, device=cfg.device)
            history.append({"update": total_update, "stage": "stage2", "lambda1": lambda_stage2.item(), "b1": b1, **metrics})
            if metrics["f1"] <= b1 - cfg.stage2_margin and metrics["f2"] < theta2["metrics"]["f2"]:
                theta2 = _checkpoint(model, metrics)

    model.load_state_dict(theta2["state"])
    b2 = theta2["metrics"]["f2"] + cfg.tolerance

    selected = _checkpoint(model, theta2["metrics"])
    lambda_stage3 = torch.zeros(2, device=cfg.device)
    optimizer = _make_adam(model, cfg)
    for local_update in range(1, cfg.stage3_updates + 1):
        total_update += 1
        batch = next(state["batches"])
        losses = _batch_losses(model, batch, scales, cfg.device)
        lambda_stage3 = _dual_update(lambda_stage3, [losses.f1, losses.f2], [b1, b2], cfg)
        objective = losses.f3 + lambda_stage3[0].detach() * losses.f1 + lambda_stage3[1].detach() * losses.f2
        _adam_step(optimizer, objective)
        if _should_eval(local_update, cfg.eval_interval, cfg.stage3_updates):
            metrics = evaluate_teacher_forced(model, train_set, scales, batch_size=cfg.batch_size, device=cfg.device)
            history.append(
                {
                    "update": total_update,
                    "stage": "stage3",
                    "lambda1": lambda_stage3[0].item(),
                    "lambda2": lambda_stage3[1].item(),
                    "b1": b1,
                    "b2": b2,
                    **metrics,
                }
            )
            if metrics["f1"] <= b1 and metrics["f2"] <= b2 and metrics["f3"] < selected["metrics"]["f3"]:
                selected = _checkpoint(model, metrics)

    last_metrics = evaluate_teacher_forced(model, train_set, scales, batch_size=cfg.batch_size, device=cfg.device)
    return TrainArtifacts(
        "constrained",
        selected["state"],
        _clone_state(model),
        {"b1": b1, "b2": b2},
        selected["metrics"],
        last_metrics,
        history,
        scales,
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
        "config": asdict(cfg),
    }
    torch.save(payload, path)


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
) -> Losses:
    batch = {key: value.to(device) for key, value in batch.items()}
    outputs = model(batch["tokens"], batch["types"], batch["padding_mask"])
    return compute_losses(outputs, batch, scales)


def _dual_update(
    lambdas: torch.Tensor,
    constraints: list[torch.Tensor],
    thresholds: list[float],
    cfg: ExperimentConfig,
) -> torch.Tensor:
    values = torch.stack([constraint.detach() - constraint.new_tensor(threshold) for constraint, threshold in zip(constraints, thresholds)])
    return torch.relu((1.0 - cfg.dual_omega) * lambdas + cfg.dual_omega * cfg.dual_nu * values)


def _make_adam(model: CausalCoTTransformer, cfg: ExperimentConfig) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)


def _adam_step(optimizer: torch.optim.Optimizer, objective: torch.Tensor) -> None:
    optimizer.zero_grad(set_to_none=True)
    objective.backward()
    optimizer.step()


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
    parser.add_argument("--seed", type=int, default=11)
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
        eval_interval=args.eval_interval,
        stage1_updates=args.stage1_updates,
        stage2_updates=args.stage2_updates,
        stage3_updates=args.stage3_updates,
        uniform_updates=args.uniform_updates,
        device=args.device,
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
        "thresholds": artifacts.thresholds,
        "selected_metrics": artifacts.selected_metrics,
        "last_metrics": artifacts.last_metrics,
        "elapsed_sec": elapsed,
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
