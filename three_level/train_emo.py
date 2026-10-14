from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .data import SyntheticCoTDataset, TargetScales, compute_target_scales
from .emo import Constraint, minimize_single_objective, solve_epigraph_level
from .evaluate import evaluate_teacher_forced
from .model import CausalCoTTransformer


@dataclass
class EMOConfig:
    train_size: int = 8192
    val_size: int = 1024
    test_size: int = 256
    batch_size: int = 32
    lr: float = 1e-4
    seed: int = 11
    eval_interval: int = 250
    stage1_updates: int = 1000
    inner_updates: int = 1500
    bisection_steps: int = 6
    tolerance: float = 0.02
    device: str = "cpu"


@dataclass
class EMOArtifacts:
    method: str
    selected_state: dict[str, torch.Tensor]
    last_state: dict[str, torch.Tensor]
    thresholds: dict[str, float]
    selected_metrics: dict[str, float]
    last_metrics: dict[str, float]
    history: list[dict[str, Any]]
    scales: TargetScales


def run_emo(cfg: EMOConfig) -> EMOArtifacts:
    _set_seed(cfg.seed)
    device = torch.device(cfg.device)
    train_set = SyntheticCoTDataset(cfg.train_size, seed=10_000 + cfg.seed, sample_demo_count=True)
    scales = compute_target_scales(train_set)
    model = CausalCoTTransformer().to(device)

    stage1 = minimize_single_objective(
        model,
        train_set,
        scales,
        objective="f1",
        updates=cfg.stage1_updates,
        eval_interval=cfg.eval_interval,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        device=device,
    )
    model.load_state_dict(stage1.state)
    b1 = stage1.metrics["f1"] + cfg.tolerance

    start_stage2 = evaluate_teacher_forced(model, train_set, scales, batch_size=cfg.batch_size, device=device)
    stage2 = solve_epigraph_level(
        model,
        train_set,
        scales,
        start_state=stage1.state,
        target_objective="f2",
        constraints=[Constraint("f1", b1)],
        upper_cap=start_stage2["f2"],
        bisection_steps=cfg.bisection_steps,
        inner_updates=cfg.inner_updates,
        eval_interval=cfg.eval_interval,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        device=device,
    )
    model.load_state_dict(stage2.state)
    b2 = stage2.metrics["f2"] + cfg.tolerance

    start_stage3 = evaluate_teacher_forced(model, train_set, scales, batch_size=cfg.batch_size, device=device)
    stage3 = solve_epigraph_level(
        model,
        train_set,
        scales,
        start_state=stage2.state,
        target_objective="f3",
        constraints=[Constraint("f1", b1), Constraint("f2", b2)],
        upper_cap=start_stage3["f3"],
        bisection_steps=cfg.bisection_steps,
        inner_updates=cfg.inner_updates,
        eval_interval=cfg.eval_interval,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        device=device,
    )
    model.load_state_dict(stage3.state)
    last_state = _clone_state(model)
    last_metrics = evaluate_teacher_forced(model, train_set, scales, batch_size=cfg.batch_size, device=device)

    history: list[dict[str, Any]] = []
    history.extend({"emo_stage": "stage1", **item} for item in stage1.history)
    history.extend({"emo_stage": "stage2", **item} for item in stage2.history)
    history.extend({"emo_stage": "stage3", **item} for item in stage3.history)

    return EMOArtifacts(
        method="emo",
        selected_state=stage3.state,
        last_state=last_state,
        thresholds={"b1": b1, "b2": b2, "stage2_cap": stage2.cap, "stage3_cap": stage3.cap},
        selected_metrics=stage3.metrics,
        last_metrics=last_metrics,
        history=history,
        scales=scales,
    )


def save_artifacts(path: Path, artifacts: EMOArtifacts, cfg: EMOConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": artifacts.method,
            "selected_state": artifacts.selected_state,
            "last_state": artifacts.last_state,
            "thresholds": artifacts.thresholds,
            "selected_metrics": artifacts.selected_metrics,
            "last_metrics": artifacts.last_metrics,
            "history": artifacts.history,
            "scales": asdict(artifacts.scales),
            "config": asdict(cfg),
        },
        path,
    )


def _clone_state(model: CausalCoTTransformer) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the epigraph-based multilevel optimization method.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("runs/emo_seed11.pt"))
    parser.add_argument("--train-size", type=int, default=8192)
    parser.add_argument("--val-size", type=int, default=1024)
    parser.add_argument("--test-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--stage1-updates", type=int, default=1000)
    parser.add_argument("--inner-updates", type=int, default=1500)
    parser.add_argument("--bisection-steps", type=int, default=6)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--quick", action="store_true", help="Use tiny sizes and update counts for smoke testing.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = EMOConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        seed=args.seed,
        eval_interval=args.eval_interval,
        stage1_updates=args.stage1_updates,
        inner_updates=args.inner_updates,
        bisection_steps=args.bisection_steps,
        tolerance=args.tolerance,
        device=args.device,
    )
    if args.quick:
        cfg.train_size = 96
        cfg.val_size = 32
        cfg.test_size = 32
        cfg.batch_size = min(cfg.batch_size, 16)
        cfg.eval_interval = 2
        cfg.stage1_updates = 4
        cfg.inner_updates = 4
        cfg.bisection_steps = 2

    start = time.time()
    artifacts = run_emo(cfg)
    save_artifacts(args.output, artifacts, cfg)
    print(
        json.dumps(
            {
                "method": artifacts.method,
                "thresholds": artifacts.thresholds,
                "selected_metrics": artifacts.selected_metrics,
                "last_metrics": artifacts.last_metrics,
                "elapsed_sec": time.time() - start,
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
