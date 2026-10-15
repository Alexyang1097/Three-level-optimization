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
from .emo import minimize_level1, solve_final_nested, solve_level2_value
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
    data_seed: int = 10_000
    test_seed: int = 30_000
    eval_interval: int = 250
    oracle_eval_interval: int = 0
    condition_check_interval: int = 10
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
    """Run the K=3 EMO algorithm from the paper.

    The first two recursive optimal values E1 and E2 are estimated with the
    corresponding lower-level epigraph oracles. The final problem then uses
    the nested z1/z2 bisection and the switching update from Eq. (11).
    """

    _set_seed(cfg.seed)
    device = torch.device(cfg.device)
    train_set = SyntheticCoTDataset(cfg.train_size, seed=cfg.data_seed, sample_demo_count=True)
    scales = compute_target_scales(train_set)
    model = CausalCoTTransformer().to(device)

    # Use one deterministic minibatch order for every bisection oracle. This
    # makes feasibility comparisons differ primarily because of z, not shuffle noise.
    loader_seed = cfg.seed + 1_000_000

    level1 = minimize_level1(
        model,
        train_set,
        scales,
        updates=cfg.stage1_updates,
        eval_interval=cfg.eval_interval,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        device=device,
        loader_seed=loader_seed,
    )
    e1 = level1.metrics["f1"]

    level2, e2 = solve_level2_value(
        model,
        train_set,
        scales,
        start_state=level1.state,
        e1=e1,
        epsilon=cfg.tolerance,
        upper_hint=max(level1.metrics["f2"], 1.0),
        bisection_steps=cfg.bisection_steps,
        inner_updates=cfg.inner_updates,
        oracle_eval_interval=cfg.oracle_eval_interval,
        condition_check_interval=cfg.condition_check_interval,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        device=device,
        loader_seed=loader_seed,
    )

    final, z1, z2 = solve_final_nested(
        model,
        train_set,
        scales,
        start_state=level2.state,
        e1=e1,
        e2=e2,
        epsilon=cfg.tolerance,
        upper_hint=max(level2.metrics["f3"], 1.0),
        bisection_steps=cfg.bisection_steps,
        inner_updates=cfg.inner_updates,
        oracle_eval_interval=cfg.oracle_eval_interval,
        condition_check_interval=cfg.condition_check_interval,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        device=device,
        loader_seed=loader_seed,
    )

    model.load_state_dict(final.state)
    last_state = final.last_state
    last_metrics = final.last_metrics
    history: list[dict[str, Any]] = []
    history.extend({"emo_level": "level1", **item} for item in level1.history)
    history.extend({"emo_level": "level2", **item} for item in level2.history)
    history.extend({"emo_level": "final", **item} for item in final.history)

    return EMOArtifacts(
        method="emo",
        selected_state=final.state,
        last_state=last_state,
        thresholds={
            "epsilon": cfg.tolerance,
            "E1_estimate": e1,
            "E2_estimate": e2,
            "z1": z1,
            "z2": z2,
        },
        selected_metrics=final.metrics,
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
    parser = argparse.ArgumentParser(description="Train the paper-faithful epigraph-based multilevel optimizer.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--data-seed", type=int, default=10_000)
    parser.add_argument("--test-seed", type=int, default=30_000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("runs/emo_seed11.pt"))
    parser.add_argument("--train-size", type=int, default=8192)
    parser.add_argument("--val-size", type=int, default=1024)
    parser.add_argument("--test-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument(
        "--oracle-eval-interval",
        type=int,
        default=0,
        help="Full training-set evaluation interval inside each oracle; 0 means only final feasibility evaluation.",
    )
    parser.add_argument(
        "--condition-check-interval",
        type=int,
        default=10,
        help="Check the stochastic Eq. (10c) condition every N inner updates.",
    )
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
        data_seed=args.data_seed,
        test_seed=args.test_seed,
        eval_interval=args.eval_interval,
        oracle_eval_interval=args.oracle_eval_interval,
        condition_check_interval=args.condition_check_interval,
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
