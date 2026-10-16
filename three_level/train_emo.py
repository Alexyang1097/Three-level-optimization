from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from .data import SyntheticCoTDataset, TargetScales, compute_target_scales
from .emo import EMOBudgetExhausted, EMOWarmStart, minimize_level1, solve_final_nested, solve_level2_value
from .model import CausalCoTTransformer
from .tracking import ComputeTracker, record_metric_snapshot, synchronized_time


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
    backward_budget: int = 0
    tolerance: float = 0.02
    device: str = "cpu"
    warm_start: bool = False


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
    compute: dict[str, Any]
    status: str
    diagnostics: list[dict[str, Any]]
    selection: dict[str, Any] = field(default_factory=dict)


def run_emo(cfg: EMOConfig) -> EMOArtifacts:
    """Run the K=3 EMO algorithm from the paper.

    The first two recursive optimal values E1 and E2 are estimated with the
    corresponding lower-level epigraph oracles. The final problem then uses
    the nested z1/z2 bisection and the switching update from Eq. (11).
    """

    _set_seed(cfg.seed)
    device = torch.device(cfg.device)
    tracker = ComputeTracker(backward_budget=cfg.backward_budget or None)
    data_started = synchronized_time(device)
    train_set = SyntheticCoTDataset(cfg.train_size, seed=cfg.data_seed, sample_demo_count=True)
    scales = compute_target_scales(train_set)
    tracker.training_data_generation_sec = synchronized_time(device) - data_started
    model_started = synchronized_time(device)
    model = CausalCoTTransformer().to(device)
    tracker.model_setup_sec = synchronized_time(device) - model_started
    started = synchronized_time(device)
    diagnostics: list[dict[str, Any]] = []
    e1: float | None = None
    e2: float | None = None
    warm_start: EMOWarmStart | None = None

    # Use one deterministic minibatch order for every bisection oracle. This
    # makes feasibility comparisons differ primarily because of z, not shuffle noise.
    loader_seed = cfg.seed + 1_000_000

    try:
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
            tracker=tracker,
        )
        diagnostics.extend({"emo_level": "level1", **item} for item in level1.history)
        e1 = level1.metrics["f1"]
        if cfg.warm_start:
            warm_start = EMOWarmStart("f2", {"f1": e1 + cfg.tolerance})
            warm_start.observe(model, level1.metrics, tracker)
        record_metric_snapshot(
            tracker,
            level1.metrics,
            phase="level2",
            reason="phase_transition",
            extra={"E1_estimate": e1},
        )

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
            eval_interval=cfg.eval_interval,
            tracker=tracker,
            warm_start=warm_start,
        )
        diagnostics.extend({"emo_level": "level2", **item} for item in level2.history)
        if cfg.warm_start:
            model.load_state_dict(level2.state)
            warm_start = EMOWarmStart("f3", {"f1": e1 + cfg.tolerance, "f2": e2 + cfg.tolerance})
            warm_start.observe(model, level2.metrics, tracker)
        record_metric_snapshot(
            tracker,
            level2.metrics,
            phase="final",
            reason="phase_transition",
            extra={"E1_estimate": e1, "E2_estimate": e2},
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
            eval_interval=cfg.eval_interval,
            tracker=tracker,
            warm_start=warm_start,
        )
        diagnostics.extend({"emo_level": "final", **item} for item in final.history)
    except EMOBudgetExhausted as exhausted:
        if warm_start is not None:
            warm_start.observe(model, exhausted.metrics, tracker)
        tracker.elapsed_train_sec = synchronized_time(device) - started
        thresholds: dict[str, float] = {"epsilon": cfg.tolerance}
        if e1 is not None:
            thresholds["E1_estimate"] = e1
        if e2 is not None:
            thresholds["E2_estimate"] = e2
        return EMOArtifacts(
            method="emo",
            selected_state=warm_start.state if warm_start is not None and warm_start.state is not None else exhausted.state,
            last_state=exhausted.state,
            thresholds=thresholds,
            selected_metrics=warm_start.metrics if warm_start is not None and warm_start.metrics is not None else exhausted.metrics,
            last_metrics=exhausted.metrics,
            history=tracker.history,
            scales=scales,
            compute=tracker.summary(),
            status=f"budget_exhausted:{exhausted.phase}",
            diagnostics=diagnostics,
            selection={
                "rule": "best_observed_upstream_feasible" if warm_start is not None else "budget_last",
                "objective": warm_start.objective if warm_start is not None else None,
                "backward_calls": warm_start.backward_calls if warm_start is not None else tracker.backward_calls,
            },
        )

    model.load_state_dict(final.state)
    last_state = final.last_state
    last_metrics = final.last_metrics
    tracker.elapsed_train_sec = synchronized_time(device) - started

    return EMOArtifacts(
        method="emo",
        selected_state=warm_start.state if warm_start is not None and warm_start.state is not None else final.state,
        last_state=last_state,
        thresholds={
            "epsilon": cfg.tolerance,
            "E1_estimate": e1,
            "E2_estimate": e2,
            "z1": z1,
            "z2": z2,
        },
        selected_metrics=warm_start.metrics if warm_start is not None and warm_start.metrics is not None else final.metrics,
        last_metrics=last_metrics,
        history=tracker.history,
        scales=scales,
        compute=tracker.summary(),
        status="complete",
        diagnostics=diagnostics,
        selection={
            "rule": "best_observed_upstream_feasible" if warm_start is not None else "final_oracle",
            "objective": warm_start.objective if warm_start is not None else "f3",
            "backward_calls": warm_start.backward_calls if warm_start is not None else None,
        },
    )


def save_artifacts(path: Path, artifacts: EMOArtifacts, cfg: EMOConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "method": artifacts.method,
            "selected_state": artifacts.selected_state,
            "last_state": artifacts.last_state,
            "thresholds": artifacts.thresholds,
            "selected_metrics": artifacts.selected_metrics,
            "last_metrics": artifacts.last_metrics,
            "history": artifacts.history,
            "diagnostics": artifacts.diagnostics,
            "selection": artifacts.selection,
            "scales": asdict(artifacts.scales),
            "compute": artifacts.compute,
            "status": artifacts.status,
            "config": asdict(cfg),
        },
        temporary,
    )
    os.replace(temporary, path)


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
    parser.add_argument(
        "--backward-budget",
        type=int,
        default=0,
        help="Maximum cumulative backward calls; 0 runs EMO to algorithm completion.",
    )
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--warm-start", action="store_true", help="Reuse observed candidate models across epigraph searches; restart Adam.")
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
        backward_budget=args.backward_budget,
        tolerance=args.tolerance,
        device=args.device,
        warm_start=args.warm_start,
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
        if cfg.backward_budget > 0:
            cfg.backward_budget = 15

    start = time.time()
    artifacts = run_emo(cfg)
    save_artifacts(args.output, artifacts, cfg)
    print(
        json.dumps(
            {
                "method": artifacts.method,
                "status": artifacts.status,
                "thresholds": artifacts.thresholds,
                "selected_metrics": artifacts.selected_metrics,
                "last_metrics": artifacts.last_metrics,
                "compute": artifacts.compute,
                "selection": artifacts.selection,
                "elapsed_sec": time.time() - start,
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
