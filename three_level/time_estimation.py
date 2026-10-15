from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


METHODS = ("uniform", "staged", "constrained", "emo")


@dataclass(frozen=True)
class FormalWorkload:
    train_size: int = 8192
    val_size: int = 1024
    test_size: int = 256
    batch_size: int = 32
    eval_interval: int = 250
    stage1_updates: int = 1000
    stage2_updates: int = 1500
    stage3_updates: int = 1500
    uniform_updates: int = 4000
    emo_inner_updates: int = 1500
    emo_bisection_steps: int = 6
    emo_backward_budget: int = 4000
    seeds: tuple[int, ...] = (11, 22, 33)


@dataclass(frozen=True)
class SampleWorkload:
    requested_fraction: float
    train_size: int
    val_size: int
    test_size: int
    batch_size: int
    eval_interval: int
    stage1_updates: int
    stage2_updates: int
    stage3_updates: int
    uniform_updates: int
    emo_inner_updates: int
    emo_bisection_steps: int
    emo_backward_budget: int
    seed: int


def main() -> None:
    args = build_arg_parser().parse_args()
    formal = FormalWorkload(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
        stage1_updates=args.stage1_updates,
        stage2_updates=args.stage2_updates,
        stage3_updates=args.stage3_updates,
        uniform_updates=args.uniform_updates,
        emo_inner_updates=args.emo_inner_updates,
        emo_bisection_steps=args.emo_bisection_steps,
        emo_backward_budget=args.emo_backward_budget,
        seeds=tuple(args.seeds),
    )
    sample = make_sample_workload(formal, args.fraction, args.sample_seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    sample_root = output_dir / f"sample_{stamp}"
    sample_root.mkdir(parents=True)

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "formal_workload": asdict(formal),
        "sample_workload": asdict(sample),
        "effective_fractions": effective_fractions(formal, sample),
        "sample_root": str(sample_root),
        "assumptions": [
            "The model, batch size, optimizer, losses, bisection depth, and prompt lengths match the formal run.",
            "Only dataset sizes, update counts, evaluation interval, EMO budget, and number of sampled seeds are reduced.",
            "Optimization time is projected by backward calls; full-pool and test evaluation time is projected by tasks processed.",
            "Train/test data construction is projected by task count; model setup, process startup, and checkpoint I/O are fixed costs.",
            "Preflight, aggregation, and plotting are measured once and retained as suite-level fixed costs.",
            "Complete-EMO control flow is data dependent; its range is a sensitivity range, not a statistical confidence interval.",
        ],
    }

    main_dir = sample_root / "main_budget_sample"
    main_command = suite_command(
        run_dir=main_dir,
        methods=METHODS,
        sample=sample,
        device=args.device,
        backward_budget=sample.emo_backward_budget,
    )
    main_wall = run_sample(main_command, sample_root / "main_suite.stdout.log", sample_root / "main_suite.stderr.log")
    main_estimate = estimate_suite(
        run_dir=main_dir,
        suite_wall_sec=main_wall,
        methods=METHODS,
        formal=formal,
        sample=sample,
        complete_emo=False,
    )
    report["main_compute_matched"] = main_estimate

    if not args.main_only:
        complete_dir = sample_root / "emo_complete_sample"
        complete_command = suite_command(
            run_dir=complete_dir,
            methods=("emo",),
            sample=sample,
            device=args.device,
            backward_budget=0,
        )
        try:
            complete_wall = run_sample(
                complete_command,
                sample_root / "complete_emo.stdout.log",
                sample_root / "complete_emo.stderr.log",
            )
            complete_estimate = estimate_suite(
                run_dir=complete_dir,
                suite_wall_sec=complete_wall,
                methods=("emo",),
                formal=formal,
                sample=sample,
                complete_emo=True,
            )
            report["complete_emo_extension"] = complete_estimate
            report["estimated_full_experiment_sec"] = (
                main_estimate["estimated_suite_total_sec"]
                + complete_estimate["estimated_suite_total_sec"]
            )
        except RuntimeError as error:
            report["complete_emo_extension"] = {
                "status": "sample_failed",
                "error": str(error),
            }
            report["estimated_full_experiment_sec"] = None
    else:
        report["complete_emo_extension"] = {"status": "not_requested"}
        report["estimated_full_experiment_sec"] = main_estimate["estimated_suite_total_sec"]

    report["estimated_full_experiment_human"] = format_duration(report["estimated_full_experiment_sec"])
    atomic_write_json(sample_root / "estimate.json", report)
    write_markdown(sample_root / "estimate.md", report)
    atomic_write_json(output_dir / "latest_estimate.json", report)
    write_markdown(output_dir / "latest_estimate.md", report)
    print(json.dumps(report, indent=2, sort_keys=True))


def make_sample_workload(formal: FormalWorkload, fraction: float, seed: int) -> SampleWorkload:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if formal.emo_backward_budget <= 0:
        raise ValueError("The compute-matched formal EMO budget must be positive")

    return SampleWorkload(
        requested_fraction=fraction,
        train_size=max(formal.batch_size, scaled(formal.train_size, fraction)),
        val_size=max(1, scaled(formal.val_size, fraction)),
        test_size=min(formal.test_size, max(formal.batch_size * 2, scaled(formal.test_size, fraction))),
        batch_size=formal.batch_size,
        eval_interval=scaled(formal.eval_interval, fraction),
        stage1_updates=scaled(formal.stage1_updates, fraction),
        stage2_updates=scaled(formal.stage2_updates, fraction),
        stage3_updates=scaled(formal.stage3_updates, fraction),
        uniform_updates=scaled(formal.uniform_updates, fraction),
        emo_inner_updates=scaled(formal.emo_inner_updates, fraction),
        emo_bisection_steps=formal.emo_bisection_steps,
        emo_backward_budget=scaled(formal.emo_backward_budget, fraction),
        seed=seed,
    )


def scaled(value: int, fraction: float) -> int:
    return max(1, int(round(value * fraction)))


def effective_fractions(formal: FormalWorkload, sample: SampleWorkload) -> dict[str, float]:
    return {
        "train_tasks": sample.train_size / formal.train_size,
        "test_tasks": sample.test_size / formal.test_size,
        "uniform_updates": sample.uniform_updates / formal.uniform_updates,
        "stage1_updates": sample.stage1_updates / formal.stage1_updates,
        "stage2_updates": sample.stage2_updates / formal.stage2_updates,
        "stage3_updates": sample.stage3_updates / formal.stage3_updates,
        "emo_inner_updates": sample.emo_inner_updates / formal.emo_inner_updates,
        "emo_backward_budget": sample.emo_backward_budget / formal.emo_backward_budget,
    }


def suite_command(
    *,
    run_dir: Path,
    methods: tuple[str, ...],
    sample: SampleWorkload,
    device: str,
    backward_budget: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "three_level.experiment_suite",
        "--run-dir",
        str(run_dir),
        "--methods",
        *methods,
        "--seeds",
        str(sample.seed),
        "--device",
        device,
        "--train-size",
        str(sample.train_size),
        "--val-size",
        str(sample.val_size),
        "--test-size",
        str(sample.test_size),
        "--batch-size",
        str(sample.batch_size),
        "--eval-interval",
        str(sample.eval_interval),
        "--stage1-updates",
        str(sample.stage1_updates),
        "--stage2-updates",
        str(sample.stage2_updates),
        "--stage3-updates",
        str(sample.stage3_updates),
        "--uniform-updates",
        str(sample.uniform_updates),
        "--emo-inner-updates",
        str(sample.emo_inner_updates),
        "--emo-bisection-steps",
        str(sample.emo_bisection_steps),
        "--emo-backward-budget",
        str(backward_budget),
        "--stop-on-error",
    ]


def run_sample(command: list[str], stdout_path: Path, stderr_path: Path) -> float:
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        stdout.write("$ " + " ".join(command) + "\n")
        stdout.flush()
        completed = subprocess.run(command, stdout=stdout, stderr=stderr, text=True)
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"Sample suite failed with return code {completed.returncode}; "
            f"see {stdout_path} and {stderr_path}"
        )
    return elapsed


def estimate_suite(
    *,
    run_dir: Path,
    suite_wall_sec: float,
    methods: tuple[str, ...],
    formal: FormalWorkload,
    sample: SampleWorkload,
    complete_emo: bool,
) -> dict[str, Any]:
    events = read_manifest(run_dir / "manifest.jsonl")
    train_events = {event["method"]: event for event in events if event.get("event") == "train_done"}
    eval_events = {
        (event["method"], event["state"]): event
        for event in events
        if event.get("event") == "eval_done"
    }
    measured_job_wall = sum(float(event["elapsed_sec"]) for event in train_events.values())
    measured_job_wall += sum(float(event["elapsed_sec"]) for event in eval_events.values())
    suite_fixed = max(0.0, suite_wall_sec - measured_job_wall)

    per_method: dict[str, Any] = {}
    aggregate = empty_components()
    seed_count = len(formal.seeds)
    for method in methods:
        checkpoint = run_dir / "checkpoints" / f"{method}_seed{sample.seed}.pt"
        payload = load_checkpoint(checkpoint)
        compute = payload["compute"]
        train_wall = float(train_events[method]["elapsed_sec"])
        train_measured = float(compute["elapsed_train_sec"])
        full_eval_measured = float(compute.get("full_train_evaluation_sec", 0.0))
        non_eval_measured = max(0.0, train_measured - full_eval_measured)
        sample_calls = int(compute["backward_calls"])

        if complete_emo:
            projected_calls = project_complete_emo_calls(sample_calls, formal, sample)
            projected_eval_count = int(compute["full_train_evaluations"])
        else:
            projected_calls = target_backward_calls(method, formal)
            projected_eval_count = target_full_eval_count(method, formal, compute)

        update_ratio = projected_calls / max(sample_calls, 1)
        optimization_sec = non_eval_measured * update_ratio
        sample_full_prompts = int(compute["full_train_prompts"])
        full_eval_rate = full_eval_measured / max(sample_full_prompts, 1)
        full_pool_sec = full_eval_rate * projected_eval_count * formal.train_size
        train_data_measured = float(compute.get("training_data_generation_sec", 0.0))
        model_setup_measured = float(compute.get("model_setup_sec", 0.0))
        train_data_sec = train_data_measured * formal.train_size / sample.train_size
        train_fixed_sec = max(
            0.0,
            train_wall - train_measured - train_data_measured - model_setup_measured,
        ) + model_setup_measured

        test_kernel_sec = 0.0
        test_data_sec = 0.0
        test_fixed_sec = 0.0
        sample_eval_wall = 0.0
        for state in ("selected", "last"):
            eval_path = run_dir / "eval" / f"{method}_seed{sample.seed}_eval_{state}.json"
            eval_payload = json.loads(eval_path.read_text(encoding="utf-8"))
            eval_compute = eval_payload["evaluation_compute"]
            kernel = float(eval_compute["elapsed_eval_sec"])
            data_measured = float(eval_compute.get("test_data_generation_sec", 0.0))
            wall = float(eval_events[(method, state)]["elapsed_sec"])
            test_kernel_sec += kernel * formal.test_size / sample.test_size
            test_data_sec += data_measured * formal.test_size / sample.test_size
            test_fixed_sec += max(0.0, wall - kernel - data_measured)
            sample_eval_wall += wall

        one_seed = {
            "optimization_and_oracle_sec": optimization_sec,
            "full_training_pool_evaluation_sec": full_pool_sec,
            "training_data_generation_sec": train_data_sec,
            "training_model_process_checkpoint_sec": train_fixed_sec,
            "selected_and_last_test_kernel_sec": test_kernel_sec,
            "test_data_generation_sec": test_data_sec,
            "test_model_process_setup_sec": test_fixed_sec,
        }
        one_seed["total_sec"] = sum(one_seed.values())
        all_seeds = {key: value * seed_count for key, value in one_seed.items()}
        sample_measurement = {
            "training_wall_sec": train_wall,
            "training_algorithm_sec": train_measured,
            "full_training_pool_evaluation_sec": full_eval_measured,
            "training_data_generation_sec": train_data_measured,
            "model_setup_sec": model_setup_measured,
            "selected_and_last_eval_wall_sec": sample_eval_wall,
            "backward_calls": sample_calls,
            "full_train_evaluations": int(compute["full_train_evaluations"]),
            "full_train_prompts": sample_full_prompts,
        }
        per_method[method] = {
            "sample_measurement": sample_measurement,
            "projection": {
                "projected_backward_calls_per_seed": projected_calls,
                "projected_full_train_evaluations_per_seed": projected_eval_count,
                "one_seed": with_human(one_seed),
                "all_seeds": with_human(all_seeds),
            },
        }
        for key in aggregate:
            aggregate[key] += all_seeds.get(key, 0.0)

    aggregate["suite_preflight_aggregation_plotting_sec"] = suite_fixed
    total = sum(aggregate.values())
    result: dict[str, Any] = {
        "status": "estimated",
        "sample_suite_wall_sec": suite_wall_sec,
        "sample_suite_wall_human": format_duration(suite_wall_sec),
        "methods": per_method,
        "estimated_components_all_seeds": with_human(aggregate),
        "estimated_suite_total_sec": total,
        "estimated_suite_total_human": format_duration(total),
    }
    if complete_emo:
        variable = aggregate["optimization_and_oracle_sec"] + aggregate["full_training_pool_evaluation_sec"]
        fixed = total - variable
        low = fixed + 0.5 * variable
        high = fixed + 2.0 * variable
        result["control_flow_sensitivity"] = {
            "description": "0.5x to 2.0x the projected EMO algorithm-dependent work; not a confidence interval.",
            "low_sec": low,
            "low_human": format_duration(low),
            "high_sec": high,
            "high_human": format_duration(high),
        }
    return result


def target_backward_calls(method: str, formal: FormalWorkload) -> int:
    if method == "uniform":
        return formal.uniform_updates
    if method in ("staged", "constrained"):
        return formal.stage1_updates + formal.stage2_updates + formal.stage3_updates
    if method == "emo":
        return formal.emo_backward_budget
    raise ValueError(f"Unknown method {method!r}")


def target_full_eval_count(method: str, formal: FormalWorkload, sample_compute: dict[str, Any]) -> int:
    if method == "uniform":
        return 1 + evaluations_during(formal.uniform_updates, formal.eval_interval)
    if method in ("staged", "constrained"):
        return 1 + sum(
            evaluations_during(updates, formal.eval_interval)
            for updates in (formal.stage1_updates, formal.stage2_updates, formal.stage3_updates)
        )
    if method == "emo":
        # The temporal update counts and global evaluation interval are scaled
        # together, so the sampled oracle path is the best available predictor.
        return int(sample_compute["full_train_evaluations"])
    raise ValueError(f"Unknown method {method!r}")


def evaluations_during(updates: int, interval: int) -> int:
    return updates // interval + int(updates % interval != 0)


def project_complete_emo_calls(
    sample_calls: int,
    formal: FormalWorkload,
    sample: SampleWorkload,
) -> int:
    sampled_level1 = min(sample_calls, sample.stage1_updates)
    sampled_inner = max(0, sample_calls - sampled_level1)
    projected = sampled_level1 * formal.stage1_updates / sample.stage1_updates
    projected += sampled_inner * formal.emo_inner_updates / sample.emo_inner_updates
    return max(1, int(round(projected)))


def read_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def empty_components() -> dict[str, float]:
    return {
        "optimization_and_oracle_sec": 0.0,
        "full_training_pool_evaluation_sec": 0.0,
        "training_data_generation_sec": 0.0,
        "training_model_process_checkpoint_sec": 0.0,
        "selected_and_last_test_kernel_sec": 0.0,
        "test_data_generation_sec": 0.0,
        "test_model_process_setup_sec": 0.0,
    }


def with_human(values: dict[str, float]) -> dict[str, Any]:
    result: dict[str, Any] = dict(values)
    if "total_sec" not in result:
        result["total_sec"] = sum(values.values())
    result["total_human"] = format_duration(float(result["total_sec"]))
    return result


def format_duration(seconds: float | None) -> str | None:
    if seconds is None or not math.isfinite(seconds):
        return None
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    main = report["main_compute_matched"]
    lines = [
        "# Experiment Time Estimate",
        "",
        f"- Device: `{report['device']}`",
        f"- Requested sample fraction: `{report['sample_workload']['requested_fraction']}`",
        f"- Sample artifacts: `{report['sample_root']}`",
        f"- Main compute-matched suite: **{main['estimated_suite_total_human']}**",
    ]
    complete = report.get("complete_emo_extension", {})
    if complete.get("status") == "estimated":
        lines.extend(
            [
                f"- Complete EMO extension: **{complete['estimated_suite_total_human']}**",
                f"- Complete EMO sensitivity: **{complete['control_flow_sensitivity']['low_human']} to "
                f"{complete['control_flow_sensitivity']['high_human']}**",
            ]
        )
    elif complete.get("status") == "sample_failed":
        lines.append(f"- Complete EMO extension: sample failed (`{complete['error']}`)")
    lines.extend(
        [
            f"- Full requested workflow: **{report['estimated_full_experiment_human']}**",
            "",
            "## Main Suite by Method",
            "",
            "| Method | One seed | All formal seeds | Projected backward calls/seed |",
            "|---|---:|---:|---:|",
        ]
    )
    for method, values in main["methods"].items():
        projection = values["projection"]
        lines.append(
            f"| {method} | {projection['one_seed']['total_human']} | "
            f"{projection['all_seeds']['total_human']} | "
            f"{projection['projected_backward_calls_per_seed']} |"
        )
    lines.extend(
        [
            "",
            "## Main Suite Components",
            "",
            "| Component | Estimated time |",
            "|---|---:|",
        ]
    )
    component_labels = {
        "optimization_and_oracle_sec": "Optimization and EMO oracle work",
        "full_training_pool_evaluation_sec": "Full training-pool evaluations",
        "training_data_generation_sec": "Training data construction",
        "training_model_process_checkpoint_sec": "Training model/process/checkpoint setup",
        "selected_and_last_test_kernel_sec": "Selected/last ID and OOD test kernels",
        "test_data_generation_sec": "ID and OOD test data construction",
        "test_model_process_setup_sec": "Test process and model setup",
        "suite_preflight_aggregation_plotting_sec": "Preflight, aggregation, and plotting",
    }
    components = main["estimated_components_all_seeds"]
    for key, label in component_labels.items():
        lines.append(f"| {label} | {format_duration(float(components[key]))} |")
    lines.extend(["", "## Assumptions", ""])
    lines.extend(f"- {item}" for item in report["assumptions"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a proportional formal experiment and estimate full-suite wall time by cost component."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/time_estimate"))
    parser.add_argument("--fraction", type=float, default=0.05)
    parser.add_argument("--sample-seed", type=int, default=11)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--main-only", action="store_true", help="Skip the algorithm-complete EMO extension estimate.")
    parser.add_argument("--train-size", type=int, default=8192)
    parser.add_argument("--val-size", type=int, default=1024)
    parser.add_argument("--test-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--stage1-updates", type=int, default=1000)
    parser.add_argument("--stage2-updates", type=int, default=1500)
    parser.add_argument("--stage3-updates", type=int, default=1500)
    parser.add_argument("--uniform-updates", type=int, default=4000)
    parser.add_argument("--emo-inner-updates", type=int, default=1500)
    parser.add_argument("--emo-bisection-steps", type=int, default=6)
    parser.add_argument("--emo-backward-budget", type=int, default=4000)
    return parser


if __name__ == "__main__":
    main()
