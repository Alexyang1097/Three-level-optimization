from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


BASELINE_METHODS = ("uniform", "staged", "constrained")
ALL_METHODS = ("uniform", "staged", "constrained", "emo")
DEFAULT_SEEDS = (11, 22, 33)


@dataclass(frozen=True)
class SuiteConfig:
    run_dir: str
    methods: tuple[str, ...]
    seeds: tuple[int, ...]
    device: str
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
    quick: bool
    preflight: bool
    overwrite: bool
    stop_on_error: bool


@dataclass(frozen=True)
class JobPaths:
    checkpoint: Path
    train_json: Path
    selected_eval_json: Path
    last_eval_json: Path
    stdout_log: Path
    stderr_log: Path


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = SuiteConfig(
        run_dir=str(args.run_dir),
        methods=tuple(args.methods),
        seeds=tuple(args.seeds),
        device=args.device,
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
        quick=args.quick,
        preflight=args.preflight,
        overwrite=args.overwrite,
        stop_on_error=args.stop_on_error,
    )
    run_dir = Path(cfg.run_dir)
    initialize_run_dir(run_dir, cfg)

    if cfg.preflight and not cfg.quick:
        run_preflight(cfg)

    failures: list[dict[str, Any]] = []
    for seed in cfg.seeds:
        for method in cfg.methods:
            ok = run_train_and_eval(method, seed, cfg)
            if not ok:
                failures.append({"method": method, "seed": seed})
                if cfg.stop_on_error:
                    summarize_results(run_dir)
                    raise SystemExit(f"Stopping after failed job: method={method}, seed={seed}")

    summarize_results(run_dir)
    if failures:
        raise SystemExit(f"Completed with failed jobs: {failures}")


def run_preflight(cfg: SuiteConfig) -> None:
    preflight_cfg = SuiteConfig(
        **{
            **asdict(cfg),
            "run_dir": str(Path(cfg.run_dir) / "_preflight"),
            "seeds": (cfg.seeds[0],),
            "quick": True,
            "preflight": False,
            "overwrite": True,
            "stop_on_error": True,
        }
    )
    initialize_run_dir(Path(preflight_cfg.run_dir), preflight_cfg)
    for method in preflight_cfg.methods:
        ok = run_train_and_eval(method, preflight_cfg.seeds[0], preflight_cfg)
        if not ok:
            raise SystemExit(f"Preflight failed for method={method}. Fix this before a long run.")


def run_train_and_eval(method: str, seed: int, cfg: SuiteConfig) -> bool:
    run_dir = Path(cfg.run_dir)
    paths = job_paths(run_dir, method, seed)
    paths.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    paths.train_json.parent.mkdir(parents=True, exist_ok=True)
    paths.stdout_log.parent.mkdir(parents=True, exist_ok=True)

    if paths.checkpoint.exists() and paths.selected_eval_json.exists() and paths.last_eval_json.exists() and not cfg.overwrite:
        append_manifest(run_dir, {"event": "skip_complete", "method": method, "seed": seed, "time": time.time()})
        return True

    append_manifest(run_dir, {"event": "train_start", "method": method, "seed": seed, "time": time.time()})
    train_command = build_train_command(method, seed, paths.checkpoint, cfg)
    train_result = run_command(train_command, paths.stdout_log, paths.stderr_log)
    if train_result["returncode"] != 0:
        append_manifest(run_dir, {"event": "train_failed", "method": method, "seed": seed, **train_result, "time": time.time()})
        return False

    train_payload = parse_json_output(paths.stdout_log)
    atomic_write_json(paths.train_json, train_payload)
    append_manifest(run_dir, {"event": "train_done", "method": method, "seed": seed, **train_result, "time": time.time()})

    for state, output_path in [("selected", paths.selected_eval_json), ("last", paths.last_eval_json)]:
        if output_path.exists() and not cfg.overwrite:
            append_manifest(run_dir, {"event": "eval_skip", "method": method, "seed": seed, "state": state, "time": time.time()})
            continue
        append_manifest(run_dir, {"event": "eval_start", "method": method, "seed": seed, "state": state, "time": time.time()})
        eval_stdout = paths.stdout_log.with_name(f"{method}_seed{seed}_{state}_eval.stdout.log")
        eval_stderr = paths.stderr_log.with_name(f"{method}_seed{seed}_{state}_eval.stderr.log")
        eval_result = run_command(build_eval_command(paths.checkpoint, state, cfg), eval_stdout, eval_stderr)
        if eval_result["returncode"] != 0:
            append_manifest(
                run_dir,
                {"event": "eval_failed", "method": method, "seed": seed, "state": state, **eval_result, "time": time.time()},
            )
            return False
        atomic_write_json(output_path, parse_json_output(eval_stdout))
        append_manifest(run_dir, {"event": "eval_done", "method": method, "seed": seed, "state": state, **eval_result, "time": time.time()})

    return True


def build_train_command(method: str, seed: int, checkpoint: Path, cfg: SuiteConfig) -> list[str]:
    if method == "emo":
        command = [
            sys.executable,
            "-m",
            "three_level.train_emo",
            "--seed",
            str(seed),
            "--device",
            cfg.device,
            "--output",
            str(checkpoint),
            "--train-size",
            str(cfg.train_size),
            "--val-size",
            str(cfg.val_size),
            "--test-size",
            str(cfg.test_size),
            "--batch-size",
            str(cfg.batch_size),
            "--eval-interval",
            str(cfg.eval_interval),
            "--stage1-updates",
            str(cfg.stage1_updates),
            "--inner-updates",
            str(cfg.emo_inner_updates),
            "--bisection-steps",
            str(cfg.emo_bisection_steps),
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "three_level.train",
            "--method",
            method,
            "--seed",
            str(seed),
            "--device",
            cfg.device,
            "--output",
            str(checkpoint),
            "--train-size",
            str(cfg.train_size),
            "--val-size",
            str(cfg.val_size),
            "--test-size",
            str(cfg.test_size),
            "--batch-size",
            str(cfg.batch_size),
            "--eval-interval",
            str(cfg.eval_interval),
            "--stage1-updates",
            str(cfg.stage1_updates),
            "--stage2-updates",
            str(cfg.stage2_updates),
            "--stage3-updates",
            str(cfg.stage3_updates),
            "--uniform-updates",
            str(cfg.uniform_updates),
        ]
    if cfg.quick:
        command.append("--quick")
    return command


def build_eval_command(checkpoint: Path, state: str, cfg: SuiteConfig) -> list[str]:
    return [
        sys.executable,
        "-m",
        "three_level.test",
        "--checkpoint",
        str(checkpoint),
        "--state",
        state,
        "--device",
        cfg.device,
        "--batch-size",
        str(cfg.batch_size),
    ]


def run_command(command: list[str], stdout_log: Path, stderr_log: Path) -> dict[str, Any]:
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with stdout_log.open("w", encoding="utf-8") as out, stderr_log.open("w", encoding="utf-8") as err:
        out.write("$ " + " ".join(command) + "\n")
        out.flush()
        completed = subprocess.run(command, stdout=out, stderr=err, text=True)
    return {
        "returncode": completed.returncode,
        "elapsed_sec": time.time() - started,
        "command": command,
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
    }


def parse_json_output(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"No JSON object found in {path}")
    return json.loads(text[start : end + 1])


def summarize_results(run_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    eval_dir = run_dir / "eval"
    for path in sorted(eval_dir.glob("*_eval_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        name_parts = path.stem.split("_eval_")
        method_seed = name_parts[0]
        state = name_parts[1]
        method, seed_text = method_seed.rsplit("_seed", 1)
        row = {
            "method": method,
            "seed": int(seed_text),
            "state": state,
            **flatten_metrics("teacher_forced_id", payload.get("teacher_forced_id", {})),
        }
        for n, metrics in payload.get("recursive_id", {}).items():
            row.update(flatten_metrics(f"id_n{n}", metrics))
        for n, metrics in payload.get("recursive_ood", {}).items():
            row.update(flatten_metrics(f"ood_n{n}", metrics))
        rows.append(row)

    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    summary_csv = run_dir / "summary.csv"
    with (summary_csv.with_suffix(".tmp")).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(summary_csv.with_suffix(".tmp"), summary_csv)
    atomic_write_json(run_dir / "summary.json", rows)


def flatten_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}.{key}": value for key, value in metrics.items()}


def job_paths(run_dir: Path, method: str, seed: int) -> JobPaths:
    stem = f"{method}_seed{seed}"
    return JobPaths(
        checkpoint=run_dir / "checkpoints" / f"{stem}.pt",
        train_json=run_dir / "train" / f"{stem}.json",
        selected_eval_json=run_dir / "eval" / f"{stem}_eval_selected.json",
        last_eval_json=run_dir / "eval" / f"{stem}_eval_last.json",
        stdout_log=run_dir / "logs" / f"{stem}_train.stdout.log",
        stderr_log=run_dir / "logs" / f"{stem}_train.stderr.log",
    )


def initialize_run_dir(run_dir: Path, cfg: SuiteConfig) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for child in ["checkpoints", "eval", "logs", "train"]:
        (run_dir / child).mkdir(exist_ok=True)
    atomic_write_json(run_dir / "suite_config.json", asdict(cfg))


def append_manifest(run_dir: Path, event: dict[str, Any]) -> None:
    with (run_dir / "manifest.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full research-note experiment suite robustly.")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/research_note_suite"))
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=list(ALL_METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    parser.add_argument("--quick", action="store_true", help="Use tiny per-method settings for smoke testing.")
    parser.add_argument("--no-preflight", dest="preflight", action="store_false", help="Skip the automatic quick preflight.")
    parser.add_argument("--overwrite", action="store_true", help="Rerun jobs even if checkpoint and eval JSON already exist.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop on the first failed job instead of recording and continuing.")
    parser.set_defaults(preflight=True)
    return parser


if __name__ == "__main__":
    main()
