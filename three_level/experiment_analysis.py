from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


METHOD_LABELS = {
    "uniform": "Uniform Sum",
    "staged": "Unconstrained Stages",
    "constrained": "Primal-Dual Stages",
    "emo": "EMO",
}
METHOD_COLORS = {
    "uniform": "#2878B5",
    "staged": "#9B5DE5",
    "constrained": "#D1495B",
    "emo": "#2A9D8F",
}


def generate_analysis(run_dir: Path, *, make_figures: bool = True) -> None:
    tables_dir = run_dir / "tables"
    figures_dir = run_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    payloads = _load_checkpoints(run_dir / "checkpoints")
    paired_thresholds = {
        seed: payload.get("thresholds", {})
        for (method, seed), payload in payloads.items()
        if method == "constrained"
    }

    training_rows = _training_rows(payloads, paired_thresholds)
    risk_rows = _risk_rows(run_dir / "eval")
    compute_rows = _compute_rows(payloads, run_dir / "eval")
    n32_rows = _n32_rows(risk_rows)
    compact_rows = _compact_n32_rows(payloads, paired_thresholds, risk_rows, compute_rows)

    _write_csv(tables_dir / "training_history.csv", training_rows)
    _write_csv(tables_dir / "risk_long.csv", risk_rows)
    _write_csv(tables_dir / "compute.csv", compute_rows)
    _write_csv(tables_dir / "n32_summary.csv", n32_rows)
    _write_csv(tables_dir / "n32_compact.csv", compact_rows)

    if make_figures and training_rows and risk_rows:
        _plot_training_losses(training_rows, run_dir, figures_dir)
        _plot_constraints_and_duals(training_rows, figures_dir)
        _plot_recursive_risks(risk_rows, figures_dir)


def _training_rows(
    payloads: dict[tuple[str, int], dict[str, Any]],
    paired_thresholds: dict[int, dict[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (method, seed), payload in sorted(payloads.items()):
        thresholds = paired_thresholds.get(seed, {})
        b1 = thresholds.get("b1")
        b2 = thresholds.get("b2")
        for item in payload.get("history", []):
            if not all(key in item for key in ("e1", "e2", "e3", "f1", "f2", "f3")):
                continue
            row = {
                "method": method,
                "seed": seed,
                "status": payload.get("status", "complete"),
                "backward_calls": item.get("backward_calls", item.get("update")),
                "optimizer_updates": item.get("optimizer_updates", item.get("update")),
                "training_prompts": item.get("training_prompts"),
                "full_train_evaluations": item.get("full_train_evaluations"),
                "phase": item.get("phase", item.get("stage", item.get("emo_level", ""))),
                "evaluation_reason": item.get("evaluation_reason", "legacy"),
                "e1": item["e1"],
                "e2": item["e2"],
                "e3": item["e3"],
                "f1": item["f1"],
                "f2": item["f2"],
                "f3": item["f3"],
                "b1": b1,
                "b2": b2,
                "lambda1": item.get("lambda1"),
                "lambda2": item.get("lambda2"),
            }
            if b1 is not None:
                row["constraint1"] = item["f1"] - b1
                row["violation1"] = max(0.0, row["constraint1"])
            if b2 is not None:
                row["constraint2"] = item["f2"] - b2
                row["violation2"] = max(0.0, row["constraint2"])
            rows.append(row)
    return rows


def _risk_rows(eval_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(eval_dir.glob("*_eval_*.json")):
        method_seed, state = path.stem.split("_eval_", 1)
        method, seed_text = method_seed.rsplit("_seed", 1)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for distribution, key in (("id", "recursive_id"), ("ood", "recursive_ood")):
            for n_text, metrics in payload.get(key, {}).items():
                for step in (1, 2, 3):
                    recursive = float(metrics[f"rec{step}"])
                    teacher_forced = float(metrics[f"tf{step}"])
                    base = {
                        "method": method,
                        "seed": int(seed_text),
                        "state": state,
                        "distribution": distribution,
                        "n": int(n_text),
                        "step": step,
                    }
                    rows.append({**base, "mode": "recursive", "risk": recursive})
                    rows.append({**base, "mode": "teacher_forced", "risk": teacher_forced})
                    rows.append({**base, "mode": "gap", "risk": recursive - teacher_forced})
    return rows


def _compute_rows(
    payloads: dict[tuple[str, int], dict[str, Any]],
    eval_dir: Path,
) -> list[dict[str, Any]]:
    eval_times: dict[tuple[str, int, str], float] = {}
    for path in sorted(eval_dir.glob("*_eval_*.json")):
        method_seed, state = path.stem.split("_eval_", 1)
        method, seed_text = method_seed.rsplit("_seed", 1)
        payload = json.loads(path.read_text(encoding="utf-8"))
        elapsed = payload.get("evaluation_compute", {}).get("elapsed_eval_sec", 0.0)
        eval_times[(method, int(seed_text), state)] = float(elapsed)

    rows: list[dict[str, Any]] = []
    for (method, seed), payload in sorted(payloads.items()):
        compute = payload.get("compute", {})
        selected_eval = eval_times.get((method, seed, "selected"), 0.0)
        last_eval = eval_times.get((method, seed, "last"), 0.0)
        train_elapsed = float(compute.get("elapsed_train_sec", 0.0))
        rows.append(
            {
                "method": method,
                "seed": seed,
                "status": payload.get("status", "complete"),
                "optimizer_updates": compute.get("optimizer_updates"),
                "backward_calls": compute.get("backward_calls"),
                "training_prompts": compute.get("training_prompts"),
                "full_train_evaluations": compute.get("full_train_evaluations"),
                "full_train_prompts": compute.get("full_train_prompts"),
                "threshold_evaluations": compute.get("threshold_evaluations"),
                "threshold_evaluation_prompts": compute.get("threshold_evaluation_prompts"),
                "condition_checks": compute.get("condition_checks"),
                "condition_check_prompts": compute.get("condition_check_prompts"),
                "oracle_calls": compute.get("oracle_calls"),
                "total_processed_prompts": compute.get("total_processed_prompts"),
                "elapsed_train_sec": train_elapsed,
                "elapsed_selected_eval_sec": selected_eval,
                "elapsed_last_eval_sec": last_eval,
                "elapsed_total_sec": train_elapsed + selected_eval + last_eval,
            }
        )
    return rows


def _n32_rows(risk_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in risk_rows:
        if row["n"] != 32:
            continue
        key = (row["method"], row["state"], row["distribution"], row["mode"], row["step"])
        grouped[key].append(float(row["risk"]))

    rows: list[dict[str, Any]] = []
    for (method, state, distribution, mode, step), values in sorted(grouped.items()):
        rows.append(
            {
                "method": method,
                "state": state,
                "distribution": distribution,
                "mode": mode,
                "step": step,
                "seed_count": len(values),
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
        )
    return rows


def _compact_n32_rows(
    payloads: dict[tuple[str, int], dict[str, Any]],
    paired_thresholds: dict[int, dict[str, float]],
    risk_rows: list[dict[str, Any]],
    compute_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    compute_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in compute_rows:
        compute_by_method[str(row["method"])].append(row)

    rows: list[dict[str, Any]] = []
    for method in sorted({key[0] for key in payloads}, key=lambda value: list(METHOD_LABELS).index(value)):
        for state in ("selected", "last"):
            row: dict[str, Any] = {"method": method, "state": state}
            method_risks = [
                item
                for item in risk_rows
                if item["method"] == method
                and item["state"] == state
                and item["n"] == 32
                and item["mode"] == "recursive"
            ]
            for distribution in ("id", "ood"):
                for step in (1, 2, 3):
                    values = [
                        float(item["risk"])
                        for item in method_risks
                        if item["distribution"] == distribution and item["step"] == step
                    ]
                    row[f"{distribution}_rec{step}_mean"] = statistics.fmean(values) if values else None
                    row[f"{distribution}_rec{step}_std"] = (
                        statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
                    )

            violations1: list[float] = []
            violations2: list[float] = []
            feasible: list[float] = []
            for (candidate, seed), payload in payloads.items():
                if candidate != method:
                    continue
                metrics = payload[f"{state}_metrics"]
                thresholds = paired_thresholds.get(seed, {})
                if "b1" not in thresholds or "b2" not in thresholds:
                    continue
                v1 = max(0.0, float(metrics["f1"]) - float(thresholds["b1"]))
                v2 = max(0.0, float(metrics["f2"]) - float(thresholds["b2"]))
                violations1.append(v1)
                violations2.append(v2)
                feasible.append(float(v1 == 0.0 and v2 == 0.0))
            row["violation1_mean"] = statistics.fmean(violations1) if violations1 else None
            row["violation2_mean"] = statistics.fmean(violations2) if violations2 else None
            row["empirical_feasibility_rate"] = statistics.fmean(feasible) if feasible else None

            method_compute = compute_by_method.get(method, [])
            for key in (
                "backward_calls",
                "training_prompts",
                "full_train_evaluations",
                "full_train_prompts",
                "total_processed_prompts",
                "elapsed_train_sec",
                "elapsed_total_sec",
            ):
                values = [float(item[key]) for item in method_compute if item.get(key) not in (None, "")]
                row[f"{key}_mean"] = statistics.fmean(values) if values else None
                row[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
            rows.append(row)
    return rows


def _plot_training_losses(rows: list[dict[str, Any]], run_dir: Path, figures_dir: Path) -> None:
    plt = _pyplot()
    cfg_path = run_dir / "suite_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    transitions = sorted(
        {
            int(row["backward_calls"])
            for row in rows
            if row["method"] == "constrained" and row.get("evaluation_reason") == "stage_transition"
        }
    )
    boundaries = transitions[:2]
    if len(boundaries) < 2:
        if cfg.get("quick"):
            boundaries = [5, 10]
        else:
            boundaries = [cfg.get("stage1_updates", 1000)]
            boundaries.append(boundaries[0] + cfg.get("stage2_updates", 1500))
    baseline_calls = [
        int(row["backward_calls"])
        for row in rows
        if row["method"] in ("uniform", "staged", "constrained") and row.get("backward_calls") is not None
    ]
    baseline_end = max(baseline_calls, default=cfg.get("uniform_updates", 4000))

    plot_rows = [
        row
        for row in rows
        if "scheduled" in str(row.get("evaluation_reason", ""))
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    for axis, metric in zip(axes, ("e1", "e2", "e3")):
        _plot_grouped_curves(axis, plot_rows, metric, state_filter=None)
        for boundary in boundaries:
            axis.axvline(boundary, color="#777777", linestyle="--", linewidth=1, alpha=0.7)
        for x_value, label in zip(
            (boundaries[0] / 2, (boundaries[0] + boundaries[1]) / 2, (boundaries[1] + baseline_end) / 2),
            ("Stage 1", "Stage 2", "Stage 3"),
        ):
            axis.text(x_value, 0.98, label, transform=axis.get_xaxis_transform(), ha="center", va="top", fontsize=8)
        axis.set_title(metric)
        axis.set_xlabel("Cumulative backward calls")
        axis.set_ylabel("Normalized training loss")
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(figures_dir / "training_losses.png", dpi=180)
    plt.close(fig)


def _plot_constraints_and_duals(rows: list[dict[str, Any]], figures_dir: Path) -> None:
    plt = _pyplot()
    plot_rows = [row for row in rows if "scheduled" in str(row.get("evaluation_reason", ""))]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for axis, metric in zip(axes.flat, ("constraint1", "constraint2", "lambda1", "lambda2")):
        valid = [row for row in plot_rows if row.get(metric) not in (None, "")]
        _plot_grouped_curves(axis, valid, metric, state_filter=None)
        axis.axhline(0.0, color="#333333", linewidth=1, alpha=0.6)
        axis.set_title(metric)
        axis.set_xlabel("Cumulative backward calls")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "constraints_and_duals.png", dpi=180)
    plt.close(fig)


def _plot_recursive_risks(rows: list[dict[str, Any]], figures_dir: Path) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(3, 2, figsize=(11, 11), sharex=True)
    for row_idx, step in enumerate((1, 2, 3)):
        for col_idx, distribution in enumerate(("id", "ood")):
            axis = axes[row_idx, col_idx]
            selected = [
                row
                for row in rows
                if row["state"] == "selected"
                and row["mode"] == "recursive"
                and row["step"] == step
                and row["distribution"] == distribution
            ]
            _plot_grouped_curves(axis, selected, "risk", state_filter=None, x_key="n")
            axis.set_xscale("log", base=2)
            axis.set_xticks([1, 2, 4, 8, 16, 32])
            axis.set_xticklabels(["1", "2", "4", "8", "16", "32"])
            axis.set_title(f"Step {step} - {distribution.upper()}")
            axis.set_xlabel("Demonstrations n")
            axis.set_ylabel("Raw recursive risk")
            axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "recursive_risks.png", dpi=180)
    plt.close(fig)


def _plot_grouped_curves(
    axis: Any,
    rows: list[dict[str, Any]],
    metric: str,
    *,
    state_filter: str | None,
    x_key: str = "backward_calls",
) -> None:
    by_method_x: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in rows:
        if state_filter is not None and row.get("state") != state_filter:
            continue
        x_value = row.get(x_key)
        y_value = row.get(metric)
        if x_value in (None, "") or y_value in (None, ""):
            continue
        by_method_x[(str(row["method"]), float(x_value))].append(float(y_value))

    methods = sorted({method for method, _ in by_method_x}, key=lambda value: list(METHOD_LABELS).index(value))
    for method in methods:
        points = sorted((x, values) for (candidate, x), values in by_method_x.items() if candidate == method)
        xs = [x for x, _ in points]
        means = [statistics.fmean(values) for _, values in points]
        stds = [statistics.stdev(values) if len(values) > 1 else 0.0 for _, values in points]
        color = METHOD_COLORS[method]
        axis.plot(xs, means, label=METHOD_LABELS[method], color=color, linewidth=2, marker="o", markersize=3)
        axis.fill_between(
            xs,
            [mean - std for mean, std in zip(means, stds)],
            [mean + std for mean, std in zip(means, stds)],
            color=color,
            alpha=0.15,
        )


def _load_checkpoints(checkpoint_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    payloads: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(checkpoint_dir.glob("*_seed*.pt")):
        method, seed_text = path.stem.rsplit("_seed", 1)
        payloads[(method, int(seed_text))] = _torch_load(path)
    return payloads


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _pyplot() -> Any:
    cache_dir = Path(tempfile.gettempdir()) / "three_level_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate experiment tables and figures from saved artifacts.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    generate_analysis(args.run_dir, make_figures=not args.no_figures)


if __name__ == "__main__":
    main()
