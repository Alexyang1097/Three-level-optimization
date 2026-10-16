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
    families = {payload.get("config", {}).get("objective_family", "cumulative") for payload in payloads.values()}
    if len(families) > 1:
        raise ValueError("Cannot mix objective families in one analysis; use separate run directories.")
    paired_thresholds = {
        seed: payload.get("thresholds", {})
        for (method, seed), payload in payloads.items()
        if method == "constrained"
    }

    training_rows = _training_rows(payloads, paired_thresholds)
    dual_rows = _dual_rows(payloads)
    risk_rows = _risk_rows(run_dir / "eval")
    if len({row["objective_family"] for row in risk_rows}) > 1:
        raise ValueError("Cannot mix objective families in evaluation data; use separate run directories.")
    if families and any(row["objective_family"] not in families for row in risk_rows):
        raise ValueError("Evaluation objective family does not match checkpoints; regenerate evaluations in a separate run directory.")
    compute_rows = _compute_rows(payloads, run_dir / "eval")
    n32_rows = _n32_rows(risk_rows)
    compact_rows = _compact_n32_rows(payloads, paired_thresholds, risk_rows, compute_rows)

    _write_csv(tables_dir / "training_history.csv", training_rows)
    _write_csv(tables_dir / "dual_history.csv", dual_rows)
    _write_csv(tables_dir / "dual_feedback_coverage.csv", _dual_coverage_rows(payloads))
    _write_csv(tables_dir / "risk_long.csv", risk_rows)
    _write_csv(tables_dir / "gap_summary.csv", _gap_summary_rows(risk_rows))
    _write_csv(tables_dir / "compute.csv", compute_rows)
    _write_csv(tables_dir / "n32_summary.csv", n32_rows)
    _write_csv(tables_dir / "n32_compact.csv", compact_rows)

    if make_figures and training_rows:
        _plot_training_losses(training_rows, run_dir, figures_dir)
        _plot_training_losses(
            training_rows, run_dir, figures_dir,
            metrics=("f1", "f2", "f3"), filename="training_objectives.png", show_bounds=True,
        )
        if any(row.get("constraint1") is not None for row in training_rows):
            _plot_constraints_and_duals(training_rows, figures_dir)
    if make_figures and risk_rows:
        _plot_prediction_risks(risk_rows, figures_dir)
        _plot_risk_gaps(risk_rows, figures_dir)
    if make_figures and dual_rows:
        _plot_dual_feedback(dual_rows, training_rows, figures_dir)


def _dual_rows(payloads: dict[tuple[str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {**item, "method": method, "seed": seed,
         "objective_family": payload.get("config", {}).get("objective_family", "cumulative")}
        for (method, seed), payload in sorted(payloads.items())
        if method == "constrained"
        for item in payload.get("dual_history", [])
    ]


def _dual_coverage_rows(payloads: dict[tuple[str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for (method, seed), payload in sorted(payloads.items()):
        if method != "constrained":
            continue
        history = payload.get("dual_history", [])
        cfg = payload.get("config", {})
        counts = {phase: sum(item["phase"] == phase for item in history) for phase in ("stage2", "stage3")}
        expected = {phase: cfg.get(f"{phase}_updates") for phase in counts}
        complete = all(expected[phase] is not None and counts[phase] == expected[phase] for phase in counts)
        rows.append({
            "method": method, "seed": seed, "recorded_rows": len(history),
            "objective_family": cfg.get("objective_family", "cumulative"),
            "stage2_rows": counts["stage2"], "stage3_rows": counts["stage3"],
            "expected_stage2_rows": expected["stage2"], "expected_stage3_rows": expected["stage3"],
            "trace_status": "missing" if "dual_history" not in payload else "complete" if complete else "partial",
        })
    return rows


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
                "objective_family": payload.get("config", {}).get("objective_family", "cumulative"),
                "backward_calls": item.get("backward_calls", item.get("update")),
                "optimizer_updates": item.get("optimizer_updates", item.get("update")),
                "training_prompts": item.get("training_prompts"),
                "full_train_evaluations": item.get("full_train_evaluations"),
                "phase": item.get("phase", item.get("stage", item.get("emo_level", ""))),
                "evaluation_reason": item.get("evaluation_reason", "legacy"),
                "oracle": item.get("oracle"),
                "cap": item.get("cap"),
                "active": item.get("active"),
                "warm_start": payload.get("config", {}).get("warm_start", False) if method == "emo" else None,
                "zero_step_feasible": item.get("zero_step_feasible"),
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
                        "objective_family": payload.get("objective_family", "cumulative"),
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
                "objective_family": payload.get("config", {}).get("objective_family", "cumulative"),
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


def _gap_summary_rows(risk_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate differences paired within each seed, not independent errors."""
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in risk_rows:
        if row["mode"] == "gap":
            key = (row["method"], row["state"], row["distribution"], row["step"], row["n"])
            grouped[key].append(float(row["risk"]))
    return [
        {
            "method": method, "state": state, "distribution": distribution, "step": step, "n": n,
            "seed_count": len(values), "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "mean_absolute_seed_gap": statistics.fmean(abs(value) for value in values),
        }
        for (method, state, distribution, step, n), values in sorted(grouped.items())
    ]


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
            family = next(payload.get("config", {}).get("objective_family", "cumulative")
                          for (candidate, _), payload in payloads.items() if candidate == method)
            row: dict[str, Any] = {"method": method, "state": state, "objective_family": family}
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


def _plot_training_losses(
    rows: list[dict[str, Any]], run_dir: Path, figures_dir: Path, *,
    metrics: tuple[str, ...] = ("e1", "e2", "e3"),
    filename: str = "training_losses.png", show_bounds: bool = False,
) -> None:
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
    staged_methods_present = any(row["method"] in ("staged", "constrained") for row in rows)
    per_step = any(row.get("objective_family") == "per_step" for row in rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8 if show_bounds else 4.2), sharex=True)
    for axis, metric in zip(axes, metrics):
        _plot_grouped_curves(axis, plot_rows, metric, state_filter=None)
        bound_key = {"f1": "b1", "f2": "b2"}.get(metric)
        if show_bounds and bound_key is not None:
            by_seed = {
                row["seed"]: float(row[bound_key])
                for row in rows if row.get(bound_key) is not None
            }
            if by_seed and plot_rows:
                values = list(by_seed.values())
                mean = statistics.fmean(values)
                std = statistics.stdev(values) if len(values) > 1 else 0.0
                start = boundaries[0 if bound_key == "b1" else 1]
                end = max(float(row["backward_calls"]) for row in plot_rows)
                axis.plot([start, end], [mean, mean], color="#444444", linestyle="--", linewidth=1.4)
                axis.fill_between([start, end], mean - std, mean + std, color="#444444", alpha=0.12)
                axis.annotate(
                    f"{bound_key} (mean)", xy=(0.98, mean), xycoords=axis.get_yaxis_transform(),
                    xytext=(0, 5), textcoords="offset points", ha="right", fontsize=8, color="#444444",
                )
        for boundary in boundaries if staged_methods_present else []:
            axis.axvline(boundary, color="#777777", linestyle="--", linewidth=1, alpha=0.7)
        stage_labels = zip(
            (boundaries[0] / 2, (boundaries[0] + boundaries[1]) / 2, (boundaries[1] + baseline_end) / 2),
            ("Stage 1", "Stage 2", "Stage 3"),
        ) if staged_methods_present else []
        for x_value, label in stage_labels:
            axis.text(x_value, 0.98, label, transform=axis.get_xaxis_transform(), ha="center", va="top", fontsize=8)
        axis.set_title(metric)
        axis.set_xlabel("Cumulative backward calls")
        axis.set_ylabel(("Normalized per-step objective" if per_step else "Normalized cumulative objective")
                        if show_bounds else "Normalized training loss")
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    mixed_emo = staged_methods_present and any(row["method"] == "emo" for row in rows)
    legend_y = (0.09 if mixed_emo else 0.045) if show_bounds else (0.045 if mixed_emo else 0)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, legend_y), ncol=4, frameon=False)
    if mixed_emo:
        fig.text(0.5, 0.015, "Stage boundaries apply to Staged and Primal-Dual only; EMO has its own search phases.", ha="center", fontsize=8)
    if show_bounds:
        fig.text(
            0.5, 0.06 if mixed_emo else 0.015,
            "Grey bounds: paired Primal-Dual thresholds, mean +/- seed SD. Mean curves do not establish per-seed feasibility.",
            ha="center", fontsize=8,
        )
    bottom = (0.22 if mixed_emo else 0.17) if show_bounds else (0.14 if mixed_emo else 0.09)
    if per_step:
        fig.suptitle("Per-step setting: f1 = e1, f2 = e2, f3 = e3", fontsize=12)
        if show_bounds:
            fig.text(0.5, 0.125, "Uniform optimizes mean(e1, e2, e3); staged methods optimize f1, then f2, then f3.",
                     ha="center", fontsize=8)
    fig.tight_layout(rect=(0, bottom, 1, 0.93 if per_step else 1))
    fig.savefig(figures_dir / filename, dpi=180)
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


def _plot_dual_feedback(
    rows: list[dict[str, Any]], training_rows: list[dict[str, Any]], figures_dir: Path,
) -> None:
    """Keep every update and each seed; never connect across dual resets."""
    plt = _pyplot()
    metrics = (
        ("c_hat", "Minibatch residual c = f_hat - b", "#2878B5"),
        ("q", "Dual drive q = c - nu * lambda_before", "#E69F00"),
        ("delta_lambda", "Projected change in lambda", "#2A9D8F"),
        ("lambda_after", "Lambda after dual update (before Adam)", "#D1495B"),
    )
    for seed in sorted({row["seed"] for row in rows}):
        seed_rows = sorted((row for row in rows if row["seed"] == seed), key=lambda row: row["backward_calls"])
        phases = list(dict.fromkeys(row["phase"] for row in seed_rows))
        windows = [("", seed_rows, "all dual updates")]
        final_phase_rows = [row for row in seed_rows if row["phase"] == phases[-1]]
        if len(final_phase_rows) > 100:
            peak = max(
                range(len(final_phase_rows)),
                key=lambda index: max(abs(final_phase_rows[index].get(f"delta_lambda{i}", 0)) for i in (1, 2)),
            )
            start = min(max(0, peak - 50), len(final_phase_rows) - 100)
            windows.append((
                "_zoom", final_phase_rows[start:start + 100],
                f"100 updates around largest |delta lambda| in {phases[-1]}",
            ))
        full_rows = [row for row in training_rows if row["method"] == "constrained" and row["seed"] == seed]
        for suffix, window, title in windows:
            fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True, squeeze=False)
            xmin = window[0]["backward_calls_before"]
            xmax = window[-1]["backward_calls"]
            for col, constraint in enumerate((1, 2)):
                active_phases = {
                    row["phase"] for row in window if f"c_hat{constraint}" in row
                }
                active_rows = [row for row in window if f"c_hat{constraint}" in row]
                active_start = active_rows[0]["backward_calls_before"] if active_rows else xmax
                for line, (metric, ylabel, color) in enumerate(metrics):
                    axis = axes[line, col]
                    if active_start > xmin:
                        axis.axvspan(xmin, active_start, color="#EEEEEE", zorder=0)
                        if line == 0:
                            axis.text(
                                (xmin + active_start) / 2, 0.5, "Not active",
                                transform=axis.get_xaxis_transform(), ha="center", color="#777777",
                            )
                    key = f"{metric}{constraint}"
                    labeled = False
                    for phase in phases:
                        points = [row for row in window if row["phase"] == phase and key in row]
                        if not points:
                            continue
                        axis.plot(
                            [row["backward_calls"] for row in points], [row[key] for row in points],
                            color=color, linewidth=0.85, marker=".", markersize=2,
                            label="Every dual update" if not labeled else None,
                        )
                        labeled = True
                        if metric == "lambda_after" and points[0]["dual_reset"]:
                            axis.scatter(
                                [points[0]["backward_calls_before"]], [points[0][f"lambda_before{constraint}"]],
                                color=color, marker="s", s=24, zorder=4,
                            )
                    for row in seed_rows:
                        boundary = row["backward_calls_before"]
                        if row["dual_reset"] and row["phase"] in active_phases and xmin <= boundary <= xmax:
                            axis.axvline(boundary, color="#777777", linestyle="--", linewidth=1)
                            if line == 0:
                                axis.text(
                                    boundary, 0.98, f" {row['phase']}: reset", fontsize=8,
                                    transform=axis.get_xaxis_transform(), va="top",
                                )
                    if metric == "c_hat":
                        for transition, marker, label in (
                            (False, "D", "Full train: after Adam"),
                            (True, "s", "Full train: restored snapshot"),
                        ):
                            snapshots = [
                                row for row in full_rows
                                if row["phase"] in active_phases and f"constraint{constraint}" in row
                                and xmin <= row["backward_calls"] <= xmax
                                and ("stage_transition" in row["evaluation_reason"]) == transition
                            ]
                            if snapshots:
                                axis.scatter(
                                    [row["backward_calls"] for row in snapshots],
                                    [row[f"constraint{constraint}"] for row in snapshots],
                                    marker=marker, s=24, color="#222222", zorder=5, label=label,
                                )
                    if labeled:
                        axis.legend(loc="best", fontsize=7, frameon=False)
                    else:
                        axis.text(0.5, 0.5, "Constraint inactive in this window", ha="center", transform=axis.transAxes)
                    axis.axhline(0, color="#555555", linewidth=0.7, alpha=0.6)
                    axis.set_ylabel(ylabel, fontsize=9)
                    axis.set_xlim(xmin, xmax)
                    axis.grid(alpha=0.18)
                    if line == 0:
                        axis.set_title(f"Constraint {constraint}")
                    if line == 3:
                        axis.set_xlabel("Cumulative primal update index")
            fig.suptitle(f"Primal-Dual feedback | seed {seed} | {title}", fontsize=14)
            fig.text(
                0.5, 0.015,
                "Each trace point uses the minibatch BEFORE that Adam update. Black markers are full-pool snapshots.\n"
                "Dashed lines / square initial lambda markers indicate stage resets; phases are not connected.",
                ha="center", fontsize=9,
            )
            fig.tight_layout(rect=(0, 0.055, 1, 0.97))
            fig.savefig(figures_dir / f"dual_feedback_seed{seed}{suffix}.png", dpi=180)
            plt.close(fig)


def _plot_prediction_risks(rows: list[dict[str, Any]], figures_dir: Path) -> None:
    """Plot recursive and teacher-forced risks with matching panel scales."""
    plt = _pyplot()
    plots = []
    for mode, ylabel, filename in (
        ("recursive", "Raw recursive risk", "recursive_risks.png"),
        ("teacher_forced", "Raw teacher-forced risk (RTF)", "teacher_forced_risks.png"),
    ):
        fig, axes = plt.subplots(3, 2, figsize=(11, 11), sharex=True)
        for row_idx, step in enumerate((1, 2, 3)):
            for col_idx, distribution in enumerate(("id", "ood")):
                axis = axes[row_idx, col_idx]
                selected = [
                    row
                    for row in rows
                    if row["state"] == "selected"
                    and row["mode"] == mode
                    and row["step"] == step
                    and row["distribution"] == distribution
                ]
                _plot_grouped_curves(axis, selected, "risk", state_filter=None, x_key="n")
                axis.set_xscale("log", base=2)
                axis.set_xticks([1, 2, 4, 8, 16, 32])
                axis.set_xticklabels(["1", "2", "4", "8", "16", "32"])
                axis.set_title(f"Step {step} - {distribution.upper()}")
                axis.set_xlabel("Demonstrations n")
                axis.set_ylabel(ylabel)
                axis.grid(alpha=0.2)
        axes[0, 0].legend(frameon=False)
        plots.append((fig, axes, filename))

    # Match each step/distribution across figures, including the seed-variation
    # bands, so differences cannot be obscured by independent autoscaling.
    for paired_axes in zip(*(axes.flat for _, axes, _ in plots)):
        limits = [axis.get_ylim() for axis in paired_axes]
        bottom = min(low for low, _ in limits)
        top = max(high for _, high in limits)
        for axis in paired_axes:
            axis.set_ylim(bottom, top)

    for fig, _, filename in plots:
        fig.tight_layout()
        fig.savefig(figures_dir / filename, dpi=180)
        plt.close(fig)


def _plot_risk_gaps(rows: list[dict[str, Any]], figures_dir: Path) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(3, 2, figsize=(11, 11), sharex=True)
    for line, step in enumerate((1, 2, 3)):
        for col, distribution in enumerate(("id", "ood")):
            axis = axes[line, col]
            selected = [
                row for row in rows
                if row["state"] == "selected" and row["mode"] == "gap"
                and row["step"] == step and row["distribution"] == distribution
            ]
            _plot_grouped_curves(axis, selected, "risk", state_filter=None, x_key="n")
            axis.axhline(0, color="#333333", linewidth=1, linestyle="--", alpha=0.7)
            axis.set_xscale("log", base=2)
            axis.set_xticks([1, 2, 4, 8, 16, 32], labels=["1", "2", "4", "8", "16", "32"])
            axis.set_title(f"Step {step} - {distribution.upper()}")
            axis.set_xlabel("Demonstrations n")
            axis.set_ylabel("Signed gap: Rrec - RTF (raw risk)")
            axis.grid(alpha=0.2)
            if step == 1:
                # Avoid magnifying floating-point noise into an apparent effect.
                axis.set_ylim(-0.05, 0.05)
                axis.text(0.5, 0.9, "Rrec = RTF up to numerical precision", ha="center", transform=axis.transAxes, fontsize=9)
    axes[0, 0].legend(frameon=False, loc="lower left")
    fig.suptitle("Recursive minus teacher-forced risk | selected checkpoints", fontsize=13)
    fig.text(
        0.5, 0.015,
        "Differences are paired within each seed; bands show +/- one sample standard deviation.\n"
        "A smaller gap alone does not imply a lower recursive risk.", ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(figures_dir / "recursive_rtf_gap.png", dpi=180)
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
