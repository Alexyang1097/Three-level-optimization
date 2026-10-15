# Experiment Suite

This directory documents the robust runner for the research-note experiment.

The suite runs each `(method, seed)` as an independent job and writes durable outputs after every
train/eval step:

- `checkpoints/*.pt`: train artifacts compatible with `three_level.test`
- `train/*.json`: train summary JSON captured from the training entry point
- `eval/*_selected.json` and `eval/*_last.json`: test outputs for both selected and last states
- `logs/*.log`: stdout/stderr for each subprocess
- `manifest.jsonl`: append-only job status stream
- `summary.csv` and `summary.json`: flattened evaluation metrics
- `tables/training_history.csv`: full-training losses against cumulative backward calls
- `tables/risk_long.csv`: recursive, teacher-forced, and propagation-gap risks
- `tables/compute.csv`: prompts, backward calls, evaluations, oracle calls, and elapsed time
- `tables/n32_summary.csv`: seed statistics at `n=32`
- `tables/n32_compact.csv`: compact selected/last performance, feasibility, and computation table
- `figures/training_losses.png`: `e1/e2/e3` curves with staged boundaries
- `figures/constraints_and_duals.png`: paired-threshold violations and dual weights
- `figures/recursive_risks.png`: three-step ID/OOD recursive-risk curves

Default methods are:

- `uniform`
- `staged`
- `constrained`
- `emo`

Default seeds are `11 22 33`.

All four methods use cumulative backward calls as the optimization-compute axis. Uniform, staged, and
primal-dual training use 4,000 calls. The default suite gives EMO the same 4,000-call budget and saves
its current state with a `budget_exhausted:<phase>` status if the nested solve has not completed. This
is a valid compute-matched outcome, not a failed job. Use `--emo-backward-budget 0` for a separate
algorithm-completion run.

For every seed, the constrained run supplies `b1` and `b2`. Aggregation applies those paired thresholds
to every method when computing `f1-b1`, `f2-b2`, and empirical feasibility. EMO's own `E1`, `E2`,
`z1`, and `z2` remain separate internal diagnostics.

Run a smoke check:

```bash
conda run -n pytorch_env python -m three_level.experiment_suite --quick --run-dir runs/suite_smoke
```

Run the full suite:

```bash
conda run -n pytorch_env python -m three_level.experiment_suite --run-dir runs/research_note_suite
```

Resume is automatic: completed `(method, seed)` jobs are skipped unless `--overwrite` is set.
If training completed but one evaluation failed, resume reuses the checkpoint and runs only the missing
selected or last evaluation. Checkpoints are written atomically.

Regenerate analysis products from existing artifacts:

```bash
conda run -n pytorch_env python -m three_level.experiment_analysis \
  --run-dir runs/research_note_suite
```

## Runtime Estimation

Run the estimator on the same device that will execute the formal suites:

```bash
conda run -n pytorch_env python -m three_level.time_estimation \
  --fraction 0.05 \
  --output-dir runs/time_estimate
```

The sample uses the formal architecture, batch size, optimizer, losses, bisection depth, prompt
lengths, and evaluation modes. It reduces the train/test pools, updates, evaluation interval, EMO
inner updates, EMO budget, and sampled seeds. The report separates:

- optimization and EMO oracle work
- full training-pool evaluations
- training and test data construction
- model/process setup and checkpoint I/O
- selected/last ID and OOD test kernels
- test process and model setup
- preflight, aggregation, and plotting

Outputs are written to `runs/time_estimate/latest_estimate.md` and `latest_estimate.json`; raw sample
artifacts are retained in a timestamped subdirectory. The complete-EMO estimate includes a sensitivity
range because feasibility decisions can change its number of oracle calls. Use `--main-only` when only
the 4-by-3 compute-matched suite is planned.
