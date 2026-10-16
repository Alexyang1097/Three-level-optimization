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
- `tables/dual_history.csv`: Primal-Dual only, one row per dual-vector update (new checkpoints)
- `tables/dual_feedback_coverage.csv`: per-seed trace coverage, including missing legacy traces
- `tables/risk_long.csv`: recursive, teacher-forced, and propagation-gap risks
- `tables/compute.csv`: prompts, backward calls, evaluations, oracle calls, and elapsed time
- `tables/n32_summary.csv`: seed statistics at `n=32`
- `tables/n32_compact.csv`: compact selected/last performance, feasibility, and computation table
- `figures/training_losses.png`: `e1/e2/e3` curves with staged boundaries
- `figures/constraints_and_duals.png`: paired-threshold violations and dual weights
- `figures/dual_feedback_seed*.png`: per-seed, every-update Primal-Dual feedback, plus a
  `_zoom.png` of 100 updates around the largest absolute multiplier change in the final active
  phase (when that phase has more than 100 updates)
- `figures/recursive_risks.png`: three-step ID/OOD recursive-risk curves
- `figures/teacher_forced_risks.png`: matching three-step ID/OOD teacher-forced (RTF) risk curves

The recursive and RTF figures use selected checkpoints, identical method colors, and a base-2
demonstration-count axis. Corresponding step/distribution panels share the same y-axis limits for
direct comparison. Lines show seed means and shaded bands show plus/minus one sample standard
deviation. Both figures report raw step risks in the original target units.

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

## Primal-Dual feedback traces

New `constrained` checkpoints save `dual_history` automatically. Uniform, unconstrained stages,
and EMO do not collect this trace. The default 1,000/1,500/1,500 schedule produces **3,000 rows per
seed**: 1,500 with constraint 1 active and 1,500 with both constraints active. Stage 1 has no dual
rows; inactive constraint-2 fields in Stage 2 are empty, not zero.

Each row records `backward_calls` (the upcoming primal update index), `backward_calls_before`,
`optimizer_updates`, `phase`, `local_update`, `dual_reset`, `batch_size`, `dual_nu`, `dual_omega`,
and, for each active constraint `i`:

- `f_hat{i}`, `b{i}`, `c_hat{i}`: minibatch loss, threshold, and signed residual
- `lambda_before{i}`, `q{i}`: incoming multiplier and drive `c_hat - nu * lambda_before`
- `lambda_unprojected{i}`, `projection_active{i}`: proposed multiplier and whether it was clipped
- `lambda_after{i}`, `delta_lambda{i}`: multiplier used in the upcoming primal update and its change

All these signals are measured **before Adam** on the existing minibatch. The recorder uses detached
values and requires no additional model forward, backward, or full-pool evaluation. It does add scalar
logging/device-transfer overhead. The optimizer, selection rules, minibatch order, and compute counters
are unchanged. Multipliers are reset to zero when entering Stage 2 and again when entering Stage 3.

The new figures show residual, dual drive, projected multiplier change, and multiplier on separate
axes, without downsampling or averaging seeds. They mark resets and do not connect phases. Black
markers separately show full-training-pool residuals at the original evaluation frequency: diamonds
refer to post-Adam evaluations and squares to restored stage-transition snapshots. Minibatch signals
and full-pool residuals therefore differ both in data scope and parameter timing. No fixed-probe
evaluation is added. The existing `constraints_and_duals.png` remains a sparse, seed-averaged overview.

Feedback tables and figures can be generated as soon as training completes, without test-evaluation
JSON files. The usual analysis command above creates them automatically. `--no-figures` exports the
tables only. Old checkpoints have no per-update trace: their coverage is marked `missing` and no new
feedback curve is inferred from their sparse history. To collect the trace, rerun `constrained` training
in a new run directory; automatic suite resume otherwise skips already completed jobs.

## Optional EMO warm start

Cold start remains the default. Enable the optional strategy with `three_level.train_emo --warm-start`,
or `three_level.experiment_suite --emo-warm-start`. Use a new run directory, for example:

```bash
conda run -n pytorch_env python -m three_level.experiment_suite \
  --methods emo --seeds 11 --emo-warm-start --emo-backward-budget 4000 \
  --run-dir runs/emo_warm_start_suite
```

Warm start reuses observed candidate weights across threshold trials, including progress made in a
trial that failed its attempted threshold. Candidate selection first protects the current trial's
upstream constraints, then minimizes its final residual. Both inner and outer searches pass candidate
weights forward. Adam and the deterministic minibatch sequence still restart for each trained oracle.
Before training, cached full-pool metrics for the exact candidate state are checked against the new
thresholds; a feasible candidate returns without additional forward/backward calls. Cached snapshots
are explicitly labeled in history and do not increment full-pool evaluation counts.

The strategy also retains an independent incumbent: the lowest observed `f2` satisfying `f1 <= E1 +
epsilon` during Level 2, or the lowest observed `f3` satisfying both original upstream constraints
during the final solve. It supplies `selected_state`, including on budget exhaustion; on budget
exhaustion `last_state` remains the actual last iterate. Checkpoint/summary `selection` records the objective, rule, and update
where the selected state was observed. Thus the warm-start option changes both initialization and
budget-time selection; compare last iterates separately when isolating the initialization effect.

The returned recursive reference values still use feasible bisection thresholds, rather than replacing
`E2` with a measured loss. An unsuccessful finite Adam solve still does not certify mathematical
infeasibility. Warm start is an experimental option, not a guarantee of a better final model.

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
