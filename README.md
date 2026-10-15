# Three-Level Synthetic CoT Optimization

This repo implements the Section 7 experimental pipeline from the research note:

- synthetic three-step tasks `s1 = ReLU(W1 x)`, `s2 = ReLU(W2 s1)`, `y = w3^T s2`
- teacher-forced Transformer training with normalized query-only losses
- recursive test-time prediction for `n in {1, 2, 4, 8, 16, 32}`
- training methods:
  - `uniform`: train `f3` for all updates
  - `staged`: train `f1`, then `f2`, then `f3` without constraints
  - `constrained`: recursive constrained stages with adaptive dual weights
  - `emo`: epigraph-based multilevel optimization with bisection over caps

## Setup

```bash
conda activate pytorch_env
python -m pip install -r requirements.txt
```

## Train

```bash
python -m three_level.train --method constrained --seed 11 --output runs/constrained_seed11.pt
python -m three_level.train --method uniform --seed 11 --output runs/uniform_seed11.pt
python -m three_level.train --method staged --seed 11 --output runs/staged_seed11.pt
python -m three_level.train_emo --seed 11 --output runs/emo_seed11.pt
```

For a tiny smoke run:

```bash
python -m three_level.train --method constrained --quick --output runs/smoke.pt
```

## Test

```bash
python -m three_level.test --checkpoint runs/constrained_seed11.pt --state selected
```

The test command reports teacher-forced losses on the ID test set plus recursive raw step risks on ID
and OOD test prompts.

## Full Suite

The robust suite runner trains and evaluates every `(method, seed)` as an independent job. The default
matrix is four methods by paired seeds `11 22 33`. Baselines use 4,000 backward calls; EMO is stopped
at the same global backward-call budget for the compute-matched comparison.

```bash
python -m three_level.experiment_suite --quick --run-dir runs/suite_smoke
python -m three_level.experiment_suite --run-dir runs/research_note_suite
```

The preflight quick run includes training, selected/last evaluation, aggregation, and plotting before
the full jobs begin. A resumed suite reuses completed checkpoints and only fills missing evaluations.
Use `--overwrite` to rerun completed work.

Run EMO to algorithm completion, without the 4,000-call cap, in a separate extension directory:

```bash
python -m three_level.experiment_suite \
  --methods emo \
  --emo-backward-budget 0 \
  --run-dir runs/emo_complete
```

Regenerate tables and figures without retraining:

```bash
python -m three_level.experiment_analysis --run-dir runs/research_note_suite
```

See `experiments/README.md` for the output layout.

## Estimate Runtime

Before starting the full experiment, run a proportional sample on the same device:

```bash
conda run -n pytorch_env python -m three_level.time_estimation \
  --fraction 0.05 \
  --output-dir runs/time_estimate
```

The estimator keeps the formal model, batch size, optimizer, losses, bisection depth, and prompt
lengths. It samples one paired seed and scales only dataset sizes, update counts, evaluation intervals,
and the EMO backward budget. It estimates the compute-matched main suite and the algorithm-complete
EMO extension separately. Read `runs/time_estimate/latest_estimate.md` for the concise report and
`latest_estimate.json` for the full component breakdown. Use `--main-only` to skip the complete-EMO
sample. A 5% to 10% sample is recommended; very small samples overemphasize startup and warmup costs.


python -m three_level.experiment_suite \
  --run-dir runs/main_budget_4000

pytorch_env python -m three_level.experiment_suite \
  --methods emo \
  --emo-backward-budget 0 \
  --run-dir runs/emo_complete
