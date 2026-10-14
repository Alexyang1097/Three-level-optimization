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

The robust suite runner trains and evaluates every `(method, seed)` as an independent job. It writes
checkpoints, JSON outputs, logs, an append-only manifest, and flattened summaries under `--run-dir`.
Completed jobs are skipped on resume unless `--overwrite` is set.

```bash
python -m three_level.experiment_suite --quick --run-dir runs/suite_smoke
python -m three_level.experiment_suite --run-dir runs/research_note_suite
```

See `experiments/README.md` for the output layout.
