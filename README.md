# Three-Level Synthetic CoT Optimization

This repo implements the Section 7 experimental pipeline from the research note:

- synthetic three-step tasks `s1 = ReLU(W1 x)`, `s2 = ReLU(W2 s1)`, `y = w3^T s2`
- teacher-forced Transformer training with normalized query-only losses
- recursive test-time prediction for `n in {1, 2, 4, 8, 16, 32}`
- three training methods:
  - `uniform`: train `f3` for all updates
  - `staged`: train `f1`, then `f2`, then `f3` without constraints
  - `constrained`: recursive constrained stages with adaptive dual weights

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
