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

Default methods are:

- `uniform`
- `staged`
- `constrained`
- `emo`

Default seeds are `11 22 33`.

Run a smoke check:

```bash
conda run -n pytorch_env python -m three_level.experiment_suite --quick --run-dir runs/suite_smoke
```

Run the full suite:

```bash
conda run -n pytorch_env python -m three_level.experiment_suite --run-dir runs/research_note_suite
```

Resume is automatic: completed `(method, seed)` jobs are skipped unless `--overwrite` is set.
