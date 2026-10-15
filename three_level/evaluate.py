from __future__ import annotations

from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from .data import (
    STATE_DIM,
    SyntheticCoTDataset,
    TargetScales,
    collate_teacher_forcing,
    make_eval_prompt_batch,
    padded,
)
from .model import CausalCoTTransformer
from .objectives import compute_losses


@torch.no_grad()
def evaluate_teacher_forced(
    model: CausalCoTTransformer,
    dataset: SyntheticCoTDataset,
    scales: TargetScales,
    *,
    batch_size: int,
    device: torch.device | str,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = defaultdict(float)
    count = 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_teacher_forcing)
    for batch in loader:
        actual = batch["tokens"].shape[0]
        batch = _move_batch(batch, device)
        outputs = model(batch["tokens"], batch["types"], batch["padding_mask"])
        losses = compute_losses(outputs, batch, scales).floats()
        for key, value in losses.items():
            totals[key] += value * actual
        count += actual
    return {key: value / count for key, value in totals.items()}


@torch.no_grad()
def evaluate_recursive(
    model: CausalCoTTransformer,
    dataset: SyntheticCoTDataset,
    *,
    n_values: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    batch_size: int,
    device: torch.device | str,
) -> dict[int, dict[str, float]]:
    model.eval()
    results: dict[int, dict[str, float]] = {}
    all_indices = torch.arange(len(dataset))

    for n in n_values:
        totals = {"rec1": 0.0, "rec2": 0.0, "rec3": 0.0, "tf1": 0.0, "tf2": 0.0, "tf3": 0.0}
        count = 0
        for start in range(0, len(dataset), batch_size):
            indices = all_indices[start : start + batch_size]
            prompt = make_eval_prompt_batch(dataset, indices, n, device=device)
            rec1, rec2, rec3 = recursive_predict(model, prompt["tokens"], prompt["types"])
            tf1, tf2, tf3 = teacher_forced_predict(model, prompt, n)
            target1 = prompt["target1"]
            target2 = prompt["target2"]
            target3 = prompt["target3"]
            actual = len(indices)

            totals["rec1"] += (rec1 - target1).square().sum(dim=-1).sum().item()
            totals["rec2"] += (rec2 - target2).square().sum(dim=-1).sum().item()
            totals["rec3"] += (rec3 - target3).square().sum(dim=-1).sum().item()
            totals["tf1"] += (tf1 - target1).square().sum(dim=-1).sum().item()
            totals["tf2"] += (tf2 - target2).square().sum(dim=-1).sum().item()
            totals["tf3"] += (tf3 - target3).square().sum(dim=-1).sum().item()
            count += actual

        averaged = {key: value / count for key, value in totals.items()}
        for step in (1, 2, 3):
            averaged[f"delta{step}"] = averaged[f"rec{step}"] - averaged[f"tf{step}"]
        results[n] = averaged
    return results


def recursive_predict(
    model: CausalCoTTransformer,
    tokens: torch.Tensor,
    types: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out1 = model(tokens, types)
    s1 = out1[:, -1, :STATE_DIM]

    tokens2 = torch.cat([tokens, padded(s1, STATE_DIM).unsqueeze(1)], dim=1)
    types2 = torch.cat([types, torch.ones(types.shape[0], 1, dtype=torch.long, device=types.device)], dim=1)
    out2 = model(tokens2, types2)
    s2 = out2[:, -1, :STATE_DIM]

    tokens3 = torch.cat([tokens2, padded(s2, STATE_DIM).unsqueeze(1)], dim=1)
    type_two = torch.full((types.shape[0], 1), 2, dtype=torch.long, device=types.device)
    types3 = torch.cat([types2, type_two], dim=1)
    out3 = model(tokens3, types3)
    y = out3[:, -1, :1]
    return s1, s2, y


def teacher_forced_predict(
    model: CausalCoTTransformer,
    prompt: dict[str, torch.Tensor],
    n: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = prompt["tokens"]
    types = prompt["types"]
    tokens_tf = torch.cat(
        [
            tokens,
            padded(prompt["target1"], STATE_DIM).unsqueeze(1),
            padded(prompt["target2"], STATE_DIM).unsqueeze(1),
        ],
        dim=1,
    )
    type_tail = torch.tensor([1, 2], dtype=torch.long, device=types.device).expand(types.shape[0], 2)
    types_tf = torch.cat([types, type_tail], dim=1)
    outputs = model(tokens_tf, types_tf)
    xq_pos = 4 * n
    s1_pos = xq_pos + 1
    s2_pos = xq_pos + 2
    return outputs[:, xq_pos, :4], outputs[:, s1_pos, :4], outputs[:, s2_pos, :1]


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device | str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}
