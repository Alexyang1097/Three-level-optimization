from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch.utils.data import Dataset


INPUT_DIM = 10
STATE_DIM = 4
MAX_DEMOS = 32
NUM_TYPES = 4


@dataclass(frozen=True)
class TargetScales:
    """Fixed training-target normalization scales A_j."""

    a1: float
    a2: float
    a3: float

    def as_tensor(self, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.a1, device=device),
            torch.tensor(self.a2, device=device),
            torch.tensor(self.a3, device=device),
        )


class SyntheticCoTDataset(Dataset):
    """Prompt records from the synthetic three-step task in Section 7.1."""

    def __init__(
        self,
        size: int,
        seed: int,
        *,
        max_demos: int = MAX_DEMOS,
        sample_demo_count: bool = True,
        ood: bool = False,
    ) -> None:
        self.size = size
        self.max_demos = max_demos
        gen = torch.Generator().manual_seed(seed)

        self.w1 = torch.randn(size, STATE_DIM, INPUT_DIM, generator=gen) * (0.5**0.5)
        self.w2 = torch.randn(size, STATE_DIM, STATE_DIM, generator=gen) * (0.5**0.5)
        self.w3 = torch.randn(size, STATE_DIM, generator=gen)

        if ood:
            xi = torch.empty(size, INPUT_DIM).exponential_(1.0, generator=gen)
            std = (4.0 * xi).sqrt()
            demo_x = torch.randn(size, max_demos, INPUT_DIM, generator=gen) * std[:, None, :]
            query_x = torch.randn(size, INPUT_DIM, generator=gen) * std
        else:
            demo_x = torch.randn(size, max_demos, INPUT_DIM, generator=gen)
            query_x = torch.randn(size, INPUT_DIM, generator=gen)

        self.demo_x = demo_x
        self.query_x = query_x
        self.demo_s1, self.demo_s2, self.demo_y = apply_task(self.w1, self.w2, self.w3, demo_x)
        self.query_s1, self.query_s2, self.query_y = apply_task(self.w1, self.w2, self.w3, query_x)

        if sample_demo_count:
            self.demo_count = torch.randint(1, max_demos + 1, (size,), generator=gen)
        else:
            self.demo_count = torch.full((size,), max_demos, dtype=torch.long)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "demo_x": self.demo_x[idx],
            "demo_s1": self.demo_s1[idx],
            "demo_s2": self.demo_s2[idx],
            "demo_y": self.demo_y[idx],
            "query_x": self.query_x[idx],
            "query_s1": self.query_s1[idx],
            "query_s2": self.query_s2[idx],
            "query_y": self.query_y[idx],
            "n": self.demo_count[idx],
        }


def apply_task(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply s1=ReLU(W1x), s2=ReLU(W2s1), y=w3^T s2."""

    s1 = torch.relu(torch.einsum("bji,b...i->b...j", w1, x))
    s2 = torch.relu(torch.einsum("bji,b...i->b...j", w2, s1))
    y = torch.einsum("bi,b...i->b...", w3, s2).unsqueeze(-1)
    return s1, s2, y


def padded(values: torch.Tensor, dims: int) -> torch.Tensor:
    out = values.new_zeros(*values.shape[:-1], INPUT_DIM)
    out[..., :dims] = values
    return out


def compute_target_scales(dataset: SyntheticCoTDataset) -> TargetScales:
    """Compute A_j on the fixed training query targets."""

    a1 = dataset.query_s1.square().sum(dim=-1).mean().clamp_min(1e-8).item()
    a2 = dataset.query_s2.square().sum(dim=-1).mean().clamp_min(1e-8).item()
    a3 = dataset.query_y.square().sum(dim=-1).mean().clamp_min(1e-8).item()
    return TargetScales(a1=a1, a2=a2, a3=a3)


def collate_teacher_forcing(records: Iterable[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Create variable-length causal batches for teacher-forced training."""

    items = list(records)
    batch = len(items)
    max_n = max(int(item["n"]) for item in items)
    seq_len = 4 * max_n + 3

    tokens = items[0]["query_x"].new_zeros(batch, seq_len, INPUT_DIM)
    types = torch.zeros(batch, seq_len, dtype=torch.long)
    padding_mask = torch.ones(batch, seq_len, dtype=torch.bool)
    pred_pos = torch.zeros(batch, 3, dtype=torch.long)

    targets1 = items[0]["query_s1"].new_zeros(batch, STATE_DIM)
    targets2 = items[0]["query_s2"].new_zeros(batch, STATE_DIM)
    targets3 = items[0]["query_y"].new_zeros(batch, 1)

    for row, item in enumerate(items):
        n = int(item["n"])
        pos = 0
        for demo_idx in range(n):
            tokens[row, pos] = item["demo_x"][demo_idx]
            types[row, pos] = 0
            pos += 1
            tokens[row, pos] = padded(item["demo_s1"][demo_idx], STATE_DIM)
            types[row, pos] = 1
            pos += 1
            tokens[row, pos] = padded(item["demo_s2"][demo_idx], STATE_DIM)
            types[row, pos] = 2
            pos += 1
            tokens[row, pos] = padded(item["demo_y"][demo_idx], 1)
            types[row, pos] = 3
            pos += 1

        pred_pos[row, 0] = pos
        tokens[row, pos] = item["query_x"]
        types[row, pos] = 0
        pos += 1
        pred_pos[row, 1] = pos
        tokens[row, pos] = padded(item["query_s1"], STATE_DIM)
        types[row, pos] = 1
        pos += 1
        pred_pos[row, 2] = pos
        tokens[row, pos] = padded(item["query_s2"], STATE_DIM)
        types[row, pos] = 2
        pos += 1

        padding_mask[row, :pos] = False
        targets1[row] = item["query_s1"]
        targets2[row] = item["query_s2"]
        targets3[row] = item["query_y"]

    return {
        "tokens": tokens,
        "types": types,
        "padding_mask": padding_mask,
        "pred_pos": pred_pos,
        "target1": targets1,
        "target2": targets2,
        "target3": targets3,
    }


def make_eval_prompt_batch(
    dataset: SyntheticCoTDataset,
    indices: torch.Tensor,
    n: int,
    *,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Build fixed-length recursive-evaluation prompts with n demonstrations."""

    demo_x = dataset.demo_x[indices, :n].to(device)
    demo_s1 = dataset.demo_s1[indices, :n].to(device)
    demo_s2 = dataset.demo_s2[indices, :n].to(device)
    demo_y = dataset.demo_y[indices, :n].to(device)
    query_x = dataset.query_x[indices].to(device)

    batch = len(indices)
    seq_len = 4 * n + 1
    tokens = query_x.new_zeros(batch, seq_len, INPUT_DIM)
    types = torch.zeros(batch, seq_len, dtype=torch.long, device=device)

    pos = 0
    for demo_idx in range(n):
        tokens[:, pos] = demo_x[:, demo_idx]
        types[:, pos] = 0
        pos += 1
        tokens[:, pos] = padded(demo_s1[:, demo_idx], STATE_DIM)
        types[:, pos] = 1
        pos += 1
        tokens[:, pos] = padded(demo_s2[:, demo_idx], STATE_DIM)
        types[:, pos] = 2
        pos += 1
        tokens[:, pos] = padded(demo_y[:, demo_idx], 1)
        types[:, pos] = 3
        pos += 1

    tokens[:, pos] = query_x
    types[:, pos] = 0
    return {
        "tokens": tokens,
        "types": types,
        "target1": dataset.query_s1[indices].to(device),
        "target2": dataset.query_s2[indices].to(device),
        "target3": dataset.query_y[indices].to(device),
    }
