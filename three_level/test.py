from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import SyntheticCoTDataset, TargetScales
from .evaluate import evaluate_recursive, evaluate_teacher_forced
from .model import CausalCoTTransformer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained three-level checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state", choices=["selected", "last"], default="selected")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu")
    cfg = payload["config"]
    device = torch.device(args.device)

    model = CausalCoTTransformer().to(device)
    state_key = "selected_state" if args.state == "selected" else "last_state"
    model.load_state_dict(payload[state_key])

    scales = TargetScales(**payload["scales"])
    test_seed = int(cfg.get("test_seed", 30_000))
    test_id = SyntheticCoTDataset(int(cfg["test_size"]), seed=test_seed, sample_demo_count=False, ood=False)
    test_ood = SyntheticCoTDataset(int(cfg["test_size"]), seed=test_seed, sample_demo_count=False, ood=True)

    result = {
        "checkpoint": str(args.checkpoint),
        "state": args.state,
        "thresholds": payload.get("thresholds", {}),
        "teacher_forced_id": evaluate_teacher_forced(
            model,
            test_id,
            scales,
            batch_size=args.batch_size,
            device=device,
        ),
        "recursive_id": evaluate_recursive(model, test_id, batch_size=args.batch_size, device=device),
        "recursive_ood": evaluate_recursive(model, test_ood, batch_size=args.batch_size, device=device),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
