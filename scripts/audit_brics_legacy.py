#!/usr/bin/env python3
"""Re-evaluate saved legacy checkpoints with pooled metrics; no training."""

import argparse
import json
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from brics_audit_suite import evaluate_pairs, load_checkpoint, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = OmegaConf.load(args.run_dir / "resolved_config.yaml")
    dm = instantiate(cfg.data)
    model = instantiate(cfg.model)
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "source_run": str(args.run_dir),
        "note": "Original overlapping splits, corrected pooled metrics only",
        "checkpoints": {},
    }
    for path in sorted((args.run_dir / "checkpoints").glob("*.ckpt")):
        epoch = load_checkpoint(model, path)
        values = {"epoch": epoch, "path": str(path)}
        for split in ("val", "test"):
            values[split] = evaluate_pairs(
                model, dm, split, torch.device(args.device), args.output / f"{path.stem}_{split}.csv"
            )
        result["checkpoints"][path.name] = values
    write_json(args.output / "pooled_metrics.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
