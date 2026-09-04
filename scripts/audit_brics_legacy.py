#!/usr/bin/env python3
"""Re-evaluate saved legacy checkpoints with pooled metrics; no training."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import pandas as pd
from hydra.utils import instantiate
from omegaconf import OmegaConf

from brics_audit_suite import binary_metrics, evaluate_pairs, load_checkpoint, write_json


def saved_metrics(path, train_csv):
    frame = pd.read_csv(path)
    train = pd.read_csv(train_csv)
    known = {hashlib.sha256(p.strip().encode()).hexdigest() for p in train["Target Sequence"]}
    mask = torch.tensor(frame.protein_sha256.isin(known).tolist())
    labels = torch.tensor(frame.label.tolist())
    result = {}
    for view in ("mean", "fragment", "whole"):
        pred = torch.tensor(frame[view].tolist())
        result[view] = binary_metrics(pred, labels)
        for cohort, selected in (("seen_protein", mask), ("unseen_protein", ~mask)):
            result[f"{view}_{cohort}"] = binary_metrics(pred[selected], labels[selected])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true", help="Reuse saved row predictions in this output directory")
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
            output = args.output / f"{path.stem}_{split}.csv"
            if args.resume and output.exists():
                values[split] = saved_metrics(output, dm.train_csv)
            else:
                values[split] = evaluate_pairs(model, dm, split, torch.device(args.device), output)
            result["checkpoints"][path.name] = values
            write_json(args.output / "pooled_metrics.json", result)
            print(f"Evaluated {path.name} {split}", flush=True)
    result["complete"] = True
    write_json(args.output / "pooled_metrics.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
