#!/usr/bin/env python3
"""Training-only receptor priors and exact fragment-input ambiguity diagnostics."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    frames = {s: pd.read_csv(args.data / f"{s}.csv") for s in ("train", "val", "test")}
    train = frames["train"]
    prior = float(train.Label.mean())
    counts = train.groupby("Target Sequence").Label.agg(["sum", "count"])
    smoothed = (counts["sum"] + 10 * prior) / (counts["count"] + 10)
    result = {"training_prevalence": prior, "protein_prior_pseudocount": 10, "splits": {}}
    for split in ("val", "test"):
        df = frames[split]
        pred = df["Target Sequence"].map(smoothed).fillna(prior)
        result["splits"][split] = {
            "constant_prior_auprc": float(df.Label.mean()),
            "protein_only_prior": {
                "auprc": float(average_precision_score(df.Label, pred)),
                "auroc": float(roc_auc_score(df.Label, pred)),
                "brier": float(brier_score_loss(df.Label, pred)),
            },
        }
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    unique = pd.concat(frames.values()).drop_duplicates("canonical_smiles")
    buckets = defaultdict(list)
    for _, row in unique.iterrows():
        fragment_hashes = sorted(hashlib.sha256(fp.numpy().tobytes()).hexdigest() for fp in cache[row.SMILES])
        buckets[tuple(fragment_hashes)].append(row.canonical_smiles)
    collisions = [values for values in buckets.values() if len(values) > 1]
    result["identical_fragment_input"] = {
        "unique_molecules": len(unique),
        "unique_fragment_multisets": len(buckets),
        "ambiguous_molecules": sum(map(len, collisions)),
        "ambiguous_groups": len(collisions),
        "examples": collisions[:10],
        "note": "Distinct canonical molecules with exactly equal multisets of cached fingerprints before FSQ.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
