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
    signatures = {}
    for _, row in unique.iterrows():
        fragment_hashes = sorted(hashlib.sha256(fp.numpy().tobytes()).hexdigest() for fp in cache[row.SMILES])
        buckets[tuple(fragment_hashes)].append(row.canonical_smiles)
        signatures[row.canonical_smiles] = hashlib.sha256("".join(fragment_hashes).encode()).hexdigest()
    collisions = [values for values in buckets.values() if len(values) > 1]
    train_molecules = set(frames["train"].canonical_smiles)
    shared_with_train = set()
    for values in buckets.values():
        if train_molecules.intersection(values):
            shared_with_train.update(values)
    result["same_fragment_input_as_training"] = {
        s: {
            "molecules": len(set(frames[s].canonical_smiles) & shared_with_train),
            "total_molecules": frames[s].canonical_smiles.nunique(),
            "rows": int(frames[s].canonical_smiles.isin(shared_with_train).sum()),
        }
        for s in ("val", "test")
    }
    result["identical_fragment_input"] = {
        "unique_molecules": len(unique),
        "unique_fragment_multisets": len(buckets),
        "ambiguous_molecules": sum(map(len, collisions)),
        "ambiguous_groups": len(collisions),
        "examples": collisions[:10],
        "note": "Distinct canonical molecules with exactly equal multisets of cached fingerprints before FSQ.",
    }
    for split in ("val", "test"):
        frame = frames[split]
        groups = {}
        for seen in (False, True):
            group = frame[frame.canonical_smiles.isin(shared_with_train) == seen]
            pred = group["Target Sequence"].map(smoothed).fillna(prior)
            both = group.Label.nunique() == 2
            groups[f"fragment_input_seen_in_train_{seen}"] = {
                "rows": len(group),
                "positive_rate": float(group.Label.mean()) if len(group) else None,
                "auprc": float(average_precision_score(group.Label, pred)) if both else None,
                "auroc": float(roc_auc_score(group.Label, pred)) if both else None,
                "brier": float(brier_score_loss(group.Label, pred)) if len(group) else None,
            }
        result["splits"][split]["protein_prior_by_fragment_input_novelty"] = groups
    args.output.parent.mkdir(parents=True, exist_ok=True)
    novelty_rows = []
    for split, frame in frames.items():
        molecules = frame.drop_duplicates("canonical_smiles")[["SMILES", "canonical_smiles"]].copy()
        molecules["split"] = split
        molecules["fragment_input_sha256"] = molecules.canonical_smiles.map(signatures)
        molecules["fragment_input_seen_in_train"] = molecules.canonical_smiles.isin(shared_with_train)
        novelty_rows.append(molecules)
    pd.concat(novelty_rows).to_csv(args.output.parent / "fragment_input_groups.csv", index=False)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
