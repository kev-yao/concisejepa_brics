#!/usr/bin/env python3
"""Build disjoint canonical-molecule splits; retain raw SMILES for cache lookup."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem


def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    # Preserve stereochemistry and all components, including counterions.
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def partition_pairs(frames, seed=17):
    frame = pd.concat([df.assign(source_split=name) for name, df in frames.items()], ignore_index=True)
    frame = frame.copy()
    frame["SMILES"] = frame["SMILES"].astype(str).str.strip()
    frame["Target Sequence"] = frame["Target Sequence"].astype(str).str.strip()
    if not frame.Label.isin([0, 1]).all():
        raise ValueError("Expected binary labels")
    canonical = {s: canonicalize(s) for s in frame.SMILES.unique()}
    frame["canonical_smiles"] = frame.SMILES.map(canonical)
    invalid = int(frame.canonical_smiles.isna().sum())
    frame = frame.dropna(subset=["canonical_smiles"])
    keys = ["canonical_smiles", "Target Sequence"]
    conflicts = frame.groupby(keys).Label.transform("nunique") > 1
    excluded = frame.loc[conflicts].copy()
    frame = frame.loc[~conflicts]
    n_pre_dedup = len(frame)
    # Stable representative raw SMILES retains compatibility with the caches.
    frame = frame.sort_values(keys + ["SMILES", "source_split"]).drop_duplicates(keys).reset_index(drop=True)
    frame["SMILES"] = frame.groupby("canonical_smiles").SMILES.transform("min")
    molecules = sorted(frame.canonical_smiles.unique())
    random.Random(seed).shuffle(molecules)
    n_train, n_val = int(0.7 * len(molecules)), int(0.15 * len(molecules))
    groups = {
        "train": set(molecules[:n_train]),
        "val": set(molecules[n_train : n_train + n_val]),
        "test": set(molecules[n_train + n_val :]),
    }
    result = {s: frame[frame.canonical_smiles.isin(group)].reset_index(drop=True) for s, group in groups.items()}
    for s in groups:
        for t in groups:
            if s != t:
                assert not groups[s] & groups[t]
    stats = {
        "seed": seed,
        "grouping": "canonical_isomeric_full_molecule",
        "fractions": [0.7, 0.15, 0.15],
        "invalid_smiles_rows": invalid,
        "contradictory_rows_excluded": len(excluded),
        "contradictory_pairs_excluded": len(excluded.drop_duplicates(keys)),
        "duplicate_consistent_rows_removed": n_pre_dedup - len(frame),
        "splits": {
            s: {
                "rows": len(df),
                "molecules": df.canonical_smiles.nunique(),
                "proteins": df["Target Sequence"].nunique(),
                "positives": int(df.Label.sum()),
                "positive_rate": float(df.Label.mean()),
            }
            for s, df in result.items()
        },
        "limitations": "Cold molecule, not cold scaffold/protein. Original source was preprocessed; not population prevalence.",
    }
    return result, excluded, stats


def build(source, output, seed):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")
    frames = {s: pd.read_csv(source / f"{s}.csv") for s in ("train", "val", "test")}
    cache_names = ["fragment_fps_r4_2048.pt", "morgan_embeddings.pt", "coati_embeddings.pt"]
    valid = None
    for name in cache_names:
        keys = set(torch.load(source / name, map_location="cpu", weights_only=False))
        valid = keys if valid is None else valid & keys
    proteins = set(torch.load(source / "raygun_embeddings.pt", map_location="cpu", weights_only=False))
    missing = {}
    for s, df in frames.items():
        df.SMILES = df.SMILES.astype(str).str.strip()
        df["Target Sequence"] = df["Target Sequence"].astype(str).str.strip()
        keep = df.SMILES.isin(valid) & df["Target Sequence"].isin(proteins)
        missing[s] = int((~keep).sum())
        frames[s] = df[keep]
    splits, excluded, stats = partition_pairs(frames, seed)
    stats["source_root"] = str(source.resolve())
    stats["missing_embedding_rows"] = missing
    stats["source_csv_sha256"] = {s: hashlib.sha256((source / f"{s}.csv").read_bytes()).hexdigest() for s in frames}
    output.mkdir(parents=True)
    for s, df in splits.items():
        df.to_csv(output / f"{s}.csv", index=False)
    excluded.to_csv(output / "excluded_conflicts.csv", index=False)
    stats["output_csv_sha256"] = {s: hashlib.sha256((output / f"{s}.csv").read_bytes()).hexdigest() for s in splits}
    (output / "manifest.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    build(args.source, args.output, args.seed)
