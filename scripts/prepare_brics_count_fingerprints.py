#!/usr/bin/env python3
"""Create a separate, verified count-fingerprint cache; never mutate legacy data."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from molfeat.trans.fp import FPVecTransformer

from concisejepa.datamodules.dataloader import _build_morgan_embeddings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--legacy-whole", type=Path, required=True)
    parser.add_argument("--fragment-cache", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    csvs = [args.data / f"{s}.csv" for s in ("train", "val", "test")]
    smiles = sorted(set(pd.concat([pd.read_csv(p) for p in csvs]).SMILES))
    output = args.data / "morgan_ecfp_count4_2048.pt"
    if output.exists():
        raise FileExistsError(output)
    _build_morgan_embeddings(smiles, output, "ecfp-count:4", 2048, csvs)
    new = torch.load(output, map_location="cpu", weights_only=False)
    old = torch.load(args.legacy_whole, map_location="cpu", weights_only=False)
    fragments = torch.load(args.fragment_cache, map_location="cpu", weights_only=False)
    if set(new) != set(smiles):
        raise RuntimeError("Generated count cache does not cover every molecule")
    matrix = torch.stack([new[s] for s in smiles])
    legacy = torch.stack([old[s] for s in smiles])
    if not bool((matrix > 1).any()) or not bool((matrix >= 0).all()):
        raise RuntimeError("Generated fingerprints are not nonnegative counts")
    # Verify independent tiny-molecule behavior, not just a metadata label.
    probe = FPVecTransformer(kind="ecfp-count:4", length=2048)(["c1ccccc1"])[0]
    if max(probe) <= 1:
        raise RuntimeError("Count generator unexpectedly returned a binary benzene fingerprint")
    audit = {
        "n_molecules": len(smiles),
        "legacy_whole_all_binary": bool(((legacy == 0) | (legacy == 1)).all()),
        "legacy_whole_max": float(legacy.max()),
        "new_whole_max": float(matrix.max()),
        "new_whole_rows_with_counts_above_one": int((matrix > 1).any(1).sum()),
        "binarized_new_equals_legacy_rows": int(((matrix > 0) == legacy).all(1).sum()),
        "fragment_max": max(float(fragments[s].max()) for s in smiles),
        "new_whole_equal_to_fragment_token_molecules": sum(bool((fragments[s] == new[s]).all(1).any()) for s in smiles),
        "new_cache": str(output.resolve()),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "legacy_cache": str(args.legacy_whole.resolve()),
        "note": "New cache only. Legacy whole-molecule values and metadata were not modified.",
    }
    (args.data / "count_fingerprint_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
