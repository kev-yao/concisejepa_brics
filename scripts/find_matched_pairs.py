#!/usr/bin/env python3
"""
B2: Find matched molecular pairs in BindingDB.

Finds pairs (A=binder, B=non-binder) for the same protein where:
  - Tanimoto similarity of Morgan FPs >= threshold (default 0.6)
  - BRICS fragment sets differ (at least one fragment unique to each molecule)

These pairs are the evaluation set for B3 (contrastive attribution).

The scientific question: given A binds and B doesn't (but A and B look similar),
which fragment in B is responsible?

Usage:
  python scripts/find_matched_pairs.py --threshold 0.6 --output matched_pairs.csv

Output CSV columns:
  smiles_binder, smiles_nonbinder, target_sequence, tanimoto,
  frags_binder ("|"-joined), frags_nonbinder ("|"-joined),
  unique_to_binder ("|"-joined), unique_to_nonbinder ("|"-joined)
"""

import argparse
import csv
import json
import signal
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

BRICS_TIMEOUT_SEC = 3
TRAIN_CSV = "/hpc/group/singhlab/user/me196/projects/moleculerep/runs/REVICE/data/bindingdb/train.csv"
VAL_CSV   = "/hpc/group/singhlab/user/me196/projects/moleculerep/runs/REVICE/data/bindingdb/val.csv"
TEST_CSV  = "/hpc/group/singhlab/user/me196/projects/moleculerep/runs/REVICE/data/bindingdb/test.csv"

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "scripts"


def _alarm_handler(signum, frame):
    raise TimeoutError


def brics_fragment_set(smiles: str) -> frozenset[str]:
    """BRICS fragment canonical SMILES, with 3s timeout. Returns frozenset."""
    try:
        from rdkit import Chem
        from rdkit.Chem import BRICS
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return frozenset([smiles])
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(BRICS_TIMEOUT_SEC)
        raw = list(BRICS.BRICSDecompose(mol))
        signal.alarm(0)
        frags = set()
        for f in raw:
            fm = Chem.MolFromSmiles(f)
            if fm is not None:
                frags.add(Chem.MolToSmiles(fm))
        return frozenset(frags) if frags else frozenset([smiles])
    except Exception:
        signal.alarm(0)
        return frozenset([smiles])


def get_morgan_fp(smiles: str, radius: int = 4, nbits: int = 2048):
    """RDKit Morgan bit vector FP."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def tanimoto_bulk(query_fp, fps):
    """Return list of Tanimoto similarities of query vs all fps."""
    from rdkit import DataStructs
    return DataStructs.BulkTanimotoSimilarity(query_fp, fps)


def load_bindingdb() -> pd.DataFrame:
    dfs = []
    for path in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
        try:
            df = pd.read_csv(path, usecols=["SMILES", "Target Sequence", "Label"])
            dfs.append(df)
            print(f"  Loaded {len(df)} rows from {Path(path).name}")
        except Exception as e:
            print(f"  Warning: could not load {path}: {e}")
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total: {len(combined)} rows before dedup")
    # Keep one row per (SMILES, Target Sequence); label = 1 if any row is positive
    combined = (
        combined.groupby(["SMILES", "Target Sequence"], as_index=False)
        .agg({"Label": "max"})
    )
    print(f"  After dedup: {len(combined)} unique (SMILES, Target) pairs")
    return combined


def main():
    p = argparse.ArgumentParser(description="Find matched molecular pairs in BindingDB (B2)")
    p.add_argument("--threshold", type=float, default=0.6,
                   help="Minimum Tanimoto similarity (default 0.6)")
    p.add_argument("--max_pairs_per_protein", type=int, default=200,
                   help="Cap pairs per protein to limit output size (default 200)")
    p.add_argument("--output", default=None, help="Output CSV path")
    p.add_argument("--output_json", default=None, help="Also write a JSON with BRICS details")
    args = p.parse_args()

    out_csv = args.output or str(OUTPUT_DIR / "matched_pairs.csv")
    out_json = args.output_json or out_csv.replace(".csv", ".json")

    print("Loading BindingDB ...")
    df = load_bindingdb()

    # Group by protein
    groups = defaultdict(list)
    for _, row in df.iterrows():
        groups[row["Target Sequence"]].append((row["SMILES"], int(row["Label"])))

    print(f"Unique proteins: {len(groups)}")

    # Filter proteins that have both binders and non-binders
    viable = {prot: mols for prot, mols in groups.items()
              if any(l == 1 for _, l in mols) and any(l == 0 for _, l in mols)}
    print(f"Proteins with both binders and non-binders: {len(viable)}")

    all_pairs = []
    total_proteins = len(viable)
    for pi, (protein, mols) in enumerate(viable.items()):
        if pi % 50 == 0:
            print(f"  Processing protein {pi+1}/{total_proteins} ({len(mols)} molecules) ...")

        binders    = [s for s, l in mols if l == 1]
        nonbinders = [s for s, l in mols if l == 0]

        # Compute Morgan FPs
        binder_fps = [(s, get_morgan_fp(s)) for s in binders]
        binder_fps = [(s, fp) for s, fp in binder_fps if fp is not None]
        nonbinder_fps = [(s, get_morgan_fp(s)) for s in nonbinders]
        nonbinder_fps = [(s, fp) for s, fp in nonbinder_fps if fp is not None]
        if not binder_fps or not nonbinder_fps:
            continue

        nb_smiles_list = [s for s, _ in nonbinder_fps]
        nb_fp_list     = [fp for _, fp in nonbinder_fps]

        protein_pairs = []
        for b_smiles, b_fp in binder_fps:
            sims = tanimoto_bulk(b_fp, nb_fp_list)
            for nb_smiles, sim in zip(nb_smiles_list, sims):
                if sim >= args.threshold:
                    protein_pairs.append((b_smiles, nb_smiles, sim))

        if not protein_pairs:
            continue

        # Sort by Tanimoto desc, cap per protein
        protein_pairs.sort(key=lambda x: x[2], reverse=True)
        protein_pairs = protein_pairs[: args.max_pairs_per_protein]

        # Compute BRICS fragments for unique SMILES in pairs
        unique_smiles = set()
        for b, nb, _ in protein_pairs:
            unique_smiles.add(b)
            unique_smiles.add(nb)

        print(f"    Computing BRICS for {len(unique_smiles)} SMILES in {len(protein_pairs)} pairs ...")
        frag_cache = {}
        for s in unique_smiles:
            frag_cache[s] = brics_fragment_set(s)

        for b_smiles, nb_smiles, sim in protein_pairs:
            b_frags  = frag_cache[b_smiles]
            nb_frags = frag_cache[nb_smiles]
            unique_b  = b_frags - nb_frags   # in binder but not non-binder
            unique_nb = nb_frags - b_frags   # in non-binder but not binder
            if not unique_b and not unique_nb:
                continue  # identical fragment sets — skip
            all_pairs.append({
                "smiles_binder": b_smiles,
                "smiles_nonbinder": nb_smiles,
                "target_sequence": protein[:40] + "...",  # truncate for CSV
                "target_sequence_full": protein,
                "tanimoto": round(sim, 4),
                "n_frags_binder": len(b_frags),
                "n_frags_nonbinder": len(nb_frags),
                "frags_binder": "|".join(sorted(b_frags)),
                "frags_nonbinder": "|".join(sorted(nb_frags)),
                "unique_to_binder": "|".join(sorted(unique_b)),
                "unique_to_nonbinder": "|".join(sorted(unique_nb)),
            })

    print(f"\nFound {len(all_pairs)} matched pairs with differing fragments.")

    # Write CSV (exclude full protein sequence from main CSV)
    csv_keys = [
        "smiles_binder", "smiles_nonbinder", "target_sequence",
        "tanimoto", "n_frags_binder", "n_frags_nonbinder",
        "unique_to_binder", "unique_to_nonbinder",
        "frags_binder", "frags_nonbinder",
    ]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_pairs)
    print(f"Saved CSV: {out_csv}")

    # Write JSON (full, includes full protein sequence)
    with open(out_json, "w") as f:
        json.dump(all_pairs, f, indent=2)
    print(f"Saved JSON: {out_json}")

    # Summary stats
    if all_pairs:
        tanimotos = [p["tanimoto"] for p in all_pairs]
        print(f"\nTanimoto stats:")
        print(f"  min={min(tanimotos):.3f}  max={max(tanimotos):.3f}  "
              f"mean={sum(tanimotos)/len(tanimotos):.3f}")
        n_high = sum(1 for t in tanimotos if t >= 0.7)
        print(f"  Pairs with Tanimoto >= 0.7: {n_high}/{len(all_pairs)} ({n_high/len(all_pairs)*100:.1f}%)")


if __name__ == "__main__":
    main()
