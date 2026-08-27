#!/usr/bin/env python3
"""
B3: Contrastive fragment attribution experiment.

For each matched pair (A=binder, B=non-binder with same protein, Tanimoto ≥ 0.6):
  1. Run attribution on A and B with a trained weighted_sum / mlp_weighted_sum model
  2. Measure whether the fragment unique to B gets a distinctive weight
  3. Aggregate: is unique-to-nonbinder systematically up/down-weighted vs. shared fragments?

Scientific question: given A binds protein P and B doesn't (ABCD share most fragments),
does the model assign a distinctive weight to the fragment that makes B different?

Usage:
  python scripts/contrastive_attribution.py \\
    --checkpoint /path/to/80epoch.ckpt \\
    --pooling weighted_sum \\
    --pairs scripts/matched_pairs.json \\
    --output scripts/contrastive_results.json

    [--max_pairs N]  limit for quick tests
"""

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))  # needed for `from scripts.attribute_fragments import ...`

from scripts.attribute_fragments import AttributionExtractor


def load_pairs(path: str, max_pairs: int | None = None) -> list[dict]:
    with open(path) as f:
        pairs = json.load(f)
    # Filter: must have at least one unique-to-nonbinder fragment
    pairs = [p for p in pairs if p.get("unique_to_nonbinder")]
    if max_pairs:
        pairs = pairs[:max_pairs]
    return pairs


def pipe_split(s: str) -> list[str]:
    """Split |-joined fragment strings."""
    return [x for x in s.split("|") if x.strip()] if s else []


def analyse_pair(attr_A: dict, attr_B: dict, pair: dict) -> dict:
    """
    Compute attribution statistics for one matched pair.

    Returns a record with per-pair stats.
    """
    # Fragment → weight mappings
    w_A = dict(zip(attr_A["fragments"], attr_A["weights"]))
    w_B = dict(zip(attr_B["fragments"], attr_B["weights"]))

    unique_nb = set(pipe_split(pair["unique_to_nonbinder"]))  # in B not A
    unique_b  = set(pipe_split(pair["unique_to_binder"]))     # in A not B

    # Weights for shared fragments (in both A and B)
    shared_A = [w for frag, w in w_A.items() if frag not in unique_b]
    shared_B = [w for frag, w in w_B.items() if frag not in unique_nb]

    # Weights for unique-to-nonbinder fragments in B
    unique_nb_weights = [w_B.get(f, 0.0) for f in unique_nb]

    # Uniform baseline: 1 / n_frags
    uniform_A = 1.0 / max(len(attr_A["fragments"]), 1)
    uniform_B = 1.0 / max(len(attr_B["fragments"]), 1)

    # Attribution ratio: unique-to-nonbinder weight vs. uniform expectation
    # > 1 → model up-weights the unique fragment in B
    # < 1 → model down-weights the unique fragment in B
    avg_unique_nb_w = mean(unique_nb_weights) if unique_nb_weights else 0.0
    attribution_ratio = avg_unique_nb_w / uniform_B if uniform_B > 0 else 0.0

    return {
        "smiles_binder":      pair["smiles_binder"],
        "smiles_nonbinder":   pair["smiles_nonbinder"],
        "tanimoto":           pair["tanimoto"],
        "n_frags_binder":     attr_A["n_frags"],
        "n_frags_nonbinder":  attr_B["n_frags"],
        "unique_to_binder":   sorted(unique_b),
        "unique_to_nonbinder": sorted(unique_nb),
        # Weights
        "mean_shared_weight_binder":    mean(shared_A) if shared_A else 0.0,
        "mean_shared_weight_nonbinder": mean(shared_B) if shared_B else 0.0,
        "mean_unique_nb_weight":        avg_unique_nb_w,
        "uniform_weight_nonbinder":     uniform_B,
        "attribution_ratio":            attribution_ratio,
        # > 1 = unique-to-nonbinder fragment is up-weighted vs. uniform
        # Top fragments
        "top_frag_binder":    attr_A["sorted"][0][0] if attr_A["sorted"] else "",
        "top_weight_binder":  attr_A["sorted"][0][1] if attr_A["sorted"] else 0.0,
        "top_frag_nonbinder": attr_B["sorted"][0][0] if attr_B["sorted"] else "",
        "top_weight_nonbinder": attr_B["sorted"][0][1] if attr_B["sorted"] else 0.0,
    }


def aggregate_results(records: list[dict]) -> dict:
    """Compute aggregate stats across all pairs."""
    if not records:
        return {}

    ratios = [r["attribution_ratio"] for r in records]
    shared_gaps = [
        r["mean_shared_weight_binder"] - r["mean_shared_weight_nonbinder"]
        for r in records
    ]

    # How often is unique-to-nonbinder above/below uniform?
    n_above = sum(1 for r in ratios if r > 1.0)
    n_below = sum(1 for r in ratios if r < 1.0)

    return {
        "n_pairs":                    len(records),
        # Attribution ratio stats (unique-to-nonbinder weight / uniform)
        "attribution_ratio_mean":     mean(ratios),
        "attribution_ratio_stdev":    stdev(ratios) if len(ratios) > 1 else 0.0,
        "attribution_ratio_above1":   n_above,
        "attribution_ratio_below1":   n_below,
        "attribution_ratio_above1_pct": n_above / len(records) * 100,
        # Shared fragment weight gap (binder − nonbinder)
        # Positive → binder focuses more on shared (pharmacophoric) fragments
        "shared_weight_gap_mean":     mean(shared_gaps),
        "shared_weight_gap_stdev":    stdev(shared_gaps) if len(shared_gaps) > 1 else 0.0,
        "n_shared_gap_positive":      sum(1 for g in shared_gaps if g > 0),
        "n_shared_gap_positive_pct":  sum(1 for g in shared_gaps if g > 0) / len(records) * 100,
        # Interpretation:
        # If attribution_ratio_mean > 1: model up-weights the "bad" fragment in non-binders
        # If shared_weight_gap_mean > 0: model allocates more attention to pharmacophoric
        #   fragments in binders than in non-binders
    }


def main():
    p = argparse.ArgumentParser(description="Contrastive fragment attribution (B3)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pooling", default="weighted_sum",
                   choices=["weighted_sum", "mlp_weighted_sum"])
    p.add_argument("--pairs", required=True, help="matched_pairs.json from B2")
    p.add_argument("--output", default=None)
    p.add_argument("--max_pairs", type=int, default=None,
                   help="Limit number of pairs (for quick tests)")
    args = p.parse_args()

    out_path = args.output or args.pairs.replace(".json", "_contrastive.json")

    # Load pairs
    pairs = load_pairs(args.pairs, args.max_pairs)
    print(f"Loaded {len(pairs)} matched pairs with unique-to-nonbinder fragments.")

    # Load attribution model
    extractor = AttributionExtractor(args.checkpoint, args.pooling)

    # Run attribution for each pair
    records = []
    for i, pair in enumerate(pairs):
        if i % 50 == 0:
            print(f"  Processing pair {i+1}/{len(pairs)} ...")

        try:
            attr_A = extractor.attribute(pair["smiles_binder"])
            attr_B = extractor.attribute(pair["smiles_nonbinder"])
            rec = analyse_pair(attr_A, attr_B, pair)
            records.append(rec)
        except Exception as e:
            print(f"  Warning: pair {i} failed: {e}")
            continue

    print(f"\nProcessed {len(records)} pairs successfully.")

    # Aggregate stats
    agg = aggregate_results(records)

    output = {
        "checkpoint": args.checkpoint,
        "pooling": args.pooling,
        "aggregate": agg,
        "per_pair": records,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved results to {out_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Contrastive Attribution Summary ({args.pooling}, {args.checkpoint[-30:]})")
    print(f"{'='*60}")
    print(f"Pairs analyzed: {agg.get('n_pairs', 0)}")
    print(f"\nAttribution ratio (unique-to-nonbinder weight / uniform):")
    print(f"  Mean: {agg.get('attribution_ratio_mean', 0):.3f}  "
          f"(>1 = model up-weights the unique fragment)")
    print(f"  Above 1: {agg.get('attribution_ratio_above1_pct', 0):.1f}% of pairs")
    print(f"\nShared-fragment weight gap (binder − nonbinder):")
    print(f"  Mean: {agg.get('shared_weight_gap_mean', 0):.4f}  "
          f"(>0 = binder focuses more on shared fragments)")
    print(f"  Positive in: {agg.get('n_shared_gap_positive_pct', 0):.1f}% of pairs")
    print(f"{'='*60}")

    # Show top examples
    if records:
        top_examples = sorted(records, key=lambda r: r["attribution_ratio"], reverse=True)[:5]
        print(f"\nTop 5 pairs by attribution ratio (model most focuses on unique-to-nonbinder):")
        for r in top_examples:
            print(f"  ratio={r['attribution_ratio']:.3f}  tanimoto={r['tanimoto']:.3f}")
            print(f"    binder:    {r['smiles_binder'][:60]}")
            print(f"    nonbinder: {r['smiles_nonbinder'][:60]}")
            print(f"    unique_to_nonbinder: {r['unique_to_nonbinder']}")
            print(f"    unique-frag weight: {r['mean_unique_nb_weight']:.4f}  "
                  f"(uniform={r['uniform_weight_nonbinder']:.4f})")


if __name__ == "__main__":
    main()
