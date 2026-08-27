#!/usr/bin/env python3
"""
Option 1: Gradient-based fragment attribution.

For each molecule, computes d(binding_score) / d(fragment_fps[i]) and uses
the gradient L2 norm as fragment importance. Unlike pooler softmax weights,
gradients flow through the full model including the cross-attention with the
protein — so this is protein-conditional attribution.

Usage:
  python scripts/gradient_attribution.py \\
    --checkpoint /path/to/ckpt.ckpt \\
    --pooling weighted_sum \\
    --pairs scripts/matched_pairs_clean.json \\
    --protein_emb /hpc/group/singhlab/user/cy244/projects/peptides/count_combined_embeddings/raygun_embeddings.pt \\
    --output scripts/grad_results_clean.json \\
    [--pairs_full scripts/matched_pairs.json --output_full scripts/grad_results_full.json]
"""

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.attribute_fragments import brics_fragment, morgan_fp


class GradientAttributionExtractor:
    def __init__(self, checkpoint_path: str, pooling: str, protein_emb_path: str):
        from spikes.phase1.fragment_encoder import ConciseFragment, ConciseJEPAFragment

        backbone = ConciseFragment(
            drug_layers=[[32, 32, 32]],
            pooling=pooling,
            ligand_dim=2048,
            residue_dim=1280,
            drug_dim=128,
            proj_dim=256,
            nheads=32,
            activation="tanh",
            drug_quantizer={"type": "fsq"},
        )
        self.model = ConciseJEPAFragment(
            concise_fragment=backbone,
            smiles_target_dim=256,
            jepa_hidden_dim=512,
        )

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt["state_dict"]
        model_state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
        self.model.load_state_dict(model_state)
        self.model.eval()

        print(f"Loaded {type(backbone.pooling_layer).__name__} checkpoint")
        print(f"Loading protein embeddings from {protein_emb_path} ...")
        self.protein_emb = torch.load(protein_emb_path, map_location="cpu", weights_only=False)
        print(f"  {len(self.protein_emb)} protein sequences loaded")

    def _get_protein_emb(self, protein_seq: str) -> torch.Tensor:
        """Return protein embedding; fall back to mean of N cache entries if seq not found."""
        if protein_seq in self.protein_emb:
            return self.protein_emb[protein_seq].unsqueeze(0)
        # Protein not in training cache (cross-dataset). Use mean of 32 random proxies.
        keys = list(self.protein_emb.keys())
        import random
        sample = random.sample(keys, min(32, len(keys)))
        mean_emb = torch.stack([self.protein_emb[k] for k in sample]).mean(0)
        return mean_emb.unsqueeze(0)

    def attribute(self, smiles: str, protein_seq: str) -> dict | None:
        frags = brics_fragment(smiles)
        fps = torch.stack([morgan_fp(f) for f in frags])         # [F, 2048]
        frag_fps = fps.unsqueeze(0).requires_grad_(True)          # [1, F, 2048]
        frag_mask = torch.ones(1, len(frags), dtype=torch.bool)
        protein_t = self._get_protein_emb(protein_seq)            # [1, 50, 1280]

        # Forward (don't use no_grad — we need gradients)
        out = self.model(protein_t, frag_fps, frag_mask)
        score = out["binding"].sum()   # scalar

        score.backward()

        grad = frag_fps.grad.squeeze(0)    # [F, 2048]
        attrs = grad.norm(dim=1)           # [F] — L2 norm per fragment

        # Normalize to sum=1 for comparability with pooler weights
        attrs_norm = (attrs / attrs.sum().clamp(min=1e-12)).tolist()
        uniform = 1.0 / max(len(frags), 1)

        sorted_pairs = sorted(zip(frags, attrs_norm), key=lambda x: x[1], reverse=True)
        return {
            "smiles":    smiles,
            "n_frags":   len(frags),
            "fragments": frags,
            "weights":   attrs_norm,
            "sorted":    [(f, w) for f, w in sorted_pairs],
            "uniform":   uniform,
        }


def pipe_split(s: str) -> list[str]:
    return [x for x in s.split("|") if x.strip()] if s else []


def analyse_pair(attr_A: dict, attr_B: dict, pair: dict) -> dict | None:
    if attr_A is None or attr_B is None:
        return None

    w_A = dict(zip(attr_A["fragments"], attr_A["weights"]))
    w_B = dict(zip(attr_B["fragments"], attr_B["weights"]))

    unique_nb = set(pipe_split(pair["unique_to_nonbinder"]))
    unique_b  = set(pipe_split(pair["unique_to_binder"]))

    shared_A = [w for frag, w in w_A.items() if frag not in unique_b]
    shared_B = [w for frag, w in w_B.items() if frag not in unique_nb]

    unique_nb_weights = [w_B.get(f, 0.0) for f in unique_nb]
    uniform_B = attr_B["uniform"]
    avg_unique_nb_w = mean(unique_nb_weights) if unique_nb_weights else 0.0
    attribution_ratio = avg_unique_nb_w / uniform_B if uniform_B > 0 else 0.0

    return {
        "smiles_binder":              pair["smiles_binder"],
        "smiles_nonbinder":           pair["smiles_nonbinder"],
        "tanimoto":                   pair["tanimoto"],
        "unique_to_nonbinder":        sorted(unique_nb),
        "mean_unique_nb_weight":      avg_unique_nb_w,
        "uniform_weight_nonbinder":   uniform_B,
        "attribution_ratio":          attribution_ratio,
        "mean_shared_weight_binder":  mean(shared_A) if shared_A else 0.0,
        "mean_shared_weight_nonbinder": mean(shared_B) if shared_B else 0.0,
    }


def aggregate(records: list[dict]) -> dict:
    if not records:
        return {}
    ratios = [r["attribution_ratio"] for r in records]
    gaps   = [r["mean_shared_weight_binder"] - r["mean_shared_weight_nonbinder"] for r in records]
    n_above = sum(1 for r in ratios if r > 1.0)
    return {
        "n_pairs": len(records),
        "attribution_ratio_mean":       mean(ratios),
        "attribution_ratio_stdev":      stdev(ratios) if len(ratios) > 1 else 0.0,
        "attribution_ratio_above1_pct": n_above / len(records) * 100,
        "attribution_ratio_median":     sorted(ratios)[len(ratios) // 2],
        "shared_weight_gap_mean":       mean(gaps),
        "n_shared_gap_positive_pct":    sum(1 for g in gaps if g > 0) / len(records) * 100,
    }


def run_pairs(ext: GradientAttributionExtractor, pairs_path: str, output_path: str):
    with open(pairs_path) as f:
        pairs = json.load(f)
    pairs = [p for p in pairs if p.get("unique_to_nonbinder")]
    print(f"\n=== {pairs_path} ({len(pairs)} pairs) ===")

    records = []
    skipped = 0
    for i, pair in enumerate(pairs):
        if i % 50 == 0:
            print(f"  Processing pair {i+1}/{len(pairs)} ...")
        try:
            attr_A = ext.attribute(pair["smiles_binder"],    pair["target_sequence"])
            attr_B = ext.attribute(pair["smiles_nonbinder"], pair["target_sequence"])
            rec = analyse_pair(attr_A, attr_B, pair)
            if rec:
                records.append(rec)
            else:
                skipped += 1
        except Exception as e:
            skipped += 1

    print(f"Processed {len(records)} pairs ({skipped} errors)")

    agg = aggregate(records)
    output = {"aggregate": agg, "per_pair": records}
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {output_path}")

    print(f"\n--- Summary ---")
    print(f"  attribution_ratio mean={agg.get('attribution_ratio_mean',0):.3f}  "
          f"median={agg.get('attribution_ratio_median',0):.3f}  "
          f"stdev={agg.get('attribution_ratio_stdev',0):.3f}")
    print(f"  Above 1: {agg.get('attribution_ratio_above1_pct',0):.1f}%")
    print(f"  shared_weight_gap mean={agg.get('shared_weight_gap_mean',0):.4f}  "
          f"positive={agg.get('n_shared_gap_positive_pct',0):.1f}%")
    return agg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--pooling",      default="weighted_sum",
                   choices=["weighted_sum", "mlp_weighted_sum"])
    p.add_argument("--protein_emb",  required=True)
    p.add_argument("--pairs",        required=True,  help="Clean cliff pairs JSON")
    p.add_argument("--output",       required=True)
    p.add_argument("--pairs_full",   default=None,   help="Full pairs JSON (optional)")
    p.add_argument("--output_full",  default=None)
    args = p.parse_args()

    ext = GradientAttributionExtractor(args.checkpoint, args.pooling, args.protein_emb)

    run_pairs(ext, args.pairs, args.output)

    if args.pairs_full and args.output_full:
        run_pairs(ext, args.pairs_full, args.output_full)


if __name__ == "__main__":
    main()
