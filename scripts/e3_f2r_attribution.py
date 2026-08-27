#!/usr/bin/env python3
"""
E3: Fragment attribution analysis for F2RPool.

Runs contrastive attribution on matched pairs using F2R (protein-conditional pooler).
Key analyses:
  1. Attribution ratio on all pairs and clean-cliff pairs (compare vs WeightedSumPool)
  2. Protein-conditionality demo: same molecule, different proteins → different weights

F2R uses a proxy protein (mean of training proteins) for fair comparison with
WeightedSumPool results (which also used a proxy for C1/C2). Attribution
protein-conditionality is demonstrated by running the same molecule through
multiple real training proteins.

See e4_realprotein_attribution.py for the follow-up that compares F2R/weighted_sum/max using
REAL individual proteins instead of this proxy-mean approach -- that's the version whose
numbers are reported as the project headline (F2R median attribution ratio 0.802; see
docs/PROJECT_HANDOFF.md §6). NOTE also §8's caveat: attribution numbers throughout this project
are single-run point estimates with no error bars, and a rigorous adversarial review found the
run-to-run variance can exceed the effects being reported.

Usage:
  python scripts/e3_f2r_attribution.py \\
    --checkpoint <f2r_ckpt.ckpt> \\
    --pairs scripts/matched_pairs.json \\
    --clean_pairs scripts/matched_pairs_clean.json \\
    --protein_embs /path/to/raygun_embeddings.pt \\
    --output scripts/contrastive_results_f2r.json
"""

import argparse
import json
import signal
import sys
from pathlib import Path
from statistics import mean, stdev

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


BRICS_TIMEOUT_SEC = 3


def _alarm_handler(signum, frame):
    raise TimeoutError


def brics_fragment(smiles: str) -> list[str]:
    from rdkit import Chem
    from rdkit.Chem import BRICS
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]
    try:
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(BRICS_TIMEOUT_SEC)
        raw = list(BRICS.BRICSDecompose(mol))
        signal.alarm(0)
    except (TimeoutError, Exception):
        signal.alarm(0)
        return [smiles]
    cleaned = []
    for f in raw:
        fm = Chem.MolFromSmiles(f)
        if fm is not None:
            cleaned.append(Chem.MolToSmiles(fm))
    return cleaned if cleaned else [smiles]


def morgan_fp(smiles: str, nbits: int = 2048) -> torch.Tensor:
    try:
        from molfeat.trans.fp import FPVecTransformer
        tr = FPVecTransformer(kind="ecfp-count:4", length=nbits)
        arr, valid = tr([smiles], ignore_errors=True)
        if len(arr) == 0:
            return torch.zeros(nbits, dtype=torch.float32)
        return torch.tensor(arr[0], dtype=torch.float32)
    except Exception:
        return torch.zeros(nbits, dtype=torch.float32)


def load_f2r_model(checkpoint_path: str):
    from spikes.phase1.fragment_encoder import ConciseFragment, ConciseJEPAFragment
    backbone = ConciseFragment(
        drug_layers=[[32, 32, 32]],
        pooling="f2r",
        ligand_dim=2048,
        residue_dim=1280,
        drug_dim=128,
        proj_dim=256,
        nheads=32,
        activation="tanh",
        drug_quantizer={"type": "fsq"},
    )
    model = ConciseJEPAFragment(
        concise_fragment=backbone,
        smiles_target_dim=256,
        jepa_hidden_dim=512,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    model_state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    model.load_state_dict(model_state)
    model.eval()
    return model


def load_protein_subset(protein_embs_path: str, n: int = 64) -> tuple[torch.Tensor, list, dict]:
    """
    Load only first n proteins from raygun_embeddings.pt.
    Returns (proxy [50,1280], all_keys_sample, subset_dict).
    Loads the full file but immediately discards what we don't need.
    """
    data = torch.load(protein_embs_path, map_location="cpu", weights_only=False)
    all_keys = list(data.keys())
    selected = all_keys[:n]
    subset = {k: data[k] for k in selected}
    del data  # release the 850MB dict immediately
    proxy = torch.stack(list(subset.values())).mean(0)  # [50, 1280]
    return proxy, selected, subset


@torch.no_grad()
def attribute_molecule(model, smiles: str, protein_emb: torch.Tensor) -> dict:
    """
    Attribute a molecule with F2R using a given protein embedding.

    protein_emb: [50, 1280]
    Returns dict with fragments, weights, etc.
    """
    frags = brics_fragment(smiles)
    fps = torch.stack([morgan_fp(f) for f in frags])  # [F, 2048]
    frag_fps = fps.unsqueeze(0)    # [1, F, 2048]
    frag_mask = torch.ones(1, len(frags), dtype=torch.bool)
    r_emb = protein_emb.unsqueeze(0)  # [1, 50, 1280]

    model.concise._encode_fragments(frag_fps, frag_mask, r_emb)
    weights = model.concise.pooling_layer.last_weights.squeeze(0).tolist()

    sorted_pairs = sorted(zip(frags, weights), key=lambda x: x[1], reverse=True)
    return {
        "smiles": smiles,
        "n_frags": len(frags),
        "fragments": frags,
        "weights": weights,
        "sorted": [(f, w) for f, w in sorted_pairs],
    }


def load_pairs(path: str, max_pairs=None) -> list[dict]:
    with open(path) as f:
        pairs = json.load(f)
    pairs = [p for p in pairs if p.get("unique_to_nonbinder")]
    if max_pairs:
        pairs = pairs[:max_pairs]
    return pairs


def pipe_split(s: str) -> list[str]:
    return [x for x in s.split("|") if x.strip()] if s else []


def analyse_pair(attr_A: dict, attr_B: dict, pair: dict) -> dict:
    w_A = dict(zip(attr_A["fragments"], attr_A["weights"]))
    w_B = dict(zip(attr_B["fragments"], attr_B["weights"]))

    unique_nb = set(pipe_split(pair["unique_to_nonbinder"]))
    unique_b  = set(pipe_split(pair["unique_to_binder"]))

    shared_A = [w for frag, w in w_A.items() if frag not in unique_b]
    shared_B = [w for frag, w in w_B.items() if frag not in unique_nb]
    unique_nb_weights = [w_B.get(f, 0.0) for f in unique_nb]

    uniform_B = 1.0 / max(len(attr_B["fragments"]), 1)
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
        "mean_shared_weight_binder":    mean(shared_A) if shared_A else 0.0,
        "mean_shared_weight_nonbinder": mean(shared_B) if shared_B else 0.0,
        "mean_unique_nb_weight":        avg_unique_nb_w,
        "uniform_weight_nonbinder":     uniform_B,
        "attribution_ratio":            attribution_ratio,
        "top_frag_nonbinder": attr_B["sorted"][0][0] if attr_B["sorted"] else "",
        "top_weight_nonbinder": attr_B["sorted"][0][1] if attr_B["sorted"] else 0.0,
    }


def aggregate_results(records: list[dict]) -> dict:
    if not records:
        return {}
    ratios = [r["attribution_ratio"] for r in records]
    shared_gaps = [
        r["mean_shared_weight_binder"] - r["mean_shared_weight_nonbinder"]
        for r in records
    ]
    n_above = sum(1 for r in ratios if r > 1.0)
    n_below = sum(1 for r in ratios if r < 1.0)
    sorted_ratios = sorted(ratios)
    n = len(sorted_ratios)
    median = sorted_ratios[n // 2] if n % 2 == 1 else (sorted_ratios[n // 2 - 1] + sorted_ratios[n // 2]) / 2
    return {
        "n_pairs": len(records),
        "attribution_ratio_mean":     mean(ratios),
        "attribution_ratio_median":   median,
        "attribution_ratio_stdev":    stdev(ratios) if len(ratios) > 1 else 0.0,
        "attribution_ratio_above1":   n_above,
        "attribution_ratio_below1":   n_below,
        "attribution_ratio_above1_pct": n_above / len(records) * 100,
        "shared_weight_gap_mean":     mean(shared_gaps),
    }


def protein_conditionality_demo(model, data: dict, test_smiles: str, n_proteins: int = 8) -> dict:
    """
    Show that F2R gives different fragment weights for the same molecule
    when different proteins are used.

    data: pre-loaded raygun_embeddings dict (protein_seq → [50, 1280] tensor).
    Returns per-protein weight arrays + variance across proteins.
    """
    selected_keys = list(data.keys())[:n_proteins]

    results = []
    for prot_seq in selected_keys:
        prot_emb = data[prot_seq]  # [50, 1280]
        attr = attribute_molecule(model, test_smiles, prot_emb)
        results.append({
            "protein_seq_prefix": prot_seq[:40] + "...",
            "weights": attr["weights"],
            "fragments": attr["fragments"],
        })

    # Variance in per-fragment weights across proteins
    all_weights = [r["weights"] for r in results]
    n_frags = len(all_weights[0]) if all_weights else 0
    weight_variances = []
    for fi in range(n_frags):
        ws = [w[fi] for w in all_weights]
        weight_variances.append(stdev(ws) if len(ws) > 1 else 0.0)

    return {
        "test_smiles": test_smiles,
        "fragments": results[0]["fragments"] if results else [],
        "n_proteins_tested": len(results),
        "per_protein": results,
        "per_fragment_weight_stdev": weight_variances,
        "mean_weight_stdev_across_proteins": mean(weight_variances) if weight_variances else 0.0,
    }


def main():
    p = argparse.ArgumentParser(description="E3: F2R fragment attribution analysis")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pairs", required=True, help="matched_pairs.json (all 2141 pairs)")
    p.add_argument("--clean_pairs", required=True, help="matched_pairs_clean.json (755 pairs)")
    p.add_argument("--protein_embs", required=True, help="raygun_embeddings.pt")
    p.add_argument("--output", default=None)
    p.add_argument("--max_pairs", type=int, default=None, help="Limit for quick tests")
    args = p.parse_args()

    out_path = args.output or str(Path(args.pairs).parent / "contrastive_results_f2r.json")

    print("Loading F2R model ...")
    model = load_f2r_model(args.checkpoint)

    print("Loading proxy protein (mean of first 64 training proteins) ...")
    proxy_protein, all_keys, data_dict = load_protein_subset(args.protein_embs, n=64)
    print(f"  Proxy protein: mean of 64 proteins. Shape: {proxy_protein.shape}")

    # --- 1. Attribution on all pairs ---
    print("\n[1/3] Attribution on all matched pairs ...")
    pairs_all = load_pairs(args.pairs, args.max_pairs)
    print(f"  Loaded {len(pairs_all)} pairs")

    records_all = []
    for i, pair in enumerate(pairs_all):
        if i % 100 == 0:
            print(f"  {i+1}/{len(pairs_all)} ...")
        try:
            attr_A = attribute_molecule(model, pair["smiles_binder"], proxy_protein)
            attr_B = attribute_molecule(model, pair["smiles_nonbinder"], proxy_protein)
            records_all.append(analyse_pair(attr_A, attr_B, pair))
        except Exception as e:
            print(f"  Warning: pair {i} failed: {e}")
    agg_all = aggregate_results(records_all)
    print(f"  Done. attribution_ratio mean={agg_all['attribution_ratio_mean']:.3f} "
          f"median={agg_all['attribution_ratio_median']:.3f} "
          f"above1={agg_all['attribution_ratio_above1_pct']:.1f}%")

    # --- 2. Attribution on clean cliff pairs ---
    print("\n[2/3] Attribution on clean cliff pairs ...")
    pairs_clean = load_pairs(args.clean_pairs, args.max_pairs)
    print(f"  Loaded {len(pairs_clean)} clean pairs")

    records_clean = []
    for i, pair in enumerate(pairs_clean):
        if i % 100 == 0:
            print(f"  {i+1}/{len(pairs_clean)} ...")
        try:
            attr_A = attribute_molecule(model, pair["smiles_binder"], proxy_protein)
            attr_B = attribute_molecule(model, pair["smiles_nonbinder"], proxy_protein)
            records_clean.append(analyse_pair(attr_A, attr_B, pair))
        except Exception as e:
            print(f"  Warning: pair {i} failed: {e}")
    agg_clean = aggregate_results(records_clean)
    print(f"  Done. attribution_ratio mean={agg_clean['attribution_ratio_mean']:.3f} "
          f"median={agg_clean['attribution_ratio_median']:.3f} "
          f"above1={agg_clean['attribution_ratio_above1_pct']:.1f}%")

    # --- 3. Protein-conditionality demo ---
    # Use a clean-cliff molecule as demo — a nonbinder with a known unique fragment
    print("\n[3/3] Protein-conditionality demo (same molecule, N different proteins) ...")
    demo_smiles = pairs_clean[0]["smiles_nonbinder"] if pairs_clean else "c1ccc(NC(=O)c2ccco2)cc1"
    demo = protein_conditionality_demo(model, data_dict, demo_smiles, n_proteins=8)
    print(f"  Mean weight stdev across proteins: {demo['mean_weight_stdev_across_proteins']:.4f}")
    print(f"  (0.0 = identical weights for all proteins = size-bias only)")

    # Save results
    output = {
        "checkpoint": args.checkpoint,
        "pooling": "f2r",
        "proxy_protein_n": len(all_keys),
        "all_pairs": {
            "aggregate": agg_all,
        },
        "clean_pairs": {
            "aggregate": agg_clean,
            "per_pair": records_clean,
        },
        "protein_conditionality": demo,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print("F2R ATTRIBUTION SUMMARY")
    print(f"{'='*60}")
    print(f"ALL PAIRS ({agg_all['n_pairs']}):")
    print(f"  attr_ratio mean={agg_all['attribution_ratio_mean']:.3f}  "
          f"median={agg_all['attribution_ratio_median']:.3f}  "
          f"above1={agg_all['attribution_ratio_above1_pct']:.1f}%")
    print(f"\nCLEAN PAIRS ({agg_clean['n_pairs']}):")
    print(f"  attr_ratio mean={agg_clean['attribution_ratio_mean']:.3f}  "
          f"median={agg_clean['attribution_ratio_median']:.3f}  "
          f"above1={agg_clean['attribution_ratio_above1_pct']:.1f}%")
    print(f"\nPROTEIN CONDITIONALITY:")
    print(f"  Mean weight stdev across 8 proteins: {demo['mean_weight_stdev_across_proteins']:.4f}")
    print(f"  Fragments: {demo['fragments']}")
    for pr in demo['per_protein']:
        ws = [f"{w:.3f}" for w in pr['weights']]
        print(f"  [{pr['protein_seq_prefix']}] weights: {ws}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
