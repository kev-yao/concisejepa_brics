#!/usr/bin/env python3
"""
R4/E4: Real-protein fragment attribution — the canonical attribution comparison, and the
source of this project's headline attribution numbers (docs/PROJECT_HANDOFF.md §6:
F2R median 0.802, max median 0.599, weighted_sum median 0.257).

E3 (e3_f2r_attribution.py) used a mean-of-64 PROXY protein for every matched pair, which
bypasses the whole point of a protein-conditional pooler. This script looks up each pair's
ACTUAL binding protein in raygun_embeddings and feeds each pooler the real partner instead.

Runs 2 or 3 poolers on the same matched pairs so attribution ratios are directly comparable
(--ws_ckpt and --f2r_ckpt are required; --max_ckpt is optional and adds max as a 3rd arm):
  - weighted_sum: protein-independent (scorer ignores the protein) — real vs. proxy should
    give the same numbers; this reconfirms the size-bias on a freshly-trained model.
  - F2R: protein-conditional — this is where the real protein should matter. Best attribution
    of any pooler tested.
  - max (optional): added later, once max pooling's own reconstruction strength was discovered
    (see docs/history/solution1-chemgroup-results.md) — included here to see how it attributes,
    not just reconstructs.

attribution_ratio = mean weight on the fragment(s) unique to the non-binder / uniform.
Ratio > 1 ⇒ the model up-weights the distinguishing fragment (correct attribution).

⚠️ See docs/PROJECT_HANDOFF.md §8: attribution numbers throughout this project are single-run
point estimates with no error bars, and a rigorous adversarial review found the run-to-run
variance can exceed the effects being reported here.

Usage:
  python scripts/e4_realprotein_attribution.py \
    --ws_ckpt  <weighted_sum final.ckpt> \
    --f2r_ckpt <f2r final.ckpt> \
    --max_ckpt <max final.ckpt> \
    --pairs        scripts/matched_pairs.json \
    --clean_pairs  scripts/matched_pairs_clean.json \
    --protein_embs /hpc/group/singhlab/user/cy244/projects/peptides/BindingDB_embeddings/raygun_embeddings.pt \
    --output       scripts/realprotein_results.json
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


# molfeat transformer is expensive to build; construct once and reuse.
_TRANSFORMER = None


def morgan_fp(smiles: str, nbits: int = 2048) -> torch.Tensor:
    global _TRANSFORMER
    try:
        if _TRANSFORMER is None:
            from molfeat.trans.fp import FPVecTransformer
            _TRANSFORMER = FPVecTransformer(kind="ecfp-count:4", length=nbits, verbose=False)
        arr, valid = _TRANSFORMER([smiles], ignore_errors=True)
        if len(arr) == 0:
            return torch.zeros(nbits, dtype=torch.float32)
        return torch.tensor(arr[0], dtype=torch.float32)
    except Exception:
        return torch.zeros(nbits, dtype=torch.float32)


def load_model(checkpoint_path: str, pooling: str):
    """Load a ConciseJEPAFragment checkpoint with the given pooling head."""
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


@torch.no_grad()
def attribute_molecule(model, smiles: str, protein_emb: torch.Tensor) -> dict:
    """Fragment weights for one molecule under a given protein.

    protein_emb: [50, 1280]. WeightedSum ignores it; F2R cross-attends against it.
    """
    frags = brics_fragment(smiles)
    fps = torch.stack([morgan_fp(f) for f in frags])   # [F, 2048]
    frag_fps = fps.unsqueeze(0)                          # [1, F, 2048]
    frag_mask = torch.ones(1, len(frags), dtype=torch.bool)
    r_emb = protein_emb.unsqueeze(0)                     # [1, 50, 1280]

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


def pipe_split(s: str) -> list[str]:
    return [x for x in s.split("|") if x.strip()] if s else []


def load_pairs(path: str, max_pairs=None) -> list[dict]:
    with open(path) as f:
        pairs = json.load(f)
    pairs = [p for p in pairs if p.get("unique_to_nonbinder")]
    if max_pairs:
        pairs = pairs[:max_pairs]
    return pairs


def analyse_pair(attr_A: dict, attr_B: dict, pair: dict) -> dict:
    w_A = dict(zip(attr_A["fragments"], attr_A["weights"]))
    w_B = dict(zip(attr_B["fragments"], attr_B["weights"]))

    unique_nb = set(pipe_split(pair["unique_to_nonbinder"]))
    unique_b = set(pipe_split(pair["unique_to_binder"]))

    shared_A = [w for frag, w in w_A.items() if frag not in unique_b]
    shared_B = [w for frag, w in w_B.items() if frag not in unique_nb]
    unique_nb_weights = [w_B.get(f, 0.0) for f in unique_nb]

    uniform_B = 1.0 / max(len(attr_B["fragments"]), 1)
    avg_unique_nb_w = mean(unique_nb_weights) if unique_nb_weights else 0.0
    attribution_ratio = avg_unique_nb_w / uniform_B if uniform_B > 0 else 0.0

    return {
        "smiles_binder": pair["smiles_binder"],
        "smiles_nonbinder": pair["smiles_nonbinder"],
        "tanimoto": pair["tanimoto"],
        "n_frags_nonbinder": attr_B["n_frags"],
        "unique_to_nonbinder": sorted(unique_nb),
        "mean_shared_weight_binder": mean(shared_A) if shared_A else 0.0,
        "mean_shared_weight_nonbinder": mean(shared_B) if shared_B else 0.0,
        "mean_unique_nb_weight": avg_unique_nb_w,
        "uniform_weight_nonbinder": uniform_B,
        "attribution_ratio": attribution_ratio,
    }


def aggregate_results(records: list[dict]) -> dict:
    if not records:
        return {}
    ratios = [r["attribution_ratio"] for r in records]
    gaps = [r["mean_shared_weight_binder"] - r["mean_shared_weight_nonbinder"] for r in records]
    n_above = sum(1 for r in ratios if r > 1.0)
    sr = sorted(ratios)
    n = len(sr)
    median = sr[n // 2] if n % 2 == 1 else (sr[n // 2 - 1] + sr[n // 2]) / 2
    return {
        "n_pairs": len(records),
        "attribution_ratio_mean": mean(ratios),
        "attribution_ratio_median": median,
        "attribution_ratio_stdev": stdev(ratios) if len(ratios) > 1 else 0.0,
        "attribution_ratio_above1": n_above,
        "attribution_ratio_above1_pct": n_above / len(records) * 100,
        "shared_weight_gap_mean": mean(gaps),
    }


def run_pooler(model, pairs: list[dict], raygun: dict, label: str) -> tuple[dict, int]:
    """Attribute all pairs with a real per-pair protein. Returns (aggregate, n_skipped)."""
    records, skipped = [], 0
    for i, pair in enumerate(pairs):
        seq = pair.get("target_sequence_full")
        if seq is None or seq not in raygun:
            skipped += 1
            continue
        prot = raygun[seq]                      # [50, 1280] — the REAL binding partner
        try:
            attr_A = attribute_molecule(model, pair["smiles_binder"], prot)
            attr_B = attribute_molecule(model, pair["smiles_nonbinder"], prot)
            records.append(analyse_pair(attr_A, attr_B, pair))
        except Exception as e:
            print(f"  [{label}] pair {i} failed: {e}")
        if i % 200 == 0:
            print(f"  [{label}] {i+1}/{len(pairs)} (skipped so far: {skipped})")
    return aggregate_results(records), skipped, records


def protein_conditionality_demo(model, pairs: list[dict], raygun: dict, n_proteins: int = 8) -> dict:
    """F2R with a fixed molecule across N distinct REAL proteins drawn from the pairs."""
    demo_smiles = pairs[0]["smiles_nonbinder"]
    seen, real_seqs = set(), []
    for p in pairs:
        s = p.get("target_sequence_full")
        if s and s in raygun and s not in seen:
            seen.add(s)
            real_seqs.append(s)
        if len(real_seqs) >= n_proteins:
            break

    results = []
    for seq in real_seqs:
        attr = attribute_molecule(model, demo_smiles, raygun[seq])
        results.append({"protein_seq_prefix": seq[:40] + "...",
                        "weights": attr["weights"], "fragments": attr["fragments"]})
    n_frags = len(results[0]["weights"]) if results else 0
    variances = [stdev([r["weights"][fi] for r in results]) if len(results) > 1 else 0.0
                 for fi in range(n_frags)]
    return {
        "test_smiles": demo_smiles,
        "fragments": results[0]["fragments"] if results else [],
        "n_proteins_tested": len(results),
        "per_protein": results,
        "per_fragment_weight_stdev": variances,
        "mean_weight_stdev_across_proteins": mean(variances) if variances else 0.0,
    }


def main():
    p = argparse.ArgumentParser(description="R4: real-protein fragment attribution (Gap-1 fix)")
    p.add_argument("--ws_ckpt", required=True, help="WeightedSum best checkpoint")
    p.add_argument("--f2r_ckpt", required=True, help="F2R best checkpoint")
    p.add_argument("--max_ckpt", default=None, help="MaxPool best checkpoint (optional 3rd arm)")
    p.add_argument("--pairs", required=True, help="matched_pairs.json (all)")
    p.add_argument("--clean_pairs", required=True, help="matched_pairs_clean.json")
    p.add_argument("--protein_embs", required=True, help="BindingDB raygun_embeddings.pt")
    p.add_argument("--output", default=str(PROJECT_ROOT / "scripts" / "realprotein_results.json"))
    p.add_argument("--max_pairs", type=int, default=None)
    args = p.parse_args()

    print("Loading raygun (real BindingDB proteins) ...")
    raygun = torch.load(args.protein_embs, map_location="cpu", weights_only=False)
    print(f"  raygun keys: {len(raygun)}")

    print("Loading models ...")
    ws = load_model(args.ws_ckpt, "weighted_sum")
    f2r = load_model(args.f2r_ckpt, "f2r")
    poolers = [("weighted_sum", ws), ("f2r", f2r)]
    if args.max_ckpt:
        poolers.append(("max", load_model(args.max_ckpt, "max")))

    pairs_all = load_pairs(args.pairs, args.max_pairs)
    pairs_clean = load_pairs(args.clean_pairs, args.max_pairs)
    print(f"Pairs: all={len(pairs_all)}  clean={len(pairs_clean)}")

    out = {"ws_ckpt": args.ws_ckpt, "f2r_ckpt": args.f2r_ckpt,
           "protein_mode": "real_per_pair", "results": {}}

    for pooler_name, model in poolers:
        print(f"\n===== {pooler_name.upper()} (real protein) =====")
        agg_all, sk_all, _ = run_pooler(model, pairs_all, raygun, f"{pooler_name}/all")
        agg_clean, sk_clean, rec_clean = run_pooler(model, pairs_clean, raygun, f"{pooler_name}/clean")
        out["results"][pooler_name] = {
            "all_pairs": {"aggregate": agg_all, "n_skipped_no_protein": sk_all},
            "clean_pairs": {"aggregate": agg_clean, "n_skipped_no_protein": sk_clean,
                            "per_pair": rec_clean},
        }
        print(f"  ALL   n={agg_all.get('n_pairs')}  mean={agg_all.get('attribution_ratio_mean'):.3f} "
              f"median={agg_all.get('attribution_ratio_median'):.3f} "
              f"above1={agg_all.get('attribution_ratio_above1_pct'):.1f}%")
        print(f"  CLEAN n={agg_clean.get('n_pairs')}  mean={agg_clean.get('attribution_ratio_mean'):.3f} "
              f"median={agg_clean.get('attribution_ratio_median'):.3f} "
              f"above1={agg_clean.get('attribution_ratio_above1_pct'):.1f}%")

    print("\n===== F2R protein-conditionality (REAL proteins) =====")
    out["results"]["f2r"]["protein_conditionality"] = protein_conditionality_demo(f2r, pairs_clean, raygun, 8)
    print(f"  mean weight stdev across 8 real proteins: "
          f"{out['results']['f2r']['protein_conditionality']['mean_weight_stdev_across_proteins']:.4f}")

    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {args.output}")

    # Comparison vs earlier PROXY-based numbers (from session-handoff / journal)
    print(f"\n{'='*64}")
    print("COMPARISON: real-protein (this run) vs proxy-protein (E3/C1)")
    print(f"{'='*64}")
    ws_c = out["results"]["weighted_sum"]["clean_pairs"]["aggregate"]
    f2_c = out["results"]["f2r"]["clean_pairs"]["aggregate"]
    print(f"WS  clean: real median={ws_c.get('attribution_ratio_median'):.3f} "
          f"above1={ws_c.get('attribution_ratio_above1_pct'):.1f}%  | proxy(C1): median=0.255 above1=10.1%")
    print(f"F2R clean: real median={f2_c.get('attribution_ratio_median'):.3f} "
          f"above1={f2_c.get('attribution_ratio_above1_pct'):.1f}%  | proxy(E3): median=0.975 above1=44.6%")
    if args.max_ckpt:
        max_c = out["results"]["max"]["clean_pairs"]["aggregate"]
        print(f"MAX clean: real median={max_c.get('attribution_ratio_median'):.3f} "
              f"above1={max_c.get('attribution_ratio_above1_pct'):.1f}%  | (new pooler, no prior proxy run)")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
