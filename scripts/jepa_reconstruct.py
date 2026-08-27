#!/usr/bin/env python3
"""
JEPA generative reconstruction harness — the main reconstruction-fidelity eval, and also a
shared library: load_model/load_coati/canonicalize_largest_fragment/batch_tanimoto/build_pairs/
run_forward/_stats/metric_block are all imported directly by other scripts (jepa_denovo.py,
jepa_embedding_retrieval.py, jepa_embedding_retrieval_codebook.py, test_fragment_screening.py) —
treat changes to these functions' signatures as touching the whole family of eval scripts, not
just this file.

For (drug, protein) pairs, runs the FRAGMENT ConciseJEPAFragment forward → jepa_pred
(256-d COATI embedding) → COATI decode (hclip_to_2d_batch) → SMILES, then measures
Tanimoto( decoded , true drug ). This is the first time the fragment model's JEPA head
is actually decoded (LitFragment trained it via MSE but never decoded it).

Decode + Tanimoto logic mirrors lit_jepa.py's validation path exactly (same COATI
encoder, same RDKFingerprint Tanimoto), so results are comparable to the whole-mol model.

Modes:
  reconstruction (default): feed each pair's real drug+protein, decode, Tanimoto vs that drug.
  --unseen_only : keep only pairs whose protein sequence is NOT in --seen_csv (B2 generalization).

A shuffled-target Tanimoto is also computed as a chance baseline (decoded_i vs a random
other true drug) — reconstruction is only meaningful if true-Tanimoto >> shuffled-Tanimoto.
See docs/PROJECT_HANDOFF.md §6 for what "lift" means and the headline numbers this produces.

load_model()'s --fsq_levels/--predictor_type flags exist because load_model used to hardcode
hierarchical FSQ + the MLP predictor — that would have silently loaded wrong/untrained weights
for flat-codebook or cross-attention checkpoints (see docs/PROJECT_HANDOFF.md §7, bug #3).
Always pass the flags matching how the checkpoint was actually trained.

Usage — use the current "final.ckpt" checkpoints (NOT "last.ckpt", which is a stale, frozen-
early-epoch file left over from a checkpoint-saving bug — see docs/PROJECT_HANDOFF.md §7, bug #2):
  python scripts/jepa_reconstruct.py \
    --checkpoint <fragment_f2r .../checkpoints/final.ckpt> --pooling f2r \
    --frag_fps  .../BindingDB_embeddings/fragment_fps_r4_2048.pt \
    --protein_embs .../BindingDB_embeddings/raygun_embeddings.pt \
    --pairs_csv .../BindingDB_embeddings/test.csv \
    --n_pairs 400 --output scripts/jepa_recon_p0.json
"""

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean, median

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ----------------------------------------------------------------------------- model
def load_model(checkpoint_path: str, pooling: str, fsq_levels=None, predictor_type: str = "mlp"):
    from spikes.phase1.fragment_encoder import ConciseFragment, ConciseJEPAFragment
    if fsq_levels is None:
        fsq_levels = [32, 32, 32]
    backbone = ConciseFragment(
        drug_layers=[fsq_levels], pooling=pooling,
        ligand_dim=2048, residue_dim=1280, drug_dim=128, proj_dim=256,
        nheads=32, activation="tanh", drug_quantizer={"type": "fsq"},
    )
    if predictor_type == "xattn":
        from spikes.phase1.fragment_xattn import ConciseJEPAFragmentXAttn
        model = ConciseJEPAFragmentXAttn(concise_fragment=backbone, smiles_target_dim=256)
    else:
        model = ConciseJEPAFragment(concise_fragment=backbone, smiles_target_dim=256, jepa_hidden_dim=512)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    model_state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
    model.eval()
    return model


# ------------------------------------------------------------------------- coati decode
def load_coati(doc_url: str, device: str):
    from coati.models.io import load_e3gnn_smiles_clip_e2e
    encoder, tokenizer = load_e3gnn_smiles_clip_e2e(freeze=True, device=device, doc_url=doc_url)
    return encoder, tokenizer


def canonicalize_largest_fragment(smiles):
    from rdkit import Chem
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True)
    if not frags:
        return None
    largest = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    try:
        return Chem.CanonSmiles(Chem.MolToSmiles(largest))
    except Exception:
        return None


def batch_tanimoto(target_smiles, predicted_smiles, fp="rdk"):
    """Tanimoto(pred_i, target_i). fp='rdk' (RDKFingerprint, matches lit_jepa) or
    'ecfp' (Morgan radius-2, 2048 bits — lower chance floor, project-standard)."""
    from rdkit import Chem
    from rdkit.Chem import RDKFingerprint, AllChem
    from rdkit.DataStructs import TanimotoSimilarity

    def fingerprint(mol):
        if fp == "ecfp":
            return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        return RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=2048)

    sims = []
    for tgt, pred in zip(target_smiles, predicted_smiles):
        if pred is None or tgt is None:
            sims.append(0.0)
            continue
        tm, pm = Chem.MolFromSmiles(tgt), Chem.MolFromSmiles(pred)
        if tm is None or pm is None:
            sims.append(0.0)
            continue
        sims.append(float(TanimotoSimilarity(fingerprint(tm), fingerprint(pm))))
    return sims


# ------------------------------------------------------------------------------- data
def build_pairs(csv_path, frag_fps, raygun, n_pairs, seed, positives_only=True,
                unseen_seqs=None):
    df = pd.read_csv(csv_path, usecols=["SMILES", "Target Sequence", "Label"])
    if positives_only:
        df = df[df["Label"] == 1]
    df = df.drop_duplicates(subset=["SMILES", "Target Sequence"])
    pairs = []
    for smi, seq in zip(df["SMILES"], df["Target Sequence"]):
        if smi not in frag_fps or seq not in raygun:
            continue
        if unseen_seqs is not None and seq in unseen_seqs:
            continue  # keep only proteins NOT seen in training
        pairs.append((smi, seq))
    random.Random(seed).shuffle(pairs)
    return pairs[:n_pairs]


@torch.no_grad()
def run_forward(model, pairs, frag_fps, raygun, batch_size, device):
    """Returns jepa_pred [N,256] and binding [N] for the pairs."""
    model = model.to(device)
    all_pred, all_bind = [], []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        fps = [frag_fps[s] for s, _ in chunk]
        max_f = max(f.shape[0] for f in fps)
        B = len(chunk)
        frag_fps_t = torch.zeros(B, max_f, 2048)
        frag_mask = torch.zeros(B, max_f, dtype=torch.bool)
        for i, f in enumerate(fps):
            frag_fps_t[i, :f.shape[0]] = f
            frag_mask[i, :f.shape[0]] = True
        prot = torch.stack([raygun[seq] for _, seq in chunk], dim=0)  # [B,50,1280]
        out = model(prot.to(device), frag_fps_t.to(device), frag_mask.to(device))
        all_pred.append(out["jepa_pred"].cpu())
        all_bind.append(out["binding"].cpu())
    return torch.cat(all_pred), torch.cat(all_bind)


def _stats(xs):
    if not xs:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0}
    return {"mean": round(mean(xs), 4), "median": round(median(xs), 4),
            "p90": round(sorted(xs)[min(int(0.9 * len(xs)), len(xs) - 1)], 4)}


def metric_block(tanimotos, baseline, valid_flags):
    """Stats for one fingerprint type, all pairs and valid-decode-only, vs baseline."""
    idx_valid = [i for i, v in enumerate(valid_flags) if v]
    tv = [tanimotos[i] for i in idx_valid]
    bv = [baseline[i] for i in idx_valid]
    return {
        "all": _stats(tanimotos),
        "valid_only": _stats(tv),
        "baseline_all_mean": round(mean(baseline), 4) if baseline else 0.0,
        "baseline_valid_only_mean": round(mean(bv), 4) if bv else 0.0,
        "lift_valid_only": round((mean(tv) if tv else 0.0) - (mean(bv) if bv else 0.0), 4),
    }


def summarize(rdk, rdk_base, ecfp, ecfp_base, valid_flags, bindings):
    n = len(valid_flags)
    return {
        "n_pairs": n,
        "n_valid_decode": int(sum(valid_flags)),
        "pct_valid_decode": round(100 * sum(valid_flags) / max(n, 1), 1),
        "ecfp": metric_block(ecfp, ecfp_base, valid_flags),
        "rdk": metric_block(rdk, rdk_base, valid_flags),
        "binding_mean": round(float(mean(bindings)), 4) if bindings else 0.0,
        "binding_min": round(float(min(bindings)), 4) if bindings else 0.0,
        "binding_max": round(float(max(bindings)), 4) if bindings else 0.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pooling", default="f2r")
    p.add_argument("--fsq_levels", default="32,32,32", help="comma-separated FSQ factors, e.g. '32768' for flat")
    p.add_argument("--predictor_type", default="mlp", choices=["mlp", "xattn"])
    p.add_argument("--frag_fps", required=True)
    p.add_argument("--protein_embs", required=True)
    p.add_argument("--pairs_csv", required=True)
    p.add_argument("--seen_csv", default=None, help="train.csv; with --unseen_only, exclude its proteins")
    p.add_argument("--unseen_only", action="store_true")
    p.add_argument("--n_pairs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--coati_doc_url", default="s3://terray-public/models/grande_closed.pkl")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    print(f"Loading frag_fps + raygun ...")
    frag_fps = torch.load(args.frag_fps, map_location="cpu", weights_only=False)
    raygun = torch.load(args.protein_embs, map_location="cpu", weights_only=False)

    unseen_seqs = None
    if args.unseen_only:
        assert args.seen_csv, "--unseen_only needs --seen_csv"
        seen = pd.read_csv(args.seen_csv, usecols=["Target Sequence"])
        unseen_seqs = set(seen["Target Sequence"].tolist())
        print(f"  seen (train) proteins: {len(unseen_seqs)} — will EXCLUDE these")

    pairs = build_pairs(args.pairs_csv, frag_fps, raygun, args.n_pairs, args.seed,
                        positives_only=True, unseen_seqs=unseen_seqs)
    print(f"  usable pairs: {len(pairs)}")

    fsq_levels = [int(x) for x in str(args.fsq_levels).split(",")]
    print(f"Loading model ({args.pooling}, fsq_levels={fsq_levels}, predictor={args.predictor_type}) ...")
    model = load_model(args.checkpoint, args.pooling, fsq_levels=fsq_levels, predictor_type=args.predictor_type)
    print(f"Loading COATI decoder ({args.coati_doc_url}) ...")
    encoder, tokenizer = load_coati(args.coati_doc_url, args.device)

    print("Forward pass (jepa_pred + binding) ...")
    jepa_pred, binding = run_forward(model, pairs, frag_fps, raygun, args.batch_size, args.device)

    print("COATI decode ...")
    with torch.no_grad():
        gen_raw = encoder.hclip_to_2d_batch(
            h_clip=jepa_pred.to(args.device).float(), tokenizer=tokenizer, noise_scale=0.0)

    true_smiles = [canonicalize_largest_fragment(s) for s, _ in pairs]
    gen_smiles = [canonicalize_largest_fragment(s) for s in gen_raw]
    valid_flags = [g is not None for g in gen_smiles]

    shuffled_true = true_smiles[1:] + true_smiles[:1]  # rotated → chance baseline
    rdk = batch_tanimoto(true_smiles, gen_smiles, fp="rdk")
    rdk_base = batch_tanimoto(shuffled_true, gen_smiles, fp="rdk")
    ecfp = batch_tanimoto(true_smiles, gen_smiles, fp="ecfp")
    ecfp_base = batch_tanimoto(shuffled_true, gen_smiles, fp="ecfp")

    agg = summarize(rdk, rdk_base, ecfp, ecfp_base, valid_flags, binding.tolist())
    examples = []
    order = sorted(range(len(pairs)), key=lambda i: ecfp[i], reverse=True)
    for i in order[:15]:
        examples.append({
            "protein_prefix": pairs[i][1][:30] + "...",
            "true_smiles": true_smiles[i],
            "generated_smiles": gen_smiles[i],
            "tanimoto_ecfp": round(ecfp[i], 4),
            "tanimoto_rdk": round(rdk[i], 4),
            "binding": round(float(binding[i]), 4),
        })

    out = {
        "checkpoint": args.checkpoint, "pooling": args.pooling,
        "fsq_levels": fsq_levels, "predictor_type": args.predictor_type,
        "pairs_csv": args.pairs_csv, "unseen_only": args.unseen_only,
        "n_requested": args.n_pairs,
        "aggregate": agg, "top_examples": examples,
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {args.output}")
    tag = "UNSEEN proteins" if args.unseen_only else "all"
    print(f"\n{'='*64}\nRECON SUMMARY ({args.pooling}, {tag}, {Path(args.pairs_csv).name})\n{'='*64}")
    print(f"  pairs={agg['n_pairs']}  valid decode={agg['pct_valid_decode']}%")
    for m in ("ecfp", "rdk"):
        b = agg[m]
        print(f"  [{m.upper():4}] all mean={b['all']['mean']}  valid-only mean={b['valid_only']['mean']}"
              f"  p90={b['valid_only']['p90']}  | baseline(valid)={b['baseline_valid_only_mean']}"
              f"  lift={b['lift_valid_only']}")
    print(f"  DTI binding  mean={agg['binding_mean']}  range=[{agg['binding_min']}, {agg['binding_max']}]")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
