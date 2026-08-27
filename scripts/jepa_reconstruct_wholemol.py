#!/usr/bin/env python3
"""
JEPA reconstruction on the ORIGINAL pre-fragmentation ConciseJEPA model (no BRICS, no
pooling — whole-molecule Morgan fingerprint straight into the DrugEncoder). Baseline/
control for the fragment-pooling investigation: does JEPA reconstruction work as intended
before fragmentation was ever introduced?

Same protocol as jepa_reconstruct.py (drug+protein given -> jepa_pred -> COATI decode ->
Tanimoto vs true drug, ECFP+RDK, shuffled-target baseline) — reuses those exact helpers so
numbers are directly comparable. Only the model class and drug input differ: ConciseJEPA +
Concise backbone, morgan_fingerprint [B,2048] instead of frag_fps+mask.

Headline result this produces (0.246 absolute ECFP Tanimoto, 0.134 lift): see
docs/PROJECT_HANDOFF.md §6 — this is the "no fragmentation" baseline every fragment pooler
gets compared against.

Usage:
  python scripts/jepa_reconstruct_wholemol.py \
    --checkpoint <concisejepa_orig run>/checkpoints/.../final.ckpt \
    --morgan_embeddings .../BindingDB_embeddings/morgan_embeddings.pt \
    --protein_embs .../BindingDB_embeddings/raygun_embeddings.pt \
    --pairs_csv .../BindingDB_embeddings/test.csv \
    --n_pairs 400 --output scripts/wholemol_recon_test.json
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

SCRIPTS = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS))
from jepa_reconstruct import (  # reuse exact decode/metric helpers for direct comparability
    load_coati, canonicalize_largest_fragment, batch_tanimoto, summarize,
)


def load_model(checkpoint_path: str):
    from concisejepa.models.concise import Concise
    from concisejepa.models.concise_jepa import ConciseJEPA
    backbone = Concise(
        drug_layers=[[32, 32, 32]],
        drug_quantizer={"type": "fsq"},
        ligand_dim=2048, residue_dim=1280, drug_dim=128, proj_dim=256,
        nheads=32, activation="tanh", cosine_prediction=False,
    )
    model = ConciseJEPA(concise_backbone=backbone, smiles_target_dim=256, jepa_hidden_dim=512)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print(f"  checkpoint epoch={ckpt.get('epoch')}  global_step={ckpt.get('global_step')}")
    state = ckpt["state_dict"]
    model_state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
    model.eval()
    return model


def build_pairs(csv_path, morgan, raygun, n_pairs, seed):
    df = pd.read_csv(csv_path, usecols=["SMILES", "Target Sequence", "Label"])
    df = df[df["Label"] == 1].drop_duplicates(subset=["SMILES", "Target Sequence"])
    pairs = [(s, q) for s, q in zip(df["SMILES"], df["Target Sequence"])
             if s in morgan and q in raygun]
    import random
    random.Random(seed).shuffle(pairs)
    return pairs[:n_pairs]


@torch.no_grad()
def run_forward(model, pairs, morgan, raygun, batch_size, device):
    model = model.to(device)
    all_pred, all_bind = [], []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        fps = torch.stack([morgan[s] for s, _ in chunk]).to(device)          # [B,2048]
        prot = torch.stack([raygun[seq] for _, seq in chunk]).to(device)    # [B,50,1280]
        out = model(prot, fps)
        all_pred.append(out["jepa_pred"].cpu())
        all_bind.append(out["binding"].cpu())
    return torch.cat(all_pred), torch.cat(all_bind)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--morgan_embeddings", required=True)
    p.add_argument("--protein_embs", required=True)
    p.add_argument("--pairs_csv", required=True)
    p.add_argument("--n_pairs", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--coati_doc_url", default="s3://terray-public/models/grande_closed.pkl")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    print("Loading morgan_embeddings + raygun ...")
    morgan = torch.load(args.morgan_embeddings, map_location="cpu", weights_only=False)
    raygun = torch.load(args.protein_embs, map_location="cpu", weights_only=False)

    pairs = build_pairs(args.pairs_csv, morgan, raygun, args.n_pairs, args.seed)
    print(f"  usable pairs: {len(pairs)}")

    print("Loading model (original whole-mol ConciseJEPA) ...")
    model = load_model(args.checkpoint)
    print(f"Loading COATI decoder ({args.coati_doc_url}) ...")
    encoder, tokenizer = load_coati(args.coati_doc_url, args.device)

    print("Forward pass (jepa_pred + binding) ...")
    jepa_pred, binding = run_forward(model, pairs, morgan, raygun, args.batch_size, args.device)

    print("COATI decode ...")
    with torch.no_grad():
        gen_raw = encoder.hclip_to_2d_batch(
            h_clip=jepa_pred.to(args.device).float(), tokenizer=tokenizer, noise_scale=0.0)

    true_smiles = [canonicalize_largest_fragment(s) for s, _ in pairs]
    gen_smiles = [canonicalize_largest_fragment(s) for s in gen_raw]
    valid_flags = [g is not None for g in gen_smiles]

    shuffled_true = true_smiles[1:] + true_smiles[:1]
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
            "true_smiles": true_smiles[i], "generated_smiles": gen_smiles[i],
            "tanimoto_ecfp": round(ecfp[i], 4), "tanimoto_rdk": round(rdk[i], 4),
            "binding": round(float(binding[i]), 4),
        })

    out = {"checkpoint": args.checkpoint, "model": "original_wholemol_concisejepa",
           "pairs_csv": args.pairs_csv, "aggregate": agg, "top_examples": examples}
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {args.output}")
    print(f"\n{'='*64}\nWHOLE-MOL (no fragmentation) RECON SUMMARY\n{'='*64}")
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
