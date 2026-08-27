#!/usr/bin/env python3
"""
Embedding-space retrieval vs. the model's OWN per-fragment codebook (fragment models only —
F2R/max; doesn't apply to the non-fragmented original model, which has no fragment vocabulary).

Different from jepa_embedding_retrieval.py, which ranks the real prediction against REAL
molecules' true (COATI-encoder) embeddings. Here the candidate pool is the model's own
hypothetical "if the whole molecule were just this one fragment" prediction, for every FSQ
vocabulary code (~2000) -- the exact same per-code mechanism jepa_denovo.py uses for de-novo
design (pool a single fragment with protein context -> d_project -> cross-attend -> predictor),
just computed for every code instead of only the DTI-argmax winner.

Query: jepa_pred from the REAL multi-fragment forward pass (the actual known drug's own
fragments + the real protein) -- same as jepa_embedding_retrieval.py.
Candidate pool: per protein, one jepa_pred per FSQ vocabulary code, standing alone.
"True match": the FSQ code(s) that are literally among the real drug's own BRICS fragments
(re-derived the same way as scripts/test_fragment_screening.py).
Rank reported: the BEST (lowest) rank among the true drug's own fragment codes -- i.e., did the
real multi-fragment prediction land close to what the model itself predicts for at least one of
its own true building blocks, compared to ~2000 other possible fragments?

Result: much weaker than ranking against real molecules (expected -- ~2000 synthetic single-
fragment constructs are a harder, noisier candidate pool), and the one place in the whole
project where max doesn't cleanly beat F2R -- F2R has the better mean/median rank here, max has
the better MRR/top-10. Full numbers: docs/history/embedding-retrieval-eval.md,
docs/PROJECT_HANDOFF.md §6/§8.

Usage:
  python scripts/jepa_embedding_retrieval_codebook.py \
    --checkpoint <fragment .../final.ckpt> --pooling f2r \
    --frag_fps .../fragment_fps_r4_2048.pt --protein_embs .../raygun_embeddings.pt \
    --pairs_csv .../test.csv --n_pairs 200 --output scripts/embed_retrieval_codebook_f2r.json
"""

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean, median

import pandas as pd
import torch
import torch.nn.functional as Fn
from einops import rearrange

SCRIPTS = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from jepa_reconstruct import load_model  # noqa: E402
from jepa_denovo import build_vocab, score_candidates  # noqa: E402
from spikes.phase1.fragment_datamodule import brics_fragment_mols  # noqa: E402


def build_pairs(csv_path, frag_fps, raygun, n_pairs, seed):
    df = pd.read_csv(csv_path, usecols=["SMILES", "Target Sequence", "Label"])
    df = df[df["Label"] == 1].drop_duplicates(subset=["SMILES", "Target Sequence"])
    pairs = [(s, q) for s, q in zip(df["SMILES"], df["Target Sequence"])
             if s in frag_fps and q in raygun]
    random.Random(seed).shuffle(pairs)
    return pairs[:n_pairs]


@torch.no_grad()
def run_forward_one(model, smi, seq, frag_fps, raygun, device):
    """Real multi-fragment jepa_pred for one (drug, protein) pair."""
    fps = frag_fps[smi].unsqueeze(0).to(device)          # [1,F,2048]
    mask = torch.ones(1, fps.shape[1], dtype=torch.bool, device=device)
    prot = raygun[seq].unsqueeze(0).to(device)           # [1,50,1280]
    out = model(prot, fps, mask)
    return out["jepa_pred"][0].cpu()                     # [256]


def true_fragment_codes(model, smi, frag_fps, device):
    """This drug's own real BRICS fragments -> their FSQ codes (same order as fragment_fps
    precompute, per scripts/test_fragment_screening.py)."""
    fps = frag_fps.get(smi)
    if fps is None or fps.numel() == 0:
        return []
    try:
        mols = brics_fragment_mols(smi)
    except Exception:
        return []
    n = min(len(mols), fps.shape[0])
    if n == 0:
        return []
    enc = model.concise.d_encoder
    codes = enc(fps[:n].to(device))["codes"].cpu()
    return [tuple(codes[i].tolist()) for i in range(n)]


@torch.no_grad()
def per_code_predictions(model, vocab_embs, protein_emb, device, chunk=4096):
    """jepa_pred for EVERY vocabulary code, standing alone, for one protein (same mechanism as
    jepa_denovo.py's design(), applied to all K candidates instead of only the DTI-argmax one)."""
    scores, d_pair, r_pooled1 = score_candidates(model, vocab_embs, protein_emb, device, chunk=chunk)
    K = d_pair.shape[0]
    preds = []
    for s in range(0, K, chunk):
        d_chunk = rearrange(d_pair[s:s + chunk], "b n k -> b (n k)").to(device)   # [n,proj]
        r_chunk = r_pooled1.expand(d_chunk.shape[0], -1).to(device)              # [n,proj]
        context = torch.cat([d_chunk, r_chunk], dim=-1)                          # [n,2*proj]
        preds.append(model.jepa_predictor(context).cpu())
    return torch.cat(preds, dim=0)  # [K,256]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pooling", default="f2r")
    p.add_argument("--fsq_levels", default="32,32,32")
    p.add_argument("--frag_fps", required=True)
    p.add_argument("--protein_embs", required=True)
    p.add_argument("--pairs_csv", required=True)
    p.add_argument("--n_pairs", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    print("Loading frag_fps + raygun + model ...")
    frag_fps = torch.load(args.frag_fps, map_location="cpu", weights_only=False)
    raygun = torch.load(args.protein_embs, map_location="cpu", weights_only=False)
    fsq_levels = [int(x) for x in str(args.fsq_levels).split(",")]
    model = load_model(args.checkpoint, args.pooling, fsq_levels=fsq_levels, predictor_type="mlp").to(args.device)
    model.eval()

    pairs = build_pairs(args.pairs_csv, frag_fps, raygun, args.n_pairs, args.seed)
    print(f"  usable pairs: {len(pairs)}")

    print("Building FSQ vocabulary ...")
    vocab_codes, vocab_embs = build_vocab(model, frag_fps, args.device)
    vocab_index = {tuple(vocab_codes[i].tolist()): i for i in range(vocab_codes.shape[0])}
    print(f"  vocab size: {vocab_codes.shape[0]}")

    results = []
    skipped_no_true_code = 0
    for i, (smi, seq) in enumerate(pairs):
        true_codes = true_fragment_codes(model, smi, frag_fps, args.device)
        true_idxs = [vocab_index[c] for c in true_codes if c in vocab_index]
        if not true_idxs:
            skipped_no_true_code += 1
            continue

        query = run_forward_one(model, smi, seq, frag_fps, raygun, args.device)          # [256]
        cand = per_code_predictions(model, vocab_embs, raygun[seq], args.device)          # [K,256]

        q = Fn.normalize(query.float(), dim=-1)
        c = Fn.normalize(cand.float(), dim=-1)
        sims = q @ c.T                                                                    # [K]
        order = torch.argsort(sims, descending=True)
        rank_of = {}
        for idx in true_idxs:
            rank_of[idx] = int((order == idx).nonzero(as_tuple=True)[0].item()) + 1
        best_rank = min(rank_of.values())
        results.append({"smiles": smi, "n_true_fragments": len(true_idxs), "best_rank": best_rank})

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(pairs)} pairs done")

    ranks = [r["best_rank"] for r in results]
    n = len(ranks)
    K = vocab_codes.shape[0]
    agg = {
        "n_queries": n,
        "n_skipped_no_true_code_in_vocab": skipped_no_true_code,
        "vocab_size": K,
        "mean_best_rank": round(mean(ranks), 2) if ranks else None,
        "median_best_rank": median(ranks) if ranks else None,
        "mean_reciprocal_rank": round(mean(1.0 / r for r in ranks), 4) if ranks else None,
        "top1_acc": round(sum(1 for r in ranks if r <= 1) / n, 4) if n else None,
        "top5_acc": round(sum(1 for r in ranks if r <= 5) / n, 4) if n else None,
        "top10_acc": round(sum(1 for r in ranks if r <= 10) / n, 4) if n else None,
        "top1pct_acc": round(sum(1 for r in ranks if r <= max(1, K // 100)) / n, 4) if n else None,
        "chance_mean_rank": round((K + 1) / 2, 2),
        "chance_top1_acc": round(1.0 / K, 4),
    }
    out = {"args": vars(args), "aggregate": agg, "per_query": results}
    Path(args.output).write_text(json.dumps(out, indent=2))

    print(f"\nSaved -> {args.output}")
    print("\n" + "=" * 62)
    print(f"EMBEDDING RETRIEVAL vs. OWN CODEBOOK ({args.pooling})")
    print("=" * 62)
    print(f"  queries={n}  (skipped, no true code in vocab={skipped_no_true_code})  vocab={K}")
    print(f"  mean best-rank={agg['mean_best_rank']}  (chance={agg['chance_mean_rank']})")
    print(f"  MRR={agg['mean_reciprocal_rank']}")
    print(f"  top-1={agg['top1_acc']} (chance={agg['chance_top1_acc']})  "
          f"top-5={agg['top5_acc']}  top-10={agg['top10_acc']}  top-1%={agg['top1pct_acc']}")
    print("=" * 62)


if __name__ == "__main__":
    main()
