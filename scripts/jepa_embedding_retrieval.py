#!/usr/bin/env python3
"""
Embedding-space retrieval evaluation — complements jepa_reconstruct.py.

Instead of decoding jepa_pred into a SMILES string (via COATI's autoregressive decoder) and
comparing Tanimoto similarity to the true molecule, this sidesteps the decoder entirely:

  1. Compute jepa_pred for each (drug, protein) test pair, same as jepa_reconstruct.py.
  2. Compute the TRUE COATI embedding for every unique molecule in the test set directly from
     its SMILES (coati's own encoder — no JEPA, no decode).
  3. Rank the true target's embedding against the whole candidate pool by cosine similarity to
     jepa_pred. Report: what rank does the true match get among all distractors?

This tests whether the JEPA predictor's output is close, in embedding space, specifically to
the correct molecule -- rather than testing whether COATI's decoder can turn that output back
into a valid, correct SMILES string. A model could fail the decode-based test (Tanimoto) either
because jepa_pred is genuinely wrong, OR because it's right but lands slightly off the decoder's
learned manifold. This isolates which one it is.

Result: all 3 models (F2R, max, original) rank far above chance here even where the Tanimoto
test looked mediocre -- suggesting the COATI decoder, not the predictor, is the bigger
bottleneck. Full numbers: docs/history/embedding-retrieval-eval.md, docs/PROJECT_HANDOFF.md §6.
See jepa_embedding_retrieval_codebook.py for a harder variant (candidates = the model's own
per-fragment-code predictions, not real molecules).

Usage (fragment models, e.g. F2R/max):
  python scripts/jepa_embedding_retrieval.py \
    --checkpoint <fragment .../final.ckpt> --pooling f2r \
    --frag_fps .../fragment_fps_r4_2048.pt --protein_embs .../raygun_embeddings.pt \
    --pairs_csv .../test.csv --n_pairs 400 --output scripts/embed_retrieval_f2r.json

Usage (original, no-fragmentation model):
  python scripts/jepa_embedding_retrieval.py \
    --checkpoint <original .../final.ckpt> --whole_mol \
    --morgan_embeddings .../morgan_embeddings.pt --protein_embs .../raygun_embeddings.pt \
    --pairs_csv .../test.csv --n_pairs 400 --output scripts/embed_retrieval_original.json
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

SCRIPTS = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def build_pairs_fragment(csv_path, frag_fps, raygun, n_pairs, seed):
    df = pd.read_csv(csv_path, usecols=["SMILES", "Target Sequence", "Label"])
    df = df[df["Label"] == 1].drop_duplicates(subset=["SMILES", "Target Sequence"])
    pairs = [(s, q) for s, q in zip(df["SMILES"], df["Target Sequence"])
             if s in frag_fps and q in raygun]
    random.Random(seed).shuffle(pairs)
    return pairs[:n_pairs]


@torch.no_grad()
def run_forward_fragment(model, pairs, frag_fps, raygun, batch_size, device):
    model = model.to(device)
    all_pred = []
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
        prot = torch.stack([raygun[seq] for _, seq in chunk], dim=0)
        out = model(prot.to(device), frag_fps_t.to(device), frag_mask.to(device))
        all_pred.append(out["jepa_pred"].cpu())
    return torch.cat(all_pred)


@torch.no_grad()
def run_forward_wholemol(model, pairs, morgan, raygun, batch_size, device):
    model = model.to(device)
    all_pred = []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        fps = torch.stack([morgan[s] for s, _ in chunk]).to(device)
        prot = torch.stack([raygun[seq] for _, seq in chunk]).to(device)
        out = model(prot, fps)
        all_pred.append(out["jepa_pred"].cpu())
    return torch.cat(all_pred)


@torch.no_grad()
def embed_candidate_pool(smiles_list, encoder, tokenizer, device, batch_size=128):
    """True COATI embeddings for a list of real molecules, straight from their SMILES."""
    from coati.generative.coati_purifications import embed_smiles_batch
    from rdkit import Chem
    all_embeds = []
    for s in range(0, len(smiles_list), batch_size):
        chunk = smiles_list[s:s + batch_size]
        canon = []
        for smi in chunk:
            m = Chem.MolFromSmiles(smi)
            canon.append(Chem.MolToSmiles(m) if m is not None else smi)
        emb = embed_smiles_batch(canon, encoder, tokenizer)
        all_embeds.append(emb.detach().cpu())
    return torch.cat(all_embeds, dim=0)


def rank_metrics(jepa_pred, pool_embeds, true_idx_per_query):
    """cosine-similarity ranking of jepa_pred[i] against the whole pool; true_idx_per_query[i]
    is the pool index of the correct match for query i."""
    q = Fn.normalize(jepa_pred.float(), dim=-1)
    c = Fn.normalize(pool_embeds.float(), dim=-1)
    sims = q @ c.T  # [n_queries, n_pool]
    ranks = []
    for i, true_idx in enumerate(true_idx_per_query):
        order = torch.argsort(sims[i], descending=True)
        rank = int((order == true_idx).nonzero(as_tuple=True)[0].item()) + 1  # 1-indexed
        ranks.append(rank)
    return ranks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pooling", default="f2r")
    p.add_argument("--fsq_levels", default="32,32,32")
    p.add_argument("--predictor_type", default="mlp", choices=["mlp", "xattn"])
    p.add_argument("--whole_mol", action="store_true", help="use the original (no-fragmentation) model")
    p.add_argument("--morgan_embeddings", default=None, help="required if --whole_mol")
    p.add_argument("--frag_fps", default=None, help="required unless --whole_mol")
    p.add_argument("--protein_embs", required=True)
    p.add_argument("--pairs_csv", required=True)
    p.add_argument("--n_pairs", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--coati_doc_url", default="s3://terray-public/models/grande_closed.pkl")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    print("Loading protein embeddings ...")
    raygun = torch.load(args.protein_embs, map_location="cpu", weights_only=False)

    if args.whole_mol:
        assert args.morgan_embeddings, "--whole_mol requires --morgan_embeddings"
        from jepa_reconstruct_wholemol import load_model as load_wholemol, build_pairs as build_pairs_wm
        morgan = torch.load(args.morgan_embeddings, map_location="cpu", weights_only=False)
        print("Loading whole-molecule (no-fragmentation) model ...")
        model = load_wholemol(args.checkpoint)
        pairs = build_pairs_wm(args.pairs_csv, morgan, raygun, args.n_pairs, args.seed)
        print(f"  usable pairs: {len(pairs)}")
        print("Forward pass ...")
        jepa_pred = run_forward_wholemol(model, pairs, morgan, raygun, args.batch_size, args.device)
    else:
        assert args.frag_fps, "fragment models require --frag_fps"
        from jepa_reconstruct import load_model as load_frag
        frag_fps = torch.load(args.frag_fps, map_location="cpu", weights_only=False)
        fsq_levels = [int(x) for x in str(args.fsq_levels).split(",")]
        print(f"Loading fragment model (pooling={args.pooling}, fsq_levels={fsq_levels}, "
              f"predictor={args.predictor_type}) ...")
        model = load_frag(args.checkpoint, args.pooling, fsq_levels=fsq_levels, predictor_type=args.predictor_type)
        pairs = build_pairs_fragment(args.pairs_csv, frag_fps, raygun, args.n_pairs, args.seed)
        print(f"  usable pairs: {len(pairs)}")
        print("Forward pass ...")
        jepa_pred = run_forward_fragment(model, pairs, frag_fps, raygun, args.batch_size, args.device)

    print(f"Loading COATI encoder ({args.coati_doc_url}) ...")
    from coati.models.io import load_e3gnn_smiles_clip_e2e
    encoder, tokenizer = load_e3gnn_smiles_clip_e2e(freeze=True, device=args.device, doc_url=args.coati_doc_url)

    # candidate pool = every unique molecule appearing among the query pairs (the true target
    # for each query is guaranteed to be in this pool)
    pool_smiles = sorted({s for s, _ in pairs})
    true_idx_per_query = {s: i for i, s in enumerate(pool_smiles)}
    query_true_idx = [true_idx_per_query[s] for s, _ in pairs]
    print(f"Embedding candidate pool ({len(pool_smiles)} unique molecules) via COATI encoder ...")
    pool_embeds = embed_candidate_pool(pool_smiles, encoder, tokenizer, args.device)

    print("Ranking jepa_pred against the pool by cosine similarity ...")
    ranks = rank_metrics(jepa_pred, pool_embeds, query_true_idx)

    n = len(ranks)
    n_pool = len(pool_smiles)
    agg = {
        "n_queries": n,
        "n_pool": n_pool,
        "mean_rank": round(mean(ranks), 2),
        "median_rank": median(ranks),
        "mean_reciprocal_rank": round(mean(1.0 / r for r in ranks), 4),
        "top1_acc": round(sum(1 for r in ranks if r <= 1) / n, 4),
        "top5_acc": round(sum(1 for r in ranks if r <= 5) / n, 4),
        "top10_acc": round(sum(1 for r in ranks if r <= 10) / n, 4),
        "top1pct_acc": round(sum(1 for r in ranks if r <= max(1, n_pool // 100)) / n, 4),
        "chance_mean_rank": round((n_pool + 1) / 2, 2),
        "chance_top1_acc": round(1.0 / n_pool, 4),
    }
    out = {"args": vars(args), "aggregate": agg, "ranks": ranks}
    Path(args.output).write_text(json.dumps(out, indent=2))

    print(f"\nSaved -> {args.output}")
    print("\n" + "=" * 60)
    print(f"EMBEDDING RETRIEVAL ({'whole-mol' if args.whole_mol else args.pooling})")
    print("=" * 60)
    print(f"  queries={n}  pool={n_pool}")
    print(f"  mean rank={agg['mean_rank']}  (chance={agg['chance_mean_rank']})")
    print(f"  MRR={agg['mean_reciprocal_rank']}")
    print(f"  top-1={agg['top1_acc']} (chance={agg['chance_top1_acc']})  "
          f"top-5={agg['top5_acc']}  top-10={agg['top10_acc']}  top-1%={agg['top1pct_acc']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
