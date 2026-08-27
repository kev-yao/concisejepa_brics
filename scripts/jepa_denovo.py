#!/usr/bin/env python3
"""
Protein-ONLY de-novo generation (contrast arm to jepa_reconstruct.py).

Reconstruction (jepa_reconstruct.py) gives the model the real drug (fragment FPs) AND the
protein, then decodes. This script gives the model ONLY the protein: it searches the FSQ
drug codebook for the single-fragment candidate the DTI binding head scores highest against
the protein, then decodes THAT latent through the JEPA head -> COATI -> SMILES, and compares
to the known binder. This is "design a binder from the protein alone."

Pipeline per target protein:
  vocab = utilized FSQ codes (from real training fragments)         # drug vocabulary
  for each candidate code c:  score = DTI_head(drug=embed(c), protein)   # protein-only
  best = argmax_c score
  jepa_pred = jepa_predictor([ d_emb(best) , r_emb(protein) ]) -> COATI -> decode -> SMILES
  Tanimoto(decoded, known_binder)                                    # ECFP + RDK

Diagnostics: designed-molecule DTI score, and #distinct winning codes across proteins
(protein-conditioning check — if one code wins for everyone, de-novo has collapsed).

Also a shared library: build_vocab() and score_candidates() are imported directly by
scripts/jepa_embedding_retrieval_codebook.py, which reuses this exact per-code scoring
mechanism but computes a prediction for EVERY vocabulary code instead of only the argmax
winner (see that file, and docs/history/embedding-retrieval-eval.md for what it found).

Note: design() below calls model.jepa_predictor(context) directly with a concatenated
[drug, protein] vector — this is the MLP predictor's calling convention (ConciseJEPAFragment).
It is NOT wired up for the cross-attention predictor (ConciseJEPAFragmentXAttn, see
fragment_xattn.py), which expects (frag_embs, frag_mask, protein_embedding) instead. Adapting
de-novo design to the cross-attention predictor would need a small change here.
"""

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median

import pandas as pd
import torch
from einops import rearrange

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from jepa_reconstruct import (  # reuse the exact model/coati/metric helpers
    load_model, load_coati, canonicalize_largest_fragment, batch_tanimoto,
    build_pairs, _stats, metric_block,
)


# ------------------------------------------------------------------ drug vocabulary
@torch.no_grad()
def build_vocab(model, frag_fps, device, max_frags=60000):
    """Unique FSQ codes used by real training fragments -> (codes [K,3], embs [K,1,128])."""
    enc = model.concise.d_encoder
    all_fps = []
    for f in frag_fps.values():
        all_fps.append(f)
        if sum(x.shape[0] for x in all_fps) >= max_frags:
            break
    fps = torch.cat(all_fps, dim=0)[:max_frags].to(device)
    codes = []
    for s in range(0, fps.shape[0], 8192):
        codes.append(enc(fps[s:s + 8192])["codes"].cpu())
    codes = torch.cat(codes, dim=0).long()               # [M,3]
    uniq = torch.unique(codes, dim=0)                     # [K,3]
    embs = enc.embed(uniq.to(device))                     # [K,1,128]
    return uniq, embs


# ------------------------------------------------------------------ protein-only scoring
@torch.no_grad()
def score_candidates(model, cand_embs, protein_emb, device, chunk=4096):
    """DTI binding score of every candidate (single-fragment) vs ONE protein.
    Returns scores [K] and post-cross-attn drug emb d_pair [K,1,proj]. Protein path computed once."""
    concise = model.concise
    prot1 = protein_emb.unsqueeze(0).to(device)                          # [1,50,1280]
    r_proj1 = concise.r_project(prot1)                                   # [1,50,proj]
    r_mix, _ = concise.r_to_r_attention(rearrange(r_proj1, "b n k -> n b k"))
    r_proj1 = r_proj1 + rearrange(r_mix, "n b k -> b n k")
    r_pooled1 = concise._pool_residue_embeddings(r_proj1)                # [1,proj]

    K = cand_embs.shape[0]
    all_scores, all_dpair = [], []
    for s in range(0, K, chunk):
        ce = cand_embs[s:s + chunk].to(device)                          # [n,1,128]
        n = ce.shape[0]
        mask = torch.ones(n, 1, dtype=torch.bool, device=device)
        prot = prot1.expand(n, -1, -1)
        if concise.pooling_needs_context:
            mol_emb = concise.pooling_layer(ce, mask, prot)             # [n,128]
        else:
            mol_emb = concise.pooling_layer(ce, mask)
        d_emb = concise.d_project(mol_emb.unsqueeze(1))                 # [n,1,proj]
        d_mix, _ = concise.d_to_d_attention(rearrange(d_emb, "b n k -> n b k"))
        d_emb = d_emb + rearrange(d_mix, "n b k -> b n k")
        r_pair = r_proj1.expand(n, -1, -1)
        r_pooled = r_pooled1.expand(n, -1)
        scores, d_pair = concise._attend_and_score_pair_chunk(d_emb, r_pair, r_pooled)
        all_scores.append(scores.cpu())
        all_dpair.append(d_pair.cpu())
    return torch.cat(all_scores), torch.cat(all_dpair), r_pooled1.cpu()


@torch.no_grad()
def design(model, pairs, vocab_codes, vocab_embs, raygun, device):
    """For each (known_smiles, seq): protein-only best candidate -> jepa context -> jepa_pred.
    Returns jepa_preds [P,256], best_scores [P], best_code_idx [P]."""
    preds, best_scores, best_idx = [], [], []
    for _, seq in pairs:
        scores, d_pair, r_pooled1 = score_candidates(model, vocab_embs, raygun[seq], device)
        b = int(scores.argmax())
        d_best = rearrange(d_pair[b:b + 1], "b n k -> b (n k)")          # [1,proj]
        context = torch.cat([d_best, r_pooled1], dim=-1).to(device)     # [1,2*proj]
        preds.append(model.jepa_predictor(context).cpu())
        best_scores.append(float(scores[b]))
        best_idx.append(b)
    return torch.cat(preds), best_scores, best_idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pooling", default="f2r")
    p.add_argument("--frag_fps", required=True)
    p.add_argument("--protein_embs", required=True)
    p.add_argument("--pairs_csv", required=True)
    p.add_argument("--seen_csv", default=None)
    p.add_argument("--unseen_only", action="store_true")
    p.add_argument("--n_pairs", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--coati_doc_url", default="s3://terray-public/models/grande_closed.pkl")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    frag_fps = torch.load(args.frag_fps, map_location="cpu", weights_only=False)
    raygun = torch.load(args.protein_embs, map_location="cpu", weights_only=False)

    unseen_seqs = None
    if args.unseen_only:
        assert args.seen_csv
        unseen_seqs = set(pd.read_csv(args.seen_csv, usecols=["Target Sequence"])["Target Sequence"])
    pairs = build_pairs(args.pairs_csv, frag_fps, raygun, args.n_pairs, args.seed,
                        positives_only=True, unseen_seqs=unseen_seqs)
    print(f"targets: {len(pairs)}")

    model = load_model(args.checkpoint, args.pooling).to(args.device)
    print("building drug vocabulary (utilized FSQ codes) ...")
    vocab_codes, vocab_embs = build_vocab(model, frag_fps, args.device)
    print(f"  vocabulary size (distinct codes): {vocab_codes.shape[0]}")

    print("protein-only design (DTI codebook search) ...")
    jepa_pred, best_scores, best_idx = design(model, pairs, vocab_codes, vocab_embs, raygun, args.device)

    print("COATI decode ...")
    encoder, tokenizer = load_coati(args.coati_doc_url, args.device)
    with torch.no_grad():
        gen_raw = encoder.hclip_to_2d_batch(h_clip=jepa_pred.to(args.device).float(),
                                            tokenizer=tokenizer, noise_scale=0.0)

    known = [canonicalize_largest_fragment(s) for s, _ in pairs]
    gen = [canonicalize_largest_fragment(s) for s in gen_raw]
    valid_flags = [g is not None for g in gen]
    shuffled = known[1:] + known[:1]
    rdk = batch_tanimoto(known, gen, fp="rdk")
    rdk_b = batch_tanimoto(shuffled, gen, fp="rdk")
    ecfp = batch_tanimoto(known, gen, fp="ecfp")
    ecfp_b = batch_tanimoto(shuffled, gen, fp="ecfp")

    n_distinct = len(set(best_idx))
    agg = {
        "n_targets": len(pairs),
        "pct_valid_decode": round(100 * sum(valid_flags) / max(len(pairs), 1), 1),
        "vocab_size": int(vocab_codes.shape[0]),
        "distinct_winning_codes": n_distinct,
        "protein_conditioning": round(n_distinct / max(len(pairs), 1), 3),  # 1.0=fully distinct, ~0=collapsed
        "designed_dti_mean": round(mean(best_scores), 4),
        "designed_dti_min": round(min(best_scores), 4),
        "designed_dti_max": round(max(best_scores), 4),
        "ecfp": metric_block(ecfp, ecfp_b, valid_flags),
        "rdk": metric_block(rdk, rdk_b, valid_flags),
    }
    examples = []
    order = sorted(range(len(pairs)), key=lambda i: ecfp[i], reverse=True)
    for i in order[:15]:
        examples.append({"protein_prefix": pairs[i][1][:30] + "...", "known_binder": known[i],
                         "designed": gen[i], "tanimoto_ecfp": round(ecfp[i], 4),
                         "tanimoto_rdk": round(rdk[i], 4), "designed_dti": round(best_scores[i], 4),
                         "winning_code": int(best_idx[i])})
    # full per-target dump for case-study mining (cluster by winning_code, identify proteins)
    all_targets = [{"seq": pairs[i][1], "known_binder": known[i], "designed": gen[i],
                    "tanimoto_ecfp": round(ecfp[i], 4), "tanimoto_rdk": round(rdk[i], 4),
                    "designed_dti": round(best_scores[i], 4), "winning_code": int(best_idx[i])}
                   for i in range(len(pairs))]
    out = {"checkpoint": args.checkpoint, "pairs_csv": args.pairs_csv,
           "unseen_only": args.unseen_only, "mode": "denovo_protein_only",
           "aggregate": agg, "top_examples": examples, "all_targets": all_targets}
    Path(args.output).write_text(json.dumps(out, indent=2))

    tag = "UNSEEN" if args.unseen_only else "all"
    print(f"\nSaved -> {args.output}")
    print(f"\n{'='*64}\nDE-NOVO (protein-only) [{tag}, {Path(args.pairs_csv).name}]\n{'='*64}")
    print(f"  targets={agg['n_targets']}  valid={agg['pct_valid_decode']}%  vocab={agg['vocab_size']}")
    print(f"  distinct winning codes={agg['distinct_winning_codes']}  "
          f"protein-conditioning={agg['protein_conditioning']} (1=distinct per protein, ~0=collapsed)")
    print(f"  designed DTI  mean={agg['designed_dti_mean']}  range=[{agg['designed_dti_min']},{agg['designed_dti_max']}]")
    for m in ("ecfp", "rdk"):
        b = agg[m]
        print(f"  [{m.upper():4}] valid-only mean={b['valid_only']['mean']}  base(valid)={b['baseline_valid_only_mean']}"
              f"  lift={b['lift_valid_only']}")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
