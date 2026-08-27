#!/usr/bin/env python3
"""
Retrospective validation of the "fragment-guided virtual screening" pipeline idea:

  1. Score every vocabulary fragment (protein-only, no real drug) against a held-out test
     protein — reuses jepa_denovo.py's own scoring mechanism.
  2. "Retrieve" molecules containing that fragment. Operationalized at the FSQ-CODE level
     (two BRICS fragments that quantize to the same discrete code are, by the model's own
     construction, the same fragment identity) rather than a raw RDKit SMARTS substructure
     match — this tests the model's own native notion of fragment equivalence directly, and
     avoids BRICS-attachment-point [n*] matching ambiguity.
  3. Check enrichment for true positive (protein, binder) pairs vs. a RANDOM-fragment-code
     control — the critical comparison: does the model's "important" fragment actually
     retrieve real binders better than an arbitrary fragment would?
  4. Rescore retrieved candidates with the model's own full-molecule DTI head; check whether
     true binders rank near the top of the retrieved pool (does step 4 close the loop, or was
     fragment-presence alone already doing all the work?).

Fully retrospective: BindingDB train/test already has ground-truth positive/negative labels.
No new training. Uses the F2R fixedckpt checkpoint already used throughout the attribution/
reconstruction investigation (best attribution ratio -> most trustworthy "important fragment"
signal available).

Result: directionally encouraging (2/117 hits vs. 1/117 for the random-fragment control, ~7.65x
vs ~0.05x enrichment) but statistically thin -- only 2-3 real hit events total. Full detail +
the 3 individual hit cases: docs/history/fragment-screening-validation.md,
docs/PROJECT_HANDOFF.md §5.7/§8 (this is flagged there as underpowered, not a confirmed finding).
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

SCRIPTS = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from jepa_reconstruct import load_model  # noqa: E402
from jepa_denovo import build_vocab, score_candidates  # noqa: E402
from spikes.phase1.fragment_datamodule import brics_fragment_mols  # noqa: E402


# --------------------------------------------------------------------- fragment index
def brics_frag_smiles_ordered(smiles: str) -> list[str]:
    """Same order as compute_fragment_fps() used to build fragment_fps_r4_2048.pt, so
    frag_fps[smiles][i] corresponds to this list's i-th entry."""
    from rdkit import Chem
    mols = brics_fragment_mols(smiles)
    return [Chem.MolToSmiles(m) for m in mols]


@torch.no_grad()
def encode_fragment_codes(model, fps: torch.Tensor, device: str, chunk: int = 8192) -> torch.Tensor:
    enc = model.concise.d_encoder
    codes = []
    for s in range(0, fps.shape[0], chunk):
        codes.append(enc(fps[s:s + chunk].to(device))["codes"].cpu())
    return torch.cat(codes, dim=0).long()


def build_index(model, frag_fps: dict, smiles_list: list[str], device: str):
    """Returns mol_to_codes[smiles]=set(code tuples), code_to_mols[code]=set(smiles)."""
    mol_to_codes, code_to_mols = {}, defaultdict(set)
    n_ok, n_skip, n_brics_err = 0, 0, 0
    for smi in smiles_list:
        fps = frag_fps.get(smi)
        if fps is None or fps.numel() == 0:
            n_skip += 1
            continue
        try:
            frag_smis = brics_frag_smiles_ordered(smi)
        except Exception:
            # RDKit's BRICS.BRICSDecompose has known internal edge-case bugs (e.g. a bare
            # AttributeError on 'pSmi' for certain structures) that aren't always caught by
            # fragment_datamodule.py's own timeout-only handler. Same defensive skip pattern
            # already used there for outright failures.
            n_brics_err += 1
            continue
        n = min(len(frag_smis), fps.shape[0])
        if n == 0:
            n_skip += 1
            continue
        codes = encode_fragment_codes(model, fps[:n], device)
        code_set = {tuple(codes[i].tolist()) for i in range(n)}
        mol_to_codes[smi] = code_set
        for c in code_set:
            code_to_mols[c].add(smi)
        n_ok += 1
    print(f"  index built: {n_ok} molecules indexed, {n_skip} skipped (no frag_fps), "
          f"{n_brics_err} skipped (BRICS re-decompose error)")
    return mol_to_codes, code_to_mols


# --------------------------------------------------------------------- full-molecule rescoring
@torch.no_grad()
def rescore_full(model, candidate_smiles: list[str], frag_fps: dict, protein_emb: torch.Tensor,
                  device: str, chunk: int = 48) -> dict:
    scores = {}
    for s in range(0, len(candidate_smiles), chunk):
        batch = candidate_smiles[s:s + chunk]
        fps = [frag_fps[c] for c in batch]
        max_f = max(f.shape[0] for f in fps)
        B = len(batch)
        frag_fps_t = torch.zeros(B, max_f, 2048)
        frag_mask = torch.zeros(B, max_f, dtype=torch.bool)
        for i, f in enumerate(fps):
            frag_fps_t[i, :f.shape[0]] = f
            frag_mask[i, :f.shape[0]] = True
        prot_b = protein_emb.unsqueeze(0).expand(B, -1, -1)
        out = model.concise(frag_fps_t.to(device), frag_mask.to(device), prot_b.to(device))
        for c, sc in zip(batch, out["binding"].cpu().tolist()):
            scores[c] = sc
    return scores


# --------------------------------------------------------------------- per-protein eval
def evaluate_protein(model, seq, vocab_codes, vocab_embs, raygun, frag_fps, code_to_mols,
                      positives: set, corpus_size: int, device: str, rng: random.Random,
                      max_rescore: int = 300):
    scores, _, _ = score_candidates(model, vocab_embs, raygun[seq], device)
    top_idx = int(scores.argmax())
    top_code = tuple(vocab_codes[top_idx].tolist())
    rand_idx = rng.randrange(vocab_codes.shape[0])
    rand_code = tuple(vocab_codes[rand_idx].tolist())

    result = {}
    for label, code in (("top", top_code), ("random", rand_code)):
        retrieved = code_to_mols.get(code, set())
        n_ret = len(retrieved)
        n_hit = len(retrieved & positives)
        precision = (n_hit / n_ret) if n_ret else 0.0
        baseline_rate = (len(positives) / corpus_size) if corpus_size else 0.0
        enrichment = (precision / baseline_rate) if baseline_rate > 0 else None

        rescore_metric = None
        if n_ret > 0 and n_hit > 0:
            cand_list = list(retrieved)
            if len(cand_list) > max_rescore:
                keep_hits = list(retrieved & positives)
                keep_rest = [c for c in cand_list if c not in positives]
                rng.shuffle(keep_rest)
                cand_list = keep_hits + keep_rest[:max(0, max_rescore - len(keep_hits))]
            scored = rescore_full(model, cand_list, frag_fps, raygun[seq], device)
            ranked = sorted(cand_list, key=lambda c: -scored[c])
            ranks = [ranked.index(c) + 1 for c in cand_list if c in positives]
            rescore_metric = {
                "n_rescored": len(cand_list),
                "mean_rank_of_true_binder": round(sum(ranks) / len(ranks), 2),
                "top10_precision": round(sum(1 for c in ranked[:10] if c in positives) / min(10, len(ranked)), 4),
            }
        result[label] = {
            "n_retrieved": n_ret, "n_hit": n_hit, "precision": round(precision, 4),
            "enrichment": round(enrichment, 3) if enrichment is not None else None,
            "rescore": rescore_metric,
        }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pooling", default="f2r")
    p.add_argument("--frag_fps", required=True)
    p.add_argument("--protein_embs", required=True)
    p.add_argument("--train_csv", required=True)
    p.add_argument("--test_csv", required=True)
    p.add_argument("--n_test_proteins", type=int, default=120)
    p.add_argument("--max_corpus", type=int, default=None, help="cap corpus size (smoke-test only)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    rng = random.Random(args.seed)

    print("Loading frag_fps + raygun + model ...")
    frag_fps = torch.load(args.frag_fps, map_location="cpu", weights_only=False)
    raygun = torch.load(args.protein_embs, map_location="cpu", weights_only=False)
    model = load_model(args.checkpoint, args.pooling).to(args.device)
    model.eval()

    train_df = pd.read_csv(args.train_csv, usecols=["SMILES", "Target Sequence", "Label"])
    test_df = pd.read_csv(args.test_csv, usecols=["SMILES", "Target Sequence", "Label"])
    corpus_smiles = sorted(set(train_df["SMILES"]) | set(test_df["SMILES"]))
    if args.max_corpus:
        rng.shuffle(corpus_smiles)
        corpus_smiles = corpus_smiles[:args.max_corpus]
    print(f"Corpus: {len(corpus_smiles)} unique molecules (train+test)")

    print("Building fragment-code index over the corpus (BRICS + FSQ encode) ...")
    mol_to_codes, code_to_mols = build_index(model, frag_fps, corpus_smiles, args.device)
    corpus_size = len(mol_to_codes)

    print("Building drug vocabulary for protein-only scoring ...")
    vocab_codes, vocab_embs = build_vocab(model, frag_fps, args.device)
    print(f"  vocab size: {vocab_codes.shape[0]} distinct codes")

    pos_test = test_df[test_df["Label"] == 1]
    protein_to_pos = defaultdict(set)
    for smi, seq in zip(pos_test["SMILES"], pos_test["Target Sequence"]):
        if smi in mol_to_codes:
            protein_to_pos[seq].add(smi)
    candidate_proteins = [seq for seq, pos in protein_to_pos.items() if len(pos) >= 1 and seq in raygun]
    rng.shuffle(candidate_proteins)
    test_proteins = candidate_proteins[:args.n_test_proteins]
    print(f"Evaluating {len(test_proteins)} held-out test proteins with >=1 indexed positive binder")

    per_protein = {}
    for i, seq in enumerate(test_proteins):
        per_protein[seq[:40] + "..."] = evaluate_protein(
            model, seq, vocab_codes, vocab_embs, raygun, frag_fps, code_to_mols,
            protein_to_pos[seq], corpus_size, args.device, rng)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(test_proteins)} proteins done")

    # Save the raw (expensive) per-protein results immediately, before any aggregation —
    # a bug in summary math should never risk losing 100+ proteins of real computation.
    Path(args.output).write_text(json.dumps({"args": vars(args), "per_protein": per_protein}, indent=2))
    print(f"(raw per-protein results saved -> {args.output}, before summary aggregation)")

    def agg(label, key, filt=lambda v: True):
        vals = [v[label][key] for v in per_protein.values() if v[label][key] is not None and filt(v[label])]
        return (round(sum(vals) / len(vals), 4), len(vals)) if vals else (None, 0)

    summary = {
        "n_test_proteins": len(test_proteins),
        "corpus_size": corpus_size,
        "vocab_size": int(vocab_codes.shape[0]),
        "top_fragment": {
            "mean_precision": agg("top", "precision")[0],
            "mean_enrichment": agg("top", "enrichment")[0],
            "n_with_enrichment": agg("top", "enrichment")[1],
            "hit_rate": round(sum(1 for v in per_protein.values() if v["top"]["n_hit"] > 0) / len(per_protein), 4),
        },
        "random_fragment": {
            "mean_precision": agg("random", "precision")[0],
            "mean_enrichment": agg("random", "enrichment")[0],
            "n_with_enrichment": agg("random", "enrichment")[1],
            "hit_rate": round(sum(1 for v in per_protein.values() if v["random"]["n_hit"] > 0) / len(per_protein), 4),
        },
    }
    # fix nested top10 precision aggregation (needs its own pass over rescore sub-dict)
    for label in ("top", "random"):
        rescored = [v[label]["rescore"] for v in per_protein.values() if v[label]["rescore"] is not None]
        summary[f"{label}_fragment"]["n_rescored_cases"] = len(rescored)
        if rescored:
            summary[f"{label}_fragment"]["mean_rank_of_true_binder"] = round(
                sum(r["mean_rank_of_true_binder"] for r in rescored) / len(rescored), 2)
            summary[f"{label}_fragment"]["mean_top10_precision"] = round(
                sum(r["top10_precision"] for r in rescored) / len(rescored), 4)

    out = {"args": vars(args), "summary": summary, "per_protein": per_protein}
    Path(args.output).write_text(json.dumps(out, indent=2))

    print(f"\nSaved -> {args.output}")
    print("\n" + "=" * 70)
    print("FRAGMENT-GUIDED SCREENING VALIDATION")
    print("=" * 70)
    for label in ("top_fragment", "random_fragment"):
        s = summary[label]
        print(f"[{label}] hit_rate={s['hit_rate']}  mean_precision={s['mean_precision']}  "
              f"mean_enrichment={s['mean_enrichment']} (n={s['n_with_enrichment']})")
        if "mean_rank_of_true_binder" in s:
            print(f"           rescored cases={s['n_rescored_cases']}  "
                  f"mean_rank_of_true_binder={s['mean_rank_of_true_binder']}  "
                  f"mean_top10_precision={s['mean_top10_precision']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
