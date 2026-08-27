#!/usr/bin/env python3
"""
D2: FSQ codebook utilization analysis.

Loads the latent_query_q4 checkpoint, passes all fragment FPs through
DrugEncoder, and reports how many of the 32^3 = 32,768 possible code
triplets are actually used.

Usage (on login node with env activated):
  python scripts/analyze_codebook_utilization.py
"""
import sys
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CKPT = (
    "/hpc/group/singhlab/user/cy244/projects/peptide_evals/fragment_pool_study"
    "/fragment_latent_query_719fa566/checkpoints/epoch=29-val/dti_auprc=0.5587.ckpt"
)
FRAG_CACHE = (
    "/hpc/group/singhlab/user/cy244/projects/peptides"
    "/count_combined_embeddings/fragment_fps_r4_2048.pt"
)
BATCH_SIZE = 1024
N_CODEBOOK_DIMS = 3
LEVELS_PER_DIM = 32
N_POSSIBLE = LEVELS_PER_DIM ** N_CODEBOOK_DIMS  # 32,768


def load_drug_encoder(ckpt_path: str):
    from concisejepa.models.drug_decoder import DrugEncoder

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"]

    enc = DrugEncoder(
        layers=[[32, 32, 32]],
        dim=2048,
        latent_dim=128,
        activation=torch.nn.Tanh,
        quantizer={"type": "fsq"},
    )
    prefix = "model.concise.d_encoder."
    enc_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    enc.load_state_dict(enc_state)
    enc.eval()
    return enc


def main():
    print(f"Loading checkpoint: {CKPT}")
    enc = load_drug_encoder(CKPT)
    print("DrugEncoder loaded.")

    print(f"Loading fragment FP cache: {FRAG_CACHE}")
    frag_cache = torch.load(FRAG_CACHE, map_location="cpu")
    print(f"Cache: {len(frag_cache)} SMILES")

    # Collect per-molecule fragment stats
    frag_counts = [fps.shape[0] for fps in frag_cache.values()]
    total_frags = sum(frag_counts)
    print(f"\nFragment count per molecule:")
    print(f"  total fragments : {total_frags}")
    print(f"  min / max / mean: {min(frag_counts)} / {max(frag_counts)} / {total_frags/len(frag_counts):.2f}")
    from collections import Counter as C
    dist = C(frag_counts)
    print(f"  distribution    : {dict(sorted(dist.items()))}")

    # Concatenate all fragment FPs → [N_total, 2048]
    all_fps = torch.cat(list(frag_cache.values()), dim=0)
    print(f"\nRunning {len(all_fps)} fragment FPs through DrugEncoder ...")

    all_codes = []
    with torch.no_grad():
        for i in tqdm(range(0, len(all_fps), BATCH_SIZE), desc="encoding"):
            batch = all_fps[i : i + BATCH_SIZE]
            out = enc(batch)
            all_codes.append(out["codes"].cpu())  # [B, 3]

    all_codes = torch.cat(all_codes, dim=0)  # [N_total, 3]
    print(f"Code tensor shape: {all_codes.shape}")

    # --- Utilization ---
    code_tuples = [tuple(c.tolist()) for c in all_codes.long()]
    counter = Counter(code_tuples)
    n_unique = len(counter)
    utilization_pct = n_unique / N_POSSIBLE * 100

    print(f"\n{'='*50}")
    print(f"FSQ Codebook Utilization (latent_query_q4 checkpoint)")
    print(f"{'='*50}")
    print(f"  Unique triplets used : {n_unique:,} / {N_POSSIBLE:,}  ({utilization_pct:.1f}%)")
    print(f"  Codes used once      : {sum(1 for v in counter.values() if v == 1):,}")
    print(f"  Codes used >10x      : {sum(1 for v in counter.values() if v > 10):,}")
    print(f"  Codes used >100x     : {sum(1 for v in counter.values() if v > 100):,}")
    print(f"  Most common code     : {counter.most_common(1)[0]}")

    # Per-dimension utilization
    print(f"\nPer-dimension utilization:")
    for i in range(N_CODEBOOK_DIMS):
        vals = all_codes[:, i].long().unique()
        print(f"  Dim {i}: {len(vals)}/{LEVELS_PER_DIM} values used ({len(vals)/LEVELS_PER_DIM*100:.0f}%)")

    # Entropy of distribution (max = log2(N_POSSIBLE) = 15 bits)
    import math
    total = len(code_tuples)
    entropy = -sum((v / total) * math.log2(v / total) for v in counter.values())
    max_entropy = math.log2(N_POSSIBLE)
    print(f"\nCode distribution entropy : {entropy:.2f} / {max_entropy:.2f} bits ({entropy/max_entropy*100:.1f}% of max)")
    print(f"{'='*50}")

    # Save
    out_path = PROJECT_ROOT / "codebook_utilization.json"
    import json
    json.dump(
        {
            "n_unique": n_unique,
            "n_possible": N_POSSIBLE,
            "utilization_pct": utilization_pct,
            "entropy_bits": entropy,
            "max_entropy_bits": max_entropy,
            "entropy_pct": entropy / max_entropy * 100,
            "n_frags_total": total_frags,
            "frag_count_dist": {str(k): v for k, v in dist.items()},
        },
        open(out_path, "w"),
        indent=2,
    )
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
