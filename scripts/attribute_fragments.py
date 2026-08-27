#!/usr/bin/env python3
"""
B1: Fragment-level attribution extractor — the original, simplest entry point for attributing
a single molecule (or a CSV of them). weighted_sum/mlp_weighted_sum only; e3_f2r_attribution.py
and e4_realprotein_attribution.py extend this same idea to F2R and max, and are the scripts
that produced this project's actually-reported attribution numbers (docs/PROJECT_HANDOFF.md §6)
using real per-pair proteins instead of the single fixed molecule this script demonstrates on.

Loads a trained WeightedSumPool or MLPWeightedSumPool checkpoint and returns
per-fragment softmax attention weights for any SMILES string.

The scientific value: for molecules A, B, C that bind a protein and molecule D
that doesn't (all structurally similar), the fragment unique to D that receives
a distinctive weight profile is the one preventing binding.

Usage — single molecule:
  python scripts/attribute_fragments.py \\
    --checkpoint /path/to/ckpt.ckpt \\
    --pooling weighted_sum \\
    --smiles "CCO"

Usage — batch from CSV (must have a 'SMILES' column):
  python scripts/attribute_fragments.py \\
    --checkpoint /path/to/ckpt.ckpt \\
    --pooling weighted_sum \\
    --input_csv matched_pairs.csv \\
    --output attribution_results.json
"""

import argparse
import json
import signal
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

BRICS_TIMEOUT_SEC = 3


def _alarm_handler(signum, frame):
    raise TimeoutError


def brics_fragment(smiles: str) -> list[str]:
    """BRICS decompose with 3s timeout; fallback to whole molecule."""
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
    # Strip dummy atoms and canonicalize
    cleaned = []
    for f in raw:
        fm = Chem.MolFromSmiles(f)
        if fm is not None:
            cleaned.append(Chem.MolToSmiles(fm))
    return cleaned if cleaned else [smiles]


def morgan_fp(smiles: str, nbits: int = 2048) -> torch.Tensor:
    """Count ECFP4 fingerprint matching precompute.sbatch (kind='ecfp-count:4')."""
    try:
        from molfeat.trans.fp import FPVecTransformer
        tr = FPVecTransformer(kind="ecfp-count:4", length=nbits)
        arr, valid = tr([smiles], ignore_errors=True)
        if len(arr) == 0:
            return torch.zeros(nbits, dtype=torch.float32)
        return torch.tensor(arr[0], dtype=torch.float32)
    except Exception:
        return torch.zeros(nbits, dtype=torch.float32)


class AttributionExtractor:
    """
    Extracts per-fragment attribution weights from a trained fragment model.

    Supported pooling types: weighted_sum, mlp_weighted_sum.
    Both save softmax fragment weights as `pooling_layer.last_weights` during forward.
    """

    def __init__(self, checkpoint_path: str, pooling: str):
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
        # LitFragment stores model under "model." prefix
        model_state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
        self.model.load_state_dict(model_state)
        self.model.eval()

        self.pooler = self.model.concise.pooling_layer
        pooler_type = type(self.pooler).__name__
        if pooler_type not in ("WeightedSumPool", "MLPWeightedSumPool"):
            raise ValueError(
                f"Attribution requires WeightedSumPool or MLPWeightedSumPool, got {pooler_type}. "
                f"Rerun with --pooling weighted_sum or --pooling mlp_weighted_sum."
            )
        print(f"Loaded {pooler_type} checkpoint: {checkpoint_path}")

    @torch.no_grad()
    def attribute(self, smiles: str, protein_emb: torch.Tensor | None = None) -> dict:
        """
        Attribute a molecule to its BRICS fragments.

        Args:
            smiles: Input SMILES string.
            protein_emb: Optional [50, 1280] tensor. If provided, also returns binding score.

        Returns:
            {
                smiles: str,
                n_frags: int,
                fragments: [str, ...],
                weights: [float, ...],  # softmax-normalized, sums to 1
                sorted: [(frag_smiles, weight), ...]  # descending by weight
                binding_score: float | None
            }
        """
        frags = brics_fragment(smiles)
        fps = torch.stack([morgan_fp(f) for f in frags])  # [F, 2048]
        frag_fps = fps.unsqueeze(0)    # [1, F, 2048]
        frag_mask = torch.ones(1, len(frags), dtype=torch.bool)

        # Dummy protein embedding — pooling weights don't depend on it for these poolers
        dummy_r = torch.zeros(1, 50, 1280) if protein_emb is None else protein_emb.unsqueeze(0)

        # _encode_fragments calls pooler.forward() which sets pooler.last_weights
        self.model.concise._encode_fragments(frag_fps, frag_mask, dummy_r)
        weights = self.pooler.last_weights.squeeze(0).tolist()  # [F]

        binding_score = None
        if protein_emb is not None:
            out = self.model(dummy_r, frag_fps, frag_mask)
            binding_score = float(out["binding"].item())

        sorted_pairs = sorted(zip(frags, weights), key=lambda x: x[1], reverse=True)

        return {
            "smiles": smiles,
            "n_frags": len(frags),
            "fragments": frags,
            "weights": weights,
            "sorted": [(f, w) for f, w in sorted_pairs],
            "binding_score": binding_score,
        }

    def attribute_batch(self, smiles_list: list[str]) -> list[dict]:
        return [self.attribute(s) for s in smiles_list]


def main():
    p = argparse.ArgumentParser(description="Fragment attribution extractor (B1)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pooling", default="weighted_sum",
                   choices=["weighted_sum", "mlp_weighted_sum"])
    p.add_argument("--smiles", default=None, help="Single SMILES string")
    p.add_argument("--input_csv", default=None, help="CSV with 'SMILES' column (batch mode)")
    p.add_argument("--output", default=None, help="Output JSON for batch mode")
    args = p.parse_args()

    extractor = AttributionExtractor(args.checkpoint, args.pooling)

    if args.smiles:
        result = extractor.attribute(args.smiles)
        print(f"\nMolecule: {result['smiles']}")
        print(f"Fragments ({result['n_frags']} total, weights sum to {sum(result['weights']):.4f}):")
        for frag, w in result["sorted"]:
            bar = "#" * int(w * 50)
            print(f"  {w:.4f} |{bar:<50}| {frag}")

    elif args.input_csv:
        import pandas as pd
        df = pd.read_csv(args.input_csv)
        smiles_col = "SMILES" if "SMILES" in df.columns else df.columns[0]
        print(f"Attributing {len(df)} molecules ...")
        results = extractor.attribute_batch(df[smiles_col].tolist())
        out_path = args.output or str(Path(args.input_csv).with_suffix("_attribution.json"))
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {len(results)} attribution results to {out_path}")
    else:
        p.error("Provide --smiles or --input_csv")


if __name__ == "__main__":
    main()
