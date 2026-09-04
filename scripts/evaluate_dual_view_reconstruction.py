#!/usr/bin/env python3
"""Decode dual-view BRICS JEPA predictions and measure reconstruction fidelity."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean, median

import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from rdkit import Chem, DataStructs
from rdkit.Chem import RDKFingerprint, rdFingerprintGenerator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def canonicalize_largest_fragment(smiles: str | None) -> str | None:
    """Return a canonical SMILES for the largest valid component."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fragments = Chem.GetMolFrags(mol, asMols=True)
    if not fragments:
        return None
    largest = max(fragments, key=lambda item: item.GetNumHeavyAtoms())
    try:
        return Chem.CanonSmiles(Chem.MolToSmiles(largest))
    except Exception:
        return None


def _fingerprint(smiles: str | None, kind: str):
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if kind == "ecfp":
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        return generator.GetFingerprint(mol)
    if kind == "rdk":
        return RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=2048)
    raise ValueError(f"Unknown fingerprint kind: {kind}")


def tanimoto_scores(targets: list[str | None], predictions: list[str | None], kind: str) -> list[float]:
    """Score aligned SMILES pairs, assigning zero to invalid molecules."""
    scores = []
    for target, prediction in zip(targets, predictions, strict=True):
        target_fp = _fingerprint(target, kind)
        prediction_fp = _fingerprint(prediction, kind)
        if target_fp is None or prediction_fp is None:
            scores.append(0.0)
        else:
            scores.append(float(DataStructs.TanimotoSimilarity(target_fp, prediction_fp)))
    return scores


def describe(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    ordered = sorted(values)
    p90_index = min(int(0.9 * len(ordered)), len(ordered) - 1)
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "p90": float(ordered[p90_index]),
        "max": float(ordered[-1]),
    }


def reconstruction_summary(
    targets: list[str | None],
    predictions: list[str | None],
    shuffled_targets: list[str | None],
) -> tuple[dict, dict[str, list[float]]]:
    valid_indices = [index for index, value in enumerate(predictions) if value is not None]
    exact_count = sum(predictions[index] == targets[index] for index in valid_indices)
    scores: dict[str, list[float]] = {}
    metrics = {
        "n": len(targets),
        "valid_count": len(valid_indices),
        "valid_rate": len(valid_indices) / max(len(targets), 1),
        "exact_count": exact_count,
        "exact_rate": exact_count / max(len(targets), 1),
        "unique_valid_predictions": len({predictions[index] for index in valid_indices}),
    }
    for kind in ("ecfp", "rdk"):
        aligned = tanimoto_scores(targets, predictions, kind)
        shuffled = tanimoto_scores(shuffled_targets, predictions, kind)
        scores[kind] = aligned
        valid_aligned = [aligned[index] for index in valid_indices]
        valid_shuffled = [shuffled[index] for index in valid_indices]
        metrics[kind] = {
            "all": describe(aligned),
            "valid_only": describe(valid_aligned),
            "shuffled_valid_only": describe(valid_shuffled),
            "lift_over_shuffled": mean(valid_aligned) - mean(valid_shuffled)
            if valid_aligned
            else 0.0,
        }
    return metrics, scores


def pad_fragments(smiles: list[str], fragment_cache: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    fragments = [fragment_cache[value].float() for value in smiles]
    max_fragments = max(item.shape[0] for item in fragments)
    fingerprint_dim = fragments[0].shape[-1]
    padded = torch.zeros(len(fragments), max_fragments, fingerprint_dim)
    mask = torch.zeros(len(fragments), max_fragments, dtype=torch.bool)
    for index, item in enumerate(fragments):
        count = item.shape[0]
        padded[index, :count] = item
        mask[index, :count] = True
    return padded, mask


@torch.no_grad()
def predict_latents(model, smiles, fragment_cache, batch_size: int, device: torch.device) -> torch.Tensor:
    predictions = []
    model.to(device).eval()
    for start in range(0, len(smiles), batch_size):
        chunk = smiles[start : start + batch_size]
        fragments, mask = pad_fragments(chunk, fragment_cache)
        pooled, _, _, _ = model._encode_fragments(fragments.to(device), mask.to(device))
        predictions.append(model.coati_predictor(pooled).cpu())
    return torch.cat(predictions, dim=0)


@torch.no_grad()
def decode_latents(encoder, tokenizer, latents: torch.Tensor, batch_size: int, device: torch.device,
                   seed: int = 123, top_k: int = 1):
    if top_k < 1:
        raise ValueError("top_k must be positive")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    decoded = []
    for start in range(0, latents.shape[0], batch_size):
        chunk = latents[start : start + batch_size].to(device).float()
        decoded.extend(
            encoder.hclip_to_2d_batch(
                h_clip=chunk,
                tokenizer=tokenizer,
                noise_scale=0.0,
                k=top_k,
            )
        )
    if len(decoded) != len(latents):
        raise RuntimeError("COATI returned a different number of molecules than requested")
    return decoded


def load_model(config, checkpoint: Path):
    model = instantiate(config.model)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = {
        name.removeprefix("model."): value
        for name, value in payload["state_dict"].items()
        if name.startswith("model.")
    }
    model.load_state_dict(state, strict=True)
    return model


def sample_test_smiles(config, count: int, seed: int, split: str = "test"):
    data = config.data
    fragment_cache = torch.load(data.fragment_fps_path, map_location="cpu", weights_only=False)
    molecule_cache = torch.load(data.morgan_embeddings_path, map_location="cpu", weights_only=False)
    target_cache = torch.load(data.smiles_embeddings_path, map_location="cpu", weights_only=False)
    protein_cache = torch.load(data.protein_embeddings_path, map_location="cpu", weights_only=False)

    if count < 2:
        raise ValueError("At least two molecules are needed for the shuffled baseline")
    frame = pd.read_csv(data[f"{split}_csv"], usecols=[data.smiles_col, data.sequence_col])
    frame[data.smiles_col] = frame[data.smiles_col].astype(str).str.strip()
    frame[data.sequence_col] = frame[data.sequence_col].astype(str).str.strip()
    valid_smiles = set(fragment_cache) & set(molecule_cache) & set(target_cache)
    frame = frame[
        frame[data.smiles_col].isin(valid_smiles)
        & frame[data.sequence_col].isin(set(protein_cache))
    ]
    population = sorted(frame[data.smiles_col].unique().tolist())
    if count > len(population):
        raise ValueError(f"Requested {count} molecules, but only {len(population)} are available.")
    sampled = random.Random(seed).sample(population, count)
    targets = torch.stack([target_cache[value].reshape(-1).float() for value in sampled])
    return sampled, targets, fragment_cache, len(population)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n-molecules", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decode-seed", type=int, default=123)
    parser.add_argument("--top-k", type=int, default=1, help="1 gives greedy decoding")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--coati-doc-url", default="s3://terray-public/models/grande_closed.pkl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.run_dir / "resolved_config.yaml")
    device = torch.device(args.device)
    sampled, target_latents, fragment_cache, population_size = sample_test_smiles(
        config, args.n_molecules, args.seed, args.split
    )
    print(f"Sampled {len(sampled)} of {population_size} unique valid test molecules.")

    model = load_model(config, args.checkpoint)
    predicted_latents = predict_latents(model, sampled, fragment_cache, args.batch_size, device)
    sample_mse = torch.mean((predicted_latents - target_latents) ** 2).item()

    from coati.models.io import load_e3gnn_smiles_clip_e2e

    encoder, tokenizer = load_e3gnn_smiles_clip_e2e(
        freeze=True,
        device=str(device),
        doc_url=args.coati_doc_url,
    )
    encoder.eval()
    predicted_raw = decode_latents(encoder, tokenizer, predicted_latents, args.batch_size, device,
                                   args.decode_seed, args.top_k)
    target_raw = decode_latents(encoder, tokenizer, target_latents, args.batch_size, device,
                                args.decode_seed, args.top_k)

    targets = [canonicalize_largest_fragment(value) for value in sampled]
    predicted = [canonicalize_largest_fragment(value) for value in predicted_raw]
    target_decoded = [canonicalize_largest_fragment(value) for value in target_raw]
    shuffled_targets = targets[1:] + targets[:1]

    prediction_metrics, prediction_scores = reconstruction_summary(targets, predicted, shuffled_targets)
    target_metrics, target_scores = reconstruction_summary(targets, target_decoded, shuffled_targets)
    examples = []
    for index in range(len(sampled)):
        examples.append(
            {
                "target_smiles": targets[index],
                "input_raw_smiles": sampled[index],
                "predicted_raw_smiles": predicted_raw[index],
                "predicted_smiles": predicted[index],
                "target_latent_decoded_smiles": target_decoded[index],
                "prediction_ecfp_tanimoto": prediction_scores["ecfp"][index],
                "prediction_rdk_tanimoto": prediction_scores["rdk"][index],
                "target_latent_ecfp_tanimoto": target_scores["ecfp"][index],
                "target_latent_rdk_tanimoto": target_scores["rdk"][index],
            }
        )

    output = {
        "run_dir": str(args.run_dir.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "split": args.split,
        "decoding": {"seed": args.decode_seed, "top_k": args.top_k, "noise_scale": 0.0},
        "sampling": {
            "seed": args.seed,
            "n_sampled_unique_molecules": len(sampled),
            "population_unique_molecules": population_size,
        },
        "sample_jepa_mse": sample_mse,
        "prediction": prediction_metrics,
        "target_latent_decoder_control": target_metrics,
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print(f"JEPA sample MSE: {sample_mse:.6f}")
    for label, metrics in (("JEPA prediction", prediction_metrics), ("target latent control", target_metrics)):
        print(
            f"{label}: valid={metrics['valid_count']}/{metrics['n']} "
            f"({metrics['valid_rate']:.1%}), exact={metrics['exact_count']}, "
            f"ECFP={metrics['ecfp']['valid_only']['mean']:.4f}, "
            f"RDK={metrics['rdk']['valid_only']['mean']:.4f}"
        )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
