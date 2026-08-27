#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Iterable

import pandas as pd
import torch


DEFAULT_ROOT = Path("/hpc/group/singhlab/user/cy244/projects/peptides")
DEFAULT_SMALL_CSVS = [
    DEFAULT_ROOT / "BindingDB_embeddings" / "train.csv",
    DEFAULT_ROOT / "BindingDB_embeddings" / "val.csv",
    DEFAULT_ROOT / "BindingDB_embeddings" / "test.csv",
]
DEFAULT_PEPTIDE_CSVS = [
    DEFAULT_ROOT / "peptide_embeddings" / "train.csv",
    DEFAULT_ROOT / "peptide_embeddings" / "val.csv",
    DEFAULT_ROOT / "peptide_embeddings" / "test.csv",
]
DEFAULT_SMALL_BINARY = DEFAULT_ROOT / "BindingDB_embeddings" / "morgan_embeddings.pt"
DEFAULT_PEPTIDE_BINARY = DEFAULT_ROOT / "peptide_embeddings" / "morgan_embeddings.pt"
DEFAULT_COUNT_COMBINED = DEFAULT_ROOT / "count_combined_embeddings" / "morgan_embeddings.pt"


def _to_float_tensor(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().float().reshape(-1)
    else:
        tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
    if tensor.ndim != 1:
        raise ValueError(f"Expected 1D fingerprint tensor, got shape {tuple(tensor.shape)}")
    return tensor


def load_embedding_dict(path: Path) -> dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise TypeError(f"Expected {path} to contain a dict, got {type(data).__name__}")
    return {str(key): _to_float_tensor(value) for key, value in data.items()}


def collect_smiles(csv_paths: Iterable[Path], smiles_col: str = "SMILES") -> set[str]:
    smiles: set[str] = set()
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path, usecols=[smiles_col])
        values = frame[smiles_col].dropna().astype(str).str.strip()
        smiles.update(value for value in values if value)
    return smiles


def subset_embeddings(
    embeddings: dict[str, torch.Tensor],
    smiles: set[str],
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    subset = {key: embeddings[key] for key in sorted(smiles & embeddings.keys())}
    diagnostics = {
        "csv_smiles": len(smiles),
        "matched_smiles": len(subset),
        "missing_from_embeddings": len(smiles - embeddings.keys()),
    }
    return subset, diagnostics


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {"p10": 0.0, "p25": 0.0, "p75": 0.0, "p90": 0.0}
    quantiles = torch.quantile(values.float(), torch.tensor([0.10, 0.25, 0.75, 0.90]))
    return {
        "p10": float(quantiles[0].item()),
        "p25": float(quantiles[1].item()),
        "p75": float(quantiles[2].item()),
        "p90": float(quantiles[3].item()),
    }


def fingerprint_sparsity_metrics(
    embeddings: dict[str, torch.Tensor],
    *,
    count_view: bool,
    top_k_bits: int = 20,
) -> dict[str, object]:
    if not embeddings:
        raise ValueError("Cannot summarize an empty embedding dictionary")

    keys = sorted(embeddings)
    matrix = torch.stack([_to_float_tensor(embeddings[key]) for key in keys], dim=0)
    if matrix.ndim != 2:
        raise ValueError(f"Expected stacked fingerprints to be 2D, got shape {tuple(matrix.shape)}")

    nonzero = matrix != 0
    nonzero_per_row = nonzero.sum(dim=1).to(torch.float32)
    density_per_row = nonzero_per_row / matrix.shape[1]
    occupancy = nonzero.to(torch.float32).mean(dim=0)
    active_bits = occupancy > 0
    unique_fingerprints = torch.unique(matrix, dim=0).shape[0]
    top_count = min(top_k_bits, matrix.shape[1])
    top_values, top_indices = torch.topk(occupancy, k=top_count)

    metrics: dict[str, object] = {
        "num_fingerprints": int(matrix.shape[0]),
        "fingerprint_length": int(matrix.shape[1]),
        "nonzero_bits_mean": float(nonzero_per_row.mean().item()),
        "nonzero_bits_median": float(nonzero_per_row.median().item()),
        "nonzero_bits_min": int(nonzero_per_row.min().item()),
        "nonzero_bits_max": int(nonzero_per_row.max().item()),
        "nonzero_bits_quantiles": _quantiles(nonzero_per_row),
        "density_mean": float(density_per_row.mean().item()),
        "density_median": float(density_per_row.median().item()),
        "zero_fraction_mean": float((1.0 - density_per_row).mean().item()),
        "active_global_bit_count": int(active_bits.sum().item()),
        "dead_global_bit_fraction": float((~active_bits).to(torch.float32).mean().item()),
        "per_bit_occupancy_mean": float(occupancy.mean().item()),
        "per_bit_occupancy_median": float(occupancy.median().item()),
        "unique_fingerprint_count": int(unique_fingerprints),
        "duplicate_fingerprint_fraction": float(1.0 - unique_fingerprints / matrix.shape[0]),
        "top_occupied_bits": [
            {"bit": int(idx.item()), "occupancy": float(value.item())}
            for idx, value in zip(top_indices, top_values)
        ],
    }

    if count_view:
        nonzero_values = matrix[nonzero]
        row_sums = matrix.sum(dim=1).to(torch.float32)
        metrics.update(
            {
                "count_mass_mean": float(row_sums.mean().item()),
                "count_mass_median": float(row_sums.median().item()),
                "count_mass_min": float(row_sums.min().item()),
                "count_mass_max": float(row_sums.max().item()),
                "nonzero_count_value_mean": float(nonzero_values.mean().item()) if nonzero_values.numel() else 0.0,
                "nonzero_count_value_median": float(nonzero_values.median().item()) if nonzero_values.numel() else 0.0,
                "max_count_value": float(matrix.max().item()),
                "fraction_nonzero_counts_gt_1": (
                    float((nonzero_values > 1).to(torch.float32).mean().item()) if nonzero_values.numel() else 0.0
                ),
            }
        )

    return metrics


def compare_metrics(small: dict[str, object], peptide: dict[str, object]) -> dict[str, float]:
    small_density = float(small["density_mean"])
    peptide_density = float(peptide["density_mean"])
    return {
        "peptide_to_small_density_ratio": peptide_density / max(small_density, 1e-12),
        "mean_nonzero_bits_difference_peptide_minus_small": (
            float(peptide["nonzero_bits_mean"]) - float(small["nonzero_bits_mean"])
        ),
        "active_global_bit_difference_peptide_minus_small": (
            float(peptide["active_global_bit_count"]) - float(small["active_global_bit_count"])
        ),
    }


def active_bit_overlap(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict[str, float]:
    if not a or not b:
        return {"active_bit_jaccard": 0.0, "shared_active_bit_count": 0.0}
    a_matrix = torch.stack([_to_float_tensor(value) for value in a.values()], dim=0)
    b_matrix = torch.stack([_to_float_tensor(value) for value in b.values()], dim=0)
    if a_matrix.shape[1] != b_matrix.shape[1]:
        return {"active_bit_jaccard": 0.0, "shared_active_bit_count": 0.0}
    a_active = (a_matrix != 0).any(dim=0)
    b_active = (b_matrix != 0).any(dim=0)
    union = (a_active | b_active).sum().item()
    intersection = (a_active & b_active).sum().item()
    return {
        "active_bit_jaccard": float(intersection / union) if union else 0.0,
        "shared_active_bit_count": float(intersection),
    }


def build_summary(args: argparse.Namespace) -> dict[str, object]:
    small_smiles = collect_smiles(args.small_csvs, args.smiles_col)
    peptide_smiles = collect_smiles(args.peptide_csvs, args.smiles_col)

    small_binary = load_embedding_dict(args.small_binary)
    peptide_binary = load_embedding_dict(args.peptide_binary)
    count_combined = load_embedding_dict(args.count_combined)

    small_binary_subset, small_binary_diag = subset_embeddings(small_binary, small_smiles)
    peptide_binary_subset, peptide_binary_diag = subset_embeddings(peptide_binary, peptide_smiles)
    small_count_subset, small_count_diag = subset_embeddings(count_combined, small_smiles)
    peptide_count_subset, peptide_count_diag = subset_embeddings(count_combined, peptide_smiles)

    assigned_count_keys = set(small_count_subset) | set(peptide_count_subset)
    unassigned_count_keys = set(count_combined) - assigned_count_keys

    cohorts = {
        "small_molecule_binary": fingerprint_sparsity_metrics(small_binary_subset, count_view=False),
        "peptide_binary": fingerprint_sparsity_metrics(peptide_binary_subset, count_view=False),
        "small_molecule_count": fingerprint_sparsity_metrics(small_count_subset, count_view=True),
        "peptide_count": fingerprint_sparsity_metrics(peptide_count_subset, count_view=True),
    }

    comparisons = {
        "binary": {
            **compare_metrics(cohorts["small_molecule_binary"], cohorts["peptide_binary"]),
            **active_bit_overlap(small_binary_subset, peptide_binary_subset),
        },
        "count": {
            **compare_metrics(cohorts["small_molecule_count"], cohorts["peptide_count"]),
            **active_bit_overlap(small_count_subset, peptide_count_subset),
        },
    }

    return {
        "inputs": {
            "small_csvs": [str(path) for path in args.small_csvs],
            "peptide_csvs": [str(path) for path in args.peptide_csvs],
            "small_binary": str(args.small_binary),
            "peptide_binary": str(args.peptide_binary),
            "count_combined": str(args.count_combined),
        },
        "diagnostics": {
            "small_binary": small_binary_diag,
            "peptide_binary": peptide_binary_diag,
            "small_count": small_count_diag,
            "peptide_count": peptide_count_diag,
            "count_unassigned_embedding_keys": len(unassigned_count_keys),
        },
        "cohorts": cohorts,
        "comparisons": comparisons,
    }


def write_csv_summary(summary: dict[str, object], path: Path) -> None:
    rows = []
    for cohort_name, metrics in summary["cohorts"].items():
        rows.append(
            {
                "cohort": cohort_name,
                "num_fingerprints": metrics["num_fingerprints"],
                "fingerprint_length": metrics["fingerprint_length"],
                "nonzero_bits_mean": metrics["nonzero_bits_mean"],
                "density_mean": metrics["density_mean"],
                "zero_fraction_mean": metrics["zero_fraction_mean"],
                "active_global_bit_count": metrics["active_global_bit_count"],
                "dead_global_bit_fraction": metrics["dead_global_bit_fraction"],
                "unique_fingerprint_count": metrics["unique_fingerprint_count"],
                "duplicate_fingerprint_fraction": metrics["duplicate_fingerprint_fraction"],
                "count_mass_mean": metrics.get("count_mass_mean", ""),
                "fraction_nonzero_counts_gt_1": metrics.get("fraction_nonzero_counts_gt_1", ""),
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_table(summary: dict[str, object]) -> None:
    print("Morgan fingerprint sparsity")
    print(
        f"{'cohort':<24} {'n':>8} {'len':>6} {'mean_nz':>10} "
        f"{'density':>10} {'active_bits':>12} {'dup_frac':>10}"
    )
    for cohort_name, metrics in summary["cohorts"].items():
        print(
            f"{cohort_name:<24} {metrics['num_fingerprints']:>8} "
            f"{metrics['fingerprint_length']:>6} {metrics['nonzero_bits_mean']:>10.2f} "
            f"{metrics['density_mean']:>10.4f} {metrics['active_global_bit_count']:>12} "
            f"{metrics['duplicate_fingerprint_fraction']:>10.4f}"
        )
    print("\nComparisons")
    for name, metrics in summary["comparisons"].items():
        print(
            f"{name}: density_ratio={metrics['peptide_to_small_density_ratio']:.3f}, "
            f"mean_nz_diff={metrics['mean_nonzero_bits_difference_peptide_minus_small']:.2f}, "
            f"active_bit_jaccard={metrics['active_bit_jaccard']:.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Morgan fingerprint sparsity by cohort.")
    parser.add_argument("--small-csvs", nargs="+", type=Path, default=DEFAULT_SMALL_CSVS)
    parser.add_argument("--peptide-csvs", nargs="+", type=Path, default=DEFAULT_PEPTIDE_CSVS)
    parser.add_argument("--small-binary", type=Path, default=DEFAULT_SMALL_BINARY)
    parser.add_argument("--peptide-binary", type=Path, default=DEFAULT_PEPTIDE_BINARY)
    parser.add_argument("--count-combined", type=Path, default=DEFAULT_COUNT_COMBINED)
    parser.add_argument("--smiles-col", default="SMILES")
    parser.add_argument("--json-out", type=Path, default=Path("morgan_sparsity_summary.json"))
    parser.add_argument("--csv-out", type=Path, default=Path("morgan_sparsity_summary.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(args)
    print_table(summary)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv_summary(summary, args.csv_out)
    print(f"\nWrote JSON: {args.json_out}")
    print(f"Wrote CSV: {args.csv_out}")


if __name__ == "__main__":
    main()
