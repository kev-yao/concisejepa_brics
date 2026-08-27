import argparse
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import torch

from scripts.analyze_morgan_sparsity import (
    active_bit_overlap,
    build_summary,
    collect_smiles,
    compare_metrics,
    fingerprint_sparsity_metrics,
    subset_embeddings,
    write_csv_summary,
)


class MorganSparsityAnalysisTests(unittest.TestCase):
    def test_binary_sparsity_metrics(self):
        embeddings = {
            "A": torch.tensor([1.0, 0.0, 1.0, 0.0]),
            "B": torch.tensor([0.0, 0.0, 1.0, 0.0]),
            "C": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        }

        metrics = fingerprint_sparsity_metrics(embeddings, count_view=False, top_k_bits=2)

        self.assertEqual(metrics["num_fingerprints"], 3)
        self.assertEqual(metrics["fingerprint_length"], 4)
        self.assertAlmostEqual(metrics["nonzero_bits_mean"], 4 / 3)
        self.assertAlmostEqual(metrics["density_mean"], 1 / 3)
        self.assertEqual(metrics["active_global_bit_count"], 2)
        self.assertAlmostEqual(metrics["dead_global_bit_fraction"], 0.5)
        self.assertEqual(metrics["unique_fingerprint_count"], 3)
        self.assertEqual(len(metrics["top_occupied_bits"]), 2)

    def test_count_sparsity_metrics(self):
        embeddings = {
            "A": torch.tensor([2.0, 0.0, 1.0]),
            "B": torch.tensor([0.0, 3.0, 0.0]),
        }

        metrics = fingerprint_sparsity_metrics(embeddings, count_view=True)

        self.assertAlmostEqual(metrics["nonzero_bits_mean"], 1.5)
        self.assertAlmostEqual(metrics["count_mass_mean"], 3.0)
        self.assertEqual(metrics["max_count_value"], 3.0)
        self.assertAlmostEqual(metrics["fraction_nonzero_counts_gt_1"], 2 / 3)

    def test_smiles_collection_and_subset_diagnostics(self):
        with TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            pd.DataFrame({"SMILES": [" A ", "B", None, ""]}).to_csv(csv_path, index=False)

            smiles = collect_smiles([csv_path])
            subset, diagnostics = subset_embeddings(
                {"A": torch.ones(2), "C": torch.zeros(2)},
                smiles,
            )

        self.assertEqual(smiles, {"A", "B"})
        self.assertEqual(set(subset), {"A"})
        self.assertEqual(diagnostics["csv_smiles"], 2)
        self.assertEqual(diagnostics["matched_smiles"], 1)
        self.assertEqual(diagnostics["missing_from_embeddings"], 1)

    def test_comparison_and_active_bit_overlap(self):
        small = fingerprint_sparsity_metrics({"A": torch.tensor([1.0, 0.0, 0.0, 0.0])}, count_view=False)
        peptide = fingerprint_sparsity_metrics({"B": torch.tensor([1.0, 1.0, 0.0, 0.0])}, count_view=False)

        comparison = compare_metrics(small, peptide)
        overlap = active_bit_overlap(
            {"A": torch.tensor([1.0, 0.0, 0.0, 0.0])},
            {"B": torch.tensor([1.0, 1.0, 0.0, 0.0])},
        )

        self.assertAlmostEqual(comparison["peptide_to_small_density_ratio"], 2.0)
        self.assertAlmostEqual(comparison["mean_nonzero_bits_difference_peptide_minus_small"], 1.0)
        self.assertAlmostEqual(overlap["active_bit_jaccard"], 0.5)
        self.assertEqual(overlap["shared_active_bit_count"], 1.0)

    def test_build_summary_and_csv_output(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            small_csv = root / "small.csv"
            peptide_csv = root / "peptide.csv"
            small_binary_path = root / "small_binary.pt"
            peptide_binary_path = root / "peptide_binary.pt"
            count_path = root / "count.pt"
            csv_out = root / "summary.csv"

            pd.DataFrame({"SMILES": ["SMALL1", "SMALL2"]}).to_csv(small_csv, index=False)
            pd.DataFrame({"SMILES": ["PEP1"]}).to_csv(peptide_csv, index=False)
            torch.save(
                {
                    "SMALL1": torch.tensor([1.0, 0.0, 0.0]),
                    "SMALL2": torch.tensor([0.0, 1.0, 0.0]),
                },
                small_binary_path,
            )
            torch.save({"PEP1": torch.tensor([1.0, 1.0, 0.0])}, peptide_binary_path)
            torch.save(
                {
                    "SMALL1": torch.tensor([2.0, 0.0, 0.0]),
                    "SMALL2": torch.tensor([0.0, 1.0, 0.0]),
                    "PEP1": torch.tensor([1.0, 2.0, 0.0]),
                    "UNUSED": torch.tensor([0.0, 0.0, 1.0]),
                },
                count_path,
            )

            args = argparse.Namespace(
                small_csvs=[small_csv],
                peptide_csvs=[peptide_csv],
                small_binary=small_binary_path,
                peptide_binary=peptide_binary_path,
                count_combined=count_path,
                smiles_col="SMILES",
            )
            summary = build_summary(args)
            write_csv_summary(summary, csv_out)

            loaded_rows = csv_out.read_text(encoding="utf-8").splitlines()
            json.dumps(summary)

        self.assertEqual(summary["diagnostics"]["count_unassigned_embedding_keys"], 1)
        self.assertEqual(summary["cohorts"]["small_molecule_count"]["num_fingerprints"], 2)
        self.assertEqual(summary["cohorts"]["peptide_count"]["num_fingerprints"], 1)
        self.assertIn("cohort,num_fingerprints", loaded_rows[0])


if __name__ == "__main__":
    unittest.main()
