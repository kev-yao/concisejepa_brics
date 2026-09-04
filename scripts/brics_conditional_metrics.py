#!/usr/bin/env python3
"""Rank ligands within receptor groups, and receptors within ligand groups."""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss


def probability_diagnostics(frame, score_column="mean"):
    y, pred = frame.label, frame[score_column]
    positive = y == 1
    return {
        "rows": len(frame),
        "positives": int(positive.sum()),
        "positive_rate": float(y.mean()) if len(frame) else None,
        "auprc": float(average_precision_score(y, pred)) if y.nunique() == 2 else None,
        "auroc": float(roc_auc_score(y, pred)) if y.nunique() == 2 else None,
        "brier": float(brier_score_loss(y, pred)) if len(frame) else None,
        "exact_zero_positive_predictions": int((positive & (pred == 0)).sum()),
        "near_zero_positive_predictions": int((positive & (pred < 1e-6)).sum()),
        "exact_one_negative_predictions": int(((y == 0) & (pred == 1)).sum()),
    }


def conditional_metrics(frame, group_column, score_column="mean", min_rows=5):
    rows = []
    for key, group in frame.groupby(group_column, sort=True):
        if len(group) < min_rows or group.label.nunique() < 2:
            continue
        y = group.label
        pred = group[score_column]
        positives = int(y.sum())
        negatives = len(y) - positives
        rows.append(
            {
                "group": key,
                "n": len(y),
                "positive_rate": float(y.mean()),
                "auprc": float(average_precision_score(y, pred)),
                "auroc": float(roc_auc_score(y, pred)),
                "comparisons": positives * negatives,
            }
        )
    if not rows:
        return {"groups": 0, "rows": 0}
    comparisons = sum(r["comparisons"] for r in rows)
    return {
        "groups": len(rows),
        "rows": sum(r["n"] for r in rows),
        "minimum_group_rows": min_rows,
        "macro_auroc": sum(r["auroc"] for r in rows) / len(rows),
        "macro_auprc": sum(r["auprc"] for r in rows) / len(rows),
        "macro_prevalence": sum(r["positive_rate"] for r in rows) / len(rows),
        "macro_ap_lift_over_constant": sum(r["auprc"] - r["positive_rate"] for r in rows) / len(rows),
        "within_group_pair_weighted_auroc": sum(r["auroc"] * r["comparisons"] for r in rows) / comparisons,
    }


def run(root):
    result = {}
    novelty_path = root / "fragment_input_groups.csv"
    novelty = pd.read_csv(novelty_path) if novelty_path.exists() else None
    probability_results = {}
    for path in sorted(root.glob("*/seed_*/best_dti_val_predictions.csv")):
        frame = pd.read_csv(path)
        values = {}
        for group, col in (("within_protein", "protein_sha256"), ("within_molecule", "SMILES")):
            for view in ("mean", "whole", "fragment"):
                values[f"{group}_{view}"] = conditional_metrics(frame, col, view)
        result[str(path.parent.relative_to(root))] = values
        cohorts = {"all": frame}
        if novelty is not None:
            joined = frame.merge(novelty[novelty.split == "val"], on="SMILES", validate="many_to_one")
            if len(joined) != len(frame):
                raise ValueError("Fragment novelty table does not cover every validation row")
            for seen in (False, True):
                cohorts[f"fragment_input_seen_in_train_{seen}"] = joined[joined.fragment_input_seen_in_train == seen]
        probability_results[str(path.parent.relative_to(root))] = {
            name: {view: probability_diagnostics(group, view) for view in ("mean", "whole", "fragment")}
            for name, group in cohorts.items()
        }
    (root / "conditional_validation_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (root / "probability_and_novelty_validation_metrics.json").write_text(
        json.dumps(probability_results, indent=2) + "\n"
    )
    if novelty is not None and "cached_fragment_count" in novelty:
        reconstruction_results = {}
        for path in sorted(root.glob("*/seed_*/reconstruction_validation.json")):
            examples = pd.DataFrame(json.loads(path.read_text())["examples"])
            joined = examples.merge(
                novelty[novelty.split == "val"], left_on="input_smiles", right_on="SMILES", validate="one_to_one"
            )
            if len(joined) != len(examples):
                raise ValueError("Fragment table does not cover every reconstruction sample")
            cohorts = {
                "all": joined,
                "multiple_cached_fragments": joined[joined.cached_fragment_count > 1],
                "single_cached_fragment": joined[joined.cached_fragment_count == 1],
            }
            if "has_whole_molecule_fingerprint_token" in joined:
                for whole in (False, True):
                    cohorts[f"has_whole_fingerprint_token_{whole}"] = joined[
                        joined.has_whole_molecule_fingerprint_token == whole
                    ]
            reconstruction_results[str(path.parent.relative_to(root))] = {
                name: {
                    "n": len(group),
                    "valid_count": int(group.decoded.notna().sum()),
                    "ecfp_all": float(group.ecfp.mean()) if len(group) else None,
                    "retrieval_top10": float((group.retrieval_rank <= 10).mean()) if len(group) else None,
                }
                for name, group in cohorts.items()
            }
        (root / "reconstruction_fragment_subgroups.json").write_text(
            json.dumps(reconstruction_results, indent=2) + "\n"
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    run(parser.parse_args().root)
