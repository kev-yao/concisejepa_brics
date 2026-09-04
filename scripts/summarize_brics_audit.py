#!/usr/bin/env python3
"""Aggregate the fixed suite without selecting experiments on test performance."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

ARMS = ["fsq_joint", "continuous_joint", "fsq_jepa", "continuous_jepa", "fsq_dti", "continuous_dti"]


def extract(summary):
    best = summary["checkpoints"]["best_dti"]
    jepa = summary["checkpoints"]["best_jepa"]
    row = {
        "arm": summary["arm"],
        "seed": summary["seed"],
        "val_auprc": best["validation"]["mean"]["auprc"],
        "test_auprc": best["test"]["mean"]["auprc"],
        "test_auroc": best["test"]["mean"]["auroc"],
        "val_fragment_auprc": best["validation"]["fragment"]["auprc"],
        "val_whole_auprc": best["validation"]["whole"]["auprc"],
        "dti_epoch": best["epoch"],
        "jepa_epoch": jepa["epoch"],
    }
    recon = summary.get("reconstruction")
    if recon:
        row.update(
            {
                "unique_val_jepa_mse": recon["full_unique_validation"]["mse"],
                "mean_baseline_mse": summary["initial"]["train_mean_mse"],
                "ecfp_all": recon["prediction"]["ecfp"]["all"]["mean"],
                "ecfp_valid": recon["prediction"]["ecfp"]["valid_only"]["mean"],
                "ecfp_shuffled_valid": recon["prediction"]["ecfp"]["shuffled_valid_only"]["mean"],
                "valid_rate": recon["prediction"]["valid_rate"],
                "exact_count": recon["prediction"]["exact_count"],
                "retrieval_top1": recon["retrieval_top1"],
                "retrieval_top10": recon["retrieval_top10"],
                "prediction_variance": recon["full_unique_validation"]["prediction_variance"],
            }
        )
    return row


def aggregate(values):
    return {"n": len(values), "mean": mean(values), "sd": stdev(values) if len(values) > 1 else None}


def summary(root):
    rows = [extract(json.loads(path.read_text())) for path in sorted(root.glob("*/seed_*/audit_summary.json"))]
    stats = {}
    for arm in ARMS:
        subset = [r for r in rows if r["arm"] == arm]
        stats[arm] = {}
        for key in sorted({k for r in subset for k in r} - {"arm", "seed"}):
            values = [r[key] for r in subset if r.get(key) is not None]
            if values:
                stats[arm][key] = aggregate(values)
    paired = {}
    comparisons = [
        ("continuous_jepa", "fsq_jepa", "unique_val_jepa_mse"),
        ("continuous_joint", "fsq_joint", "unique_val_jepa_mse"),
        ("fsq_joint", "fsq_jepa", "unique_val_jepa_mse"),
        ("continuous_joint", "continuous_jepa", "unique_val_jepa_mse"),
        ("fsq_joint", "fsq_dti", "val_auprc"),
        ("continuous_joint", "continuous_dti", "val_auprc"),
    ]
    for a, b, key in comparisons:
        left = {r["seed"]: r.get(key) for r in rows if r["arm"] == a}
        right = {r["seed"]: r.get(key) for r in rows if r["arm"] == b}
        seeds = sorted(left.keys() & right.keys())
        differences = [left[s] - right[s] for s in seeds if left[s] is not None and right[s] is not None]
        if differences:
            paired[f"{a} minus {b}: {key}"] = {**aggregate(differences), "differences": differences, "seeds": seeds}
    result = {
        "completed_runs": len(rows),
        "expected_runs": 18,
        "per_arm": stats,
        "paired_seed_differences": paired,
        "runs": rows,
    }
    (root / "suite_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    if rows:
        fields = ["arm", "seed"] + sorted({k for r in rows for k in r} - {"arm", "seed"})
        with (root / "suite_summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def fmt(arm, key):
        value = stats[arm].get(key)
        if not value:
            return "—"
        return f"{value['mean']:.4f}" + (f" ± {value['sd']:.4f}" if value["sd"] is not None else " (n=1)")

    lines = [
        "# BRICS cold-molecule experiment results",
        "",
        f"Completed {len(rows)}/18 fixed runs. Mean ± sample SD across training seeds, not a confidence interval.",
        "",
        "All DTI scores are pooled over rows. Checkpoints selected by validation; test scores are not used for model selection.",
        "",
        "| Arm | Seeds | Val AP | Test AP | Test AUROC | Unique val JEPA MSE | ECFP all | Valid rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        n = len([r for r in rows if r["arm"] == arm])
        cells = [
            fmt(arm, k)
            for k in ("val_auprc", "test_auprc", "test_auroc", "unique_val_jepa_mse", "ecfp_all", "valid_rate")
        ]
        if arm.endswith("jepa"):
            cells[:3] = ["untrained head"] * 3
        lines.append(f"| {arm} | {n} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "JEPA-only binding heads are untrained. DTI controls retain alignment but do not train the JEPA head.",
        "",
        "## Paired-seed contrasts",
        "",
        "Negative MSE differences favor the first condition; positive AP differences favor the first condition.",
        "",
    ]
    for label, v in paired.items():
        lines.append(f"- {label}: {v['mean']:+.5f}, differences {v['differences']} (seeds {v['seeds']}).")
    lines += [
        "",
        "## Interpretation limits",
        "",
        "One canonical-molecule split and three seeds are exploratory evidence. Molecules, not scaffolds, are disjoint. Conflicting labels are excluded. Continuous bypass removes the 3-D bottleneck as well as rounding. Reconstruction uses 100 common validation molecules and greedy COATI; invalid decodes score zero in the all-sample ECFP mean. Exact matches use the largest component.",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    summary(args.root)
