#!/usr/bin/env python3
"""Export seed-level experiment comparisons and randomly selected reconstructions."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    data = json.loads((args.root / "suite_summary.json").read_text())
    rows = data["runs"]
    if not rows:
        return
    arms = [a for a in data["per_arm"] if any(r["arm"] == a for r in rows)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")
    axes = axes.flatten()
    for ax, key, title in zip(
        axes,
        ["val_auprc", "unique_val_jepa_mse", "ecfp_all", "retrieval_top10"],
        [
            "Binding: pooled validation AP",
            "Reconstruction: unique validation MSE",
            "Reconstruction: ECFP Tanimoto (all decodes)",
            "Latent identity: top-10 retrieval / 1,050 candidates",
        ],
    ):
        labels = []
        for arm in arms:
            values = [
                r[key]
                for r in rows
                if r["arm"] == arm and key in r and not (key == "val_auprc" and arm.endswith("jepa"))
            ]
            if not values:
                continue
            x = len(labels)
            labels.append(arm)
            ax.scatter(
                [x + (i - (len(values) - 1) / 2) * 0.1 for i in range(len(values))],
                values,
                s=45,
                color=f"C{arms.index(arm)}",
            )
            ax.plot([x - 0.22, x + 0.22], [sum(values) / len(values)] * 2, color="black", linewidth=2)
        ax.set_xticks(range(len(labels)), labels, rotation=50, ha="right")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylabel(key)
    baseline = args.root / "shortcut_baselines.json"
    if baseline.exists():
        b = json.loads(baseline.read_text())
        axes[0].axhline(b["splits"]["val"]["protein_only_prior"]["auprc"], ls="--", color="red", label="Protein prior")
        axes[0].legend()
    baseline_mse = next((r["mean_baseline_mse"] for r in rows if "mean_baseline_mse" in r), None)
    if baseline_mse is not None:
        axes[1].axhline(baseline_mse, ls="--", color="red", label="Training mean target")
        axes[1].legend()
    recon_path = next(iter(sorted(args.root.glob("*/seed_42/reconstruction_validation.json"))), None)
    if recon_path is not None:
        recon = json.loads(recon_path.read_text())
        axes[2].axhline(
            recon["target_latent_control"]["ecfp"]["all"]["mean"],
            ls="--",
            color="red",
            label="True COATI latent control",
        )
        axes[2].legend()
        axes[3].axhline(10 / recon["retrieval_catalog_n"], ls="--", color="red", label="Random rank baseline")
        axes[3].legend()
    scope = (
        "Matched count inputs (JEPA-only controls reused)" if "study" in data else "Legacy mixed count/binary inputs"
    )
    fig.suptitle(
        f"{scope}; {data['completed_runs']}/18 conditions complete\nFixed cold-molecule split; each dot is a training seed"
    )
    fig.savefig(args.root / "comparison.png", dpi=180)
    fig.savefig(args.root / "comparison.pdf")
    plt.close(fig)
    # Same first six sampled molecules, without choosing attractive successes.
    for path in sorted(args.root.glob("*/seed_42/reconstruction_validation.json")):
        recon = json.loads(path.read_text())
        molecules = []
        legends = []
        for i, row in enumerate(recon["examples"][:6]):
            molecules.extend(
                [
                    Chem.MolFromSmiles(row["input_smiles"]),
                    Chem.MolFromSmiles(row["decoded"]) if row["decoded"] else None,
                ]
            )
            legends.extend(
                [
                    f"Sample {i + 1}: original",
                    f"Predicted | ECFP {row['ecfp']:.3f}" if row["decoded"] else "Invalid decode",
                ]
            )
        grid = Draw.MolsToGridImage(molecules, molsPerRow=2, subImgSize=(500, 300), legends=legends, useSVG=True)
        (path.parent / "random_reconstructions.svg").write_text(grid)


if __name__ == "__main__":
    main()
