"""
Chemical codebook interpretability metrics — collected at validation epoch end.

Metrics logged:

  Structural similarity
  ─────────────────────
  val/chem_intra_code_tanimoto   — mean Morgan Tanimoto within same-code buckets
  val/chem_same_vs_cross_ratio   — intra / cross Tanimoto ratio (>1 = codes cluster
                                   similar molecules better than chance)
  val/chem_occupied_codes        — number of occupied FSQ slots (≥ min_code_size mols)

  Chemical homogeneity
  ─────────────────────
  val/chem_prop_entropy_{name}   — mean Shannon entropy of property-bin distribution
                                   within each occupied code (lower = more homogeneous)
  val/chem_scaffold_purity       — fraction of majority Murcko scaffold per code,
                                   averaged across occupied codes (higher = more
                                   scaffold-specific bucketing)

  Linear probe: how much chemical info lives in the code integers
  ───────────────────────────────────────────────────────────────
  val/probe_{prop}_acc           — balanced accuracy of LogisticRegression trained on
                                   one-hot(f0, f1, f2) → property bin (per property)
  val/probe_mean_acc             — mean across properties (headline interpretability score)
  val/probe_factor{i}_mean_acc   — per-factor mean probe accuracy (which FSQ scalar
                                   carries the most chemical information)
"""

from collections import Counter, defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def _morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def _mean_tanimoto(fps: list) -> float:
    n = len(fps)
    if n < 2:
        return 1.0 if n == 1 else 0.0
    total, count = 0.0, 0
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        total += sum(sims)
        count += len(sims)
    return total / count if count > 0 else 0.0


class ChemCodebookEvalCallback(pl.Callback):
    """
    Accumulates (code_tuple, smiles) pairs during validation and computes
    chemical interpretability metrics at epoch end.

    Args:
        max_pairs_per_code: cap on pairwise Tanimoto comparisons per bucket
            to bound wall-time; larger buckets are randomly sub-sampled.
        min_code_size:      buckets smaller than this are skipped for
                            Tanimoto and scaffold purity (too noisy).
        morgan_radius:      Morgan fingerprint radius.
        morgan_n_bits:      Morgan fingerprint bit length.
    """

    def __init__(
        self,
        max_pairs_per_code: int = 200,
        min_code_size: int = 3,
        morgan_radius: int = 2,
        morgan_n_bits: int = 2048,
    ) -> None:
        super().__init__()
        self.max_pairs_per_code = max_pairs_per_code
        self.min_code_size = min_code_size
        self.morgan_radius = morgan_radius
        self.morgan_n_bits = morgan_n_bits

        self._code_to_smiles: dict[tuple, list[str]] = defaultdict(list)
        self._fp_cache: dict[str, object] = {}
        # Persistent property-bin cache across epochs (numpy arrays)
        self._prop_bins_cache: dict[str, np.ndarray] = {}

    def _reset(self) -> None:
        self._code_to_smiles = defaultdict(list)

    def _get_fp(self, smiles: str):
        if smiles not in self._fp_cache:
            self._fp_cache[smiles] = _morgan_fp(smiles, self.morgan_radius, self.morgan_n_bits)
        return self._fp_cache[smiles]

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self._reset()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        if len(batch) <= 4:
            return
        smiles_list: list[str] = batch[4]

        with torch.no_grad():
            morgan = batch[1].to(pl_module.device)
            prot = batch[0].to(pl_module.device)
            fwd = pl_module.model(protein_embedding=prot, morgan_fingerprint=morgan)
            codes = fwd["codes"].detach().cpu()  # [B, D]

        for smi, code in zip(smiles_list, codes.tolist()):
            self._code_to_smiles[tuple(code)].append(smi)

    # ------------------------------------------------------------------
    # Epoch-end: compute all interpretability metrics
    # ------------------------------------------------------------------

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return

        occupied_codes = {
            code: smiles
            for code, smiles in self._code_to_smiles.items()
            if len(smiles) >= self.min_code_size
        }

        if not occupied_codes:
            return

        # ── Intra-code Tanimoto ──────────────────────────────────────
        intra_tanimotos: list[float] = []
        all_fps_flat: list = []

        for code, smiles in occupied_codes.items():
            fps = [self._get_fp(s) for s in smiles]
            fps = [fp for fp in fps if fp is not None]
            all_fps_flat.extend(fps)

            if len(fps) < 2:
                continue
            if len(fps) > self.max_pairs_per_code:
                rng = np.random.default_rng(seed=0)
                idxs = rng.choice(len(fps), self.max_pairs_per_code, replace=False)
                fps = [fps[i] for i in idxs]
            intra_tanimotos.append(_mean_tanimoto(fps))

        mean_intra = float(np.mean(intra_tanimotos)) if intra_tanimotos else 0.0

        rng = np.random.default_rng(seed=42)
        max_cross = min(len(all_fps_flat), 500)
        if max_cross >= 2:
            idxs = rng.choice(len(all_fps_flat), max_cross, replace=False)
            mean_cross = _mean_tanimoto([all_fps_flat[i] for i in idxs])
        else:
            mean_cross = 0.0
        ratio = mean_intra / max(mean_cross, 1e-8)

        # ── Property entropy ─────────────────────────────────────────
        try:
            from concisejepa.models.chem_property_head import (
                PROPERTY_SPECS,
                compute_property_bins,
            )
            prop_names = [name for name, _ in PROPERTY_SPECS]
            n_bins_list = [len(edges) - 1 for _, edges in PROPERTY_SPECS]
            prop_entropies: dict[str, list[float]] = {name: [] for name in prop_names}

            for smiles in occupied_codes.values():
                bin_counts = [np.zeros(n_bins, dtype=np.float64) for n_bins in n_bins_list]
                for smi in smiles:
                    bins = compute_property_bins(smi)
                    for p_idx in range(len(PROPERTY_SPECS)):
                        b = int(bins[p_idx].item())
                        if b >= 0:
                            bin_counts[p_idx][b] += 1.0
                for p_idx, name in enumerate(prop_names):
                    counts = bin_counts[p_idx]
                    total = counts.sum()
                    if total < 1:
                        continue
                    probs = counts / total
                    probs = probs[probs > 0]
                    prop_entropies[name].append(float(-(probs * np.log(probs)).sum()))

            for name in prop_names:
                if prop_entropies[name]:
                    pl_module.log(
                        f"val/chem_prop_entropy_{name}",
                        float(np.mean(prop_entropies[name])),
                        on_epoch=True,
                    )
        except Exception:
            pass

        # ── Scaffold purity ──────────────────────────────────────────
        # For each occupied code bucket, what fraction of molecules share
        # the most common Murcko scaffold? Higher = more scaffold-specific codes.
        try:
            from rdkit.Chem.Scaffolds import MurckoScaffold

            scaffold_purities: list[float] = []
            for smiles in occupied_codes.values():
                scaffolds: list[str] = []
                for smi in smiles:
                    try:
                        mol = Chem.MolFromSmiles(smi)
                        if mol is not None:
                            scaf = MurckoScaffold.GetScaffoldForMol(mol)
                            scaffolds.append(Chem.MolToSmiles(scaf))
                    except Exception:
                        pass
                if len(scaffolds) < 2:
                    continue
                most_common = Counter(scaffolds).most_common(1)[0][1]
                scaffold_purities.append(most_common / len(scaffolds))

            if scaffold_purities:
                pl_module.log(
                    "val/chem_scaffold_purity",
                    float(np.mean(scaffold_purities)),
                    on_epoch=True,
                )
        except Exception:
            pass

        # ── Linear probe: code integers → chemical property ──────────
        # Fits a LogisticRegression on one-hot(f0, f1, f2) to predict
        # property bins. Measures how much chemical information the discrete
        # code integers alone carry, independent of the continuous embedding.
        # Baseline (random codes): balanced_accuracy ≈ 1 / n_bins per property.
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score
            from concisejepa.models.chem_property_head import PROPERTY_SPECS, compute_property_bins

            all_codes: list[tuple] = []
            all_smiles_flat: list[str] = []
            for code_tuple, smiles_list in self._code_to_smiles.items():
                for smi in smiles_list:
                    all_codes.append(code_tuple)
                    all_smiles_flat.append(smi)

            n_samples = len(all_codes)
            if n_samples >= 30:
                n_factors = len(all_codes[0])   # 3 for [32,32,32]
                factor_size = 32
                n_features = n_factors * factor_size  # 96

                X = np.zeros((n_samples, n_features), dtype=np.float32)
                for i, code_tuple in enumerate(all_codes):
                    for fi, fval in enumerate(code_tuple):
                        X[i, fi * factor_size + int(fval) % factor_size] = 1.0

                # Property bins — lazily cached as numpy arrays
                for smi in all_smiles_flat:
                    if smi not in self._prop_bins_cache:
                        self._prop_bins_cache[smi] = compute_property_bins(smi).numpy()
                bin_matrix = np.array(
                    [self._prop_bins_cache[s] for s in all_smiles_flat], dtype=np.int64
                )  # [N, n_props]

                prop_names_probe = [name for name, _ in PROPERTY_SPECS]
                n_props = bin_matrix.shape[1]

                # Full probe: all 3 factors → each property
                all_probe_scores: list[float] = []
                for p_idx, prop_name in enumerate(prop_names_probe):
                    y = bin_matrix[:, p_idx]
                    valid = y >= 0
                    if valid.sum() < 30:
                        continue
                    try:
                        clf = LogisticRegression(
                            max_iter=300, C=1.0, solver="lbfgs", multi_class="auto"
                        )
                        scores = cross_val_score(
                            clf, X[valid], y[valid], cv=3, scoring="balanced_accuracy"
                        )
                        score = float(np.mean(scores))
                        pl_module.log(f"val/probe_{prop_name}_acc", score, on_epoch=True)
                        all_probe_scores.append(score)
                    except Exception:
                        pass

                if all_probe_scores:
                    pl_module.log(
                        "val/probe_mean_acc",
                        float(np.mean(all_probe_scores)),
                        on_epoch=True,
                    )

                # Per-factor probe: factor_i alone → property bins (avg over props)
                # Reveals which FSQ scalar carries the most chemical information.
                for fi in range(n_factors):
                    X_fi = X[:, fi * factor_size : (fi + 1) * factor_size]
                    fi_scores: list[float] = []
                    for p_idx in range(n_props):
                        y = bin_matrix[:, p_idx]
                        valid = y >= 0
                        if valid.sum() < 30:
                            continue
                        try:
                            clf = LogisticRegression(
                                max_iter=300, C=1.0, solver="lbfgs", multi_class="auto"
                            )
                            scores = cross_val_score(
                                clf, X_fi[valid], y[valid], cv=3, scoring="balanced_accuracy"
                            )
                            fi_scores.append(float(np.mean(scores)))
                        except Exception:
                            pass
                    if fi_scores:
                        pl_module.log(
                            f"val/probe_factor{fi}_mean_acc",
                            float(np.mean(fi_scores)),
                            on_epoch=True,
                        )
        except Exception:
            pass

        # ── Scalar logs (always) ─────────────────────────────────────
        pl_module.log("val/chem_intra_code_tanimoto", mean_intra, on_epoch=True)
        pl_module.log("val/chem_same_vs_cross_ratio", ratio, on_epoch=True)
        pl_module.log("val/chem_occupied_codes", float(len(occupied_codes)), on_epoch=True)

        print(
            f"\n[ChemCodebook] occupied={len(occupied_codes)} | "
            f"intra-Tan={mean_intra:.4f} | cross-Tan={mean_cross:.4f} | "
            f"ratio={ratio:.3f}"
        )
