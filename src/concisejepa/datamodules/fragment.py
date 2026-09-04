"""Hydra-compatible BRICS fragment data module."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from spikes.phase1.fragment_datamodule import (
    MAX_FRAGS,
    FragmentCollator,
    FragmentDataModule as _SpikeFragmentDataModule,
    brics_fragment_mols,
    build_fragment_fp_cache,
    compute_fragment_fps,
)

from .dataloader import _validate_fingerprint_metadata


def validate_fragment_fingerprint_metadata(
    fragment_fps_path: str | Path,
    fingerprint_kind: str,
    fingerprint_length: int,
    max_fragments: int,
) -> None:
    """Reject caches whose fragment fingerprint contract does not match the model."""
    cache_path = Path(fragment_fps_path)
    metadata_path = Path(f"{cache_path}.meta.json")
    if not metadata_path.exists():
        raise RuntimeError(
            f"Fragment fingerprint metadata is missing for {cache_path}. Expected {metadata_path}."
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read fragment metadata from {metadata_path}: {exc}") from exc

    expected = {
        "type": "fragment_fps",
        "fingerprint_kind": fingerprint_kind,
        "fingerprint_length": fingerprint_length,
        "max_frags": max_fragments,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: cache={actual!r}, requested={requested!r}"
            for key, (actual, requested) in mismatches.items()
        )
        raise RuntimeError(f"Fragment fingerprint metadata mismatch for {cache_path}: {details}.")


class FragmentDataModule(_SpikeFragmentDataModule):
    """Explicit-keyword adapter around the reference fragment data module."""

    def __init__(
        self,
        train_csv: str,
        val_csv: str,
        test_csv: str,
        protein_embeddings_path: str,
        fragment_fps_path: str,
        smiles_embeddings_path: str,
        morgan_embeddings_path: str | None = None,
        fingerprint_kind: str = "ecfp-count:4",
        fingerprint_length: int = 2048,
        max_fragments: int = MAX_FRAGS,
        sequence_col: str = "Target Sequence",
        smiles_col: str = "SMILES",
        label_col: str = "Label",
        batch_size: int = 256,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        use_whole_mol: bool = False,
        validate_metadata: bool = True,
        data_root: str | None = None,
    ) -> None:
        # ``data_root`` is retained in the resolved config for provenance; all
        # concrete paths have already been interpolated by Hydra.
        del data_root
        if max_fragments != MAX_FRAGS:
            raise ValueError(
                f"The BRICS preprocessor currently fixes max_fragments={MAX_FRAGS}; got {max_fragments}."
            )
        if validate_metadata and not use_whole_mol:
            validate_fragment_fingerprint_metadata(
                fragment_fps_path,
                fingerprint_kind,
                fingerprint_length,
                max_fragments,
            )

        cfg = SimpleNamespace(
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            protein_embeddings_path=protein_embeddings_path,
            fragment_fps_path=fragment_fps_path,
            smiles_embeddings_path=smiles_embeddings_path,
            morgan_embeddings_path=morgan_embeddings_path,
            sequence_col=sequence_col,
            smiles_col=smiles_col,
            label_col=label_col,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            use_whole_mol=use_whole_mol,
        )
        super().__init__(cfg)


class DualViewFragmentCollator(FragmentCollator):
    """Return both BRICS fragments and the original molecule fingerprint."""

    def __call__(self, batch):
        protein, fragment_fps, fragment_mask, smiles_embedding, label, smiles, sequences = (
            super().__call__(batch)
        )
        if self.fallback_morgan is None:
            raise RuntimeError("Dual-view collation requires whole-molecule Morgan fingerprints.")
        whole_molecule_fingerprint = torch.stack(
            [self.fallback_morgan[smiles_value] for smiles_value in smiles], dim=0
        )
        return (
            protein,
            fragment_fps,
            fragment_mask,
            whole_molecule_fingerprint,
            smiles_embedding,
            label,
            smiles,
            sequences,
        )


class DualViewFragmentDataModule(FragmentDataModule):
    """BRICS data module exposing fragment and whole-molecule views together."""

    def __init__(
        self,
        train_csv: str,
        val_csv: str,
        test_csv: str,
        protein_embeddings_path: str,
        fragment_fps_path: str,
        smiles_embeddings_path: str,
        morgan_embeddings_path: str,
        fingerprint_kind: str = "ecfp-count:4",
        fingerprint_length: int = 2048,
        max_fragments: int = MAX_FRAGS,
        sequence_col: str = "Target Sequence",
        smiles_col: str = "SMILES",
        label_col: str = "Label",
        batch_size: int = 256,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        validate_metadata: bool = True,
        data_root: str | None = None,
        require_count_values: bool = False,
    ) -> None:
        if validate_metadata:
            _validate_fingerprint_metadata(
                Path(morgan_embeddings_path), fingerprint_kind, fingerprint_length
            )
        super().__init__(
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            protein_embeddings_path=protein_embeddings_path,
            fragment_fps_path=fragment_fps_path,
            smiles_embeddings_path=smiles_embeddings_path,
            morgan_embeddings_path=morgan_embeddings_path,
            fingerprint_kind=fingerprint_kind,
            fingerprint_length=fingerprint_length,
            max_fragments=max_fragments,
            sequence_col=sequence_col,
            smiles_col=smiles_col,
            label_col=label_col,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            use_whole_mol=False,
            validate_metadata=validate_metadata,
            data_root=data_root,
        )
        if self.morgan_embeddings is None:
            raise RuntimeError("Dual-view training requires whole-molecule Morgan fingerprints.")
        if require_count_values:
            validate_count_fingerprint_values(self.morgan_embeddings)
        self.valid_smiles &= set(self.morgan_embeddings)
        self.collator = DualViewFragmentCollator(
            protein_embeddings=self.protein_embeddings,
            fragment_fps=self.fragment_fps,
            smiles_embeddings=self.smiles_embeddings,
            fallback_morgan=self.morgan_embeddings,
        )


def validate_count_fingerprint_values(cache: dict[str, torch.Tensor]) -> None:
    """Catch binary caches mislabeled as counts, without changing legacy defaults."""
    values = torch.stack([v.reshape(-1) for v in cache.values()])
    if not bool(torch.isfinite(values).all() and (values >= 0).all() and (values == values.round()).all()):
        raise ValueError("Expected finite, nonnegative integer count fingerprints")
    if not bool((values > 1).any()):
        raise ValueError("Count fingerprints requested, but every cached value is binary; verify cache provenance")


__all__ = [
    "FragmentCollator",
    "FragmentDataModule",
    "DualViewFragmentCollator",
    "DualViewFragmentDataModule",
    "brics_fragment_mols",
    "build_fragment_fp_cache",
    "compute_fragment_fps",
    "validate_fragment_fingerprint_metadata",
]
