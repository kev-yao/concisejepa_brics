# phase1_fragment_datamodule.py — Fragment FP precompute + collator + data module.
#
# Precompute: for each unique SMILES in the dataset, BRICS-decompose → Morgan FP per
# fragment.  Outputs fragment_fps_r4_2048.pt: dict[smiles → Tensor[n_frags, 2048]].
#
# FragmentCollator: loads the fragment FP cache; pads variable-length fragment lists to
# max_frags-in-batch; produces fragment_mask (True = valid).

import re
import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

MAX_FRAGS = 16  # cap per molecule; keep largest fragments (by heavy-atom count)


# ---------------------------------------------------------------------------
# BRICS fragmentation + Morgan FP helpers
# ---------------------------------------------------------------------------

def _sanitize_brics_mol(brics_smiles: str):
    """Convert BRICS fragment SMILES (with [n*] attachment points) to RDKit Mol."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(brics_smiles)
    if mol is None:
        # Try stripping numeric attachment labels, keeping bare [*]
        cleaned = re.sub(r'\[(\d+)\*\]', '[*]', brics_smiles)
        mol = Chem.MolFromSmiles(cleaned)
    return mol


def brics_fragment_mols(smiles: str):
    """BRICS decompose a SMILES string. Returns list of RDKit Mol objects, largest first."""
    from rdkit import Chem
    from rdkit.Chem import BRICS

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    frag_smiles_set = BRICS.BRICSDecompose(mol)
    if not frag_smiles_set:
        return [mol]  # fallback: whole molecule

    frag_mols = []
    for fsmi in frag_smiles_set:
        fmol = _sanitize_brics_mol(fsmi)
        if fmol is not None:
            frag_mols.append(fmol)

    if not frag_mols:
        return [mol]  # fallback

    # Sort largest-first (most informative), cap at MAX_FRAGS
    frag_mols.sort(key=lambda m: m.GetNumHeavyAtoms(), reverse=True)
    return frag_mols[:MAX_FRAGS]


def compute_fragment_fps(smiles: str, transformer) -> torch.Tensor:
    """
    Returns Tensor[n_frags, 2048] of count-based Morgan FPs for BRICS fragments.
    Falls back to whole-molecule FP if fragmentation fails.
    """
    from rdkit.Chem import MolToSmiles

    frag_mols = brics_fragment_mols(smiles)
    if not frag_mols:
        # Complete failure — return zero vector (filtered downstream)
        return torch.zeros(1, 2048, dtype=torch.float32)

    frag_smiles = [MolToSmiles(m) for m in frag_mols]

    # Use molfeat transformer (same params as whole-mol FP in training data)
    fps_array, valid_ids = transformer(frag_smiles, ignore_errors=True)
    if len(fps_array) == 0:
        return torch.zeros(1, 2048, dtype=torch.float32)

    return torch.tensor(np.array(fps_array), dtype=torch.float32)


def build_fragment_fp_cache(
    smiles_values: list[str],
    output_path: Path,
    fingerprint_kind: str = "ecfp-count:4",
    fingerprint_length: int = 2048,
) -> None:
    """
    Precompute fragment FPs for all unique SMILES and save to output_path.
    Output: dict[smiles → Tensor[n_frags, fingerprint_length]]
    """
    from molfeat.trans.fp import FPVecTransformer

    transformer = FPVecTransformer(kind=fingerprint_kind, length=fingerprint_length, verbose=False)
    cache: dict[str, torch.Tensor] = {}

    for smi in tqdm(smiles_values, desc="Precomputing fragment FPs"):
        try:
            fps = compute_fragment_fps(smi, transformer)
            cache[smi] = fps
        except Exception as e:
            print(f"[warn] Fragment FP failed for {smi!r}: {e}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output_path)
    meta = {
        "type": "fragment_fps",
        "fingerprint_kind": fingerprint_kind,
        "fingerprint_length": fingerprint_length,
        "max_frags": MAX_FRAGS,
        "n_smiles": len(cache),
    }
    Path(f"{output_path}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {len(cache)} fragment FP entries to {output_path}")


# ---------------------------------------------------------------------------
# Fragment collator
# ---------------------------------------------------------------------------

class FragmentCollator:
    """
    Collate function that looks up fragment FPs and pads to batch-max_frags.

    Returns:
        protein:    [B, 50, 1280]
        frag_fps:   [B, max_frags, 2048]
        frag_mask:  [B, max_frags] bool (True = valid fragment)
        smiles_emb: [B, 256]
        label:      [B]
        smiles_list: list[str]
        sequences:  list[str]
    """

    def __init__(
        self,
        protein_embeddings: Dict[str, torch.Tensor],
        fragment_fps: Dict[str, torch.Tensor],
        smiles_embeddings: Dict[str, torch.Tensor],
        fallback_morgan: Dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.protein_embeddings = protein_embeddings
        self.fragment_fps = fragment_fps
        self.smiles_embeddings = smiles_embeddings
        self.fallback_morgan = fallback_morgan  # whole-mol FP if fragment lookup fails

    def __call__(self, batch: Sequence[tuple[str, str, float]]):
        sequences, smiles_list, labels = zip(*batch)

        protein = torch.stack([self.protein_embeddings[seq] for seq in sequences], dim=0)
        smiles_emb = torch.stack([self.smiles_embeddings[smi] for smi in smiles_list], dim=0)
        smiles_emb = smiles_emb.reshape(smiles_emb.shape[0], -1)
        label = torch.tensor(labels, dtype=torch.float32)

        # Look up fragment FPs; fall back to whole-mol FP if missing
        frag_list: list[torch.Tensor] = []
        for smi in smiles_list:
            if smi in self.fragment_fps:
                frag_list.append(self.fragment_fps[smi])
            elif self.fallback_morgan and smi in self.fallback_morgan:
                frag_list.append(self.fallback_morgan[smi].unsqueeze(0))  # [1, 2048]
            else:
                fp_dim = next(iter(self.fragment_fps.values())).shape[-1]
                frag_list.append(torch.zeros(1, fp_dim))

        # Pad to max_frags in batch
        max_frags = max(fp.shape[0] for fp in frag_list)
        fp_dim = frag_list[0].shape[-1]
        frag_fps_tensor = torch.zeros(len(frag_list), max_frags, fp_dim)
        frag_mask = torch.zeros(len(frag_list), max_frags, dtype=torch.bool)
        for i, fp in enumerate(frag_list):
            n = fp.shape[0]
            frag_fps_tensor[i, :n] = fp
            frag_mask[i, :n] = True

        return protein, frag_fps_tensor, frag_mask, smiles_emb, label, list(smiles_list), list(sequences)


# ---------------------------------------------------------------------------
# Data module
# ---------------------------------------------------------------------------

class FragmentDataModule(pl.LightningDataModule):
    """
    DataModule for fragment-level FSQ experiments.

    Expects pre-built embedding caches:
      - protein_embeddings_path:  dict[sequence → Tensor[50, 1280]]
      - fragment_fps_path:        dict[smiles → Tensor[n_frags, 2048]]
      - smiles_embeddings_path:   dict[smiles → Tensor[256]]
      - morgan_embeddings_path:   dict[smiles → Tensor[2048]]  (fallback only)
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.train_csv = cfg.train_csv
        self.val_csv = cfg.val_csv
        self.test_csv = cfg.test_csv
        self.sequence_col = getattr(cfg, "sequence_col", "Target Sequence")
        self.smiles_col = getattr(cfg, "smiles_col", "SMILES")
        self.label_col = getattr(cfg, "label_col", "Label")
        self.batch_size = cfg.batch_size
        self.num_workers = getattr(cfg, "num_workers", 0)
        self.pin_memory = getattr(cfg, "pin_memory", False)
        self.persistent_workers = getattr(cfg, "persistent_workers", False) and self.num_workers > 0

        self.protein_embeddings_path = Path(cfg.protein_embeddings_path)
        self.fragment_fps_path = Path(cfg.fragment_fps_path)
        self.smiles_embeddings_path = Path(cfg.smiles_embeddings_path)
        morgan_path = getattr(cfg, "morgan_embeddings_path", None)
        self.morgan_embeddings_path = Path(morgan_path) if morgan_path else None

        self._load_embeddings()

    def _load_embeddings(self) -> None:
        from concisejepa.datamodules.dataloader import BindingDBDataset

        print(f"Loading protein embeddings from {self.protein_embeddings_path}")
        self.protein_embeddings = torch.load(self.protein_embeddings_path, map_location="cpu")
        print(f"Loading fragment FPs from {self.fragment_fps_path}")
        self.fragment_fps = torch.load(self.fragment_fps_path, map_location="cpu")
        print(f"Loading COATI embeddings from {self.smiles_embeddings_path}")
        self.smiles_embeddings = torch.load(self.smiles_embeddings_path, map_location="cpu")

        self.morgan_embeddings: dict | None = None
        if self.morgan_embeddings_path and self.morgan_embeddings_path.exists():
            self.morgan_embeddings = torch.load(self.morgan_embeddings_path, map_location="cpu")

        self.valid_smiles = set(self.fragment_fps.keys()) & set(self.smiles_embeddings.keys())
        self.valid_sequences = set(self.protein_embeddings.keys())
        print(f"Valid SMILES: {len(self.valid_smiles)}, Valid sequences: {len(self.valid_sequences)}")

        self._BindingDBDataset = BindingDBDataset
        self.collator = FragmentCollator(
            protein_embeddings=self.protein_embeddings,
            fragment_fps=self.fragment_fps,
            smiles_embeddings=self.smiles_embeddings,
            fallback_morgan=self.morgan_embeddings,
        )

    def _make_dataset(self, csv_path: str):
        return self._BindingDBDataset(
            csv_path=csv_path,
            sequence_col=self.sequence_col,
            smiles_col=self.smiles_col,
            label_col=self.label_col,
            valid_smiles=self.valid_smiles,
            valid_sequences=self.valid_sequences,
        )

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = self._make_dataset(self.train_csv)
            self.val_dataset = self._make_dataset(self.val_csv)
        if stage in (None, "test"):
            self.test_dataset = self._make_dataset(self.test_csv)

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self.collator,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False)
