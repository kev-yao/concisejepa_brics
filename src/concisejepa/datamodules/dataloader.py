from pathlib import Path
from typing import Dict, Sequence

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class BindingDBDataset(Dataset):
    """
    Reads CSV rows and returns (sequence, smiles, label).
    Embedding lookup is handled in the collate_fn.
    """

    def __init__(
        self,
        csv_path: str | Path,
        sequence_col: str = "Target Sequence",
        smiles_col: str = "SMILES",
        label_col: str = "Label",
        valid_smiles: set[str] | None = None,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.sequence_col = sequence_col
        self.smiles_col = smiles_col
        self.label_col = label_col
        self.valid_smiles = valid_smiles

        self.df[self.smiles_col] = self.df[self.smiles_col].astype(str).str.strip()
        self.df[self.sequence_col] = self.df[self.sequence_col].astype(str).str.strip()

        if self.valid_smiles is not None:
            before = len(self.df)
            self.df = self.df[self.df[self.smiles_col].isin(self.valid_smiles)].reset_index(drop=True)
            after = len(self.df)
            dropped = before - after
            if dropped > 0:
                print(f"[BindingDBDataset] Dropped {dropped} rows with missing SMILES embeddings from {csv_path}.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        return (
            str(row[self.sequence_col]).strip(),
            str(row[self.smiles_col]).strip(),
            float(row[self.label_col]),
        )


class Collator:
    """
    Loads all embedding dictionaries in memory and performs key lookup per batch.

    Keys:
      - protein embeddings: sequence string
      - morgan embeddings: smiles string
      - smiles embeddings: smiles string
    """

    def __init__(
        self,
        protein_embeddings: Dict[str, torch.Tensor],
        morgan_embeddings: Dict[str, torch.Tensor],
        smiles_embeddings: Dict[str, torch.Tensor],
    ) -> None:
        self.protein_embeddings = protein_embeddings
        self.morgan_embeddings = morgan_embeddings
        self.smiles_embeddings = smiles_embeddings

    def __call__(self, batch: Sequence[tuple[str, str, float]]):
        sequences, smiles_list, labels = zip(*batch)

        protein = torch.stack([self.protein_embeddings[seq] for seq in sequences], dim=0)
        morgan = torch.stack([self.morgan_embeddings[smi] for smi in smiles_list], dim=0)
        smiles = torch.stack([self.smiles_embeddings[smi] for smi in smiles_list], dim=0)
        smiles = smiles.reshape(smiles.shape[0], -1)
        label = torch.tensor(labels, dtype=torch.float32)
        return protein, morgan, smiles, label, list(smiles_list)


class BindingDBDictDataModule(pl.LightningDataModule):
    """
    Dictionary-backed DataModule.

    Dictionary files are loaded once into memory with torch.load(...):
      - protein_embeddings_path: dict[sequence] -> protein embedding tensor
      - morgan_embeddings_path: dict[smiles] -> morgan fingerprint tensor
      - smiles_embeddings_path: dict[smiles] -> smiles target embedding tensor
    """

    def __init__(
        self,
        cfg,
    ) -> None:
        super().__init__()
        self.train_csv = cfg.train_csv
        self.val_csv = cfg.val_csv
        self.test_csv = cfg.test_csv
        self.sequence_col = cfg.sequence_col
        self.smiles_col = cfg.smiles_col
        self.label_col = cfg.label_col
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.pin_memory = cfg.pin_memory
        self.persistent_workers = cfg.persistent_workers and self.num_workers > 0

        self.protein_embeddings = torch.load(cfg.protein_embeddings_path, map_location="cpu")
        self.morgan_embeddings = torch.load(cfg.morgan_embeddings_path, map_location="cpu")
        self.smiles_embeddings = torch.load(cfg.smiles_embeddings_path, map_location="cpu")
        self.valid_smiles = set(self.morgan_embeddings.keys()) & set(self.smiles_embeddings.keys())
        self.collator = Collator(
            protein_embeddings=self.protein_embeddings,
            morgan_embeddings=self.morgan_embeddings,
            smiles_embeddings=self.smiles_embeddings,
        )

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = BindingDBDataset(
                csv_path=self.train_csv,
                sequence_col=self.sequence_col,
                smiles_col=self.smiles_col,
                label_col=self.label_col,
                valid_smiles=self.valid_smiles,
            )
            self.val_dataset = BindingDBDataset(
                csv_path=self.val_csv,
                sequence_col=self.sequence_col,
                smiles_col=self.smiles_col,
                label_col=self.label_col,
                valid_smiles=self.valid_smiles,
            )
        if stage in (None, "test"):
            self.test_dataset = BindingDBDataset(
                csv_path=self.test_csv,
                sequence_col=self.sequence_col,
                smiles_col=self.smiles_col,
                label_col=self.label_col,
                valid_smiles=self.valid_smiles,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self.collator,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self.collator,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self.collator,
        )
