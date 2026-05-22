from pathlib import Path
from typing import Dict, Sequence

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

MIN_SEQ_LEN = 50
MAX_SEQ_LEN = 2000


def _normalize_path(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _resolve_embedding_path(path_value, csv_path: str | Path, default_filename: str) -> Path:
    normalized = _normalize_path(path_value)
    if normalized:
        return Path(normalized)
    return Path(csv_path).resolve().parent / default_filename


def _coati_tensor_from_output(output: object) -> torch.Tensor | None:
    if output is None:
        return None
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            tensor = _coati_tensor_from_output(item)
            if tensor is not None:
                return tensor
        return None
    if isinstance(output, dict):
        for key in ("h_clip", "hclip", "embedding", "embeddings"):
            if key in output:
                tensor = _coati_tensor_from_output(output[key])
                if tensor is not None:
                    return tensor
    return None


def _encode_smiles_with_coati(encoder, tokenizer, smiles: str) -> torch.Tensor:
    # Prefer the stable COATI helper API when available.
    try:
        from coati.generative.coati_purifications import embed_smiles as coati_embed_smiles

        output = coati_embed_smiles(smiles, encoder, tokenizer)
        tensor = _coati_tensor_from_output(output)
        if tensor is not None:
            if tensor.ndim > 1:
                tensor = tensor[0]
            return tensor.detach().cpu().float().reshape(-1)
    except Exception:
        pass

    attempts = [
        lambda: encoder.smiles_to_hclip(smiles, tokenizer=tokenizer),
        lambda: encoder.smiles_to_hclip([smiles], tokenizer=tokenizer),
        lambda: encoder.smiles_to_hclip_batch([smiles], tokenizer=tokenizer),
        lambda: encoder.encode_smiles(smiles, tokenizer=tokenizer),
        lambda: encoder.encode_smiles([smiles], tokenizer=tokenizer),
        lambda: encoder.encode_smiles_batch([smiles], tokenizer=tokenizer),
    ]

    for attempt in attempts:
        try:
            output = attempt()
        except (AttributeError, TypeError, ValueError):
            continue
        tensor = _coati_tensor_from_output(output)
        if tensor is not None:
            if tensor.ndim > 1:
                tensor = tensor[0]
            return tensor.detach().cpu().float().reshape(-1)

    raise RuntimeError(
        f"Could not encode SMILES with COATI encoder. Unsupported encoder API for sample: {smiles!r}."
    )


def _collect_unique_sequences_and_smiles(
    csv_paths: list[str | Path], sequence_col: str, smiles_col: str
) -> tuple[list[str], list[str]]:
    sequences: set[str] = set()
    smiles: set[str] = set()

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, usecols=[sequence_col, smiles_col])
        seq_values = df[sequence_col].astype(str).str.strip()
        smi_values = df[smiles_col].astype(str).str.strip()
        sequences.update(v for v in seq_values if v)
        smiles.update(v for v in smi_values if v)

    return sorted(sequences), sorted(smiles)


def _build_morgan_embeddings(smiles_values: list[str], output_path: Path) -> None:
    from molfeat.trans.fp import FPVecTransformer

    transformer = FPVecTransformer(kind="ecfp:4", length=2048, verbose=True)
    valid_features, valid_ids = transformer(smiles_values, ignore_errors=True)

    embeddings: dict[str, torch.Tensor] = {}
    for feature, idx in zip(valid_features, valid_ids):
        embeddings[smiles_values[int(idx)]] = torch.tensor(feature, dtype=torch.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output_path)


def _build_protein_embeddings(sequences: list[str], output_path: Path, device: str) -> None:
    import esm

    valid_sequences = [seq for seq in sequences if MIN_SEQ_LEN <= len(seq) <= MAX_SEQ_LEN]
    dropped = len(sequences) - len(valid_sequences)
    if dropped > 0:
        print(
            f"[BindingDBDictDataModule] Skipping {dropped} sequences outside [{MIN_SEQ_LEN}, {MAX_SEQ_LEN}] aa "
            "for Raygun embeddings."
        )
    if not valid_sequences:
        raise ValueError(
            "No valid protein sequences found for embedding generation after length filtering "
            f"[{MIN_SEQ_LEN}, {MAX_SEQ_LEN}] aa."
        )

    esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = esm_alphabet.get_batch_converter()
    esm_model = esm_model.to(device)
    esm_model.eval()

    esm_embeddings: list[tuple[str, torch.Tensor]] = []
    with torch.no_grad():
        for seq in tqdm(valid_sequences, desc="Generating ESM embeddings"):
            _, _, tokens = batch_converter([("seq", seq.upper())])
            embedding = esm_model(tokens.to(device), repr_layers=[33], return_contacts=False)["representations"][33]
            esm_embeddings.append((seq, embedding[:, 1:-1, :].cpu()))

    del esm_model

    # Avoid interactive trust prompt in non-interactive/batch jobs.
    try:
        raymodel, _, _ = torch.hub.load(
            "rohitsinghlab/raygun",
            "pretrained_uniref50_95000_750M",
            trust_repo=True,
        )
    except TypeError:
        raymodel, _, _ = torch.hub.load("rohitsinghlab/raygun", "pretrained_uniref50_95000_750M")
    raymodel = raymodel.to(device)
    raymodel.eval()

    embeddings: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for seq, esm_embedding in tqdm(esm_embeddings, desc="Constructing Raygun embeddings"):
            rayencode = raymodel.encoder(esm_embedding.to(device)).squeeze().detach().cpu().float()
            embeddings[seq] = rayencode

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output_path)


def _build_smiles_embeddings(smiles_values: list[str], output_path: Path, device: str, doc_url: str) -> None:
    from coati.models.io import load_e3gnn_smiles_clip_e2e

    encoder, tokenizer = load_e3gnn_smiles_clip_e2e(
        freeze=True,
        device=device,
        doc_url=doc_url,
    )
    encoder.eval()

    embeddings: dict[str, torch.Tensor] = {}
    failure_count = 0
    first_error: str | None = None
    for smi in tqdm(smiles_values, desc="Generating COATI embeddings"):
        try:
            embeddings[smi] = _encode_smiles_with_coati(encoder, tokenizer, smi)
        except (RuntimeError, TypeError, ValueError) as exc:
            failure_count += 1
            if first_error is None:
                first_error = f"{type(exc).__name__}: {exc}"
            continue

    if not embeddings:
        raise RuntimeError(
            "COATI embedding generation produced 0 embeddings. "
            f"Failed SMILES: {failure_count}/{len(smiles_values)}. "
            f"First error: {first_error or 'unknown'}. "
            "Check COATI model/doc_url compatibility and runtime dependencies."
        )
    if failure_count > 0:
        print(
            "[BindingDBDictDataModule] COATI encoding skipped "
            f"{failure_count}/{len(smiles_values)} SMILES. "
            f"First error: {first_error or 'unknown'}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output_path)


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
        valid_sequences: set[str] | None = None,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.sequence_col = sequence_col
        self.smiles_col = smiles_col
        self.label_col = label_col
        self.valid_smiles = valid_smiles
        self.valid_sequences = valid_sequences

        self.df[self.smiles_col] = self.df[self.smiles_col].astype(str).str.strip()
        self.df[self.sequence_col] = self.df[self.sequence_col].astype(str).str.strip()

        if self.valid_smiles is not None or self.valid_sequences is not None:
            before = len(self.df)
            mask = pd.Series(True, index=self.df.index)
            if self.valid_smiles is not None:
                mask &= self.df[self.smiles_col].isin(self.valid_smiles)
            if self.valid_sequences is not None:
                mask &= self.df[self.sequence_col].isin(self.valid_sequences)
            self.df = self.df[mask].reset_index(drop=True)
            after = len(self.df)
            dropped = before - after
            if dropped > 0:
                print(f"[BindingDBDataset] Dropped {dropped} rows with missing embeddings from {csv_path}.")

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
        return protein, morgan, smiles, label, list(smiles_list), list(sequences)


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

        self.embedding_device = getattr(cfg, "embedding_device", "cuda:0" if torch.cuda.is_available() else "cpu")
        self.coati_doc_url = getattr(cfg, "coati_doc_url", "s3://terray-public/models/grande_closed.pkl")
        self.protein_embeddings_path = _resolve_embedding_path(
            path_value=getattr(cfg, "protein_embeddings_path", ""),
            csv_path=self.train_csv,
            default_filename="raygun_embeddings.pt",
        )
        self.morgan_embeddings_path = _resolve_embedding_path(
            path_value=getattr(cfg, "morgan_embeddings_path", ""),
            csv_path=self.train_csv,
            default_filename="morgan_embeddings.pt",
        )
        self.smiles_embeddings_path = _resolve_embedding_path(
            path_value=getattr(cfg, "smiles_embeddings_path", ""),
            csv_path=self.train_csv,
            default_filename="coati_embeddings.pt",
        )

        self._ensure_embedding_files_exist()

        self.protein_embeddings = torch.load(self.protein_embeddings_path, map_location="cpu")
        self.morgan_embeddings = torch.load(self.morgan_embeddings_path, map_location="cpu")
        self.smiles_embeddings = torch.load(self.smiles_embeddings_path, map_location="cpu")
        self.valid_smiles = set(self.morgan_embeddings.keys()) & set(self.smiles_embeddings.keys())
        self.valid_sequences = set(self.protein_embeddings.keys())
        if len(self.smiles_embeddings) == 0:
            raise RuntimeError(
                "Loaded 0 COATI embeddings from "
                f"{self.smiles_embeddings_path}. Remove this file and regenerate; "
                "check COATI encoder/doc_url/dependencies for runtime errors."
            )
        if len(self.valid_smiles) == 0:
            raise RuntimeError(
                "No overlapping SMILES keys between Morgan and COATI embeddings. "
                f"Morgan keys: {len(self.morgan_embeddings)}, COATI keys: {len(self.smiles_embeddings)}. "
                "Embedding files are mismatched or COATI generation failed."
            )
        self.collator = Collator(
            protein_embeddings=self.protein_embeddings,
            morgan_embeddings=self.morgan_embeddings,
            smiles_embeddings=self.smiles_embeddings,
        )

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _ensure_embedding_files_exist(self) -> None:
        sequence_values, smiles_values = _collect_unique_sequences_and_smiles(
            csv_paths=[self.train_csv, self.val_csv, self.test_csv],
            sequence_col=self.sequence_col,
            smiles_col=self.smiles_col,
        )

        if not self.protein_embeddings_path.exists():
            print(f"[BindingDBDictDataModule] Building protein embeddings at {self.protein_embeddings_path}")
            _build_protein_embeddings(
                sequences=sequence_values,
                output_path=self.protein_embeddings_path,
                device=self.embedding_device,
            )
        if not self.morgan_embeddings_path.exists():
            print(f"[BindingDBDictDataModule] Building Morgan embeddings at {self.morgan_embeddings_path}")
            _build_morgan_embeddings(
                smiles_values=smiles_values,
                output_path=self.morgan_embeddings_path,
            )
        if not self.smiles_embeddings_path.exists():
            print(f"[BindingDBDictDataModule] Building COATI embeddings at {self.smiles_embeddings_path}")
            _build_smiles_embeddings(
                smiles_values=smiles_values,
                output_path=self.smiles_embeddings_path,
                device=self.embedding_device,
                doc_url=self.coati_doc_url,
            )

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = BindingDBDataset(
                csv_path=self.train_csv,
                sequence_col=self.sequence_col,
                smiles_col=self.smiles_col,
                label_col=self.label_col,
                valid_smiles=self.valid_smiles,
                valid_sequences=self.valid_sequences,
            )
            self.val_dataset = BindingDBDataset(
                csv_path=self.val_csv,
                sequence_col=self.sequence_col,
                smiles_col=self.smiles_col,
                label_col=self.label_col,
                valid_smiles=self.valid_smiles,
                valid_sequences=self.valid_sequences,
            )
        if stage in (None, "test"):
            self.test_dataset = BindingDBDataset(
                csv_path=self.test_csv,
                sequence_col=self.sequence_col,
                smiles_col=self.smiles_col,
                label_col=self.label_col,
                valid_smiles=self.valid_smiles,
                valid_sequences=self.valid_sequences,
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
