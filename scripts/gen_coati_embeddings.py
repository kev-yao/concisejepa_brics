from pathlib import Path

import hydra
from omegaconf import DictConfig

from concisejepa.datamodules.dataloader import (
    _build_smiles_embeddings,
    _collect_unique_sequences_and_smiles,
    _resolve_embedding_path,
)


@hydra.main(version_base=None, config_path="../configs", config_name="datamodule")
def main(cfg: DictConfig) -> None:
    _, smiles_values = _collect_unique_sequences_and_smiles(
        csv_paths=[cfg.train_csv, cfg.val_csv, cfg.test_csv],
        sequence_col=cfg.sequence_col,
        smiles_col=cfg.smiles_col,
    )
    output_path = _resolve_embedding_path(
        path_value=getattr(cfg, "smiles_embeddings_path", ""),
        csv_path=cfg.train_csv,
        default_filename="coati_embeddings.pt",
    )

    _build_smiles_embeddings(
        smiles_values=smiles_values,
        output_path=Path(output_path),
        device=getattr(cfg, "embedding_device", "cuda:0"),
        doc_url=getattr(cfg, "coati_doc_url", "s3://terray-public/models/grande_closed.pkl"),
    )
    print(f"Wrote COATI embeddings: {output_path}")


if __name__ == "__main__":
    main()
