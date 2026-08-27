import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from concisejepa.datamodules.dataloader import (
    _build_morgan_embeddings,
    _fingerprint_default_filename,
    _fingerprint_metadata_path,
    _resolve_embedding_path,
    _validate_fingerprint_model_dimensions,
    _validate_fingerprint_metadata,
)


class FingerprintEmbeddingTests(unittest.TestCase):
    def test_default_filename_is_variant_specific(self):
        self.assertEqual(
            _fingerprint_default_filename("ecfp-count:4", 2048),
            "morgan_ecfp_count_4_2048.pt",
        )
        self.assertEqual(_fingerprint_default_filename("ecfp:4", 2048), "morgan_ecfp_4_2048.pt")

    def test_empty_path_resolves_next_to_training_csv(self):
        path = _resolve_embedding_path("", "/data/bindingdb/train.csv", "morgan_ecfp_count_4_2048.pt")
        self.assertEqual(path, Path("/data/bindingdb/morgan_ecfp_count_4_2048.pt"))

    def test_metadata_validation_rejects_missing_and_mismatched_metadata(self):
        with TemporaryDirectory() as tmpdir:
            embedding_path = Path(tmpdir) / "morgan.pt"
            torch.save({}, embedding_path)

            with self.assertRaisesRegex(RuntimeError, "metadata is missing"):
                _validate_fingerprint_metadata(embedding_path, "ecfp-count:4", 2048)

            _fingerprint_metadata_path(embedding_path).write_text(
                json.dumps({"fingerprint_kind": "ecfp:4", "fingerprint_length": 2048}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "metadata mismatch"):
                _validate_fingerprint_metadata(embedding_path, "ecfp-count:4", 2048)

    def test_model_dimension_must_match_fingerprint_length(self):
        _validate_fingerprint_model_dimensions(2048, 2048)
        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            _validate_fingerprint_model_dimensions(1024, 2048)

    def test_generation_uses_configured_transformer_and_writes_metadata(self):
        calls = {}

        class FakeTransformer:
            def __init__(self, **kwargs):
                calls["kwargs"] = kwargs

            def __call__(self, smiles_values, ignore_errors):
                calls["smiles"] = smiles_values
                calls["ignore_errors"] = ignore_errors
                return np.asarray([[2.0, 0.0, 1.0], [0.0, 3.0, 0.0]]), [0, 1]

        fake_fp_module = types.ModuleType("molfeat.trans.fp")
        fake_fp_module.FPVecTransformer = FakeTransformer
        fake_trans_module = types.ModuleType("molfeat.trans")
        fake_trans_module.fp = fake_fp_module
        fake_molfeat_module = types.ModuleType("molfeat")
        fake_molfeat_module.trans = fake_trans_module

        with TemporaryDirectory() as tmpdir, patch.dict(
            sys.modules,
            {
                "molfeat": fake_molfeat_module,
                "molfeat.trans": fake_trans_module,
                "molfeat.trans.fp": fake_fp_module,
            },
        ):
            output_path = Path(tmpdir) / "nested" / "morgan.pt"
            _build_morgan_embeddings(
                smiles_values=["CC", "CCC"],
                output_path=output_path,
                fingerprint_kind="ecfp-count:4",
                fingerprint_length=3,
                source_csv_paths=[Path(tmpdir) / "train.csv"],
            )

            embeddings = torch.load(output_path, map_location="cpu")
            metadata = json.loads(_fingerprint_metadata_path(output_path).read_text(encoding="utf-8"))

        self.assertEqual(calls["kwargs"]["kind"], "ecfp-count:4")
        self.assertEqual(calls["kwargs"]["length"], 3)
        self.assertEqual(embeddings["CC"].tolist(), [2.0, 0.0, 1.0])
        self.assertEqual(metadata["fingerprint_kind"], "ecfp-count:4")
        self.assertEqual(metadata["fingerprint_length"], 3)


if __name__ == "__main__":
    unittest.main()
