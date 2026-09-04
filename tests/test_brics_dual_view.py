import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from concisejepa.models.drug_decoder import DrugEncoder


CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def small_config():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config", overrides=["experiment=brics_dual_view_debug"])
    cfg.model.drug_layers = [[4, 4]]
    cfg.model.ligand_dim = 16
    cfg.model.residue_dim = 12
    cfg.model.drug_dim = 8
    cfg.model.dti_dim = 8
    cfg.model.protein_heads = 2
    cfg.model.fragment_transformer_depth = 1
    cfg.model.fragment_transformer_heads = 2
    cfg.model.fragment_transformer_mlp_ratio = 2.0
    cfg.model.coati_target_dim = 8
    cfg.model.coati_hidden_dim = 16
    cfg.data.fingerprint_length = 16
    return cfg


def synthetic_batch():
    generator = torch.Generator().manual_seed(20260904)
    return (
        torch.randn(4, 5, 12, generator=generator),
        torch.randn(4, 3, 16, generator=generator),
        torch.tensor(
            [
                [True, True, False],
                [True, True, True],
                [True, False, False],
                [True, True, False],
            ]
        ),
        torch.randn(4, 16, generator=generator),
        torch.randn(4, 8, generator=generator),
        torch.tensor([1.0, 0.0, 1.0, 0.0]),
        ["CCO", "CCN", "CCC", "CCCl"],
        ["protein-a", "protein-b", "protein-c", "protein-d"],
    )


class BricsDualViewTests(unittest.TestCase):
    def test_config_selects_new_bundle_and_reference_weights(self):
        cfg = small_config()
        self.assertEqual(
            cfg.model._target_, "concisejepa.models.brics_dual_view.BricsDualViewJEPA"
        )
        self.assertEqual(
            cfg.data._target_, "concisejepa.datamodules.fragment.DualViewFragmentDataModule"
        )
        self.assertEqual(
            cfg.task._target_,
            "concisejepa.lightning_modules.lit_brics_dual_view.LitBricsDualViewJEPA",
        )
        self.assertEqual(cfg.dual_view_loss.dti_weight, 1.0)
        self.assertEqual(cfg.dual_view_loss.jepa_weight, 1.0)
        self.assertEqual(cfg.dual_view_loss.alignment_weight, 1.0)
        self.assertEqual(cfg.trainer.limit_train_batches, 2)

    def test_forward_uses_one_codebook_and_returns_cosine_probabilities(self):
        cfg = small_config()
        model = instantiate(cfg.model).eval()
        outputs = model(*synthetic_batch()[:4])

        self.assertEqual(sum(isinstance(module, DrugEncoder) for module in model.modules()), 1)
        self.assertFalse(hasattr(model, "r_to_d_attention"))
        self.assertFalse(hasattr(model, "d_to_r_attention"))
        self.assertEqual(outputs["fragment_set_vector"].shape, (4, 8))
        self.assertEqual(outputs["whole_molecule_vector"].shape, (4, 8))
        self.assertEqual(outputs["receptor_dti_vector"].shape, (4, 8))
        self.assertEqual(outputs["jepa_pred"].shape, (4, 8))
        self.assertEqual(outputs["fragment_codes"].shape[:2], (4, 3))
        for key in (
            "binding",
            "fragment_binding",
            "whole_binding",
            "fragment_molecule_similarity",
        ):
            self.assertTrue(bool((outputs[key] >= 0).all()), key)
            self.assertTrue(bool((outputs[key] <= 1).all()), key)
        torch.testing.assert_close(
            outputs["binding"],
            0.5 * (outputs["fragment_binding"] + outputs["whole_binding"]),
        )

    def test_fragment_set_is_permutation_invariant_and_ignores_padding(self):
        cfg = small_config()
        model = instantiate(cfg.model).eval()
        batch = synthetic_batch()
        with torch.no_grad():
            original = model(*batch[:4])

            permutation = torch.tensor([2, 0, 1])
            permuted = model(
                batch[0], batch[1][:, permutation], batch[2][:, permutation], batch[3]
            )
            padded_fingerprints = batch[1].clone()
            padded_fingerprints[~batch[2]] = 1000.0
            changed_padding = model(batch[0], padded_fingerprints, batch[2], batch[3])

        torch.testing.assert_close(
            original["fragment_set_vector"],
            permuted["fragment_set_vector"],
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            original["fragment_set_vector"],
            changed_padding["fragment_set_vector"],
            rtol=1e-5,
            atol=1e-6,
        )

    def test_task_uses_probability_bce_and_all_objectives_reach_shared_codebook(self):
        cfg = small_config()
        model = instantiate(cfg.model)
        task = instantiate(cfg.task, experiment_config=cfg, model=model)
        batch = synthetic_batch()
        loss, losses, _, _, outputs = task._forward_losses_metrics(batch, "train")

        expected_dti = 0.5 * (
            F.binary_cross_entropy(outputs["whole_binding"], batch[5])
            + F.binary_cross_entropy(outputs["fragment_binding"], batch[5])
        )
        torch.testing.assert_close(losses["loss_dti"], expected_dti)
        torch.testing.assert_close(
            loss,
            losses["loss_dti"] + losses["loss_jepa"] + losses["loss_alignment"],
        )

        shared_parameter = model.drug_encoder.pre_transform[1].weight
        for loss_name in ("loss_dti", "loss_jepa", "loss_alignment"):
            task.zero_grad(set_to_none=True)
            _, component_losses, _, _, _ = task._forward_losses_metrics(batch, "train")
            component_losses[loss_name].backward()
            self.assertIsNotNone(shared_parameter.grad, loss_name)
            self.assertTrue(bool(torch.isfinite(shared_parameter.grad).all()), loss_name)
            self.assertGreater(float(shared_parameter.grad.abs().sum()), 0.0, loss_name)

    def test_dual_view_datamodule_returns_both_molecule_views(self):
        cfg = small_config()
        sequence = "A" * 50
        smiles = "CCO"
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            root = Path(tmpdir)
            frame = pd.DataFrame(
                {"Target Sequence": [sequence], "SMILES": [smiles], "Label": [1]}
            )
            for split in ("train", "val", "test"):
                frame.to_csv(root / f"{split}.csv", index=False)
            torch.save({sequence: torch.zeros(50, 12)}, root / "raygun_embeddings.pt")
            torch.save({smiles: torch.zeros(2, 16)}, root / "fragment_fps_r4_2048.pt")
            torch.save({smiles: torch.ones(16)}, root / "morgan_embeddings.pt")
            torch.save({smiles: torch.zeros(8)}, root / "coati_embeddings.pt")
            metadata = {
                "fingerprint_kind": "ecfp-count:4",
                "fingerprint_length": 16,
            }
            (root / "morgan_embeddings.pt.meta.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            metadata.update({"type": "fragment_fps", "max_frags": 16})
            (root / "fragment_fps_r4_2048.pt.meta.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )

            cfg.data.data_root = str(root)
            for key, filename in {
                "train_csv": "train.csv",
                "val_csv": "val.csv",
                "test_csv": "test.csv",
                "protein_embeddings_path": "raygun_embeddings.pt",
                "fragment_fps_path": "fragment_fps_r4_2048.pt",
                "morgan_embeddings_path": "morgan_embeddings.pt",
                "smiles_embeddings_path": "coati_embeddings.pt",
            }.items():
                cfg.data[key] = str(root / filename)

            datamodule = instantiate(cfg.data)
            datamodule.setup("fit")
            batch = datamodule.collator([(sequence, smiles, 1.0)])

        self.assertEqual(batch[0].shape, (1, 50, 12))
        self.assertEqual(batch[1].shape, (1, 2, 16))
        self.assertEqual(batch[2].shape, (1, 2))
        self.assertEqual(batch[3].shape, (1, 16))
        self.assertEqual(batch[4].shape, (1, 8))
        self.assertEqual(batch[5].shape, (1,))


if __name__ == "__main__":
    unittest.main()
