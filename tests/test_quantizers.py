import math
import unittest
from unittest.mock import Mock, patch

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from concisejepa.lightning_modules.lit_jepa import (
    LitConciseJEPA,
    product_quantizer_usage_metrics,
    quantizer_usage_metrics,
)
from concisejepa.models.concise import Concise
from concisejepa.models.concise_jepa import ConciseJEPA
from concisejepa.models.diveq import DiVeQQuantizer, ProductDiVeQQuantizer
from concisejepa.models.drug_decoder import DrugEncoder


class DiVeQQuantizerTests(unittest.TestCase):
    def test_nearest_selection_shapes_eval_and_embed(self):
        quantizer = DiVeQQuantizer(codebook_size=3, code_dim=2, sigma=0.01)
        with torch.no_grad():
            quantizer.codebook.copy_(torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]))
        inputs = torch.tensor([[0.1, 0.2], [1.8, 0.1]])

        quantizer.eval()
        quantized, indices = quantizer(inputs)

        self.assertEqual(quantized.shape, (2, 2))
        self.assertEqual(indices.tolist(), [0, 1])
        torch.testing.assert_close(quantized, quantizer.codebook[indices])
        torch.testing.assert_close(quantizer.embed(indices), quantized)

    def test_training_is_noisy_and_gradients_reach_latent_and_codebook(self):
        torch.manual_seed(7)
        quantizer = DiVeQQuantizer(codebook_size=2, code_dim=3, sigma=0.1)
        with torch.no_grad():
            quantizer.codebook.copy_(torch.tensor([[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]]))
        inputs = torch.tensor([[1.0, 0.5, -0.5]], requires_grad=True)
        quantizer.train()

        quantized, indices = quantizer(inputs)
        self.assertEqual(indices.item(), 0)
        self.assertFalse(torch.equal(quantized, quantizer.codebook[indices]))
        quantized.sum().backward()

        self.assertGreater(inputs.grad.abs().sum().item(), 0.0)
        self.assertGreater(quantizer.codebook.grad[0].abs().sum().item(), 0.0)
        self.assertEqual(quantizer.codebook.grad[1].abs().sum().item(), 0.0)

    def test_training_matches_directional_reparameterization_equation(self):
        quantizer = DiVeQQuantizer(codebook_size=2, code_dim=2, sigma=0.5)
        with torch.no_grad():
            quantizer.codebook.copy_(torch.tensor([[0.0, 0.0], [5.0, 5.0]]))
        inputs = torch.tensor([[1.0, 0.0]], requires_grad=True)
        noise = torch.tensor([[0.0, 2.0]])

        with patch("torch.randn_like", return_value=noise):
            quantized, indices = quantizer(inputs)

        direction = quantizer.codebook[indices] - inputs
        noisy_direction = direction + 0.5 * noise
        expected = inputs + direction.norm(dim=-1, keepdim=True) * (
            noisy_direction / noisy_direction.norm(dim=-1, keepdim=True)
        ).detach()
        torch.testing.assert_close(quantized, expected)

    def test_inactive_codes_are_replaced_before_next_forward(self):
        quantizer = DiVeQQuantizer(
            codebook_size=4,
            code_dim=2,
            sigma=0.0,
            replacement_interval=1,
            discard_threshold=0.01,
            replacement_perturbation=1e-3,
        )
        with torch.no_grad():
            quantizer.codebook.copy_(torch.tensor([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0], [5.0, 5.0]]))
        inputs = torch.tensor([[0.1, 0.1]], requires_grad=True)

        first_output, _ = quantizer(inputs)
        first_output.sum().backward()
        before_replacement = quantizer.codebook.detach().clone()
        quantizer(inputs.detach())

        self.assertFalse(torch.equal(quantizer.codebook[1:], before_replacement[1:]))
        self.assertEqual(quantizer.assignment_counts.sum().item(), 1)


class DrugEncoderTests(unittest.TestCase):
    def _check_encoder(self, quantizer_config, expected_codes, expected_pre_quantized):
        encoder = DrugEncoder(
            layers=[[2, 2]],
            dim=8,
            latent_dim=4,
            quantizer=quantizer_config,
        )
        inputs = torch.randn(3, 8)
        outputs = encoder(inputs)
        self.assertEqual(outputs["emb"].shape, (3, 1, 4))
        self.assertEqual(outputs["codes"].shape, expected_codes)
        self.assertEqual(outputs["pre_quantized"].shape, expected_pre_quantized)
        self.assertEqual(encoder.embed(outputs["codes"]).shape, (3, 1, 4))
        if quantizer_config["type"] == "fsq":
            self.assertEqual(outputs["residual_code"].shape, (3, 1, 2))
            reconstructed = encoder.embed(outputs["codes"], outputs["residual_code"])
            torch.testing.assert_close(reconstructed, outputs["emb"])

    def test_morgan_and_code_paths_for_both_quantizers(self):
        self._check_encoder(
            {"type": "diveq", "num_groups": 2, "codebook_size": 2, "group_dim": 1, "sigma": 0.01},
            (3, 2),
            (3, 2, 1),
        )
        self._check_encoder(
            {"type": "fsq", "residual_dim": 2, "residual_alpha": 0.1},
            (3, 2),
            (3, 1, 4),
        )

    def test_fsq_factor_codes_enumeration_and_coarse_embedding(self):
        encoder = DrugEncoder(
            layers=[[2, 3, 4]],
            dim=8,
            latent_dim=4,
            quantizer={"type": "fsq", "residual_dim": 2, "residual_alpha": 0.1},
        )
        encoder.eval()
        outputs = encoder(torch.randn(5, 8))
        self.assertEqual(outputs["codes"].shape, (5, 3))
        self.assertTrue(torch.all(outputs["codes"] >= 0))
        self.assertTrue(torch.all(outputs["codes"] < torch.tensor([2, 3, 4])))

        all_codes = encoder.all_code_indices()
        self.assertEqual(all_codes.shape, (24, 3))
        self.assertEqual(torch.unique(all_codes, dim=0).shape[0], 24)
        coarse = encoder.embed(outputs["codes"])
        self.assertFalse(torch.equal(coarse, outputs["emb"]))

    def test_fsq_residual_branch_gradients_and_alpha_blend(self):
        encoder = DrugEncoder(
            layers=[[2, 2]],
            dim=8,
            latent_dim=4,
            quantizer={
                "type": "fsq",
                "residual_dim": 2,
                "residual_alpha": 0.1,
            },
        )
        inputs = torch.randn(3, 8, requires_grad=True)
        outputs = encoder(inputs)

        torch.testing.assert_close(
            outputs["residual_code"],
            encoder.residual_encoder(outputs["pre_quantized"]),
        )
        scaled_residual = 0.1 * outputs["residual_correction"]
        expected = outputs["quantized"] + scaled_residual
        blended_norm = expected.norm(dim=-1).mean()
        torch.testing.assert_close(outputs["emb"], expected)
        torch.testing.assert_close(outputs["blended_norm"], blended_norm)
        torch.testing.assert_close(
            outputs["quantized_to_blended_ratio"],
            outputs["quantized"].norm(dim=-1).mean() / blended_norm.clamp_min(1e-8),
        )
        torch.testing.assert_close(
            outputs["continuous_to_blended_ratio"],
            scaled_residual.norm(dim=-1).mean() / blended_norm.clamp_min(1e-8),
        )
        torch.testing.assert_close(
            outputs["scaled_residual_norm"],
            0.1 * outputs["residual_norm"],
        )
        torch.testing.assert_close(
            outputs["scaled_residual_to_quantized_ratio"],
            0.1 * outputs["residual_to_quantized_ratio"],
        )
        outputs["emb"].sum().backward()
        self.assertGreater(inputs.grad.abs().sum().item(), 0.0)
        self.assertGreater(encoder.residual_encoder[0].weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(encoder.residual_decoder.weight.grad.abs().sum().item(), 0.0)

    def test_diveq_capacity_validation_and_checkpoint_incompatibility(self):
        with self.assertRaisesRegex(ValueError, "capacity must match"):
            DrugEncoder(
                [[2, 2]],
                dim=8,
                latent_dim=4,
                quantizer={"type": "diveq", "num_groups": 3, "codebook_size": 2},
            )

        fsq = DrugEncoder([[2, 2]], dim=8, latent_dim=4, quantizer={"type": "fsq"})
        diveq = DrugEncoder(
            [[2, 2]],
            dim=8,
            latent_dim=4,
            quantizer={"type": "diveq", "num_groups": 2, "codebook_size": 2, "group_dim": 1},
        )
        fsq.load_state_dict(fsq.state_dict(), strict=True)
        old_fsq_state = {
            key: value
            for key, value in fsq.state_dict().items()
            if "residual_encoder" not in key and "residual_decoder" not in key
        }
        with self.assertRaises(RuntimeError):
            fsq.load_state_dict(old_fsq_state, strict=True)
        with self.assertRaises(RuntimeError):
            diveq.load_state_dict(fsq.state_dict(), strict=True)

        legacy_diveq = DiVeQQuantizer(codebook_size=4, code_dim=4)
        with self.assertRaises(RuntimeError):
            diveq.diveq.load_state_dict(legacy_diveq.state_dict(), strict=True)


class ProductDiVeQTests(unittest.TestCase):
    def test_shapes_embed_and_complete_code_enumeration(self):
        quantizer = ProductDiVeQQuantizer(dim=4, num_groups=3, codebook_size=32, group_dim=1)
        quantizer.eval()
        outputs = quantizer(torch.randn(5, 1, 4))
        self.assertEqual(outputs["codes"].shape, (5, 3))
        self.assertEqual(outputs["emb"].shape, (5, 1, 4))
        self.assertEqual(outputs["pre_quantized"].shape, (5, 3, 1))
        torch.testing.assert_close(quantizer.embed(outputs["codes"]), outputs["emb"])

        all_codes = quantizer.all_code_indices()
        self.assertEqual(all_codes.shape, (32768, 3))
        self.assertEqual(torch.unique(all_codes, dim=0).shape[0], 32768)
        self.assertEqual(all_codes.min().item(), 0)
        self.assertEqual(all_codes.max().item(), 31)
        joint_indices = quantizer.codes_to_indices(all_codes)
        self.assertEqual(torch.unique(joint_indices).numel(), 32768)
        self.assertEqual(joint_indices.min().item(), 0)
        self.assertEqual(joint_indices.max().item(), 32767)

    def test_training_gradients_reach_all_selected_group_codebooks(self):
        quantizer = ProductDiVeQQuantizer(dim=4, num_groups=3, codebook_size=4, group_dim=1, sigma=0.01)
        inputs = torch.randn(3, 1, 4, requires_grad=True)
        outputs = quantizer(inputs)
        weights = torch.tensor([[[1.0, -2.0, 3.0, -4.0]]])
        (outputs["emb"] * weights).sum().backward()

        self.assertGreater(inputs.grad.abs().sum().item(), 0.0)
        for group, group_quantizer in enumerate(quantizer.quantizers):
            selected = outputs["codes"][:, group].unique()
            self.assertGreater(group_quantizer.codebook.grad[selected].abs().sum().item(), 0.0)

    def test_training_uses_joint_three_dimensional_diveq_direction(self):
        quantizer = ProductDiVeQQuantizer(dim=4, num_groups=3, codebook_size=2, group_dim=1, sigma=0.5)
        inputs = torch.randn(2, 1, 4)
        projected = quantizer.in_proj(quantizer.ln1(inputs)).reshape(2, 3, 1)
        selected = []
        for group, group_quantizer in enumerate(quantizer.quantizers):
            indices = torch.cdist(projected[:, group], group_quantizer.codebook).argmin(dim=-1)
            selected.append(group_quantizer.embed(indices))
        selected = torch.stack(selected, dim=1).reshape(2, 3)
        projected_flat = projected.reshape(2, 3)
        noise = torch.tensor([[0.5, -0.25, 1.0], [-0.5, 0.25, -1.0]])

        with patch("torch.randn_like", return_value=noise):
            outputs = quantizer(inputs)

        direction = selected - projected_flat
        noisy_direction = direction + 0.5 * noise
        expected = projected_flat + direction.norm(dim=-1, keepdim=True) * (
            noisy_direction / noisy_direction.norm(dim=-1, keepdim=True)
        ).detach()
        torch.testing.assert_close(outputs["quantized"].reshape(2, 3), expected)

    def test_replacement_is_applied_independently_to_each_group(self):
        quantizer = ProductDiVeQQuantizer(
            dim=4,
            num_groups=3,
            codebook_size=4,
            group_dim=1,
            replacement_interval=1,
            discard_threshold=0.01,
            replacement_perturbation=1e-3,
        )
        inputs = torch.randn(2, 1, 4)
        quantizer(inputs)
        before = [group.codebook.detach().clone() for group in quantizer.quantizers]
        quantizer(inputs)

        for group, previous in zip(quantizer.quantizers, before):
            self.assertFalse(torch.equal(group.codebook, previous))
            self.assertEqual(group.assignment_counts.sum().item(), inputs.shape[0])


class ModelAndMetricsTests(unittest.TestCase):
    @staticmethod
    def _model(quantizer_type):
        quantizer = {"type": quantizer_type}
        if quantizer_type == "diveq":
            quantizer.update(num_groups=2, codebook_size=2, group_dim=1, sigma=0.01)
        backbone = Concise(
            drug_layers=[[2, 2]],
            drug_quantizer=quantizer,
            ligand_dim=8,
            residue_dim=6,
            drug_dim=4,
            proj_dim=8,
            nheads=2,
            pairwise_attention_chunk_size=4,
            use_pairwise_attention_checkpoint=False,
        )
        return ConciseJEPA(backbone, smiles_target_dim=5, jepa_hidden_dim=12)

    def test_forward_backward_shapes_for_both_quantizers(self):
        for quantizer_type in ("diveq", "fsq"):
            with self.subTest(quantizer_type=quantizer_type):
                model = self._model(quantizer_type)
                outputs = model(torch.randn(3, 4, 6), torch.randn(3, 8))
                self.assertEqual(outputs["binding"].shape, (3,))
                self.assertEqual(outputs["similarity_logits"].shape, (3, 3))
                self.assertEqual(outputs["jepa_pred"].shape, (3, 5))
                self.assertEqual(outputs["codes"].shape, (3, 2))
                if quantizer_type == "fsq":
                    self.assertIn("scaled_residual_norm", outputs)
                    self.assertIn("scaled_residual_to_quantized_ratio", outputs)
                    self.assertIn("quantized_to_blended_ratio", outputs)
                    self.assertIn("continuous_to_blended_ratio", outputs)
                outputs["binding"].sum().add(outputs["jepa_pred"].sum()).backward()

                from_codes = model.predict_from_codes(torch.randn(3, 4, 6), outputs["codes"].detach())
                self.assertEqual(from_codes["binding"].shape, (3,))
                self.assertEqual(from_codes["jepa_pred"].shape, (3, 5))
                if quantizer_type == "fsq":
                    from_codes_and_residual = model.predict_from_codes(
                        torch.randn(3, 4, 6),
                        outputs["codes"].detach(),
                        outputs["residual_code"].detach(),
                    )
                    self.assertEqual(from_codes_and_residual["binding"].shape, (3,))

    def test_controlled_usage_metrics(self):
        metrics = quantizer_usage_metrics(torch.tensor([2, 1, 0, 1]))
        self.assertAlmostEqual(metrics["code_usage_entropy"].item(), 1.5 * math.log(2), places=6)
        self.assertEqual(metrics["active_code_count"].item(), 3)
        self.assertAlmostEqual(metrics["dead_code_fraction"].item(), 0.25)
        self.assertAlmostEqual(metrics["top_code_frequency"].item(), 0.5)

        product_metrics = product_quantizer_usage_metrics(
            group_counts=torch.tensor([[2, 2], [1, 3], [4, 0]]),
            joint_counts=torch.tensor([1, 1, 0, 2, 0, 0, 0, 0]),
        )
        self.assertEqual(product_metrics["active_code_count"].item(), 3)
        self.assertEqual(product_metrics["group_0_active_code_count"].item(), 2)
        self.assertEqual(product_metrics["group_2_active_code_count"].item(), 1)

    def test_ranking_metrics_return_perfect_scores_for_perfect_ordering(self):
        logits = torch.eye(6) * 2.0
        targets = torch.arange(6)
        mrr, acc_at_1, acc_at_5 = LitConciseJEPA._ranking_metrics_from_logits(logits, targets)
        self.assertEqual(mrr.item(), 1.0)
        self.assertEqual(acc_at_1.item(), 1.0)
        self.assertEqual(acc_at_5.item(), 1.0)


class LightningSanityTests(unittest.TestCase):
    def _config(self, quantizer_type):
        quantizer = {"type": quantizer_type}
        if quantizer_type == "diveq":
            quantizer.update(num_groups=2, codebook_size=2, group_dim=1, sigma=0.01)
        else:
            quantizer.update(
                residual_dim=2,
                residual_alpha=0.1,
            )
        return OmegaConf.create(
            {
                "lr": 1e-4,
                "weight_decay": 0.0,
                "coati_validation": {"enabled": False},
                "synthetic_validation": {"enabled": False},
                "model": {
                    "_target_": "concisejepa.models.concise_jepa.ConciseJEPA",
                    "concise_backbone": {
                        "_target_": "concisejepa.models.concise.Concise",
                        "drug_layers": [[2, 2]],
                        "drug_quantizer": quantizer,
                        "ligand_dim": 8,
                        "residue_dim": 6,
                        "drug_dim": 4,
                        "proj_dim": 8,
                        "nheads": 2,
                        "pairwise_attention_chunk_size": 4,
                        "use_pairwise_attention_checkpoint": False,
                    },
                    "smiles_target_dim": 5,
                    "jepa_hidden_dim": 12,
                },
            }
        )

    def test_lightning_validation_for_both_quantizers(self):
        dataset = TensorDataset(
            torch.randn(2, 4, 6),
            torch.randn(2, 8),
            torch.randn(2, 5),
            torch.tensor([0.0, 1.0]),
        )
        loader = DataLoader(dataset, batch_size=2)
        for quantizer_type in ("diveq", "fsq"):
            with self.subTest(quantizer_type=quantizer_type):
                module = LitConciseJEPA(self._config(quantizer_type))
                trainer = pl.Trainer(
                    accelerator="cpu",
                    devices=1,
                    logger=False,
                    enable_checkpointing=False,
                    enable_progress_bar=False,
                    limit_val_batches=1,
                )
                results = trainer.validate(module, dataloaders=loader, verbose=False)
                self.assertIn("val/dti_auroc", results[0])
                self.assertIn("val/dti_auprc", results[0])
                if quantizer_type == "fsq":
                    self.assertIn("val/scaled_residual_norm", results[0])
                    self.assertIn("val/scaled_residual_to_quantized_ratio", results[0])
                    self.assertIn("val/quantized_to_blended_ratio", results[0])
                    self.assertIn("val/continuous_to_blended_ratio", results[0])

    def test_alpha_contribution_metrics_are_logged_for_train_and_val(self):
        module = LitConciseJEPA(self._config("fsq"))
        module.log = Mock()
        batch = (
            torch.randn(2, 4, 6),
            torch.randn(2, 8),
            torch.randn(2, 5),
            torch.tensor([0.0, 1.0]),
        )

        for stage in ("train", "val"):
            with self.subTest(stage=stage):
                module.log.reset_mock()
                module._step(batch, stage=stage)
                logged_names = {call.args[0] for call in module.log.call_args_list}
                self.assertIn(f"{stage}/scaled_residual_norm", logged_names)
                self.assertIn(f"{stage}/scaled_residual_to_quantized_ratio", logged_names)
                self.assertIn(f"{stage}/quantized_to_blended_ratio", logged_names)
                self.assertIn(f"{stage}/continuous_to_blended_ratio", logged_names)


if __name__ == "__main__":
    unittest.main()
