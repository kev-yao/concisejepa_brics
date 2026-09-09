"""Matched bounded-continuous FSQ control, not a high-dimensional bypass."""

import copy
import io
import unittest

import torch

from concisejepa.models.drug_decoder import DrugEncoder
from concisejepa.models.fsq import round_ste
from spikes.phase1.fragment_encoder import ConciseFragment, ConciseJEPAFragment
from spikes.phase1.lit_fragment import LitFragment


class ContinuousFSQTests(unittest.TestCase):
    @staticmethod
    def encoder(kind="continuous_fsq", levels=(32, 32, 32)):
        return DrugEncoder([list(levels)], dim=16, latent_dim=8, quantizer={"type": kind})

    def test_matches_bounded_no_round_formula_and_same_seed_parameters(self):
        torch.manual_seed(42)
        discrete = self.encoder("fsq")
        discrete_rng = torch.get_rng_state()
        torch.manual_seed(42)
        continuous = self.encoder()
        self.assertTrue(torch.equal(discrete_rng, torch.get_rng_state()))
        self.assertEqual(discrete.state_dict().keys(), continuous.state_dict().keys())
        for name, value in discrete.state_dict().items():
            torch.testing.assert_close(value, continuous.state_dict()[name], rtol=0, atol=0)

        inputs = torch.randn(5, 16)
        outputs = continuous(inputs)
        block = continuous.residualfsqs[0]
        projected = block.in_proj(block.ln1(continuous.pre_transform(inputs)))
        levels = torch.tensor([32, 32, 32])
        half_l = (levels - 1) * (1 - 1e-3) / 2
        offset = torch.where(levels % 2 == 0, 0.5, 0.0)
        bounded = (projected + (offset / half_l).tan()).tanh() * half_l - offset
        points = bounded / (levels // 2)
        expected = block.activation(block.ln2(block.out_proj(points)))
        torch.testing.assert_close(outputs["points"], points, rtol=0, atol=0)
        torch.testing.assert_close(outputs["emb"], expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(points * 16, (points * 16).round()))
        self.assertFalse(torch.equal(outputs["emb"], discrete(inputs)["emb"]))
        self.assertEqual(outputs["codes"].shape, (5, 3))
        self.assertEqual(outputs["codes"].dtype, torch.long)
        self.assertTrue(bool((outputs["codes"] == -1).all()))

    def test_continuous_has_no_discrete_embedding_or_enumeration(self):
        encoder = self.encoder()
        with self.assertRaisesRegex(RuntimeError, "continuous_fsq"):
            encoder.embed(torch.zeros(2, 3, dtype=torch.long))
        with self.assertRaisesRegex(RuntimeError, "continuous_fsq"):
            encoder.all_code_indices()
        self.assertEqual(encoder.get_levels()[0].tolist(), [32, 32, 32])

    def test_gradients_are_finite_and_nonzero_in_train_and_eval(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for training in (True, False):
                with self.subTest(device=device, training=training):
                    torch.manual_seed(42)
                    encoder = self.encoder().to(device).train(training)
                    inputs = torch.randn(5, 16, device=device, requires_grad=True)
                    outputs = encoder(inputs)
                    weights = torch.arange(1, 9, device=device, dtype=inputs.dtype)
                    (outputs["emb"] * weights).sum().backward()
                    for gradient in [inputs.grad] + [p.grad for p in encoder.parameters()]:
                        self.assertIsNotNone(gradient)
                        self.assertTrue(bool(torch.isfinite(gradient).all()))
                        self.assertGreater(gradient.abs().sum().item(), 0)

    def test_default_fsq_matches_original_composition_and_gradients(self):
        torch.manual_seed(42)
        encoder = self.encoder("fsq").double()
        reference = copy.deepcopy(encoder)
        inputs = torch.randn(5, 16, dtype=torch.float64, requires_grad=True)
        reference_inputs = inputs.detach().clone().requires_grad_()
        actual = encoder(inputs)
        block = reference.residualfsqs[0]
        projected = block.in_proj(block.ln1(reference.pre_transform(reference_inputs)))
        points = round_ste(block.fsq.bound(projected)) / (block.fsq._levels // 2)
        expected = block.activation(block.ln2(block.out_proj(points)))
        torch.testing.assert_close(actual["emb"], expected, rtol=0, atol=0)
        torch.testing.assert_close(actual["points"], points, rtol=0, atol=0)
        torch.testing.assert_close(
            actual["codes"], block.fsq.codes_to_factors(points).squeeze(1), rtol=0, atol=0
        )
        actual["emb"].square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=0, atol=0)
        for parameter, reference_parameter in zip(encoder.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=0, atol=0)

    def test_five_factor_fsq_capacity_and_roundtrip(self):
        encoder = self.encoder("fsq", levels=(8, 8, 8, 8, 8))
        all_codes = encoder.all_code_indices()
        self.assertEqual(all_codes.shape, (32768, 5))
        self.assertEqual(torch.unique(all_codes, dim=0).shape[0], 32768)
        outputs = encoder(torch.randn(5, 16))
        self.assertEqual(outputs["codes"].shape, (5, 5))
        self.assertTrue(bool(((outputs["codes"] >= 0) & (outputs["codes"] < 8)).all()))
        torch.testing.assert_close(encoder.embed(outputs["codes"]), outputs["emb"], rtol=0, atol=0)

    def test_corrected_f2r_training_and_checkpoint_reload(self):
        def make_task():
            backbone = ConciseFragment(
                [[32, 32, 32]], pooling="f2r", ligand_dim=16, residue_dim=12,
                drug_dim=8, proj_dim=8, nheads=2, pairwise_attention_chunk_size=4,
                use_pairwise_attention_checkpoint=True,
                drug_quantizer={"type": "continuous_fsq"},
            )
            return LitFragment(ConciseJEPAFragment(backbone, smiles_target_dim=8, jepa_hidden_dim=16))

        torch.manual_seed(42)
        task = make_task()
        batch = (
            torch.randn(3, 5, 12), torch.randn(3, 3, 16),
            torch.tensor([[True, True, False], [True, True, True], [True, False, False]]),
            torch.randn(3, 8), torch.tensor([1.0, 0.0, 1.0]),
            ["CCO", "CCN", "CCC"], ["protein-a", "protein-a", "protein-b"],
        )
        optimizer = task.configure_optimizers()
        loss, outputs = task._forward_step(batch, "train")
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(outputs["frag_codes"].shape, (3, 3, 3))
        self.assertTrue(bool((outputs["frag_codes"] == -1).all()))
        loss.backward()
        for parameter in task.model.concise.d_encoder.parameters():
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        optimizer.step()
        task.eval()
        with torch.no_grad():
            expected = task.model(*batch[:3])
        buffer = io.BytesIO()
        torch.save({"state_dict": task.state_dict(), "optimizer": optimizer.state_dict()}, buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=True)
        reloaded = make_task().eval()
        reloaded.load_state_dict(checkpoint["state_dict"], strict=True)
        reloaded.configure_optimizers().load_state_dict(checkpoint["optimizer"])
        with torch.no_grad():
            actual = reloaded.model(*batch[:3])
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
