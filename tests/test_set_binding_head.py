"""Typed-set readout behavior, independent of frozen backbones and data files."""
import copy
import unittest

import torch
import torch.nn.functional as F

from concisejepa.models.set_binding_head import SetTransformerBindingHead


class SetBindingHeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.head = SetTransformerBindingHead(
            feature_dim=12, model_dim=16, num_heads=4, num_blocks=2, dropout=0.1
        ).eval()
        self.tokens = torch.randn(3, 4, 12)
        self.types = torch.arange(4)

    def test_shapes_probabilities_and_attention_pooling(self):
        output = self.head(self.tokens)
        self.assertEqual(output["binding_logits"].shape, (3,))
        self.assertEqual(output["binding"].shape, (3,))
        self.assertEqual(output["pooled"].shape, (3, 16))
        torch.testing.assert_close(output["binding"], output["binding_logits"].sigmoid())
        self.assertEqual(sum(isinstance(m, torch.nn.MultiheadAttention)
                             for m in self.head.modules()), 3)
        self.assertEqual(tuple(self.head.pool_seed.shape), (1, 1, 16))
        self.assertTrue(self.head.pool_seed.requires_grad)
        for value in output.values():
            self.assertTrue(torch.isfinite(value).all())
        # The classifier exposes signed logits, not positive scaled probabilities.
        with torch.no_grad():
            self.head.readout[-1].weight.zero_()
            self.head.readout[-1].bias.fill_(-2)
        torch.testing.assert_close(self.head(self.tokens)["binding_logits"], torch.full((3,), -2.))

    def test_joint_token_type_mask_permutation(self):
        mask = torch.tensor([[1, 1, 0, 1], [1, 0, 1, 1], [1, 1, 1, 1]], dtype=torch.bool)
        order = torch.tensor([2, 0, 3, 1])
        expected = self.head(self.tokens, self.types, mask)
        actual = self.head(self.tokens[:, order], self.types[order], mask[:, order])
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=1e-6)
        # Shared [F] and per-example [B,F] type IDs mean the same thing.
        batched = self.head(self.tokens, self.types.expand(3, -1), mask)
        for key in expected:
            torch.testing.assert_close(batched[key], expected[key], rtol=0, atol=0)

    def test_padding_values_and_extra_padding_are_ignored(self):
        mask = torch.tensor([[1, 1, 0, 1], [1, 0, 1, 1], [1, 1, 1, 1]], dtype=torch.bool)
        expected = self.head(self.tokens, mask=mask)
        changed = self.tokens.clone()
        changed[~mask] = float("nan")
        actual = self.head(changed, mask=mask)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        extended = torch.cat([changed, torch.full((3, 2, 12), float("inf"))], dim=1)
        extended_mask = torch.cat([mask, torch.zeros(3, 2, dtype=torch.bool)], dim=1)
        actual = self.head(extended, torch.tensor([0, 1, 2, 3, 0, 1]), extended_mask)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=1e-6)

    def test_masked_tokens_have_no_input_gradient(self):
        mask = torch.ones(3, 4, dtype=torch.bool)
        mask[0, 2] = False
        tokens = self.tokens.clone()
        tokens[~mask] = float("nan")
        tokens.requires_grad_()
        output = self.head(tokens, mask=mask)
        F.binary_cross_entropy_with_logits(
            output["binding_logits"], torch.tensor([1., 0., 1.])
        ).backward()
        self.assertTrue(torch.isfinite(tokens.grad).all())
        torch.testing.assert_close(tokens.grad[~mask], torch.zeros_like(tokens.grad[~mask]))
        self.assertGreater(tokens.grad[mask].abs().sum(), 0)

    def test_statistics_gather_by_type_and_checkpoint_roundtrip(self):
        mean = torch.randn(4, 12, requires_grad=True)
        scale = torch.rand(4, 12, requires_grad=True) + .2
        self.head.set_input_stats(mean, scale)
        self.assertFalse(self.head.input_mean.requires_grad)
        self.assertFalse(self.head.input_scale.requires_grad)
        self.assertIn("input_mean", self.head.state_dict())
        self.assertIn("input_scale", self.head.state_dict())
        normalized_head = copy.deepcopy(self.head)
        normalized_head.set_input_stats(torch.zeros_like(mean), torch.ones_like(scale))
        order = torch.tensor([3, 0, 2, 1])
        expected = normalized_head((self.tokens - mean.detach()) / scale.detach())
        actual = self.head(self.tokens[:, order], order)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=1e-6)
        restored = SetTransformerBindingHead(feature_dim=12, model_dim=16, num_heads=4).eval()
        restored.load_state_dict(self.head.state_dict(), strict=True)
        for key, value in self.head(self.tokens).items():
            torch.testing.assert_close(restored(self.tokens)[key], value, rtol=0, atol=0)

    def test_finite_bce_gradients_and_two_optimizer_updates(self):
        self.head.train()
        optimizer = torch.optim.AdamW(self.head.parameters(), lr=1e-3)
        original = {name: p.detach().clone() for name, p in self.head.named_parameters()}
        for labels in (torch.tensor([1., 0., 1.]), torch.zeros(3)):
            optimizer.zero_grad(set_to_none=True)
            tokens = self.tokens.clone().requires_grad_()
            output = self.head(tokens)
            loss = F.binary_cross_entropy_with_logits(output["binding_logits"], labels)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(tokens.grad).all())
            self.assertGreater(tokens.grad.abs().sum(), 0)
            for name, parameter in self.head.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(self.head.pool_seed.grad.abs().sum(), 0)
            optimizer.step()
        self.assertTrue(any(not torch.equal(p, original[name])
                            for name, p in self.head.named_parameters()))
        self.head.eval()
        singleton = self.head(self.tokens[:1, :1], torch.tensor([0]))
        loss = F.binary_cross_entropy_with_logits(singleton["binding_logits"], torch.ones(1))
        self.assertTrue(torch.isfinite(loss))

    def test_invalid_architecture_parameters(self):
        for kwargs in (
            {"feature_dim": 0}, {"model_dim": 0}, {"num_heads": 0},
            {"num_blocks": 0}, {"num_blocks": 1.5}, {"model_dim": 15, "num_heads": 4},
            {"dropout": -0.1}, {"dropout": 1}, {"dropout": float("nan")},
            {"ff_multiplier": 0}, {"ff_multiplier": float("inf")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SetTransformerBindingHead(**kwargs)

    def test_invalid_input_and_mask_contracts(self):
        invalid_calls = [
            (torch.randn(3, 12), None, None),
            (torch.randn(3, 4, 13), None, None),
            (torch.randn(0, 4, 12), None, None),
            (torch.randn(3, 0, 12), torch.empty(0, dtype=torch.long), None),
            (torch.randn(3, 2, 12), None, None),
            (self.tokens, torch.tensor([0, 1, 2]), None),
            (self.tokens, torch.ones(2, 4, dtype=torch.long), None),
            (self.tokens, torch.tensor([0, 1, 2, 4]), None),
            (self.tokens, torch.tensor([-1, 1, 2, 3]), None),
            (self.tokens, torch.arange(4).float(), None),
            (self.tokens, None, torch.ones(3, 4)),
            (self.tokens, None, torch.ones(3, 3, dtype=torch.bool)),
            (self.tokens, None, torch.zeros(3, 4, dtype=torch.bool)),
            (self.tokens.long(), None, None),
            (torch.full_like(self.tokens, float("nan")), None, None),
        ]
        for tokens, types, mask in invalid_calls:
            with self.subTest(shape=tokens.shape, types=types, mask=mask), self.assertRaises(ValueError):
                self.head(tokens, types, mask)
        mask = torch.ones(3, 4, dtype=torch.bool)
        mask[1] = False
        with self.assertRaises(ValueError):
            self.head(self.tokens, mask=mask)

    def test_invalid_statistics_do_not_modify_buffers(self):
        for mean, scale in (
            (torch.zeros(3, 12), torch.ones(4, 12)),
            (torch.zeros(4, 12), torch.ones(4, 13)),
            (torch.full((4, 12), float("nan")), torch.ones(4, 12)),
            (torch.zeros(4, 12), torch.full((4, 12), float("inf"))),
            (torch.zeros(4, 12), torch.zeros(4, 12)),
            (torch.zeros(4, 12), -torch.ones(4, 12)),
        ):
            with self.subTest(), self.assertRaises(ValueError):
                self.head.set_input_stats(mean, scale)
            torch.testing.assert_close(self.head.input_mean, torch.zeros(4, 12))
            torch.testing.assert_close(self.head.input_scale, torch.ones(4, 12))


if __name__ == "__main__":
    unittest.main()
