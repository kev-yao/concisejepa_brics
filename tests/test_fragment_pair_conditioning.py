"""Pair scores must not depend on a drug's unrelated anchor protein."""

import unittest

import torch

from concisejepa.models.fragment import (
    ConciseFragment,
    ConciseJEPAFragment,
    ConciseJEPAFragmentXAttn,
)


class FragmentPairConditioningTests(unittest.TestCase):
    @staticmethod
    def make_model(pooling, chunk_size=4, checkpoint=False):
        torch.manual_seed(42)
        return ConciseFragment(
            drug_layers=[[32, 32, 32]],
            pooling=pooling,
            ligand_dim=64,
            residue_dim=12,
            drug_dim=16,
            proj_dim=16,
            nheads=4,
            pairwise_attention_chunk_size=chunk_size,
            use_pairwise_attention_checkpoint=checkpoint,
        ).double().eval()

    @staticmethod
    def inputs():
        generator = torch.Generator().manual_seed(123)
        fingerprints = 3 * torch.randn(3, 5, 64, generator=generator, dtype=torch.float64)
        proteins = 3 * torch.randn(3, 7, 12, generator=generator, dtype=torch.float64)
        mask = torch.tensor(
            [[True, True, False, False, False],
             [True, True, True, True, True],
             [True, True, True, False, False]]
        )
        return fingerprints, mask, proteins

    def test_batched_scores_match_independently_evaluated_pairs(self):
        fingerprints, mask, proteins = self.inputs()
        for pooling in ("f2r", "cross_attention", "mean", "max", "latent_query"):
            with self.subTest(pooling=pooling), torch.no_grad():
                model = self.make_model(pooling)
                outputs = model(fingerprints, mask, proteins)
                # Avoid a vacuous pass from a fixture with identical fragment codes.
                self.assertGreater(torch.unique(outputs["frag_codes"][mask], dim=0).shape[0], 1)
                independent = torch.stack([
                    model(fingerprints[i:i + 1], mask[i:i + 1], proteins[j:j + 1])["binding"][0]
                    for i in range(3) for j in range(3)
                ]).reshape(3, 3)
                torch.testing.assert_close(
                    outputs["pairwise_binding"], independent, rtol=1e-9, atol=1e-10
                )

    def test_f2r_attribution_keeps_aligned_weights_through_backward(self):
        fingerprints, mask, proteins = self.inputs()
        for checkpoint in (False, True):
            with self.subTest(checkpoint=checkpoint):
                model = self.make_model("f2r", checkpoint=checkpoint)
                with torch.no_grad():
                    expected = []
                    for i in range(3):
                        model(fingerprints[i:i + 1], mask[i:i + 1], proteins[i:i + 1])
                        expected.append(model.pooling_layer.last_weights.clone())
                    expected = torch.cat(expected)

                outputs = model(fingerprints, mask, proteins)
                torch.testing.assert_close(model.pooling_layer.last_weights, expected)
                (outputs["pairwise_binding"].square().sum() + outputs["d_emb"].square().mean()).backward()
                # Checkpoint recomputation must not expose a pair chunk's weights.
                torch.testing.assert_close(model.pooling_layer.last_weights, expected)

    def test_scores_are_invariant_to_chunk_size_protein_order_and_padding(self):
        fingerprints, mask, proteins = self.inputs()
        order = torch.tensor([2, 0, 1])
        changed_padding = fingerprints.clone()
        changed_padding[~mask] = 10000
        for pooling in ("f2r", "cross_attention"):
            with self.subTest(pooling=pooling), torch.no_grad():
                model = self.make_model(pooling)
                expected = model(fingerprints, mask, proteins)["pairwise_binding"]
                for chunk_size in (1, 4, 20):
                    model.pairwise_attention_chunk_size = chunk_size
                    actual = model(changed_padding, mask, proteins[order])["pairwise_binding"]
                    torch.testing.assert_close(actual, expected[:, order], rtol=1e-9, atol=1e-10)

    def test_pairwise_gradients_match_independent_pairs_with_and_without_checkpoint(self):
        weights = torch.tensor(
            [[0.2, -0.7, 0.1], [0.4, 0.3, -0.6], [-0.9, 0.5, 0.8]], dtype=torch.float64
        )
        for pooling in ("f2r", "cross_attention"):
            for use_checkpoint in (False, True):
                with self.subTest(pooling=pooling, checkpoint=use_checkpoint):
                    fingerprints, mask, proteins = self.inputs()
                    fingerprints.requires_grad_()
                    proteins.requires_grad_()
                    model = self.make_model(pooling, checkpoint=use_checkpoint)
                    outputs = model(fingerprints, mask, proteins)
                    loss = (weights * outputs["pairwise_binding"]).sum()
                    loss = loss + 0.1 * (
                        outputs["d_emb"].square().mean() + outputs["pooled_drug_emb"].square().mean()
                    )
                    loss.backward()

                    reference = self.make_model(pooling)
                    ref_fingerprints = fingerprints.detach().clone().requires_grad_()
                    ref_proteins = proteins.detach().clone().requires_grad_()
                    terms = []
                    for i in range(3):
                        for j in range(3):
                            pair = reference(
                                ref_fingerprints[i:i + 1], mask[i:i + 1], ref_proteins[j:j + 1]
                            )
                            term = weights[i, j] * pair["binding"].sum()
                            if i == j:
                                term = term + 0.1 / 3 * (
                                    pair["d_emb"].square().mean() + pair["pooled_drug_emb"].square().mean()
                                )
                            terms.append(term)
                    torch.stack(terms).sum().backward()

                    torch.testing.assert_close(fingerprints.grad, ref_fingerprints.grad, rtol=1e-8, atol=1e-10)
                    torch.testing.assert_close(proteins.grad, ref_proteins.grad, rtol=1e-8, atol=1e-10)
                    reference_parameters = dict(reference.named_parameters())
                    for name, parameter in model.named_parameters():
                        expected = reference_parameters[name].grad
                        self.assertEqual(parameter.grad is None, expected is None, name)
                        if expected is not None:
                            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                            # ESM attention softmax uses float32 even in this
                            # double fixture. Shared versus repeated protein
                            # forwards accumulate its backward rounding differently.
                            torch.testing.assert_close(
                                parameter.grad, expected, rtol=1e-6, atol=1e-8,
                                msg=lambda detail: f"{name}: {detail}",
                            )
                    self.assertGreater(sum(p.grad.abs().sum().item() for p in model.pooling_layer.parameters()), 0)
                    torch.testing.assert_close(fingerprints.grad[~mask], torch.zeros_like(fingerprints.grad[~mask]))

    def test_checkpointing_preserves_training_trajectory(self):
        fingerprints, mask, proteins = self.inputs()
        for pooling in ("f2r", "cross_attention"):
            with self.subTest(pooling=pooling):
                models = [self.make_model(pooling, checkpoint=flag).train() for flag in (False, True)]
                optimizers = [torch.optim.AdamW(model.parameters(), lr=1e-3) for model in models]
                for _ in range(2):
                    losses = []
                    for model, optimizer in zip(models, optimizers):
                        optimizer.zero_grad(set_to_none=True)
                        outputs = model(fingerprints, mask, proteins)
                        loss = outputs["pairwise_binding"].square().mean() + outputs["d_emb"].square().mean()
                        loss.backward()
                        optimizer.step()
                        losses.append(loss.detach())
                    torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
                    for name, expected in models[0].state_dict().items():
                        torch.testing.assert_close(models[1].state_dict()[name], expected, rtol=0, atol=0, msg=name)

    def test_jepa_and_aligned_outputs_match_single_pair_predictions(self):
        fingerprints, mask, proteins = self.inputs()
        for pooling in ("f2r", "cross_attention"):
            for predictor in ("mlp", "xattn"):
                with self.subTest(pooling=pooling, predictor=predictor), torch.no_grad():
                    backbone = self.make_model(pooling)
                    if predictor == "mlp":
                        model = ConciseJEPAFragment(backbone, smiles_target_dim=5, jepa_hidden_dim=12)
                    else:
                        model = ConciseJEPAFragmentXAttn(
                            backbone, smiles_target_dim=5, predictor_dim=8,
                            predictor_depth=1, predictor_heads=2,
                        )
                    model = model.double().eval()
                    outputs = model(proteins, fingerprints, mask)
                    pairs = [
                        model(proteins[i:i + 1], fingerprints[i:i + 1], mask[i:i + 1])
                        for i in range(3)
                    ]
                    for name in ("binding", "jepa_pred", "pooled_drug_emb", "drug_features", "protein_features"):
                        expected = torch.cat([pair[name] for pair in pairs])
                        torch.testing.assert_close(outputs[name], expected, rtol=1e-9, atol=1e-10, msg=name)


if __name__ == "__main__":
    unittest.main()
