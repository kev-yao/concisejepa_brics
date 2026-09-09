"""Repeated entities must not erase drug/protein binding discrimination."""

import unittest

import torch
import torch.nn.functional as F
from hydra.utils import instantiate

from concisejepa.lightning_modules import LitConciseJEPA
from tests.test_brics_hydra_integration import small_brics_config
from tests.test_training_refactor_parity import _small_baseline_config


class PositivePairMaskTests(unittest.TestCase):
    @staticmethod
    def make_task(kind):
        torch.manual_seed(42)
        if kind == "fragment":
            cfg = small_brics_config()
            return instantiate(cfg.task, model=instantiate(cfg.model), experiment_config=cfg).eval()
        cfg = _small_baseline_config()
        cfg.chem_supervision.enabled = False
        cfg.group_supervision.enabled = False
        cfg.forward_capture.enabled = False
        cfg.coati_validation.enabled = False
        cfg.synthetic_validation.enabled = False
        return LitConciseJEPA(cfg).eval()

    def test_shared_receptor_mixed_labels_restore_discrimination_gradients(self):
        for kind in ("fragment", "whole_molecule"):
            with self.subTest(task=kind):
                task = self.make_task(kind)
                # The positive ligand is incorrectly ranked below the negative.
                # Both columns refer to the identical receptor, so scores tie
                # within each row but must discriminate between the two rows.
                scores = torch.tensor([0.2, 0.8], requires_grad=True)
                matrix = scores[:, None].expand(2, 2)
                labels = torch.tensor([1.0, 0.0])
                loss = task._contrastive_dti_loss(14 * matrix, labels, ["CCO", "CCN"], ["P", "P"])
                loss = loss + task._negative_diagonal_cosine_loss(
                    matrix, labels, ["CCO", "CCN"], ["P", "P"]
                )
                loss.backward()
                self.assertLess(scores.grad[0].item(), 0.0)
                self.assertGreater(scores.grad[1].item(), 0.0)
                expected = 0.5 * F.softplus(14 * (scores.detach()[1] - scores.detach()[0]))
                torch.testing.assert_close(loss.detach(), expected)

    def test_identity_masks_use_the_correct_label_axis(self):
        cases = [
            ([1, 0], ["A", "B"], ["P", "P"], [[True, True], [False, False]]),
            ([1, 0], ["A", "A"], ["P", "Q"], [[True, False], [True, False]]),
            ([1, 0], ["A", "B"], ["P", "Q"], [[True, False], [False, False]]),
            ([1, 1], ["A", "B"], ["P", "P"], [[True, True], [True, True]]),
            ([0, 0], ["A", "B"], ["P", "P"], [[False, False], [False, False]]),
            ([1, 0], [], ["P", "P"], [[True, True], [False, False]]),
            ([1, 0], ["A", "A"], [], [[True, False], [True, False]]),
            ([1, 0], [], [], [[False, False], [False, False]]),
            ([1, 0], ["A"], ["P"], [[False, False], [False, False]]),
            ([1], ["A"], ["P"], [[True]]),
            ([0], ["A"], ["P"], [[False]]),
        ]
        for kind in ("fragment", "whole_molecule"):
            task = self.make_task(kind)
            for labels, smiles, sequences, expected in cases:
                with self.subTest(task=kind, labels=labels, smiles=smiles, sequences=sequences):
                    labels = torch.tensor(labels, dtype=torch.float32)
                    actual = task._known_positive_mask(labels, smiles, sequences, len(labels), labels.device)
                    expected = torch.tensor(expected)
                    torch.testing.assert_close(actual, expected)
                    # Reordering interaction rows must reorder both mask axes.
                    permuted = task._known_positive_mask(
                        labels.flip(0), smiles[::-1], sequences[::-1], len(labels), labels.device
                    )
                    torch.testing.assert_close(permuted, expected.flip((0, 1)))

    def test_shared_drug_mixed_labels_still_discriminate_proteins(self):
        for kind in ("fragment", "whole_molecule"):
            with self.subTest(task=kind):
                task = self.make_task(kind)
                scores = torch.tensor([0.2, 0.8], requires_grad=True)
                matrix = scores[None, :].expand(2, 2)
                loss = task._contrastive_dti_loss(
                    14 * matrix, torch.tensor([1.0, 0.0]), ["CCO", "CCO"], ["P", "Q"]
                )
                loss.backward()
                self.assertLess(scores.grad[0].item(), 0.0)
                self.assertGreater(scores.grad[1].item(), 0.0)
                torch.testing.assert_close(loss.detach(), 0.5 * F.softplus(torch.tensor(8.4)))

    def test_negative_reference_keeps_same_receptor_negatives(self):
        for kind in ("fragment", "whole_molecule"):
            with self.subTest(task=kind):
                task = self.make_task(kind)
                scores = torch.tensor(
                    [[0.9, 0.9, 0.4], [0.6, 0.6, 0.2], [0.1, 0.1, 0.1]], requires_grad=True
                )
                loss = task._negative_diagonal_cosine_loss(
                    scores, torch.tensor([1.0, 0.0, 0.0]), ["A", "B", "C"], ["P", "P", "Q"]
                )
                # B-P's duplicate negative remains a reference: its row mean
                # is (0.6+0.2)/2. Average its 0.2 penalty with C's zero penalty.
                torch.testing.assert_close(loss, torch.tensor(0.1))
                loss.backward()
                torch.testing.assert_close(scores.grad[1, 0], torch.tensor(-0.25))

    def test_duplicate_positives_and_negative_only_batches_are_finite(self):
        for kind in ("fragment", "whole_molecule"):
            task = self.make_task(kind)
            for batch_size in (1, 2):
                for label in (0.0, 1.0):
                    with self.subTest(task=kind, batch_size=batch_size, label=label):
                        scores = torch.full((batch_size, batch_size), 0.5)
                        labels = torch.full((batch_size,), label)
                        smiles, sequences = ["A"] * batch_size, ["P"] * batch_size
                        loss = task._contrastive_dti_loss(scores, labels, smiles, sequences)
                        loss = loss + task._negative_diagonal_cosine_loss(scores, labels, smiles, sequences)
                        torch.testing.assert_close(loss, torch.tensor(0.0))

    def test_training_step_sends_discriminative_gradients_through_real_models(self):
        for kind in ("fragment", "whole_molecule"):
            with self.subTest(task=kind):
                task = self.make_task(kind)
                generator = torch.Generator().manual_seed(20260908)
                proteins = torch.randn(1, 5, 12, generator=generator).repeat(2, 1, 1)
                if kind == "fragment":
                    inputs = torch.randn(2, 3, 16, generator=generator)
                    mask = torch.ones(2, 3, dtype=torch.bool)
                    model_inputs = (proteins, inputs, mask)
                else:
                    inputs = torch.randn(2, 16, generator=generator)
                    model_inputs = (proteins, inputs)
                with torch.no_grad():
                    # Isolate binding gradients without changing the objective:
                    # JEPA residual and its gradient are zero at this target.
                    target = task.model(*model_inputs)["jepa_pred"].detach()
                batch = (*model_inputs, target, torch.tensor([1.0, 0.0]), ["CCO", "CCN"], ["P", "P"])
                captured = []

                def capture(_module, _args, outputs):
                    outputs["similarity_logits"].retain_grad()
                    captured.append(outputs)

                handle = task.model.register_forward_hook(capture)
                try:
                    loss = task.training_step(batch, 0)
                    loss.backward()
                finally:
                    handle.remove()
                row_gradients = captured[0]["similarity_logits"].grad.sum(dim=1)
                self.assertLess(row_gradients[0].item(), 0.0)
                self.assertGreater(row_gradients[1].item(), 0.0)
                gradients = [p.grad for p in task.model.concise.d_encoder.parameters() if p.grad is not None]
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
                self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0.0)


if __name__ == "__main__":
    unittest.main()
