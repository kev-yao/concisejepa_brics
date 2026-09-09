"""Aligned supervised objectives preserve legacy defaults and branch boundaries."""
import copy
from unittest import TestCase, mock

import torch
import torch.nn.functional as F
from hydra.utils import instantiate

from tests.test_brics_dual_view import synthetic_batch
from tests.test_secondary_binding import small_config


class SecondaryObjectiveTests(TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.cfg = small_config()
        self.batch = synthetic_batch()

    def task(self, **weights):
        for key, value in weights.items():
            self.cfg.task[key] = value
        return instantiate(self.cfg.task, model=instantiate(self.cfg.model), experiment_config=self.cfg).eval()

    def test_weight_validation_and_saved_hyperparameters(self):
        names = ('binding_bce_weight', 'binding_contrastive_weight', 'binding_negative_weight', 'jepa_loss_weight')
        for name in names:
            for value in (-1., float('nan'), float('inf')):
                with self.subTest(name=name, value=value):
                    self.cfg = small_config()
                    with self.assertRaisesRegex(Exception, name):
                        self.task(**{name: value})
        self.cfg = small_config()
        task = self.task(**dict.fromkeys(names, .3))
        for name in names:
            self.assertEqual(getattr(task, name), .3)
            self.assertEqual(task.hparams[name], .3)

    def test_weighted_loss_arithmetic_and_logs(self):
        task = self.task(binding_bce_weight=1.7, binding_contrastive_weight=.2,
                         binding_negative_weight=.4, jepa_loss_weight=.1)
        task.negative_diagonal_weight = .8
        logs = {}
        task.log = lambda key, value, **kwargs: logs.update({key: value})
        loss, out = task._forward_step(self.batch, 'train')
        label, smiles, sequences = self.batch[5:]
        bces, ces, negatives = [], [], []
        for view in ('fragment', 'whole'):
            bces.append(F.binary_cross_entropy(out[f'{view}_binding'], label))
            ces.append(task._contrastive_dti_loss(out[f'{view}_similarity_logits'], label, smiles, sequences))
            negatives.append(task._negative_diagonal_cosine_loss(out[f'{view}_similarity_cosines'], label, smiles, sequences))
        def combine(values):
            return values[0] + .25 * values[1]
        expected = 1.7 * combine(bces) + .2 * combine(ces) + .4 * .8 * combine(negatives)
        expected = expected + .1 * F.mse_loss(out['jepa_pred'], self.batch[4])
        torch.testing.assert_close(loss, expected)
        for key, expected in [('bce', 1.7 * combine(bces)), ('contrastive', .2 * combine(ces)),
                              ('neg_diag', .4 * .8 * combine(negatives))]:
            torch.testing.assert_close(logs[f'train/loss_{key}_weighted'], expected)
        torch.testing.assert_close(logs['train/loss_jepa_weighted'], .1 * logs['train/loss_jepa'])
        for view, bce, ce, neg in zip(('fragment', 'whole'), bces, ces, negatives):
            torch.testing.assert_close(logs[f'train/loss_{view}_bce'], bce)
            torch.testing.assert_close(logs[f'train/loss_{view}_dti'], ce)
            torch.testing.assert_close(logs[f'train/loss_{view}_contrastive'], ce)
            torch.testing.assert_close(logs[f'train/loss_{view}_neg_diag'], neg)

    def test_bce_only_uses_aligned_labels_and_has_negative_signal(self):
        task = self.task(binding_bce_weight=1., binding_contrastive_weight=0., binding_negative_weight=0.)
        for labels in ([0.], [1.], [0., 0.], [1., 1.], [0., 1.]):
            with self.subTest(labels=labels):
                target = torch.tensor(labels)
                p = torch.full((len(labels),), .4, requires_grad=True)
                matrix = p[:, None].expand(-1, len(labels))
                out = {f'{view}_{key}': value for view in ('fragment', 'whole')
                       for key, value in [('binding', p), ('similarity_logits', 14 * matrix), ('similarity_cosines', matrix)]}
                dti, neg = task._binding_losses(out, target, [], [], 'train')
                torch.testing.assert_close(dti, 1.25 * F.binary_cross_entropy(p, target))
                torch.testing.assert_close(neg, torch.tensor(0.))
                dti.backward()
                self.assertTrue(torch.isfinite(p.grad).all())
                self.assertTrue((p.grad[target == 0] > 0).all())
                self.assertTrue((p.grad[target == 1] < 0).all())
        # Exact endpoints are finite in float32 BCE, including incorrect labels.
        p = torch.tensor([0., 1., 1e-7, 1 - 1e-7], requires_grad=True)
        target = torch.tensor([1., 0., 0., 1.])
        matrix = p[:, None].expand(-1, 4)
        out = {f'{view}_{key}': value for view in ('fragment', 'whole')
               for key, value in [('binding', p), ('similarity_logits', matrix), ('similarity_cosines', matrix)]}
        loss, _ = task._binding_losses(out, target, [], [], 'train')
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_disabled_bce_is_not_evaluated(self):
        task = self.task()
        with mock.patch('torch.nn.functional.binary_cross_entropy', side_effect=AssertionError('inactive BCE called')):
            task._forward_step(self.batch, 'train')

    def test_bce_branch_gradients_and_weighted_jepa_isolation(self):
        self.cfg.model.concise_fragment.drug_quantizer.type = 'continuous_fsq'
        task = self.task(binding_bce_weight=1., binding_contrastive_weight=0., binding_negative_weight=0., jepa_loss_weight=.1)
        p, f, mask, whole = self.batch[:4]
        whole = whole.clone().requires_grad_()
        out = task.model(p, f, mask, whole)
        shared = task.model.concise.d_encoder.pre_transform[1].weight
        pool = list(task.model.concise.pooling_layer.parameters())
        whole_loss = F.binary_cross_entropy(out['whole_binding'], self.batch[5])
        grads = torch.autograd.grad(whole_loss, [shared, *pool], retain_graph=True, allow_unused=True)
        self.assertGreater(grads[0].abs().sum(), 0)
        self.assertTrue(all(g is None for g in grads[1:]))
        fragment_loss = F.binary_cross_entropy(out['fragment_binding'], self.batch[5])
        gradients = torch.autograd.grad(fragment_loss, pool, retain_graph=True)
        self.assertGreater(sum(g.abs().sum() for g in gradients), 0)
        jepa = .1 * F.mse_loss(out['jepa_pred'], self.batch[4])
        self.assertIsNone(torch.autograd.grad(jepa, whole, allow_unused=True)[0])

    def test_default_and_explicit_weights_have_identical_updates(self):
        task = self.task()
        explicit = copy.deepcopy(task)
        explicit.binding_bce_weight = 0.
        explicit.binding_contrastive_weight = explicit.binding_negative_weight = explicit.jepa_loss_weight = 1.
        optimizers = [t.configure_optimizers() for t in (task, explicit)]
        for seed in (10, 11):
            results = []
            for t, opt in zip((task, explicit), optimizers):
                t.train()
                torch.manual_seed(seed)
                opt.zero_grad(set_to_none=True)
                loss, _ = t._forward_step(self.batch, 'train')
                loss.backward()
                results.append((loss.detach(), {n: p.grad.clone() for n, p in t.named_parameters() if p.grad is not None}))
                opt.step()
            torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
            for name in results[0][1]:
                torch.testing.assert_close(results[0][1][name], results[1][1][name], rtol=0, atol=0)
            for name in task.state_dict():
                torch.testing.assert_close(task.state_dict()[name], explicit.state_dict()[name], rtol=0, atol=0)
