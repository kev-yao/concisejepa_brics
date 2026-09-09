"""Separate whole-molecule binding must not leak into fragment pooling/JEPA."""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import binary_average_precision, binary_auroc
from hydra.utils import instantiate
from omegaconf import open_dict

from concisejepa.callbacks import EpochMetricsWriter
from concisejepa.models.drug_decoder import DrugEncoder
from tests.test_brics_hydra_integration import compose_experiment
from tests.test_brics_dual_view import synthetic_batch


def small_config():
    cfg = compose_experiment("brics_f2r_whole_debug")
    b = cfg.model.concise_fragment
    b.drug_layers = [[32, 32, 32]]
    b.ligand_dim = 16
    b.residue_dim = 12
    b.drug_dim = 8
    b.proj_dim = 8
    b.nheads = 2
    b.pairwise_attention_chunk_size = 5
    b.use_pairwise_attention_checkpoint = False
    cfg.model.smiles_target_dim = 8
    cfg.model.jepa_hidden_dim = 16
    cfg.data.fingerprint_length = 16
    return cfg


class SecondaryBindingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.cfg = small_config()
        self.batch = synthetic_batch()

    def test_forward_separation_and_shared_encoder(self):
        for quantizer in ("fsq", "continuous_fsq"):
            with self.subTest(quantizer=quantizer):
                self.cfg.model.concise_fragment.drug_quantizer.type = quantizer
                model = instantiate(self.cfg.model).eval()
                self.assertEqual(sum(isinstance(m, DrugEncoder) for m in model.modules()), 1)
                p, f, mask, whole = self.batch[:4]
                out = model(p, f, mask, whole)
                changed_whole = model(p, f, mask, -10 * whole)
                for key in ("fragment_binding", "fragment_similarity_logits", "jepa_pred",
                            "pooled_drug_emb", "drug_features", "protein_features", "frag_codes"):
                    torch.testing.assert_close(out[key], changed_whole[key], rtol=0, atol=0)
                self.assertGreater((out["whole_binding"] - changed_whole["whole_binding"]).abs().max(), 0)
                self.assertGreater((out["binding"] - changed_whole["binding"]).abs().max(), 0)
                changed_frag = model(p, -10 * f, mask, whole)
                torch.testing.assert_close(out["whole_binding"], changed_frag["whole_binding"], rtol=0, atol=0)
                for suffix in ("binding", "similarity_cosines", "similarity_logits"):
                    torch.testing.assert_close(out[suffix], .75 * out[f"fragment_{suffix}"] + .25 * out[f"whole_{suffix}"])
                torch.testing.assert_close(out["binding"], out["similarity_cosines"].diagonal())

    def test_gradient_boundaries(self):
        self.cfg.model.concise_fragment.drug_quantizer.type = "continuous_fsq"
        model = instantiate(self.cfg.model).eval()
        p, f, mask, w = self.batch[:4]
        f, w = f.clone().requires_grad_(), w.clone().requires_grad_()
        outputs = model(p, f, mask, w)
        self.assertIsNone(torch.autograd.grad(outputs["jepa_pred"].sum(), w, allow_unused=True, retain_graph=True)[0])
        whole_grad = torch.autograd.grad(outputs["whole_binding"].sum(), w, retain_graph=True)[0]
        self.assertGreater(whole_grad.abs().sum(), 0)
        pool_params = list(model.concise.pooling_layer.parameters())
        shared = model.concise.d_encoder.pre_transform[1].weight
        task = instantiate(self.cfg.task, model=model, experiment_config=self.cfg)
        label, smiles, seqs = self.batch[5:]
        def objective(view):
            return task._contrastive_dti_loss(outputs[f"{view}_similarity_logits"], label, smiles, seqs) + task._negative_diagonal_cosine_loss(outputs[f"{view}_similarity_cosines"], label, smiles, seqs)
        grads = torch.autograd.grad(objective("whole"), [shared, *pool_params], allow_unused=True, retain_graph=True)
        self.assertGreater(grads[0].abs().sum(), 0)
        self.assertTrue(all(g is None for g in grads[1:]))
        grads = torch.autograd.grad(objective("fragment"), pool_params)
        self.assertGreater(sum(g.abs().sum() for g in grads), 0)

    def test_pair_matrices_permutations_chunks_and_checkpoint_gradients(self):
        p, f, mask, w = self.batch[:4]
        order = torch.tensor([2, 0, 3, 1])
        base = instantiate(self.cfg.model).double().eval()
        p, f, w = p.double(), f.double(), w.double()
        with torch.no_grad():
            expected = base(p, f, mask, w)
            singles = [base(p[j:j+1], f[i:i+1], mask[i:i+1], w[i:i+1])
                       for i in range(4) for j in range(4)]
            for view in ("fragment_", "whole_", ""):
                scores = expected[f"{view}similarity_cosines"]
                independent = torch.stack([x[f"{view}binding"][0] for x in singles]).reshape(4, 4)
                torch.testing.assert_close(scores, independent, rtol=1e-8, atol=1e-9)
            for chunk in (1, 5, 100):
                base.concise.pairwise_attention_chunk_size = chunk
                actual = base(p[order], f, mask, w)
                for view in ("fragment_", "whole_", ""):
                    torch.testing.assert_close(actual[f"{view}similarity_cosines"], expected[f"{view}similarity_cosines"][:, order], rtol=1e-8, atol=1e-9)
        base.concise.pairwise_attention_chunk_size = 5
        checked = copy.deepcopy(base)
        checked.concise.use_pairwise_attention_checkpoint = True
        for model in (base, checked):
            out = model(p, f, mask, w)
            weights = model.concise.pooling_layer.last_weights.clone()
            (out["similarity_logits"].square().mean() + out["jepa_pred"].square().mean()).backward()
            torch.testing.assert_close(model.concise.pooling_layer.last_weights, weights)
        for (name, a), (_, b) in zip(base.named_parameters(), checked.named_parameters()):
            self.assertEqual(a.grad is None, b.grad is None, name)
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0, msg=name)

    def test_whole_pair_gradients_match_single_pairs(self):
        model = instantiate(self.cfg.model).double().eval()
        model.concise.use_pairwise_attention_checkpoint = True
        reference = copy.deepcopy(model)
        reference.concise.use_pairwise_attention_checkpoint = False
        p, f, mask, w = self.batch[:4]
        p, f, w = p.double(), f.double(), w.double()
        p0, w0 = p.clone().requires_grad_(), w.clone().requires_grad_()
        p1, w1 = p.clone().requires_grad_(), w.clone().requires_grad_()
        weights = torch.arange(16, dtype=torch.float64).reshape(4, 4) / 10 - .7
        (model(p0, f, mask, w0)["whole_similarity_cosines"] * weights).sum().backward()
        terms = [weights[i, j] * reference(p1[j:j+1], f[i:i+1], mask[i:i+1], w1[i:i+1])["whole_binding"][0]
                 for i in range(4) for j in range(4)]
        torch.stack(terms).sum().backward()
        torch.testing.assert_close(p0.grad, p1.grad, rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(w0.grad, w1.grad, rtol=1e-6, atol=1e-8)
        for (name, a), (_, b) in zip(model.named_parameters(), reference.named_parameters()):
            self.assertEqual(a.grad is None, b.grad is None, name)
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=1e-6, atol=1e-8, msg=name)

    def test_endpoints_invalid_weights_and_missing_input(self):
        for alpha in (0., 1.):
            self.cfg.model.whole_binding_weight = alpha
            model = instantiate(self.cfg.model).eval()
            out = model(*self.batch[:4])
            name = "fragment_binding" if alpha == 0 else "whole_binding"
            torch.testing.assert_close(out["binding"], out[name], rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "whole"):
                model(*self.batch[:3])
        for value in (-.1, 1.1, float("nan"), float("inf")):
            self.cfg.model.whole_binding_weight = value
            with self.assertRaisesRegex(Exception, "whole_binding_weight"):
                instantiate(self.cfg.model)
        self.cfg.model.whole_binding_weight = .25
        for value in (-.1, float("nan"), float("inf")):
            self.cfg.task.secondary_loss_weight = value
            with self.assertRaisesRegex(Exception, "secondary_loss_weight"):
                instantiate(self.cfg.task, model=instantiate(self.cfg.model), experiment_config=self.cfg)

    def test_loss_arithmetic_and_baseline_fragment_parity(self):
        task = instantiate(self.cfg.task, model=instantiate(self.cfg.model), experiment_config=self.cfg).eval()
        loss, out = task._forward_step(self.batch, "train")
        label, smiles, seqs = self.batch[5:]
        def binding_loss(view):
            return task._contrastive_dti_loss(out[f"{view}_similarity_logits"], label, smiles, seqs) + task.negative_diagonal_weight * task._negative_diagonal_cosine_loss(out[f"{view}_similarity_cosines"], label, smiles, seqs)
        expected = binding_loss("fragment") + .25 * binding_loss("whole") + torch.nn.functional.mse_loss(out["jepa_pred"], self.batch[4])
        torch.testing.assert_close(loss, expected)
        cfg = copy.deepcopy(self.cfg)
        with open_dict(cfg.model):
            cfg.model._target_ = "concisejepa.models.fragment.ConciseJEPAFragment"
            del cfg.model.whole_binding_weight
        baseline = instantiate(cfg.model).eval()
        baseline.load_state_dict(task.model.state_dict(), strict=True)
        reference = baseline(*self.batch[:3])
        for key in reference:
            equivalent = f"fragment_{key}" if key in ("binding", "similarity_cosines", "similarity_logits") else key
            torch.testing.assert_close(out[equivalent], reference[key], rtol=0, atol=0, msg=key)
        loss.backward()
        optimizer = task.configure_optimizers()
        optimizer.step()
        restored = instantiate(self.cfg.task, model=instantiate(self.cfg.model), experiment_config=self.cfg).eval()
        restored.load_state_dict(task.state_dict(), strict=True)
        for key, value in task.model(*self.batch[:4]).items():
            torch.testing.assert_close(value, restored.model(*self.batch[:4])[key], rtol=0, atol=0)

    def test_zero_secondary_supervision_preserves_fragment_training(self):
        self.cfg.model.whole_binding_weight = 0.
        self.cfg.task.secondary_loss_weight = 0.
        new_task = instantiate(self.cfg.task, model=instantiate(self.cfg.model), experiment_config=self.cfg)
        cfg = copy.deepcopy(self.cfg)
        with open_dict(cfg.model), open_dict(cfg.task):
            cfg.model._target_ = "concisejepa.models.fragment.ConciseJEPAFragment"
            del cfg.model.whole_binding_weight
            cfg.task._target_ = "concisejepa.lightning_modules.lit_fragment.LitFragment"
            for key in ("secondary_loss_weight", "binding_bce_weight", "binding_contrastive_weight",
                        "binding_negative_weight", "jepa_loss_weight"):
                del cfg.task[key]
        old_task = instantiate(cfg.task, model=instantiate(cfg.model), experiment_config=cfg)
        old_task.load_state_dict(new_task.state_dict(), strict=True)
        old_batch = (*self.batch[:3], *self.batch[4:])
        optimizers = [task.configure_optimizers() for task in (old_task, new_task)]
        for step in range(2):
            results = []
            for task, batch, optimizer in zip((old_task, new_task), (old_batch, self.batch), optimizers):
                optimizer.zero_grad(set_to_none=True)
                torch.manual_seed(200 + step)
                loss, outputs = task._forward_step(batch, "train")
                loss.backward()
                grads = {name: p.grad.clone() for name, p in task.named_parameters() if p.grad is not None}
                optimizer.step()
                results.append((loss, outputs, grads))
            torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
            for key, value in results[0][1].items():
                torch.testing.assert_close(value, results[1][1][key], rtol=0, atol=0, msg=key)
            for name, grad in results[0][2].items():
                torch.testing.assert_close(grad, results[1][2][name], rtol=0, atol=0, msg=name)
            for name, value in old_task.state_dict().items():
                torch.testing.assert_close(value, new_task.state_dict()[name], rtol=0, atol=0, msg=name)

    def test_bundle_collation_and_count_cache_guard(self):
        cfg = self.cfg
        self.assertTrue(cfg.data.require_count_values)
        self.assertEqual(cfg.model.concise_fragment.pooling, "f2r")
        self.assertEqual(cfg.callbacks.best_checkpoint.monitor, "val/fused_pooled_ap")
        self.assertEqual(cfg.trainer.limit_train_batches, 2)
        full = compose_experiment("brics_f2r_whole")
        self.assertEqual(full.trainer.max_epochs, 30)
        self.assertFalse(full.chem_supervision.enabled)
        self.assertFalse(full.group_supervision.enabled)
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            for split in ("train", "val", "test"):
                pd.DataFrame({"Target Sequence": ["P"], "SMILES": ["CCO"], "Label": [1]}).to_csv(root / f"{split}.csv", index=False)
            torch.save({"P": torch.randn(5, 12)}, root / "raygun_embeddings.pt")
            fragment = torch.arange(48).reshape(3, 16).float()
            torch.save({"CCO": fragment}, root / "fragment_fps_r4_2048.pt")
            torch.save({"CCO": torch.zeros(8)}, root / "coati_embeddings.pt")
            meta = {"fingerprint_kind": "ecfp-count:4", "fingerprint_length": 16}
            (root / "morgan_embeddings.pt.meta.json").write_text(json.dumps(meta))
            meta.update(type="fragment_fps", max_frags=16)
            (root / "fragment_fps_r4_2048.pt.meta.json").write_text(json.dumps(meta))
            cfg.data.data_root = str(root)
            torch.save({"CCO": torch.ones(16)}, root / "morgan_embeddings.pt")
            with self.assertRaisesRegex(Exception, "binary"):
                instantiate(cfg.data)
            whole = torch.arange(16).float()
            torch.save({"CCO": whole}, root / "morgan_embeddings.pt")
            dm = instantiate(cfg.data)
            batch = dm.collator([("P", "CCO", 1.)])
            self.assertEqual(len(batch), 8)
            torch.testing.assert_close(batch[1][0], fragment)
            torch.testing.assert_close(batch[3][0], whole)
            self.assertEqual(batch[4].shape, (1, 8))
            self.assertEqual(batch[5].tolist(), [1.])
        task = instantiate(cfg.task, model=instantiate(cfg.model), experiment_config=cfg)
        with self.assertRaisesRegex(ValueError, "8-item"):
            task._unpack_batch(self.batch[:7])

    def test_epoch_metrics_reset_artifacts_and_lightning_checkpoint(self):
        task = instantiate(self.cfg.task, model=instantiate(self.cfg.model), experiment_config=self.cfg)
        batch2 = list(self.batch)
        batch2[1] = -5 * batch2[1]
        batch2[3] = -5 * batch2[3]
        batch2[5] = 1 - batch2[5]
        loader = DataLoader([self.batch, tuple(batch2)], batch_size=None)
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=2,
                                 logger=False, enable_checkpointing=False,
                                 enable_progress_bar=False, enable_model_summary=False,
                                 num_sanity_val_steps=0, callbacks=[EpochMetricsWriter(tmp)])
            trainer.fit(task, train_dataloaders=loader, val_dataloaders=loader)
            records = json.loads((Path(tmp) / "epoch_metrics.json").read_text())
            self.assertEqual(len(records), 2)
            for record in records:
                for stage in ("train", "val"):
                    for view in ("fragment", "whole", "fused"):
                        for name in ("ap", "auroc"):
                            self.assertIn(f"{stage}/{view}_pooled_{name}", record)
            # Repeated validation with different data checks reset, not just aggregation.
            for batches in ([self.batch, tuple(batch2)], [tuple(batch2)]):
                task.eval()
                with torch.no_grad():
                    outputs = [task.model(*b[:4]) for b in batches]
                labels = torch.cat([b[5] for b in batches]).int()
                actual = trainer.validate(task, dataloaders=DataLoader(batches, batch_size=None), verbose=False)[0]
                for view in ("fragment", "whole", "fused"):
                    key = "binding" if view == "fused" else f"{view}_binding"
                    predictions = torch.cat([o[key] for o in outputs])
                    for name, metric in (("ap", binary_average_precision), ("auroc", binary_auroc)):
                        self.assertAlmostEqual(actual[f"val/{view}_pooled_{name}"], metric(predictions, labels).item(), places=6)
            trainer.test(task, dataloaders=loader, verbose=False)
            saved = json.loads((Path(tmp) / "test_metrics.json").read_text())
            self.assertIn("test/whole_pooled_ap", saved)
            checkpoint = Path(tmp) / "roundtrip.ckpt"
            trainer.save_checkpoint(checkpoint)
            restored = type(task).load_from_checkpoint(checkpoint, model=instantiate(self.cfg.model), experiment_config=self.cfg, weights_only=False).eval()
            task.eval()
            with torch.no_grad():
                expected = task.model(*self.batch[:4])
                actual = restored.model(*self.batch[:4])
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
