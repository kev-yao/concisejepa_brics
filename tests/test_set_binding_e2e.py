"""End-to-end means binding BCE updates parameters before discrete FSQ too."""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from hydra.utils import instantiate
from omegaconf import open_dict
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import main as runner
from tests.test_brics_dual_view import synthetic_batch
from tests.test_brics_hydra_integration import compose_experiment
from tests.test_secondary_binding import small_config as secondary_config


def small_config():
    cfg = secondary_config()
    new = compose_experiment("brics_f2r_set_e2e_debug")
    with open_dict(cfg.model):
        cfg.model._target_ = new.model._target_
        cfg.model.concise_fragment._target_ = new.model.concise_fragment._target_
        cfg.model.concise_fragment.set_head = dict(model_dim=8, num_heads=2, num_blocks=1, dropout=0.)
    cfg.model.whole_binding_weight = .05
    cfg.task.binding_bce_weight = 1.
    cfg.evaluation.run_test = False
    return cfg


class SetBindingEndToEndTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.cfg = small_config()
        self.batch = synthetic_batch()

    def test_primary_bce_updates_head_and_backbone_through_discrete_fsq(self):
        self.cfg.model.concise_fragment.use_pairwise_attention_checkpoint = True
        model = instantiate(self.cfg.model).train()
        task = instantiate(self.cfg.task, model=model, experiment_config=self.cfg)
        optimizer = task.configure_optimizers()
        self.assertEqual({id(p) for p in model.parameters()},
                         {id(p) for group in optimizer.param_groups for p in group["params"]})
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        out = model(*self.batch[:4])
        self.assertEqual(out["frag_codes"].dtype, torch.long)
        self.assertTrue(((out["frag_codes"] >= 0) & (out["frag_codes"] < 32)).all())
        encoder = model.concise.d_encoder
        self.assertTrue(encoder.residualfsqs[0].discretize)
        encoded = encoder(self.batch[1].flatten(0, 1))
        # Forward still lies on the finite scalar grid and uses real code IDs.
        lattice = encoded["points"] * 16
        torch.testing.assert_close(lattice, lattice.round(), rtol=0, atol=1e-6)
        torch.testing.assert_close(encoded["emb"], encoder.embed(encoded["codes"]))
        # Isolate binding: no JEPA or auxiliary term can mask a detached backbone.
        loss = F.binary_cross_entropy(out["fragment_binding"], self.batch[5])
        loss.backward()
        modules = {
            "pre_fsq": encoder.pre_transform,
            "fsq_in_projection": encoder.residualfsqs[0].in_proj,
            "fsq_out_projection": encoder.residualfsqs[0].out_proj,
            "f2r": model.concise.pooling_layer,
            "drug_projection": model.concise.d_project,
            "protein_projection": model.concise.r_project,
            "protein_attention": model.concise.r_to_r_attention,
            "cross_attention": model.concise.r_to_d_attention,
            "set_head": model.concise.set_head,
        }
        before = {}
        for name, module in modules.items():
            params = list(module.parameters())
            grads = [p.grad for p in params if p.grad is not None]
            self.assertTrue(grads, name)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads), name)
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0, name)
            before[name] = [p.detach().clone() for p in params]
        # Primary BCE does not supervise the legacy whole scorer or JEPA head.
        self.assertTrue(all(p.grad is None for p in model.concise.final.parameters()))
        self.assertTrue(all(p.grad is None for p in model.jepa_predictor.parameters()))
        optimizer.step()
        for name, module in modules.items():
            self.assertTrue(any(not torch.equal(old, p) for old, p in zip(before[name], module.parameters())), name)

    def test_branch_separation_and_legacy_whole_parity(self):
        model = instantiate(self.cfg.model).eval()
        baseline = instantiate(secondary_config().model).eval()
        baseline.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith("concise.set_head.")}, strict=True)
        p, f, mask, w = self.batch[:4]
        with torch.no_grad():
            expected = model(p, f, mask, w)
            legacy = baseline(p, f, mask, w)
            for key in ("whole_binding", "whole_similarity_cosines", "jepa_pred", "drug_features", "protein_features"):
                torch.testing.assert_close(expected[key], legacy[key], rtol=0, atol=0)
            changed = model(p, f, mask, -10 * w)
            for key in ("fragment_binding", "fragment_similarity_cosines", "jepa_pred", "frag_codes"):
                torch.testing.assert_close(expected[key], changed[key], rtol=0, atol=0)
            self.assertGreater((expected["whole_binding"] - changed["whole_binding"]).abs().sum(), 0)
            model.concise.set_head.readout[-1].bias.add_(3.)
            changed_head = model(p, f, mask, w)
            torch.testing.assert_close(expected["whole_binding"], changed_head["whole_binding"], rtol=0, atol=0)
            self.assertGreater((expected["fragment_binding"] - changed_head["fragment_binding"]).abs().sum(), 0)
            torch.testing.assert_close(expected["binding"], .95 * expected["fragment_binding"] + .05 * expected["whole_binding"])
            for view in ("", "fragment_", "whole_"):
                torch.testing.assert_close(expected[f"{view}binding"], expected[f"{view}similarity_cosines"].diagonal())

    def test_candidate_pairs_and_checkpoint_recomputation(self):
        base = instantiate(self.cfg.model).double().eval()
        p, f, mask, w = self.batch[:4]
        p, f, w = p.double(), f.double(), w.double()
        with torch.no_grad():
            expected = base(p, f, mask, w)
            independent = torch.stack([
                base(p[j:j+1], f[i:i+1], mask[i:i+1], w[i:i+1])["fragment_binding"][0]
                for i in range(4) for j in range(4)
            ]).reshape(4, 4)
            torch.testing.assert_close(expected["fragment_similarity_cosines"], independent, rtol=1e-7, atol=1e-9)
            order = torch.tensor([2, 0, 3, 1])
            for chunk in (1, 5, 100):
                base.concise.pairwise_attention_chunk_size = chunk
                changed = base(p[order], f, mask, w)
                torch.testing.assert_close(changed["fragment_similarity_cosines"], independent[:, order], rtol=1e-7, atol=1e-9)
        base.concise.pairwise_attention_chunk_size = 5
        # Include training dropout in checkpoint equivalence (RNG preservation).
        base.train()
        for module in base.concise.set_head.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = .2
        checked = copy.deepcopy(base)
        checked.concise.use_pairwise_attention_checkpoint = True
        values = []
        for model in (base, checked):
            torch.manual_seed(123)
            out = model(p, f, mask, w)
            values.append(out)
            weights = model.concise.pooling_layer.last_weights.clone()
            out["binding"].square().mean().backward()
            torch.testing.assert_close(model.concise.pooling_layer.last_weights, weights)
        torch.testing.assert_close(values[0]["binding"], values[1]["binding"], rtol=0, atol=0)
        for (name, a), (_, b) in zip(base.named_parameters(), checked.named_parameters()):
            self.assertEqual(a.grad is None, b.grad is None, name)
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0, msg=name)

    def test_full_objective_and_checkpoint_roundtrip(self):
        task = instantiate(self.cfg.task, model=instantiate(self.cfg.model), experiment_config=self.cfg).eval()
        loss, _ = task._forward_step(self.batch, "train")
        loss.backward()
        for module in (task.model.jepa_predictor, task.model.concise.final):
            self.assertGreater(sum(p.grad.abs().sum() for p in module.parameters() if p.grad is not None), 0)
        task.configure_optimizers().step()
        restored = instantiate(self.cfg.task, model=instantiate(self.cfg.model), experiment_config=self.cfg).eval()
        restored.load_state_dict(task.state_dict(), strict=True)
        with torch.no_grad():
            expected, actual = task.model(*self.batch[:4]), restored.model(*self.batch[:4])
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)

    def test_warm_start_strictness_and_runner_receipt(self):
        legacy = instantiate(secondary_config().model)
        state = {f"model.{k}": v for k, v in legacy.state_dict().items()}
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.ckpt"
            torch.save({"state_dict": state}, path)
            model = instantiate(self.cfg.model)
            head_before = copy.deepcopy(model.concise.set_head.state_dict())
            receipt = model.initialize_backbone(path)
            self.assertEqual(len(receipt["sha256"]), 64)
            for key, value in legacy.state_dict().items():
                torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
            for key, value in head_before.items():
                torch.testing.assert_close(model.concise.set_head.state_dict()[key], value, rtol=0, atol=0)
            for bad_state in ({}, {**state, "model.unexpected": torch.zeros(1)},
                              {**state, "model.logit_scale": torch.zeros(2)}):
                torch.save({"state_dict": bad_state}, path)
                with self.assertRaisesRegex(ValueError, "Backbone checkpoint"):
                    model.initialize_backbone(path)
            torch.save({"state_dict": state}, path)
            cfg = self.cfg
            cfg.runtime.run_dir = tmp
            with open_dict(cfg):
                cfg.initialization = {"backbone_checkpoint": str(path)}
            trainer = mock.Mock()
            with mock.patch.object(runner, "_prepare_runtime_config"), \
                 mock.patch.object(runner, "_write_run_manifest"), \
                 mock.patch.object(runner, "_instantiate_callbacks", return_value=([], {})), \
                 mock.patch.object(runner, "instantiate", side_effect=[model, mock.Mock(), mock.Mock(), False, trainer]):
                runner.main.__wrapped__(cfg)
            trainer.fit.assert_called_once()
            trainer.test.assert_not_called()
            self.assertEqual(json.loads((Path(tmp) / "initialization.json").read_text()), receipt)

    def test_real_lightning_fit_and_saved_winner(self):
        class Data(pl.LightningDataModule):
            def __init__(self, batch):
                super().__init__()
                self.batch, self.stages = batch, []

            def setup(self, stage=None):
                self.stages.append(stage)
                if stage == "test":
                    raise AssertionError("Candidate fit accessed test")

            def train_dataloader(self):
                return DataLoader([self.batch, self.batch], batch_size=None)

            def val_dataloader(self):
                return DataLoader([self.batch], batch_size=None)

        with TemporaryDirectory() as tmp:
            cfg = self.cfg
            cfg.run.output_root = tmp
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.enable_progress_bar = False
            cfg.callbacks.progress_bar.enabled = False
            cfg.model.concise_fragment.use_pairwise_attention_checkpoint = True
            data, tasks = Data(self.batch), []

            def build(component, **kwargs):
                if component is cfg.data:
                    return data
                if component is cfg.logger:
                    return False
                built = instantiate(component, **kwargs)
                if component is cfg.task:
                    tasks.append(built)
                return built

            with mock.patch.object(runner, "instantiate", side_effect=build):
                runner.main.__wrapped__(cfg)
            self.assertEqual(data.stages, ["fit"])
            run = Path(cfg.runtime.run_dir)
            self.assertFalse((run / "test_metrics.json").exists())
            winners = list((run / "checkpoints").glob("epoch=*.ckpt"))
            self.assertEqual(len(winners), 1)
            restored = type(tasks[0]).load_from_checkpoint(
                winners[0], model=instantiate(cfg.model), experiment_config=cfg, weights_only=False
            ).eval()
            tasks[0].eval()
            with torch.no_grad():
                expected, actual = tasks[0].model(*self.batch[:4]), restored.model(*self.batch[:4])
            for key in expected:
                torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
            self.assertTrue(restored.model.concise.d_encoder.residualfsqs[0].discretize)

    def test_opt_in_config_and_continuous_rejection(self):
        cfg = compose_experiment("brics_f2r_set_e2e")
        self.assertFalse(cfg.evaluation.run_test)
        self.assertEqual(cfg.model.concise_fragment.drug_quantizer.type, "fsq")
        self.assertEqual(cfg.model.concise_fragment.pooling, "f2r")
        self.assertEqual(cfg.model.whole_binding_weight, .05)
        self.assertEqual(cfg.task.binding_bce_weight, 1.)
        self.assertEqual(cfg.callbacks.best_checkpoint.monitor, "val/fused_pooled_ap")
        self.cfg.model.concise_fragment.drug_quantizer.type = "continuous_fsq"
        with self.assertRaisesRegex(Exception, "discrete fsq"):
            instantiate(self.cfg.model)


if __name__ == "__main__":
    unittest.main()
