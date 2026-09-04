import importlib.util
import unittest
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import binary_auroc, binary_average_precision

from test_brics_dual_view import small_config, synthetic_batch


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FixedPredictions(nn.Module):
    def forward(self, protein_embedding, **kwargs):
        pred = protein_embedding[:, 0, 0]
        return {
            "binding": pred,
            "whole_binding": pred,
            "fragment_binding": pred,
            "fragment_molecule_similarity": torch.ones_like(pred),
            "jepa_pred": torch.zeros(len(pred), 8),
        }


class AuditTests(unittest.TestCase):
    def test_conditional_metrics_and_probability_boundaries(self):
        mod = load_script("brics_conditional_metrics")
        frame = pd.DataFrame(
            {
                "protein": ["A"] * 6 + ["B"] * 6,
                "label": [0, 0, 0, 1, 1, 1] * 2,
                "mean": [0.2] * 6 + [0.8] * 6,
            }
        )
        result = mod.conditional_metrics(frame, "protein")
        self.assertEqual(result["groups"], 2)
        self.assertEqual(result["macro_auroc"], 0.5)
        self.assertEqual(result["macro_ap_lift_over_constant"], 0)
        frame.loc[3, "mean"] = 0.0
        frame.loc[4, "mean"] = 1e-8
        frame.loc[0, "mean"] = 1.0
        values = mod.probability_diagnostics(frame)
        self.assertEqual(values["exact_zero_positive_predictions"], 1)
        self.assertEqual(values["near_zero_positive_predictions"], 2)
        self.assertEqual(values["exact_one_negative_predictions"], 1)

    def test_unique_validation_weights_each_molecule_once(self):
        from concisejepa.evals.reconstruction import UniqueJEPAValidation

        cfg = small_config()
        task = instantiate(cfg.task, experiment_config=cfg, model=instantiate(cfg.model)).eval()
        batch = synthetic_batch()
        smiles = batch[6]
        dm = SimpleNamespace(
            smiles_col="SMILES",
            val_dataset=SimpleNamespace(df=pd.DataFrame({"SMILES": smiles + [smiles[0]] * 5})),
            fragment_fps={s: batch[1][i][batch[2][i]] for i, s in enumerate(smiles)},
            smiles_embeddings={s: batch[4][i] for i, s in enumerate(smiles)},
        )
        captured = {}
        task.log = lambda name, value, **kwargs: captured.update({name: value})
        UniqueJEPAValidation().on_validation_epoch_end(SimpleNamespace(sanity_checking=False, datamodule=dm), task)
        with torch.no_grad():
            expected = (task.model(*batch[:4])["jepa_pred"] - batch[4]).square().mean()
        torch.testing.assert_close(captured["val/unique_jepa_mse"], expected)

    def test_epoch_metrics_are_pooled_and_reset_between_evaluations(self):
        cfg = small_config()
        task = instantiate(cfg.task, experiment_config=cfg, model=FixedPredictions())
        b = list(synthetic_batch())
        preds = torch.tensor([0.1, 0.2, 0.8, 0.9])
        b[0][:, 0, 0] = preds
        b[5] = torch.tensor([0.0, 1.0, 0.0, 1.0])
        batches = [tuple(v[i : i + 2] for v in b) for i in (0, 2)]
        loader = DataLoader(batches, batch_size=None)
        with TemporaryDirectory() as tmp:
            trainer = pl.Trainer(
                accelerator="cpu",
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                default_root_dir=tmp,
            )
            for _ in range(2):
                metrics = trainer.validate(task, dataloaders=loader, verbose=False)[0]
                self.assertAlmostEqual(metrics["val/dti_auroc"], float(binary_auroc(preds, b[5].int())))
                self.assertAlmostEqual(metrics["val/dti_auprc"], float(binary_average_precision(preds, b[5].int())))
                self.assertEqual(task.auprc_by_stage["val_metric"]._update_count, 0)

    def test_continuous_bypass_preserves_masking_and_has_no_quantizer_gradient(self):
        cfg = small_config()
        cfg.model.molecular_representation = "continuous"
        model = instantiate(cfg.model).eval()
        batch = synthetic_batch()
        out = model(*batch[:4])
        bad = batch[1].clone()
        bad[~batch[2]] = 10000
        changed = model(batch[0], bad, batch[2], batch[3])
        torch.testing.assert_close(out["jepa_pred"], changed["jepa_pred"])
        out["jepa_pred"].square().mean().backward()
        self.assertGreater(model.drug_encoder.pre_transform[1].weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.drug_encoder.residualfsqs.parameters()))

    def test_canonical_split_deduplicates_and_excludes_conflicting_pairs(self):
        mod = load_script("prepare_brics_audit_data")
        rows = [("CCO", "A", 1), ("OCC", "A", 1), ("CCN", "B", 0), ("NCC", "B", 1)]
        rows += [("C" * i, "P", i % 2) for i in range(1, 15)]
        df = pd.DataFrame(rows, columns=["SMILES", "Target Sequence", "Label"])
        splits, excluded, stats = mod.partition_pairs({"train": df}, 17)
        allrows = pd.concat(splits.values())
        self.assertEqual(len(excluded), 2)
        self.assertEqual(len(allrows[allrows.canonical_smiles == "CCO"]), 1)
        for s in splits:
            for t in splits:
                if s != t:
                    self.assertFalse(set(splits[s].canonical_smiles) & set(splits[t].canonical_smiles))
        repeated, _, _ = mod.partition_pairs({"train": df.sample(frac=1, random_state=7)}, 17)
        for s in splits:
            pd.testing.assert_frame_equal(splits[s], repeated[s])


if __name__ == "__main__":
    unittest.main()
