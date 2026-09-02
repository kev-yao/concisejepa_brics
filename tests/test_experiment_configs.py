import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from main import _instantiate_callbacks, _prepare_runtime_config, _write_run_manifest


CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def compose_experiment(name: str):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="config", overrides=[f"experiment={name}"])


class ExperimentConfigTests(unittest.TestCase):
    def test_baseline_preserves_legacy_training_defaults(self):
        cfg = compose_experiment("baseline")

        self.assertEqual(cfg.seed, 42)
        self.assertEqual(cfg.lr, 1.0e-4)
        self.assertEqual(cfg.weight_decay, 1.0e-2)
        self.assertEqual(
            {
                "enabled": cfg.negative_diagonal_loss.enabled,
                "weight": cfg.negative_diagonal_loss.weight,
                "margin": cfg.negative_diagonal_loss.margin,
            },
            {"enabled": True, "weight": 1.0, "margin": 0.0},
        )

        backbone = cfg.model.concise_backbone
        self.assertEqual(backbone.drug_layers, [[32, 32, 32]])
        self.assertEqual(backbone.drug_quantizer.type, "fsq")
        self.assertEqual(backbone.ligand_dim, 2048)
        self.assertEqual(backbone.residue_dim, 1280)
        self.assertEqual(backbone.drug_dim, 128)
        self.assertEqual(backbone.proj_dim, 256)
        self.assertEqual(backbone.nheads, 32)
        self.assertEqual(backbone.activation, "tanh")
        self.assertFalse(backbone.cosine_prediction)
        self.assertEqual(cfg.model.smiles_target_dim, 256)
        self.assertEqual(cfg.model.jepa_hidden_dim, 512)

        self.assertEqual(
            cfg.data.train_csv,
            "/hpc/group/singhlab/user/me196/projects/moleculerep/runs/REVICE/data/bindingdb/train.csv",
        )
        self.assertEqual(
            cfg.data.protein_embeddings_path,
            "/hpc/group/singhlab/user/cy244/projects/peptides/raygun_embeddings.pt",
        )
        self.assertEqual(
            cfg.data.morgan_embeddings_path,
            "/hpc/group/singhlab/user/cy244/projects/peptides/count_combined_embeddings/morgan_embeddings.pt",
        )
        self.assertEqual(
            cfg.data.smiles_embeddings_path,
            "/hpc/group/singhlab/user/cy244/projects/peptides/coati_embeddings.pt",
        )
        self.assertEqual(cfg.data.fingerprint_kind, "ecfp-count:4")
        self.assertEqual(cfg.data.fingerprint_length, 2048)
        self.assertEqual(cfg.data.batch_size, 1028)
        self.assertEqual(cfg.data.num_workers, 4)
        self.assertTrue(cfg.data.pin_memory)
        self.assertTrue(cfg.data.persistent_workers)

        self.assertEqual(cfg.trainer.max_epochs, 100)
        self.assertEqual(cfg.trainer.accelerator, "auto")
        self.assertEqual(cfg.trainer.devices, 1)
        self.assertEqual(cfg.trainer.precision, 32)
        self.assertEqual(cfg.trainer.log_every_n_steps, 20)
        self.assertTrue(cfg.chem_supervision.enabled)
        self.assertEqual(cfg.chem_supervision.weight, 0.1)
        self.assertTrue(cfg.chem_supervision.post_fsq)
        self.assertTrue(cfg.group_supervision.enabled)
        self.assertEqual(cfg.group_supervision.weight, 0.1)

    def test_baseline_selects_composable_components(self):
        cfg = compose_experiment("baseline")

        self.assertEqual(cfg.model._target_, "concisejepa.models.concise_jepa.ConciseJEPA")
        self.assertEqual(cfg.model.concise_backbone.drug_quantizer.type, "fsq")
        self.assertEqual(cfg.data._target_, "concisejepa.datamodules.BindingDBDictDataModule")
        self.assertEqual(cfg.task._target_, "concisejepa.lightning_modules.LitConciseJEPA")
        self.assertFalse(cfg.task._recursive_)
        self.assertEqual(cfg.run.name_prefix, "baseline-fsq")
        self.assertIn("fsq_monitor", cfg.callbacks)

    def test_diveq_experiment_swaps_model_and_callbacks_together(self):
        cfg = compose_experiment("diveq")

        self.assertEqual(cfg.model.concise_backbone.drug_quantizer.type, "diveq")
        self.assertNotIn("fsq_monitor", cfg.callbacks)
        self.assertEqual(cfg.run.name_prefix, "diveq")

    def test_runtime_paths_callbacks_and_manifest_resolve(self):
        cfg = compose_experiment("baseline")
        with TemporaryDirectory() as tmpdir:
            cfg.run.output_root = tmpdir
            _prepare_runtime_config(cfg)
            callbacks, callbacks_by_name = _instantiate_callbacks(cfg.callbacks)
            logger = instantiate(cfg.logger)
            cfg.trainer.max_epochs = 1
            trainer = instantiate(cfg.trainer, logger=logger, callbacks=callbacks)
            _write_run_manifest(cfg)

            run_dir = Path(cfg.runtime.run_dir)
            self.assertTrue(run_dir.is_dir())
            self.assertTrue((run_dir / "resolved_config.yaml").is_file())
            self.assertTrue((run_dir / "run_manifest.json").is_file())
            self.assertEqual(len(callbacks), len(callbacks_by_name))
            self.assertIn("best_checkpoint", callbacks_by_name)
            self.assertIn("final_checkpoint", callbacks_by_name)
            self.assertEqual(trainer.max_epochs, 1)

    def test_composed_model_task_and_data_instantiate(self):
        cfg = compose_experiment("baseline")
        sequence = "A" * 50
        smiles = "CCO"

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frame = pd.DataFrame(
                {"Target Sequence": [sequence], "SMILES": [smiles], "Label": [1]}
            )
            for split in ("train", "val", "test"):
                frame.to_csv(root / f"{split}.csv", index=False)

            torch.save({sequence: torch.zeros(50, 1280)}, root / "raygun_embeddings.pt")
            torch.save({smiles: torch.zeros(2048)}, root / "morgan_embeddings.pt")
            torch.save({smiles: torch.zeros(256)}, root / "coati_embeddings.pt")
            (root / "morgan_embeddings.pt.meta.json").write_text(
                json.dumps(
                    {
                        "fingerprint_kind": "ecfp-count:4",
                        "fingerprint_length": 2048,
                    }
                ),
                encoding="utf-8",
            )

            cfg.data.csv_root = str(root)
            cfg.data.protein_embeddings_path = str(root / "raygun_embeddings.pt")
            cfg.data.morgan_embeddings_path = str(root / "morgan_embeddings.pt")
            cfg.data.smiles_embeddings_path = str(root / "coati_embeddings.pt")
            cfg.data.num_workers = 0
            cfg.data.persistent_workers = False

            model = instantiate(cfg.model)
            task = instantiate(cfg.task, experiment_config=cfg, model=model)
            datamodule = instantiate(cfg.data)
            datamodule.setup("fit")

            self.assertIs(task.model, model)
            self.assertEqual(len(datamodule.train_dataset), 1)
            self.assertEqual(datamodule.collator([(sequence, smiles, 1.0)])[1].shape, (1, 2048))


if __name__ == "__main__":
    unittest.main()
