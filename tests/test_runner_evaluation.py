"""Candidate fitting must never invoke the automatic held-out test stage."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
from pytorch_lightning.callbacks import ModelCheckpoint

import main as runner
from tests.test_brics_hydra_integration import compose_experiment
from tests.test_brics_dual_view import synthetic_batch
from tests.test_secondary_binding import small_config


class RunnerEvaluationTests(TestCase):
    def test_test_gate_and_legacy_final_checkpoint_choice(self):
        for setting in ('default', False, True, 'old_config'):
            with self.subTest(setting=setting), TemporaryDirectory() as tmp:
                cfg = compose_experiment('brics_f2r_whole_debug')
                self.assertTrue(cfg.evaluation.run_test)
                if setting == 'old_config':
                    with open_dict(cfg):
                        del cfg.evaluation
                elif setting != 'default':
                    cfg.evaluation.run_test = setting
                cfg.runtime.run_dir = str(Path(tmp))
                datamodule = mock.Mock()
                trainer = mock.Mock()
                trainer.fit.side_effect = lambda task, datamodule: datamodule.setup('fit')
                trainer.test.side_effect = lambda task, datamodule, ckpt_path: datamodule.setup('test')
                model, task, logger = mock.Mock(), mock.Mock(), mock.Mock()
                best, final = ModelCheckpoint(), ModelCheckpoint()
                best.best_model_path = '/validation-best.ckpt'
                final.best_model_path = '/final.ckpt'
                with mock.patch.object(runner, '_prepare_runtime_config'), \
                     mock.patch.object(runner, '_write_run_manifest'), \
                     mock.patch.object(runner, '_instantiate_callbacks', return_value=([], {'best_checkpoint': best, 'final_checkpoint': final})), \
                     mock.patch.object(runner, 'instantiate', side_effect=[model, datamodule, task, logger, trainer]):
                    runner.main.__wrapped__(cfg)
                trainer.fit.assert_called_once_with(task, datamodule=datamodule)
                if setting is False:
                    trainer.test.assert_not_called()
                    self.assertEqual(datamodule.setup.call_args_list, [mock.call('fit')])
                else:
                    trainer.test.assert_called_once_with(task, datamodule=datamodule, ckpt_path='/final.ckpt')
                    self.assertEqual(datamodule.setup.call_args_list, [mock.call('fit'), mock.call('test')])

    def test_real_candidate_fit_never_sets_up_test_and_restores_objective_weights(self):
        class Data(pl.LightningDataModule):
            def __init__(self):
                super().__init__()
                self.stages = []

            def setup(self, stage=None):
                self.stages.append(stage)
                if stage == 'test':
                    raise AssertionError('Candidate touched test setup')

            def train_dataloader(self):
                return DataLoader([synthetic_batch()], batch_size=None)

            def val_dataloader(self):
                return DataLoader([synthetic_batch()], batch_size=None)

        with TemporaryDirectory() as tmp:
            cfg = small_config()
            cfg.evaluation.run_test = False
            cfg.task.binding_bce_weight = 1.
            cfg.task.binding_contrastive_weight = .1
            cfg.task.binding_negative_weight = 0.
            cfg.task.jepa_loss_weight = .1
            cfg.run.output_root = tmp
            cfg.trainer.accelerator = 'cpu'
            cfg.trainer.limit_train_batches = cfg.trainer.limit_val_batches = 1
            cfg.trainer.enable_progress_bar = False
            cfg.callbacks.progress_bar.enabled = False
            data = Data()
            tasks = []

            def build(component, **kwargs):
                if component is cfg.data:
                    return data
                if component is cfg.logger:
                    return False
                built = instantiate(component, **kwargs)
                if component is cfg.task:
                    tasks.append(built)
                return built

            with mock.patch.object(runner, 'instantiate', side_effect=build):
                runner.main.__wrapped__(cfg)
            self.assertEqual(data.stages, ['fit'])
            run = Path(cfg.runtime.run_dir)
            self.assertTrue((run / 'final_metrics.json').exists())
            self.assertFalse((run / 'test_metrics.json').exists())
            checkpoint = run / 'checkpoints/final.ckpt'
            restored = type(tasks[0]).load_from_checkpoint(
                checkpoint, model=instantiate(cfg.model), experiment_config=cfg, weights_only=False)
            for key in ('binding_bce_weight', 'binding_contrastive_weight', 'binding_negative_weight', 'jepa_loss_weight'):
                self.assertEqual(getattr(restored, key), cfg.task[key])
                self.assertEqual(restored.hparams[key], cfg.task[key])

    def test_candidate_gate_is_a_resolved_boolean(self):
        cfg = compose_experiment('brics_f2r_whole')
        cfg.evaluation.run_test = False
        restored = OmegaConf.create(OmegaConf.to_yaml(cfg, resolve=True))
        self.assertIs(restored.evaluation.run_test, False)
