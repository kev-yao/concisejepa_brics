import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from hydra.errors import InstantiationException
from hydra.utils import instantiate

from spikes.phase1.fragment_train import build_model as build_spike_model
from spikes.phase1.lit_fragment import LitFragment as SpikeLitFragment


CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def compose_experiment(name: str):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="config", overrides=[f"experiment={name}"])


def small_brics_config():
    cfg = compose_experiment("brics_mean")
    backbone = cfg.model.concise_fragment
    backbone.drug_layers = [[4, 4]]
    backbone.ligand_dim = 16
    backbone.residue_dim = 12
    backbone.drug_dim = 8
    backbone.proj_dim = 8
    backbone.nheads = 2
    backbone.pairwise_attention_chunk_size = 16
    backbone.use_pairwise_attention_checkpoint = False
    cfg.model.smiles_target_dim = 8
    cfg.model.jepa_hidden_dim = 16
    cfg.data.fingerprint_length = 16
    return cfg


def spike_build_config(cfg):
    backbone = cfg.model.concise_fragment
    return SimpleNamespace(
        fsq_levels=",".join(str(level) for level in backbone.drug_layers[0]),
        pooling=backbone.pooling,
        ligand_dim=backbone.ligand_dim,
        residue_dim=backbone.residue_dim,
        drug_dim=backbone.drug_dim,
        proj_dim=backbone.proj_dim,
        nheads=backbone.nheads,
        activation=backbone.activation,
        smiles_target_dim=cfg.model.smiles_target_dim,
        jepa_hidden_dim=cfg.model.jepa_hidden_dim,
        predictor_type="mlp",
        predictor_dim=256,
        predictor_depth=3,
        predictor_heads=8,
        predictor_mlp_ratio=2.0,
    )


def synthetic_fragment_batch():
    generator = torch.Generator().manual_seed(20260901)
    fragment_mask = torch.tensor(
        [
            [True, True, False],
            [True, True, True],
            [True, False, False],
            [True, True, False],
        ]
    )
    return (
        torch.randn(4, 5, 12, generator=generator),
        torch.randn(4, 3, 16, generator=generator),
        fragment_mask,
        torch.randn(4, 8, generator=generator),
        torch.tensor([1.0, 0.0, 1.0, 0.0]),
        ["CCO", "CCN", "CCC", "CCCl"],
        ["protein-a", "protein-b", "protein-c", "protein-d"],
    )


def optimizer_trajectory(module, batch, rng_seeds):
    module.train()
    optimizer = module.configure_optimizers()
    trajectory = []
    for rng_seed in rng_seeds:
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(rng_seed)
        loss, outputs = module._forward_step(batch, stage="train")
        loss.backward()
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in module.named_parameters()
            if parameter.grad is not None
        }
        optimizer.step()
        state = {name: value.detach().clone() for name, value in module.state_dict().items()}
        outputs = {
            name: value.detach().clone()
            for name, value in outputs.items()
            if isinstance(value, torch.Tensor)
        }
        trajectory.append((loss.detach().clone(), outputs, gradients, state))
    return trajectory


class BricsHydraIntegrationTests(unittest.TestCase):
    def assert_tensor_dict_equal(self, left, right, message):
        self.assertEqual(left.keys(), right.keys(), message)
        for name in left:
            torch.testing.assert_close(
                left[name],
                right[name],
                rtol=0.0,
                atol=0.0,
                msg=f"{message}: {name}",
            )

    def test_brics_experiments_select_compatible_bundles(self):
        mean_cfg = compose_experiment("brics_mean")
        self.assertEqual(mean_cfg.model._target_, "concisejepa.models.fragment.ConciseJEPAFragment")
        self.assertEqual(mean_cfg.model.concise_fragment.pooling, "mean")
        self.assertEqual(
            mean_cfg.data._target_, "concisejepa.datamodules.fragment.FragmentDataModule"
        )
        self.assertEqual(
            mean_cfg.task._target_, "concisejepa.lightning_modules.lit_fragment.LitFragment"
        )
        self.assertEqual(mean_cfg.data.batch_size, 256)
        self.assertEqual(mean_cfg.trainer.max_epochs, 30)
        self.assertFalse(mean_cfg.chem_supervision.enabled)
        self.assertFalse(mean_cfg.group_supervision.enabled)

        xattn_cfg = compose_experiment("brics_xattn")
        self.assertEqual(
            xattn_cfg.model._target_, "concisejepa.models.fragment.ConciseJEPAFragmentXAttn"
        )
        self.assertEqual(xattn_cfg.model.predictor_depth, 3)
        self.assertEqual(xattn_cfg.model.predictor_heads, 8)

        debug_cfg = compose_experiment("brics_debug")
        self.assertEqual(debug_cfg.trainer.max_epochs, 1)
        self.assertEqual(debug_cfg.trainer.limit_train_batches, 2)

    def test_fragment_fingerprint_dimension_must_match_model(self):
        cfg = small_brics_config()
        model = instantiate(cfg.model)
        cfg.data.fingerprint_length = 17
        with self.assertRaisesRegex(InstantiationException, "dimension mismatch"):
            instantiate(cfg.task, experiment_config=cfg, model=model)

    def test_hydra_data_module_preserves_fragment_batch_contract(self):
        cfg = small_brics_config()
        sequence = "A" * 50
        smiles = "CCO"
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            root = Path(tmpdir)
            frame = pd.DataFrame(
                {"Target Sequence": [sequence], "SMILES": [smiles], "Label": [1]}
            )
            for split in ("train", "val", "test"):
                frame.to_csv(root / f"{split}.csv", index=False)

            torch.save({sequence: torch.zeros(50, 12)}, root / "raygun_embeddings.pt")
            torch.save({smiles: torch.zeros(2, 16)}, root / "fragment_fps_r4_2048.pt")
            torch.save({smiles: torch.zeros(16)}, root / "morgan_embeddings.pt")
            torch.save({smiles: torch.zeros(8)}, root / "coati_embeddings.pt")
            (root / "fragment_fps_r4_2048.pt.meta.json").write_text(
                json.dumps(
                    {
                        "type": "fragment_fps",
                        "fingerprint_kind": "ecfp-count:4",
                        "fingerprint_length": 16,
                        "max_frags": 16,
                    }
                ),
                encoding="utf-8",
            )

            cfg.data.data_root = str(root)
            cfg.data.train_csv = str(root / "train.csv")
            cfg.data.val_csv = str(root / "val.csv")
            cfg.data.test_csv = str(root / "test.csv")
            cfg.data.protein_embeddings_path = str(root / "raygun_embeddings.pt")
            cfg.data.fragment_fps_path = str(root / "fragment_fps_r4_2048.pt")
            cfg.data.morgan_embeddings_path = str(root / "morgan_embeddings.pt")
            cfg.data.smiles_embeddings_path = str(root / "coati_embeddings.pt")
            datamodule = instantiate(cfg.data)
            datamodule.setup("fit")
            batch = datamodule.collator([(sequence, smiles, 1.0)])

            self.assertEqual(batch[0].shape, (1, 50, 12))
            self.assertEqual(batch[1].shape, (1, 2, 16))
            self.assertEqual(batch[2].shape, (1, 2))
            self.assertTrue(batch[2].all())
            self.assertEqual(batch[3].shape, (1, 8))

    def test_all_promoted_poolers_and_xattn_predictor_accept_fragment_contract(self):
        cfg = small_brics_config()
        protein, fragment_fps, fragment_mask = synthetic_fragment_batch()[:3]
        pooling_names = [
            "mean",
            "max",
            "weighted_sum",
            "cross_attention",
            "latent_query",
            "latent_query_q1",
            "latent_query_q2",
            "latent_query_q8",
            "pre_attn_latent_query",
            "mlp_weighted_sum",
            "multi_head_weighted_sum_k2",
            "multi_head_weighted_sum_k4",
            "f2r",
        ]
        for pooling_name in pooling_names:
            with self.subTest(pooling=pooling_name):
                cfg.model.concise_fragment.pooling = pooling_name
                model = instantiate(cfg.model).eval()
                with torch.no_grad():
                    outputs = model(protein, fragment_fps, fragment_mask)
                self.assertEqual(outputs["binding"].shape, (4,))
                self.assertEqual(outputs["similarity_logits"].shape, (4, 4))
                self.assertEqual(outputs["jepa_pred"].shape, (4, 8))
                self.assertEqual(outputs["frag_codes"].shape[:2], (4, 3))

        xattn_cfg = compose_experiment("brics_xattn")
        backbone = xattn_cfg.model.concise_fragment
        backbone.drug_layers = [[4, 4]]
        backbone.ligand_dim = 16
        backbone.residue_dim = 12
        backbone.drug_dim = 8
        backbone.proj_dim = 8
        backbone.nheads = 2
        backbone.pairwise_attention_chunk_size = 16
        backbone.use_pairwise_attention_checkpoint = False
        xattn_cfg.model.smiles_target_dim = 8
        xattn_cfg.model.predictor_dim = 8
        xattn_cfg.model.predictor_depth = 1
        xattn_cfg.model.predictor_heads = 2
        xattn_model = instantiate(xattn_cfg.model).eval()
        with torch.no_grad():
            outputs = xattn_model(protein, fragment_fps, fragment_mask)
        self.assertEqual(outputs["binding"].shape, (4,))
        self.assertEqual(outputs["jepa_pred"].shape, (4, 8))

    def test_spike_and_hydra_paths_have_identical_training_trajectory(self):
        cfg = small_brics_config()

        torch.manual_seed(314159)
        spike_model = build_spike_model(spike_build_config(cfg))
        spike_task = SpikeLitFragment(
            spike_model,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            negative_diagonal_weight=cfg.negative_diagonal_loss.weight,
            negative_diagonal_margin=cfg.negative_diagonal_loss.margin,
            chem_supervision_weight=0.0,
            group_supervision_weight=0.0,
            jepa_loss_type="mse",
        )

        torch.manual_seed(314159)
        hydra_model = instantiate(cfg.model)
        hydra_task = instantiate(cfg.task, experiment_config=cfg, model=hydra_model)

        self.assert_tensor_dict_equal(
            spike_task.state_dict(), hydra_task.state_dict(), "initial state differs"
        )

        batch = synthetic_fragment_batch()
        spike_trajectory = optimizer_trajectory(spike_task, batch, [271828, 161803])
        hydra_trajectory = optimizer_trajectory(hydra_task, batch, [271828, 161803])
        for step, (spike_result, hydra_result) in enumerate(
            zip(spike_trajectory, hydra_trajectory, strict=True), start=1
        ):
            torch.testing.assert_close(spike_result[0], hydra_result[0], rtol=0.0, atol=0.0)
            self.assert_tensor_dict_equal(
                spike_result[1], hydra_result[1], f"step {step} outputs differ"
            )
            self.assert_tensor_dict_equal(
                spike_result[2], hydra_result[2], f"step {step} gradients differ"
            )
            self.assert_tensor_dict_equal(
                spike_result[3], hydra_result[3], f"step {step} updated state differs"
            )


if __name__ == "__main__":
    unittest.main()
