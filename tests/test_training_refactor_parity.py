from pathlib import Path
import unittest

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from concisejepa.lightning_modules import LitConciseJEPA


CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def _small_baseline_config():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config", overrides=["experiment=baseline"])

    # Preserve the architecture and objectives while making the parity smoke
    # test fast enough for CPU-only CI.
    cfg.model.concise_backbone.ligand_dim = 16
    cfg.model.concise_backbone.residue_dim = 12
    cfg.model.concise_backbone.drug_dim = 8
    cfg.model.concise_backbone.proj_dim = 8
    cfg.model.concise_backbone.nheads = 2
    cfg.model.concise_backbone.drug_layers = [[4, 4]]
    cfg.model.smiles_target_dim = 8
    cfg.model.jepa_hidden_dim = 16
    cfg.data.fingerprint_length = 16
    return cfg


def _synthetic_batch():
    generator = torch.Generator().manual_seed(20260901)
    batch_size = 4
    return (
        torch.randn(batch_size, 5, 12, generator=generator),
        torch.randn(batch_size, 16, generator=generator),
        torch.randn(batch_size, 8, generator=generator),
        torch.tensor([1.0, 0.0, 1.0, 0.0]),
        ["CCO", "CCN", "CCC", "CCCl"],
        ["protein-a", "protein-b", "protein-c", "protein-d"],
    )


def _build_legacy(cfg, seed: int):
    torch.manual_seed(seed)
    # This is the pre-refactor construction contract: LitConciseJEPA owns
    # Hydra model instantiation.
    return LitConciseJEPA(cfg)


def _build_composed(cfg, seed: int):
    torch.manual_seed(seed)
    # This is the refactored runner contract: model and task are independently
    # instantiated, and the model is injected into the task.
    model = instantiate(cfg.model)
    return instantiate(cfg.task, experiment_config=cfg, model=model)


def _optimizer_trajectory(module, batch, rng_seeds):
    module.train()
    optimizer = module.configure_optimizers()
    trajectory = []
    for rng_seed in rng_seeds:
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(rng_seed)
        loss, losses, _, _, outputs, _, _ = module._forward_losses_metrics(batch, stage="train")
        loss.backward()

        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in module.named_parameters()
            if parameter.grad is not None
        }
        optimizer.step()
        state = {name: value.detach().clone() for name, value in module.state_dict().items()}
        scalar_losses = {name: value.detach().clone() for name, value in losses.items()}
        forward_outputs = {
            name: value.detach().clone()
            for name, value in outputs.items()
            if isinstance(value, torch.Tensor)
        }
        trajectory.append((loss.detach().clone(), scalar_losses, forward_outputs, gradients, state))
    return trajectory


class TrainingRefactorParityTests(unittest.TestCase):
    def assert_tensor_dict_close(self, left, right, *, message):
        self.assertEqual(left.keys(), right.keys(), message)
        for name in left:
            torch.testing.assert_close(
                left[name],
                right[name],
                rtol=0.0,
                atol=0.0,
                msg=f"{message}: {name}",
            )

    def test_legacy_and_composed_paths_have_identical_training_update(self):
        cfg = _small_baseline_config()
        legacy = _build_legacy(cfg, seed=314159)
        composed = _build_composed(cfg, seed=314159)

        self.assert_tensor_dict_close(
            legacy.state_dict(), composed.state_dict(), message="initial state differs"
        )

        batch = _synthetic_batch()
        rng_seeds = [271828, 161803]
        legacy_trajectory = _optimizer_trajectory(legacy, batch, rng_seeds)
        composed_trajectory = _optimizer_trajectory(composed, batch, rng_seeds)

        for step, (legacy_result, composed_result) in enumerate(
            zip(legacy_trajectory, composed_trajectory, strict=True), start=1
        ):
            torch.testing.assert_close(legacy_result[0], composed_result[0], rtol=0.0, atol=0.0)
            self.assert_tensor_dict_close(
                legacy_result[1], composed_result[1], message=f"step {step} loss components differ"
            )
            self.assert_tensor_dict_close(
                legacy_result[2], composed_result[2], message=f"step {step} forward outputs differ"
            )
            self.assert_tensor_dict_close(
                legacy_result[3], composed_result[3], message=f"step {step} gradients differ"
            )
            self.assert_tensor_dict_close(
                legacy_result[4], composed_result[4], message=f"step {step} updated state differs"
            )


if __name__ == "__main__":
    unittest.main()
