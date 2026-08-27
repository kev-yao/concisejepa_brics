#!/usr/bin/env python
"""
Layer perturbation AUPRC evaluation.

Runs baseline train-split DTI AUPRC, then for each codebook layer replaces that
layer's codes with random indices (uniform) and recomputes AUPRC. Useful for
measuring sensitivity of predictions to individual code layers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch.nn as nn
import torch
import torch.utils.data as data
from omegaconf import DictConfig, ListConfig, OmegaConf
from torchmetrics.classification import BinaryAveragePrecision

from concisejepa.datamodules import BindingDBDictDataModule


def _levels_to_int_list(levels_entry: Any) -> list[int]:
    """Normalize level entries to a flat list of ints."""
    if isinstance(levels_entry, torch.Tensor):
        if levels_entry.numel() == 0:
            raise ValueError("Encountered empty tensor while parsing codebook levels")
        return [int(v) for v in levels_entry.to(torch.long).reshape(-1).tolist()]

    if isinstance(levels_entry, (list, tuple)):
        values: list[int] = []
        for value in levels_entry:
            values.extend(_levels_to_int_list(value))
        if not values:
            raise ValueError("Encountered empty sequence while parsing codebook levels")
        return values

    return [int(levels_entry)]


def _codebook_size(levels_entry: Any) -> int:
    """Derive codebook size from level entries (ints, sequences, or tensors)."""
    levels = _levels_to_int_list(levels_entry)
    total = 1
    for level in levels:
        total *= int(level)
    return int(total)


def _build_perturb_units(levels: list[Any], perturb_mode: str) -> list[dict[str, Any]]:
    """Construct logical perturbation units from residual-layer levels."""
    mode = perturb_mode.lower()
    if mode not in {"factor", "residual"}:
        raise ValueError(f"Unsupported perturb_mode={perturb_mode!r}; expected 'factor' or 'residual'")

    units: list[dict[str, Any]] = []
    logical_layer = 0
    for residual_layer, levels_entry in enumerate(levels):
        factor_levels = _levels_to_int_list(levels_entry)

        if mode == "residual":
            units.append({
                "layer": logical_layer,
                "residual_layer": residual_layer,
                "factor_index": None,
                "factor_levels": factor_levels,
                "codebook_size": _codebook_size(levels_entry),
            })
            logical_layer += 1
            continue

        for factor_index, factor_level in enumerate(factor_levels):
            units.append({
                "layer": logical_layer,
                "residual_layer": residual_layer,
                "factor_index": factor_index,
                "factor_levels": factor_levels,
                "codebook_size": int(factor_level),
            })
            logical_layer += 1

    return units


def _fsq_bases(levels: list[int]) -> list[int]:
    """Compute FSQ basis values for factor-to-index conversion."""
    if not levels:
        return []
    bases = [1]
    for level in levels[:-1]:
        bases.append(bases[-1] * int(level))
    return bases


def _perturb_codes(
    codes: torch.Tensor,
    levels: list[Any],
    unit: dict[str, Any],
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """Perturb one logical unit (residual layer or internal factor) and return modified codes."""
    residual_layer = int(unit["residual_layer"])
    perturbed = codes.clone()
    factor_index = unit["factor_index"]
    factor_levels = _levels_to_int_list(levels[residual_layer])

    # Hybrid FSQ exposes its single layer as one integer column per factor.
    if len(levels) == 1 and codes.shape[1] == len(factor_levels):
        if factor_index is None:
            for idx, level in enumerate(factor_levels):
                perturbed[:, idx] = torch.randint(
                    low=0,
                    high=int(level),
                    size=(codes.shape[0],),
                    device=device,
                    generator=generator,
                    dtype=perturbed.dtype,
                )
        else:
            factor_index = int(factor_index)
            perturbed[:, factor_index] = torch.randint(
                low=0,
                high=int(factor_levels[factor_index]),
                size=(codes.shape[0],),
                device=device,
                generator=generator,
                dtype=perturbed.dtype,
            )
        return perturbed

    if residual_layer >= codes.shape[1]:
        raise ValueError(f"Layer index {residual_layer} out of range for codes shape {codes.shape}")

    if factor_index is None:
        max_code = int(unit["codebook_size"])
        rand_codes = torch.randint(
            low=0,
            high=max_code,
            size=(codes.shape[0],),
            device=device,
            generator=generator,
        )
        if rand_codes.dtype != perturbed.dtype:
            rand_codes = rand_codes.to(perturbed.dtype)
        perturbed[:, residual_layer] = rand_codes
        return perturbed

    factor_index = int(factor_index)
    if factor_index >= len(factor_levels):
        raise ValueError(
            f"Factor index {factor_index} out of range for levels {factor_levels} in residual layer {residual_layer}"
        )

    bases = _fsq_bases(factor_levels)
    residual_codes = perturbed[:, residual_layer].to(torch.long)

    factor_digits = [
        (residual_codes // int(base)) % int(level)
        for base, level in zip(bases, factor_levels)
    ]
    digits = torch.stack(factor_digits, dim=1)

    rand_factor = torch.randint(
        low=0,
        high=int(factor_levels[factor_index]),
        size=(codes.shape[0],),
        device=device,
        generator=generator,
        dtype=torch.long,
    )
    digits[:, factor_index] = rand_factor

    recombined = torch.zeros_like(residual_codes)
    for idx, base in enumerate(bases):
        recombined = recombined + digits[:, idx] * int(base)

    if recombined.dtype != perturbed.dtype:
        recombined = recombined.to(perturbed.dtype)
    perturbed[:, residual_layer] = recombined
    return perturbed


def _build_train_eval_dataloader(cfg: DictConfig) -> data.DataLoader:
    """Build a deterministic no-shuffle train dataloader for perturbation evaluation."""
    datamodule = BindingDBDictDataModule(cfg.datamodule)
    datamodule.setup(stage="fit")
    if datamodule.train_dataset is None:
        raise RuntimeError("Datamodule train dataset is not initialized after setup('fit')")

    return data.DataLoader(
        datamodule.train_dataset,
        batch_size=datamodule.batch_size,
        shuffle=False,
        num_workers=datamodule.num_workers,
        pin_memory=datamodule.pin_memory,
        persistent_workers=datamodule.persistent_workers,
        collate_fn=datamodule.collator,
    )


def _resolve_device(device_name: str) -> torch.device:
    requested = torch.device(device_name)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("[codebook_randomization] CUDA not available, falling back to CPU.")
        return torch.device("cpu")
    return requested


def _load_concise_backbone(
    cfg: DictConfig,
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    from concisejepa.lightning_modules import LitConciseJEPA

    torch.serialization.add_safe_globals([DictConfig, ListConfig])

    lit_module = LitConciseJEPA(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        raise KeyError(
            "Lightning checkpoint is missing 'state_dict'. "
            "Please pass a .ckpt file produced by PyTorch Lightning."
        )

    lit_module.load_state_dict(state_dict, strict=True)
    model = lit_module.model.concise
    model.to(device)
    model.eval()
    return model


def _run_ap(
    model: nn.Module,
    dataloader: data.DataLoader,
    device: torch.device,
    perturb_unit: dict[str, Any] | None,
    levels: list[Any],
    seed: int,
) -> float:
    ap = BinaryAveragePrecision().to(device)
    model_mask_prev = getattr(model, "enable_layer_masking", True)
    try:
        if hasattr(model, "enable_layer_masking"):
            model.enable_layer_masking = False
        model.eval()
        generator_device = device.type if device.type == "cuda" else "cpu"
        gen = torch.Generator(device=generator_device)
        gen.manual_seed(seed)

        with torch.no_grad():
            for batch in dataloader:
                receptor, ligand, _, label = batch[:4]

                ligand = ligand.to(device)
                receptor = receptor.float().to(device)
                label = label.to(device).to(torch.int)

                if perturb_unit is None:
                    out = model(ligand, receptor, is_morgan_fingerprint=True)
                else:
                    encoded = model.d_encoder(ligand)
                    perturbed_codes = _perturb_codes(
                        codes=encoded["codes"],
                        levels=levels,
                        unit=perturb_unit,
                        device=device,
                        generator=gen,
                    )
                    out = model(
                        perturbed_codes,
                        receptor,
                        is_morgan_fingerprint=False,
                    )

                ap.update(out["binding"].detach(), label)
        return float(ap.compute().detach().cpu().item())
    finally:
        ap.reset()
        if hasattr(model, "enable_layer_masking"):
            model.enable_layer_masking = model_mask_prev


def layer_perturb_eval(
    model: nn.Module,
    dataloader: data.DataLoader,
    device: torch.device,
    seed: int,
    perturb_mode: str = "factor",
) -> dict[str, Any]:
    """
    Compute baseline AUPRC and per-layer perturbed AUPRC (random codes) on a dataloader.
    Returns a dict with baseline, per-layer values, and deltas.
    """

    levels = model.d_encoder.get_levels()
    perturb_units = _build_perturb_units(levels, perturb_mode=perturb_mode)

    baseline = _run_ap(model, dataloader, device, perturb_unit=None, levels=levels, seed=seed)

    per_layer = []
    for unit in perturb_units:
        layer_idx = int(unit["layer"])
        ap = _run_ap(
            model,
            dataloader,
            device,
            perturb_unit=unit,
            levels=levels,
            seed=seed + layer_idx + 1,
        )
        row = {
            "layer": layer_idx,
            "ap": ap,
            "delta": ap - baseline,
            "codebook_size": int(unit["codebook_size"]),
            "residual_layer": int(unit["residual_layer"]),
        }
        if unit["factor_index"] is not None:
            row["factor"] = int(unit["factor_index"])
        per_layer.append(row)

    return {
        "baseline_ap": baseline,
        "per_layer": per_layer,
        "seed": seed,
        "perturb_mode": perturb_mode,
    }


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Layer perturbation AUPRC eval")
    p.add_argument("--checkpoint", required=True, help="Path to Lightning checkpoint (.ckpt)")
    p.add_argument("--config", default="configs/config.yaml", help="Path to concisejepa config YAML")
    p.add_argument("--device", default="cuda:0", help="Device to use")
    p.add_argument("--seed", type=int, default=17, help="Random seed for code perturbation")
    p.add_argument(
        "--perturb-mode",
        choices=["factor", "residual"],
        default="factor",
        help="Perturbation unit: internal factors (default) or residual code layers",
    )
    p.add_argument("--output-json", required=True, help="Path to write summary JSON")
    return p.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    device = _resolve_device(args.device)
    cfg = OmegaConf.load(config_path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected DictConfig from {config_path}, got {type(cfg).__name__}")

    model = _load_concise_backbone(cfg=cfg, checkpoint_path=checkpoint_path, device=device)
    dataloader = _build_train_eval_dataloader(cfg)
    summary = layer_perturb_eval(
        model=model,
        dataloader=dataloader,
        device=device,
        seed=args.seed,
        perturb_mode=args.perturb_mode,
    )

    result = {
        "split": "train",
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "device": str(device),
        **summary,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
