#!/usr/bin/env python3
"""Run a preregistered BRICS ablation, with pooled metrics and reconstruction controls."""

import argparse
import gc
import hashlib
import json
import random
import sys
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
from torchmetrics.functional.classification import binary_auroc, binary_average_precision

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from main import _write_run_manifest, _instantiate_callbacks
from evaluate_dual_view_reconstruction import (
    canonicalize_largest_fragment,
    decode_latents,
    predict_latents,
    reconstruction_summary,
)

ARMS = ["fsq_joint", "continuous_joint", "fsq_jepa", "continuous_jepa", "fsq_dti", "continuous_dti"]


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def molecule_data(dm, split):
    df = pd.read_csv(getattr(dm, f"{split}_csv"))
    df.SMILES = df.SMILES.str.strip()
    df["Target Sequence"] = df["Target Sequence"].str.strip()
    df = df[df.SMILES.isin(dm.valid_smiles) & df["Target Sequence"].isin(dm.valid_sequences)]
    if "canonical_smiles" in df:
        df = df.drop_duplicates("canonical_smiles")
    else:
        df = df.drop_duplicates("SMILES")
    smiles = sorted(df.SMILES.tolist())
    targets = torch.stack([dm.smiles_embeddings[s].reshape(-1).float() for s in smiles])
    return smiles, targets


@torch.no_grad()
def latent_metrics(model, dm, split, device):
    smiles, targets = molecule_data(dm, split)
    pred = predict_latents(model, smiles, dm.fragment_fps, 128, device)
    return {
        "n_unique_molecules": len(smiles),
        "mse": float(F.mse_loss(pred, targets)),
        "prediction_variance": float(pred.var(0, unbiased=False).mean()),
        "target_variance": float(targets.var(0, unbiased=False).mean()),
        "mean_cosine": float(F.cosine_similarity(pred, targets).mean()),
    }, (smiles, targets, pred)


def binary_metrics(pred, labels):
    labels = labels.int()
    both = bool((labels == 0).any() and (labels == 1).any())
    return {
        "n": len(labels),
        "positives": int(labels.sum()),
        "positive_rate": float(labels.float().mean()) if len(labels) else None,
        "auroc": float(binary_auroc(pred, labels)) if both else None,
        "auprc": float(binary_average_precision(pred, labels)) if both else None,
        "bce": float(F.binary_cross_entropy(pred, labels.float())) if len(labels) else None,
        "brier": float(((pred - labels) ** 2).mean()) if len(labels) else None,
    }


@torch.no_grad()
def evaluate_pairs(model, dm, split, device, output):
    model.to(device).eval()
    dataset = dm._make_dataset(getattr(dm, f"{split}_csv"))
    predictions, labels, smiles, proteins = [], [], [], []
    for batch in dm._loader(dataset, shuffle=False):
        out = model(*(x.to(device) for x in batch[:4]))
        predictions.append(torch.stack([out[k] for k in ("binding", "fragment_binding", "whole_binding")], -1).cpu())
        labels.append(batch[5])
        smiles.extend(batch[6])
        proteins.extend(batch[7])
    preds, y = torch.cat(predictions), torch.cat(labels)
    train = pd.read_csv(dm.train_csv)
    known = set(train["Target Sequence"].str.strip())
    known_mask = torch.tensor([p in known for p in proteins])
    metrics = {}
    for view, col in (("mean", 0), ("fragment", 1), ("whole", 2)):
        metrics[view] = binary_metrics(preds[:, col], y)
        for cohort, mask in (("seen_protein", known_mask), ("unseen_protein", ~known_mask)):
            metrics[f"{view}_{cohort}"] = binary_metrics(preds[mask, col], y[mask])
    # Saved row scores permit alternative metrics and later calibration checks.
    frame = pd.DataFrame(
        {
            "SMILES": smiles,
            "protein_sha256": [hashlib.sha256(p.encode()).hexdigest() for p in proteins],
            "label": y.tolist(),
            "mean": preds[:, 0].tolist(),
            "fragment": preds[:, 1].tolist(),
            "whole": preds[:, 2].tolist(),
        }
    )
    frame.to_csv(output, index=False)
    return metrics


def gradient_diagnostics(model, dm, device):
    model.to(device).eval()
    indices = random.Random(17).sample(range(len(dm.train_dataset)), min(128, len(dm.train_dataset)))
    batch = dm.collator([dm.train_dataset[i] for i in indices])
    out = model(*(x.to(device) for x in batch[:4]))
    y, target = batch[5].to(device), batch[4].to(device)
    losses = {
        "dti": 0.5
        * (F.binary_cross_entropy(out["whole_binding"], y) + F.binary_cross_entropy(out["fragment_binding"], y)),
        "jepa": F.mse_loss(out["jepa_pred"], target),
        "alignment": (1 - out["fragment_molecule_similarity"]).mean(),
    }
    parameter = model.drug_encoder.pre_transform[3].weight
    gradients = {
        name: torch.autograd.grad(loss, parameter, retain_graph=True)[0].flatten() for name, loss in losses.items()
    }
    result = {
        "parameter": "drug_encoder.pre_transform.3.weight",
        "losses": {k: float(v.detach()) for k, v in losses.items()},
        "gradient_norms": {k: float(v.norm()) for k, v in gradients.items()},
    }
    result["gradient_cosines"] = {
        f"{a}_vs_{b}": float(F.cosine_similarity(gradients[a], gradients[b], dim=0))
        for a, b in (("dti", "jepa"), ("jepa", "alignment"), ("dti", "alignment"))
    }
    return result


@torch.no_grad()
def attribution(model, dm, device, count=20):
    frame = pd.read_csv(dm.val_csv)
    candidates = []
    for smiles, group in frame.groupby("SMILES", sort=True):
        if group.Label.nunique() == 2 and len(dm.fragment_fps[smiles]) > 1:
            candidates.append((smiles, group))
    random.Random(123).shuffle(candidates)
    records = []
    model.to(device).eval()
    for smiles, group in candidates[:count]:
        rows = pd.concat([group[group.Label == label].head(1) for label in (1, 0)])
        batch = dm.collator([(r["Target Sequence"], smiles, float(r.Label)) for _, r in rows.iterrows()])
        protein, fps, mask, whole = (t.to(device) for t in batch[:4])
        base = model(protein, fps, mask, whole)
        deltas = []
        for i in range(mask.shape[1]):
            removed = mask.clone()
            removed[:, i] = False
            without = model(protein, fps, removed, whole)
            deltas.append((base["fragment_binding"] - without["fragment_binding"]).cpu())
        delta = torch.stack(deltas, 1)
        records.append(
            {
                "smiles": smiles,
                "labels": [1, 0],
                "fragment_binding": base["fragment_binding"].cpu().tolist(),
                "delta_by_receptor_and_cached_fragment_index": delta.tolist(),
                "top_fragment_indices": delta.argmax(1).tolist(),
                "mean_absolute_receptor_difference": float((delta[0] - delta[1]).abs().mean()),
            }
        )
    return {
        "n_molecules": len(records),
        "method": "leave-one-cached-fragment-out; same molecule, positive and negative receptor",
        "top_fragment_differs_count": sum(
            r["top_fragment_indices"][0] != r["top_fragment_indices"][1] for r in records
        ),
        "interpretation": "Model sensitivity, not causal binding evidence; token indices follow the saved cache.",
        "examples": records,
    }


def reconstruction(model, dm, device, decoder, tokenizer, n=100):
    values, (smiles, targets, pred) = latent_metrics(model, dm, "val", device)
    indices = random.Random(42).sample(range(len(smiles)), min(n, len(smiles)))
    sampled = [smiles[i] for i in indices]
    t = targets[indices]
    p = pred[indices]
    train_smiles, train_targets = molecule_data(dm, "train")
    del train_smiles
    mu = train_targets.mean(0)
    raw = decode_latents(decoder, tokenizer, p, 32, device, seed=123, top_k=1)
    control = decode_latents(decoder, tokenizer, t, 32, device, seed=123, top_k=1)
    target = [canonicalize_largest_fragment(s) for s in sampled]
    decoded = [canonicalize_largest_fragment(s) for s in raw]
    ct = [canonicalize_largest_fragment(s) for s in control]
    metrics, scores = reconstruction_summary(target, decoded, target[1:] + target[:1])
    control_metrics, _ = reconstruction_summary(target, ct, target[1:] + target[:1])
    # Retrieval probes latent identity without relying on the generative decoder.
    distances = torch.cdist(p, targets).square() / targets.shape[1]
    order = distances.argsort(1)
    hit = order == torch.tensor(indices)[:, None]
    rank = hit.int().argmax(1) + 1
    return {
        "split": "val",
        "sample_seed": 42,
        "decode_seed": 123,
        "top_k": 1,
        "full_unique_validation": values,
        "sample_mse": float(F.mse_loss(p, t)),
        "sample_train_mean_mse": float(((t - mu) ** 2).mean()),
        "retrieval_catalog_n": len(targets),
        "retrieval_top1": float((rank <= 1).float().mean()),
        "retrieval_top10": float((rank <= 10).float().mean()),
        "prediction": metrics,
        "target_latent_control": control_metrics,
        "examples": [
            {
                "input_smiles": s,
                "predicted_raw": raw[i],
                "decoded": decoded[i],
                "ecfp": scores["ecfp"][i],
                "rdk": scores["rdk"][i],
                "retrieval_rank": int(rank[i]),
            }
            for i, s in enumerate(sampled)
        ],
    }


def load_checkpoint(model, path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(
        {k.removeprefix("model."): v for k, v in payload["state_dict"].items() if k.startswith("model.")}, strict=True
    )
    return int(payload["epoch"])


def run(args):
    torch.set_num_threads(4)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    decoder = tokenizer = None
    for seed in args.seeds:
        run_dir = args.output / args.arm / f"seed_{seed}"
        if (run_dir / "audit_summary.json").exists():
            print(f"Already complete: {run_dir}", flush=True)
            continue
        if run_dir.exists():
            raise FileExistsError(f"Incomplete run exists; inspect before retrying: {run_dir}")
        with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
            cfg = compose(config_name="config", overrides=[f"experiment=audit_{args.arm}", f"seed={seed}"])
        with open_dict(cfg):
            cfg.runtime.run_id = f"{args.arm}-{seed}"
            cfg.runtime.run_name = cfg.runtime.run_id
            cfg.runtime.run_dir = str(run_dir.resolve())
            cfg.runtime.checkpoint_dir = str((run_dir / "checkpoints").resolve())
            cfg.run.output_root = str(args.output.resolve())
            cfg.trainer.max_epochs = args.epochs
            cfg.trainer.accelerator = "gpu" if args.device.startswith("cuda") else "cpu"
            cfg.trainer.enable_progress_bar = False
            cfg.trainer.num_sanity_val_steps = 0
        pl.seed_everything(seed, workers=True)
        model = instantiate(cfg.model)
        dm = instantiate(cfg.data)
        dm.setup("fit")
        task = instantiate(cfg.task, experiment_config=cfg, model=model)
        callbacks, named = _instantiate_callbacks(cfg.callbacks)
        trainer = instantiate(cfg.trainer, logger=instantiate(cfg.logger), callbacks=callbacks)
        _write_run_manifest(cfg)
        device = torch.device(args.device)
        model.to(device)
        # Evaluation and gradient probes must not perturb training RNG state.
        rng_devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=rng_devices):
            initial, (smiles, target, pred) = latent_metrics(model, dm, "val", device)
            _, train_targets = molecule_data(dm, "train")
            initial["zero_mse"] = float(target.square().mean())
            initial["train_mean_mse"] = float(((target - train_targets.mean(0)) ** 2).mean())
            initial["gradient_diagnostics"] = gradient_diagnostics(model, dm, device)
            write_json(run_dir / "initial_metrics.json", initial)
        task.train()
        trainer.fit(task, datamodule=dm)
        summary = {
            "arm": args.arm,
            "seed": seed,
            "epochs": args.epochs,
            "initial": initial,
            "molecular_representation": cfg.model.molecular_representation,
            "weights": OmegaConf.to_container(cfg.dual_view_loss),
            "checkpoints": {},
        }
        # JEPA selection is shared across reconstruction arms. DTI selection is separately reported.
        for name in ("best_jepa", "best_dti"):
            path = named[name].best_model_path
            epoch = load_checkpoint(model, path)
            val = evaluate_pairs(model, dm, "val", device, run_dir / f"{name}_val_predictions.csv")
            test = evaluate_pairs(model, dm, "test", device, run_dir / f"{name}_test_predictions.csv")
            summary["checkpoints"][name] = {"path": path, "epoch": epoch, "validation": val, "test": test}
        if args.arm.endswith("jepa") or args.arm.endswith("joint"):
            load_checkpoint(model, named["best_jepa"].best_model_path)
            if decoder is None:
                from coati.models.io import load_e3gnn_smiles_clip_e2e

                decoder, tokenizer = load_e3gnn_smiles_clip_e2e(freeze=True, device=str(device), doc_url=args.coati_doc)
                decoder.eval()
            recon = reconstruction(model, dm, device, decoder, tokenizer, args.decode_n)
            write_json(run_dir / "reconstruction_validation.json", recon)
            summary["reconstruction"] = {k: v for k, v in recon.items() if k != "examples"}
        if not args.arm.endswith("jepa"):
            load_checkpoint(model, named["best_dti"].best_model_path)
            sensitivity = attribution(model, dm, device)
            write_json(run_dir / "fragment_sensitivity.json", sensitivity)
            summary["fragment_sensitivity"] = {k: v for k, v in sensitivity.items() if k != "examples"}
        summary["trained_gradient_diagnostics"] = gradient_diagnostics(model, dm, device)
        write_json(run_dir / "audit_summary.json", summary)
        print(
            f"COMPLETE {args.arm} seed={seed}: val AP={summary['checkpoints']['best_dti']['validation']['mean']['auprc']:.4f}",
            flush=True,
        )
        del model, task, trainer, dm, callbacks, named, smiles, target, pred
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--decode-n", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--coati-doc", default=str(ROOT / "models/grande_closed.pkl"))
    run(parser.parse_args())
