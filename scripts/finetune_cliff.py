#!/usr/bin/env python3
"""
Option 2: Activity-cliff fine-tuning.

Fine-tunes the pooler scorer (only) with a pairwise ranking loss on matched
molecular pairs. For each (binder A, nonbinder B, protein P):
  loss = max(0, score(B, P) - score(A, P) + margin)

This teaches the pooler to assign fragment weights such that binders outscore
their structurally-similar nonbinder counterparts — directly supervised by the
activity cliff information from the matched pairs.

The DrugEncoder (FSQ bottleneck) is frozen; only pooling_layer parameters are
trained. This isolates the pooler's learning from the shared representation.

Usage:
  python scripts/finetune_cliff.py \\
    --checkpoint /path/to/80ep.ckpt \\
    --pooling weighted_sum \\
    --pairs scripts/matched_pairs.json \\
    --protein_emb /hpc/.../raygun_embeddings.pt \\
    --output_dir /hpc/.../finetuned_cliff/ \\
    [--epochs 30 --lr 1e-4 --margin 0.1 --val_frac 0.15]
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.attribute_fragments import brics_fragment, morgan_fp


def load_model(checkpoint_path: str, pooling: str):
    from spikes.phase1.fragment_encoder import ConciseFragment, ConciseJEPAFragment
    backbone = ConciseFragment(
        drug_layers=[[32, 32, 32]],
        pooling=pooling,
        ligand_dim=2048,
        residue_dim=1280,
        drug_dim=128,
        proj_dim=256,
        nheads=32,
        activation="tanh",
        drug_quantizer={"type": "fsq"},
    )
    model = ConciseJEPAFragment(
        concise_fragment=backbone,
        smiles_target_dim=256,
        jepa_hidden_dim=512,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state)
    return model


def pipe_split(s):
    return [x for x in s.split("|") if x.strip()] if s else []


def build_frag_fps(smiles: str) -> tuple[torch.Tensor, torch.Tensor]:
    frags = brics_fragment(smiles)
    fps = torch.stack([morgan_fp(f) for f in frags])         # [F, 2048]
    mask = torch.ones(len(frags), dtype=torch.bool)
    return fps, mask


def score_molecule(model, frag_fps, frag_mask, protein_emb):
    """Return binding score for a single molecule-protein pair."""
    out = model(protein_emb.unsqueeze(0), frag_fps.unsqueeze(0), frag_mask.unsqueeze(0))
    return out["binding"].squeeze(0)    # scalar


def run_epoch(model, pairs_data, protein_emb_cache, margin: float, batch_size: int,
              optimizer=None, train: bool = True):
    model.train(train)
    random.shuffle(pairs_data)

    total_loss = 0.0
    n_batches = 0
    n_correct = 0    # score_binder > score_nonbinder

    for i in range(0, len(pairs_data), batch_size):
        batch = pairs_data[i:i + batch_size]
        batch_loss = torch.tensor(0.0, requires_grad=True)
        valid = 0

        proxy_keys = list(protein_emb_cache.keys())
        proxy_mean = torch.stack([protein_emb_cache[k] for k in proxy_keys[:32]]).mean(0)

        for pair in batch:
            protein_seq = pair["target_sequence"]
            if protein_seq in protein_emb_cache:
                protein_emb = protein_emb_cache[protein_seq]
            else:
                protein_emb = proxy_mean  # cross-dataset: use mean proxy

            try:
                fps_A, mask_A = build_frag_fps(pair["smiles_binder"])
                fps_B, mask_B = build_frag_fps(pair["smiles_nonbinder"])
            except Exception:
                continue

            score_A = score_molecule(model, fps_A, mask_A, protein_emb)
            score_B = score_molecule(model, fps_B, mask_B, protein_emb)

            loss_pair = F.relu(score_B - score_A + margin)
            batch_loss = batch_loss + loss_pair
            valid += 1
            if score_A.item() > score_B.item():
                n_correct += 1

        if valid == 0:
            continue

        batch_loss = batch_loss / valid
        total_loss += batch_loss.item()
        n_batches += 1

        if train and optimizer is not None:
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

    avg_loss = total_loss / max(n_batches, 1)
    acc = n_correct / max(len(pairs_data), 1)
    return avg_loss, acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--pooling",     default="weighted_sum",
                   choices=["weighted_sum", "mlp_weighted_sum"])
    p.add_argument("--pairs",       required=True)
    p.add_argument("--protein_emb", required=True)
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--margin",      type=float, default=0.1)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--val_frac",    type=float, default=0.15)
    p.add_argument("--seed",        type=int,   default=42)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    with open(args.pairs) as f:
        all_pairs = json.load(f)
    all_pairs = [p for p in all_pairs if p.get("unique_to_nonbinder")]
    print(f"Loaded {len(all_pairs)} pairs with unique-to-nonbinder fragments")

    random.shuffle(all_pairs)
    n_val = max(1, int(len(all_pairs) * args.val_frac))
    val_pairs   = all_pairs[:n_val]
    train_pairs = all_pairs[n_val:]
    print(f"Train: {len(train_pairs)}  Val: {len(val_pairs)}")

    # Load protein embeddings
    print(f"Loading protein embeddings ...")
    protein_emb = torch.load(args.protein_emb, map_location="cpu", weights_only=False)
    print(f"  {len(protein_emb)} sequences")

    # Load model, freeze DrugEncoder
    model = load_model(args.checkpoint, args.pooling)
    pooler_type = type(model.concise.pooling_layer).__name__
    print(f"Loaded {pooler_type} from {args.checkpoint}")

    # Freeze everything except the pooling scorer
    for name, param in model.named_parameters():
        if "pooling_layer" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} (pooler only)")

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    best_val_loss = float("inf")
    best_ckpt = output_dir / "best_finetune.ckpt"
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_pairs, protein_emb,
            margin=args.margin, batch_size=args.batch_size,
            optimizer=optimizer, train=True
        )
        with torch.no_grad():
            val_loss, val_acc = run_epoch(
                model, val_pairs, protein_emb,
                margin=args.margin, batch_size=args.batch_size,
                optimizer=None, train=False
            )
        scheduler.step()

        row = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
               "val_loss": val_loss, "val_acc": val_acc}
        log_rows.append(row)

        print(f"Epoch {epoch:3d}  train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "pooling": args.pooling,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "args": vars(args),
            }, best_ckpt)
            print(f"  *** New best checkpoint saved (val_loss={val_loss:.4f}) ***")

    with open(output_dir / "finetune_log.json", "w") as f:
        json.dump(log_rows, f, indent=2)

    print(f"\nDone. Best val_loss={best_val_loss:.4f}. Checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
