"""
Training script for DiSOL (Thermal_Conduction dataset example).

This refactored version follows common "paper-code" conventions:
- Argument parsing (paths, hyperparameters, reproducibility)
- Clear separation of concerns (data, model, optimization, train/eval loops)
- Checkpointing (latest + best) and resumability
- JSON export of the model configuration and training metadata

Assumptions:
- The dataset returns a tuple (x, y, s) where s can be ignored (or logged) for PEQ.
- Input tensor x has shape (B, C_in, H, W); target y has shape (B, C_out, H, W).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ------------------------------
# Project imports
# ------------------------------
# The user renamed the refactored model file to "DiSOL.py".
# To keep compatibility with earlier naming, we try a few import patterns.
from DiSOL import DiSOL, ModelCfg  # type: ignore
from dataset import DiSOLPTDataset


# ======================================================
# Loss / metrics
# ======================================================
def field_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Default supervised loss: mean absolute error over the field."""
    return F.l1_loss(pred, target)


@torch.no_grad()
def compute_running_avg(running_sum: float, step: int) -> float:
    return running_sum / max(step, 1)


# ======================================================
# Data
# ======================================================
def build_loaders(
    pt_path: str,
    batch_size: int,
    num_workers_train: int,
    num_workers_val: int,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Construct training and validation data loaders."""
    train_ds = DiSOLPTDataset(pt_path, split="train", n_steps=10) 
    val_ds = DiSOLPTDataset(pt_path, split="val", n_steps=10)
    # Training and validation on only the first 10 time steps: n_steps=10

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers_train,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers_val,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


# ======================================================
# Model configuration
# ======================================================
def default_model_cfg() -> Dict[str, Any]:
    """
    Default DiSOL configuration that matches the user's original training script.

    You can also load an external JSON config and override these values.
    """
    return dict(
        in_channels=5,
        out_channels=1,
        # Geometry/physics fusion (FiLM + optional SE)
        if_feature_fusion=False,
        geom_idx=[0, 1],
        geom_ch=32,
        phys_ch=64,
        film_hidden=128,
        film_use_beta=True,
        fusion_use_se=True,
        fusion_expand=2,
        # Backbone / assembly pathway
        if_U_net=True,
        if_gate=True,
        base_ch=20,
        depth=4,
        norm="gn",
        # Optional spatial attention
        if_SA=True,
        SA_kernel=5,
        # Readout MLP
        mlp_hidden=64,
        mlp_layers=4,
        mlp_dropout=0.0,
    )


def load_cfg_from_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cfg_to_serializable(cfg_obj: Any) -> Dict[str, Any]:
    """
    Convert a cfg object (dataclass or plain object) to a JSON-serializable dict.
    """
    if is_dataclass(cfg_obj):
        return asdict(cfg_obj)
    if hasattr(cfg_obj, "__dict__"):
        return dict(cfg_obj.__dict__)
    if isinstance(cfg_obj, dict):
        return cfg_obj
    raise TypeError(f"Unsupported cfg type: {type(cfg_obj)}")


def build_model(cfg_dict: Dict[str, Any], device: torch.device) -> nn.Module:
    """Instantiate DiSOL from a config dictionary."""
    cfg = ModelCfg(**cfg_dict)  # type: ignore[arg-type]
    model = DiSOL(cfg).to(device)
    return model


# ======================================================
# Checkpointing
# ======================================================
def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    best_val: float,
    history: Dict[str, Any],
    amp_scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> None:
    """Save a training checkpoint."""
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "opt_state_dict": optimizer.state_dict(),
        "best_val": float(best_val),
        "history": history,
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if amp_scaler is not None:
        payload["amp_scaler_state_dict"] = amp_scaler.state_dict()

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    amp_scaler: Optional[torch.cuda.amp.GradScaler] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """Load a checkpoint into model/optimizer/scheduler/scaler and return the payload."""
    payload = torch.load(str(path), map_location=map_location)

    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "opt_state_dict" in payload:
        optimizer.load_state_dict(payload["opt_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in payload:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if amp_scaler is not None and "amp_scaler_state_dict" in payload:
        amp_scaler.load_state_dict(payload["amp_scaler_state_dict"])

    return payload


# ======================================================
# Train / eval loops
# ======================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_amp: bool = False,
    amp_scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> float:
    model.train()
    running = 0.0

    pbar = tqdm(
        loader,
        total=len(loader),
        desc=f"Epoch {epoch:04d} [Train]",
        ncols=120,
        leave=True,
    )

    for step, (x, y, _s) in enumerate(pbar, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            if amp_scaler is None:
                raise RuntimeError("AMP is enabled but GradScaler is None.")
            with torch.cuda.amp.autocast():
                pred = model(x)
                loss = field_l1_loss(pred, y)
            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            pred = model(x)
            loss = field_l1_loss(pred, y)
            loss.backward()
            optimizer.step()

        running += float(loss.item())
        avg = running / step

        pbar.set_postfix(
            loss=f"{loss.item():.4e}",
            avg=f"{avg:.4e}",
            lr=f"{optimizer.param_groups[0]['lr']:.1e}",
        )

    return running / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> float:
    model.eval()
    running = 0.0

    pbar = tqdm(
        loader,
        total=len(loader),
        desc=f"Epoch {epoch:04d} [Val ]",
        ncols=120,
        leave=False,
    )

    for step, (x, y, _s) in enumerate(pbar, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        loss = field_l1_loss(pred, y)

        running += float(loss.item())
        avg = running / step

        pbar.set_postfix(loss=f"{loss.item():.4e}", avg=f"{avg:.4e}")

    return running / max(len(loader), 1)


# ======================================================
# CLI / main
# ======================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DiSOL on PEQ dataset (.pt).")

    # Paths
    parser.add_argument("--pt_path", type=str, default="./data/PEQ_data/training_data_PEQ_difsource.pt")
    parser.add_argument("--save_dir", type=str, default="./model_ckpt")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    # Scheduler
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=10)

    # Dataloader
    parser.add_argument("--num_workers_train", type=int, default=8)
    parser.add_argument("--num_workers_val", type=int, default=4)

    # Reproducibility / device
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic CuDNN (slower).")

    # Checkpointing
    parser.add_argument("--save_every", type=int, default=100, help="Save an epoch checkpoint every N epochs.")
    parser.add_argument("--resume", type=str, default="", help="Path to a checkpoint (.pth) to resume from.")
    parser.add_argument("--save_best", action="store_true", help="Also save the best checkpoint based on val loss.")

    # Model config
    parser.add_argument("--cfg_json", type=str, default="", help="Optional JSON config to override default model cfg.")

    # Mixed precision
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision (CUDA only).")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Device selection
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(args.device)

    # Determinism
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build loaders
    train_loader, val_loader = build_loaders(
        pt_path=args.pt_path,
        batch_size=args.batch_size,
        num_workers_train=args.num_workers_train,
        num_workers_val=args.num_workers_val,
        pin_memory=(device.type == "cuda"),
    )

    # Build model cfg
    cfg_dict = default_model_cfg()
    if args.cfg_json:
        cfg_override = load_cfg_from_json(args.cfg_json)
        cfg_dict.update(cfg_override)

    # Instantiate model
    model = build_model(cfg_dict, device=device)

    # Optimizer / scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
    )

    # AMP
    use_amp = bool(args.amp and device.type == "cuda")
    amp_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Save config + metadata for reproducibility
    save_json(cfg_dict, save_dir / "model_cfg.json")
    save_json(
        {
            "pt_path": args.pt_path,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lr_factor": args.lr_factor,
            "lr_patience": args.lr_patience,
            "device": args.device,
            "amp": use_amp,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        save_dir / "train_args.json",
    )

    # Resume if requested
    start_epoch = 1
    best_val = float("inf")
    history: Dict[str, Any] = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
    }

    if args.resume:
        ckpt_path = Path(args.resume)
        payload = load_checkpoint(
            ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=None,  # scheduler is ReduceLROnPlateau (not _LRScheduler); we step manually below
            amp_scaler=amp_scaler if use_amp else None,
            map_location=device,
        )
        start_epoch = int(payload.get("epoch", 0)) + 1
        best_val = float(payload.get("best_val", best_val))
        history = payload.get("history", history)
        print(f"[Resume] Loaded checkpoint from: {ckpt_path}")
        print(f"[Resume] start_epoch={start_epoch}, best_val={best_val:.4e}")

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            use_amp=use_amp,
            amp_scaler=amp_scaler if use_amp else None,
        )

        val_loss = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            epoch=epoch,
        )

        # Scheduler step (ReduceLROnPlateau expects the monitored metric)
        scheduler.step(val_loss)

        # History
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        print(f"Epoch {epoch}/{args.epochs} | Train {train_loss:.4e} | Val {val_loss:.4e}")

        # Always save "latest" checkpoint (overwrite)
        save_checkpoint(
            path=save_dir / "ckpt_latest.pth",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=None,  # see note above
            best_val=best_val,
            history=history,
            amp_scaler=amp_scaler if use_amp else None,
        )

        # Save periodic checkpoints
        if args.save_every > 0 and (epoch % args.save_every == 0):
            save_checkpoint(
                path=save_dir / f"ckpt_epoch_{epoch:04d}.pth",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                best_val=best_val,
                history=history,
                amp_scaler=amp_scaler if use_amp else None,
            )

        # Save best (optional)
        if args.save_best and val_loss < best_val:
            best_val = float(val_loss)
            save_checkpoint(
                path=save_dir / "ckpt_best.pth",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                best_val=best_val,
                history=history,
                amp_scaler=amp_scaler if use_amp else None,
            )
            print(f"[Best] Updated best checkpoint: val={best_val:.4e}")

    print("Training finished.")


if __name__ == "__main__":
    main()
