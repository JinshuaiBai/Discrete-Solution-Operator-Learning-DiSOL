"""
DiSOL testing / evaluation script (Advection_Diffusion dataset)

Assumptions:
- Your dataset is a .pt file: torch.load(pt_path) returns a dict with:
  - "inp": Tensor [N, Cin, H, W]
  - "out": Tensor [N, Cout, H, W]
  - optional "lim": Tensor [N, K]  (not required for testing metrics)
- Geometry/active-region mask is stored in input channel `mask_channel` (default 0).
- Model file is DiSOL.py (your refactored English version), exposing DiSOL and Modelcfg/ModelCfg.
- Dataloader file is disol_pt_dataset.py exposing DiSOLPTDataset.

Outputs:
- metrics_summary.json
- metrics_per_sample.csv
- optional: prediction dumps (pred/gt) as .pt
- optional: visualization PNGs for a few samples
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    x: [B, C, H, W]
    mask: [B, 1, H, W] (0/1 or bool)
    """
    mask = mask.float()
    denom = mask.sum().clamp_min(eps)
    return (x * mask).sum() / denom


def masked_l2(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    sqrt(mean(x^2)) over masked region
    """
    return torch.sqrt(masked_mean(x * x, mask, eps=eps).clamp_min(eps))


def compute_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    pred, gt: [B, C, H, W]
    mask: [B, 1, H, W] or None
    Returns scalar metrics averaged over batch.
    """
    err = pred - gt

    if mask is None:
        # global metrics
        mae = err.abs().mean()
        rmse = torch.sqrt((err * err).mean().clamp_min(eps))
        rel_l2 = torch.sqrt((err * err).sum(dim=(1, 2, 3)) / (gt * gt).sum(dim=(1, 2, 3)).clamp_min(eps)).mean()
        rel_l1 = (err.abs().sum(dim=(1, 2, 3)) / gt.abs().sum(dim=(1, 2, 3)).clamp_min(eps)).mean()
        max_err = err.abs().amax(dim=(1, 2, 3)).mean()
    else:
        # masked metrics: compute over active region only
        mae = masked_mean(err.abs(), mask, eps=eps)
        rmse = masked_l2(err, mask, eps=eps)

        # Rel metrics computed in masked region
        num_l2 = (err * err * mask).sum(dim=(1, 2, 3))
        den_l2 = (gt * gt * mask).sum(dim=(1, 2, 3)).clamp_min(eps)
        rel_l2 = torch.sqrt(num_l2 / den_l2).mean()

        num_l1 = (err.abs() * mask).sum(dim=(1, 2, 3))
        den_l1 = (gt.abs() * mask).sum(dim=(1, 2, 3)).clamp_min(eps)
        rel_l1 = (num_l1 / den_l1).mean()

        # Max error inside mask
        masked_err = err.abs() * mask
        # if mask has zeros, masked_err max could be 0 at non-active; still fine
        max_err = masked_err.amax(dim=(1, 2, 3)).mean()

    return {
        "MAE": float(mae.item()),
        "RMSE": float(rmse.item()),
        "RelL1": float(rel_l1.item()),
        "RelL2": float(rel_l2.item()),
        "MaxAbsErr": float(max_err.item()),
    }


def finite_diff_grad(u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simple first-order finite difference gradient on a grid.
    u: [B, C, H, W]
    Returns (du/dx, du/dy) with same shape, padding at boundaries by replication.
    """
    # x direction: W axis
    dudx = u[..., :, 1:] - u[..., :, :-1]
    dudx = F.pad(dudx, (0, 1, 0, 0), mode="replicate")

    # y direction: H axis
    dudy = u[..., 1:, :] - u[..., :-1, :]
    dudy = F.pad(dudy, (0, 0, 0, 1), mode="replicate")
    return dudx, dudy


def compute_grad_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Gradient error metrics based on simple finite differences.
    """
    pdx, pdy = finite_diff_grad(pred)
    gdx, gdy = finite_diff_grad(gt)

    ex = pdx - gdx
    ey = pdy - gdy

    if mask is None:
        mae_gx = ex.abs().mean()
        mae_gy = ey.abs().mean()
        rmse_g = torch.sqrt(((ex * ex + ey * ey) / 2.0).mean().clamp_min(eps))
    else:
        mae_gx = masked_mean(ex.abs(), mask, eps=eps)
        mae_gy = masked_mean(ey.abs(), mask, eps=eps)
        rmse_g = torch.sqrt(masked_mean((ex * ex + ey * ey) / 2.0, mask, eps=eps).clamp_min(eps))

    return {
        "GradMAE_dx": float(mae_gx.item()),
        "GradMAE_dy": float(mae_gy.item()),
        "GradRMSE": float(rmse_g.item()),
    }


# -----------------------------
# Model / Dataset imports
# -----------------------------
def import_model():
    # Expect DiSOL.py in the same folder or in PYTHONPATH
    from DiSOL import DiSOL, ModelCfg  # type: ignore

    return DiSOL, ModelCfg


def load_model_cfg(model_cfg_json: Optional[str], ckpt: Optional[Dict[str, Any]], ModelCfg):
    """
    Returns an object/dict that can be fed to ModelCfg(**cfg) if available,
    else returns a plain dict.
    """
    cfg: Dict[str, Any] = {}

    if model_cfg_json is not None and os.path.isfile(model_cfg_json):
        with open(model_cfg_json, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return ModelCfg(**cfg) if ModelCfg is not None else cfg

    if ckpt is not None:
        # common patterns
        if "model_cfg" in ckpt and isinstance(ckpt["model_cfg"], dict):
            cfg = ckpt["model_cfg"]
            return ModelCfg(**cfg) if ModelCfg is not None else cfg
        if "cfg" in ckpt and isinstance(ckpt["cfg"], dict):
            cfg = ckpt["cfg"]
            return ModelCfg(**cfg) if ModelCfg is not None else cfg

    # fallback: minimal safe defaults (adjust if needed)
    cfg = dict(
        in_channels=3,
        out_channels=1,
        base_ch=8,
        depth=4,
        if_feature_fusion=True,
        fusion_expand=2,
        fusion_use_se=True,
        if_U_net=True,
        out_norm=False,
        geom_idx=(0, 1),
    )
    return ModelCfg(**cfg) if ModelCfg is not None else cfg


# -----------------------------
# Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("DiSOL Test Script (.pt dataset)")
    p.add_argument("--pt_path", type=str, required=True, help="Path to .pt test dataset")
    p.add_argument("--ckpt", type=str, required=True, help="Checkpoint path (.pth/.pt) with model weights")
    p.add_argument("--model_cfg_json", type=str, default=None, help="Optional model_cfg.json saved by training")
    p.add_argument("--out_dir", type=str, default="./output_results", help="Output directory")

    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    p.add_argument("--split", type=str, default="all", choices=["train", "val", "all"])
    p.add_argument("--train_ratio", type=float, default=0.8, help="Used when split != test to slice data")

    p.add_argument("--mask_channel", type=int, default=0, help="Input channel index used as active-region mask")
    p.add_argument("--use_mask_metrics", action="store_true", help="Compute metrics only on active region mask")
    p.add_argument("--apply_mask_to_pred", action="store_true", help="Multiply model output by mask before metrics")

    p.add_argument("--compute_grad_metrics", action="store_true", help="Also compute finite-diff gradient metrics")
    p.add_argument("--dump_predictions", action="store_true", help="Save pred/gt tensors to .pt")
    p.add_argument("--vis_n", type=int, default=0, help="Save N visualization PNGs (0 disables)")

    p.add_argument("--ood_pt_path", type=str, default=None, help="Path to OOD test .pt file (optional)")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    safe_mkdir(args.out_dir)

    # Dataset
    from dataset import DiSOLPTDataset  # your simplified PT-only dataset

    ds = DiSOLPTDataset(
        pt_file=args.pt_path,
        split=args.split,
        train_ratio=args.train_ratio,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    # Model
    DiSOL, ModelCfg = import_model()

    ckpt_obj = torch.load(args.ckpt, weights_only=False, map_location="cpu")
    # support both raw state_dict and dict checkpoints
    if isinstance(ckpt_obj, dict) and ("model_state" in ckpt_obj or "state_dict" in ckpt_obj):
        state_dict = ckpt_obj.get("model_state", ckpt_obj.get("state_dict"))
        ckpt_dict = ckpt_obj
    elif isinstance(ckpt_obj, dict):
        # could be a plain state_dict
        state_dict = ckpt_obj
        ckpt_dict = None
    else:
        raise ValueError("Unsupported checkpoint format.")

    model_cfg = load_model_cfg(args.model_cfg_json, ckpt_dict, ModelCfg)
    model = DiSOL(model_cfg) if ModelCfg is not None else DiSOL(model_cfg)  # keep same call signature
    model.load_state_dict(ckpt_obj['model_state_dict'], strict=True)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    model.to(device)
    model.eval()

    # Evaluation loop
    all_metrics: List[Dict[str, float]] = []
    all_grad_metrics: List[Dict[str, float]] = []
    preds_dump = []
    gts_dump = []

    with torch.no_grad():
        for batch in loader:
            # dataset returns (inp, out, lim) where lim may be empty
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x = batch[0].to(device, non_blocking=True)
                y = batch[1].to(device, non_blocking=True)
            else:
                raise ValueError("Dataset batch should be a tuple/list with at least (inp, out).")

            # mask extraction
            mask = x[:, args.mask_channel : args.mask_channel + 1, ...]
            if mask.dtype != torch.bool:
                # treat as probability/binary image; threshold at 0.5
                mask_bin = (mask > 0.5)
            else:
                mask_bin = mask

            pred = model(x)

            if args.apply_mask_to_pred:
                pred = pred * mask_bin.float()

            metrics = compute_metrics(
                pred, y, mask=mask_bin if args.use_mask_metrics else None
            )
            all_metrics.append(metrics)

            if args.compute_grad_metrics:
                g_metrics = compute_grad_metrics(
                    pred, y, mask=mask_bin if args.use_mask_metrics else None
                )
                all_grad_metrics.append(g_metrics)

            if args.dump_predictions:
                preds_dump.append(pred.detach().cpu())
                gts_dump.append(y.detach().cpu())

    # Aggregate metrics
    def avg_dict(list_of_dicts: List[Dict[str, float]]) -> Dict[str, float]:
        keys = list_of_dicts[0].keys()
        return {k: float(np.mean([d[k] for d in list_of_dicts])) for k in keys}

    summary = avg_dict(all_metrics)
    if args.compute_grad_metrics and len(all_grad_metrics) > 0:
        summary.update(avg_dict(all_grad_metrics))

    # Save summary
    with open(os.path.join(args.out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "pt_path": args.pt_path,
                "ckpt": args.ckpt,
                "split": args.split,
                "use_mask_metrics": args.use_mask_metrics,
                "apply_mask_to_pred": args.apply_mask_to_pred,
                "mask_channel": args.mask_channel,
                "summary": summary,
            },
            f,
            indent=2,
        )

    # Save per-batch metrics as CSV (simple, no pandas dependency)
    csv_path = os.path.join(args.out_dir, "metrics_per_sample.csv")
    keys = list(all_metrics[0].keys())
    if args.compute_grad_metrics and len(all_grad_metrics) > 0:
        keys += list(all_grad_metrics[0].keys())

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for i in range(len(all_metrics)):
            row = {**all_metrics[i]}
            if args.compute_grad_metrics and len(all_grad_metrics) > 0:
                row.update(all_grad_metrics[i])
            f.write(",".join([f"{row[k]:.8e}" for k in keys]) + "\n")

    # Dump predictions
    if args.dump_predictions and len(preds_dump) > 0:
        out_pt = os.path.join(args.out_dir, "pred_gt_dump.pt")
        torch.save(
            {
                "pred": torch.cat(preds_dump, dim=0),
                "gt": torch.cat(gts_dump, dim=0),
            },
            out_pt,
        )

    # Optional visualization
    if args.vis_n > 0:
        import matplotlib.pyplot as plt
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print("matplotlib not available; skip visualization.")
        else:
            # take first batch again for visualization
            model.eval()
            with torch.no_grad():
                batch = next(iter(loader))
                x = batch[0].to(device)
                y = batch[1].to(device)
                mask = x[:, args.mask_channel : args.mask_channel + 1, ...]
                mask_bin = (mask > 0.5) if mask.dtype != torch.bool else mask
                pred = model(x)
                if args.apply_mask_to_pred:
                    pred = pred * mask_bin.float()

            B = min(args.vis_n, x.shape[0])
            for i in range(B):
                gt_i = y[i, 0].detach().cpu().numpy()
                pd_i = pred[i, 0].detach().cpu().numpy()
                mk_i = mask_bin[i, 0].detach().cpu().numpy().astype(np.float32)

                err_i = np.abs(pd_i - gt_i)

                fig = plt.figure(figsize=(10, 3))
                ax1 = fig.add_subplot(1, 4, 1)
                ax2 = fig.add_subplot(1, 4, 2)
                ax3 = fig.add_subplot(1, 4, 3)
                ax4 = fig.add_subplot(1, 4, 4)

                ax1.set_title("GT")
                im1 = ax1.imshow(gt_i)
                plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

                ax2.set_title("Pred")
                im2 = ax2.imshow(pd_i)
                plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

                ax3.set_title("|Err|")
                im3 = ax3.imshow(err_i)
                plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

                ax4.set_title("Mask")
                im4 = ax4.imshow(mk_i)
                plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

                for ax in [ax1, ax2, ax3, ax4]:
                    ax.axis("off")

                fig.tight_layout()
                fig.savefig(os.path.join(args.out_dir, f"Validation_{i+1:03d}.png"), dpi=200)
                plt.close(fig)

    # ============================================
    # OOD Test Set Visualization
    # ============================================
    if args.ood_pt_path is not None and os.path.isfile(args.ood_pt_path):
        import matplotlib.pyplot as plt
        
        print(f"Loading OOD test set from: {args.ood_pt_path}")
        ood_data = torch.load(args.ood_pt_path, weights_only=False, map_location="cpu")
        
        # Support both formats: dict with 'inputs'/'outputs' or 'inp'/'out'
        if 'inputs' in ood_data:
            ood_inp = ood_data['inputs']
            ood_out = ood_data['outputs']
        else:
            ood_inp = ood_data['inp']
            ood_out = ood_data['out']
        
        n_ood = ood_inp.shape[0]
        print(f"Found {n_ood} OOD test samples")
        
        model.eval()
        with torch.no_grad():
            ood_inp_dev = ood_inp.to(device)
            ood_out_dev = ood_out.to(device)
            
            mask = ood_inp_dev[:, args.mask_channel : args.mask_channel + 1, ...]
            mask_bin = (mask > 0.5) if mask.dtype != torch.bool else mask
            
            ood_pred = model(ood_inp_dev)
            if args.apply_mask_to_pred:
                ood_pred = ood_pred * mask_bin.float()
        
        for i in range(n_ood):
            gt_i = ood_out_dev[i, 0].detach().cpu().numpy()
            pd_i = ood_pred[i, 0].detach().cpu().numpy()
            mk_i = mask_bin[i, 0].detach().cpu().numpy().astype(np.float32)
            err_i = np.abs(pd_i - gt_i)
            
            fig = plt.figure(figsize=(10, 3))
            ax1 = fig.add_subplot(1, 4, 1)
            ax2 = fig.add_subplot(1, 4, 2)
            ax3 = fig.add_subplot(1, 4, 3)
            ax4 = fig.add_subplot(1, 4, 4)
            
            ax1.set_title("GT")
            im1 = ax1.imshow(gt_i)
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            
            ax2.set_title("Pred")
            im2 = ax2.imshow(pd_i)
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            
            ax3.set_title("|Err|")
            im3 = ax3.imshow(err_i)
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
            
            ax4.set_title("Mask")
            im4 = ax4.imshow(mk_i)
            plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
            
            for ax in [ax1, ax2, ax3, ax4]:
                ax.axis("off")
            
            fig.tight_layout()
            fig.savefig(os.path.join(args.out_dir, f"OOD_{i+1:03d}.png"), dpi=200)
            plt.close(fig)
        
        print(f"Saved {n_ood} OOD visualization plots")

    print("Test finished.")
    print("Summary metrics:", summary)
    print("Saved to:", args.out_dir)


if __name__ == "__main__":
    main()