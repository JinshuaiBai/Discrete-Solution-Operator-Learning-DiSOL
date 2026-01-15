"""
DiSOL testing / evaluation script (Thermal_Conduction dataset) - Time-dependent version

Assumptions:
- Your dataset is a .pt file: torch.load(pt_path) returns a dict with:
  - "inp": Tensor [N, Cin, H, W]
  - "out": Tensor [N, Cout, H, W, T]
  - optional "lim": Tensor [N, K]  (not required for testing metrics)
- Geometry/active-region mask is stored in input channel `mask_channel` (default 0).
- Model file is DiSOL.py (your refactored English version), exposing DiSOL and ModelCfg.
- Dataloader file is dataset.py exposing DiSOLPTDataset.

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
    from DiSOL import DiSOL, ModelCfg
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
        if "model_cfg" in ckpt and isinstance(ckpt["model_cfg"], dict):
            cfg = ckpt["model_cfg"]
            return ModelCfg(**cfg) if ModelCfg is not None else cfg
        if "cfg" in ckpt and isinstance(ckpt["cfg"], dict):
            cfg = ckpt["cfg"]
            return ModelCfg(**cfg) if ModelCfg is not None else cfg

    # fallback: minimal safe defaults for time-dependent problem
    cfg = dict(
        in_channels=5,  # 3 (geometry/BC) + 1 (u0) + 1 (time)
        out_channels=1,
        if_feature_fusion=False,
        geom_idx=[0, 1, 2, 3, 4],
        geom_ch=64,
        phys_ch=4,
        film_hidden=64,
        film_use_beta=True,
        fusion_use_se=True,
        fusion_expand=8,
        if_U_net=True,
        if_gate=True,
        base_ch=32,
        depth=5,
        norm="gn",
        if_SA=True,
        SA_kernel=5,
        mlp_hidden=64,
        mlp_layers=4,
        mlp_dropout=0.0,
    )
    return ModelCfg(**cfg) if ModelCfg is not None else cfg


# -----------------------------
# Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("DiSOL Test Script (.pt dataset) - Time-dependent")
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

    p.add_argument("--vis_n", type=int, default=4, help="Save N visualization PNGs (0 disables)")
    p.add_argument("--ood_pt_path", type=str, default=None, help="Path to OOD test .pt file (optional)")

    # Time-dependent specific arguments
    p.add_argument("--n_steps", type=int, default=50, help="Total number of time steps")
    p.add_argument("--vis_time_steps", type=int, nargs='+', default=[10, 25, 49], 
                   help="Time steps to visualize (list of integers)")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    safe_mkdir(args.out_dir)

    # Dataset
    from dataset import DiSOLPTDataset

    ds = DiSOLPTDataset(
        pt_file=args.pt_path,
        split=args.split,
        train_ratio=args.train_ratio,
        n_steps=args.n_steps,
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
    if isinstance(ckpt_obj, dict) and ("model_state" in ckpt_obj or "state_dict" in ckpt_obj):
        state_dict = ckpt_obj.get("model_state", ckpt_obj.get("state_dict"))
        ckpt_dict = ckpt_obj
    elif isinstance(ckpt_obj, dict):
        state_dict = ckpt_obj
        ckpt_dict = None
    else:
        raise ValueError("Unsupported checkpoint format.")

    model_cfg = load_model_cfg(args.model_cfg_json, ckpt_dict, ModelCfg)
    model = DiSOL(model_cfg) if ModelCfg is not None else DiSOL(model_cfg)
    model.load_state_dict(ckpt_obj['model_state_dict'], strict=True)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    model.to(device)
    model.eval()

    # Optional visualization for time-dependent problem
    if args.vis_n > 0:
        import matplotlib.pyplot as plt

        # Time steps to visualize
        time_steps = [0, 3, 6, 9, 14, 19, 24]
        n_future_frames = 3
        
        # Time mapping: step 0 = 2s, step 49 = 100s
        T_START = 2.0
        T_END = 100.0
        
        def step_to_time(step, n_steps=50, t_start=T_START, t_end=T_END):
            """Convert step index to time in seconds."""
            return t_start + (step / (n_steps - 1)) * (t_end - t_start)

        # Load raw data for visualization (need full time series)
        raw_data = torch.load(args.pt_path, map_location="cpu")
        inp_raw = raw_data["inp"]  # [N, C_in, H, W]
        out_raw = raw_data["out"]  # [N, C_out, H, W, T]

        # Get validation indices
        n_total = inp_raw.shape[0]
        split_idx = int(round(n_total * args.train_ratio))
        if args.split == "train":
            vis_indices = list(range(min(args.vis_n, split_idx)))
        elif args.split == "val":
            vis_indices = list(range(split_idx, min(split_idx + args.vis_n, n_total)))
        else:
            vis_indices = list(range(min(args.vis_n, n_total)))

        model.eval()
        n_time_steps = len(time_steps)
        
        for sample_idx, real_idx in enumerate(vis_indices):
            inp_base = inp_raw[real_idx]  # [C_in, H, W]
            out_full = out_raw[real_idx]  # [C_out, H, W, T]
            
            n_out_channels = out_full.shape[0]
            
            # Figure layout: 3 rows (GT, Pred, Error) x n_time_steps columns
            fig, axes = plt.subplots(3, n_time_steps, figsize=(3 * n_time_steps, 9))
            
            # Get mask
            mk_i = (inp_base[0].numpy() > 0.5).astype(np.float32)
            
            # Collect all data first to get consistent color limits
            gt_list = []
            pd_list = []
            err_list = []
            
            for t_idx in time_steps:
                t_norm = t_idx / (args.n_steps - 1)
                
                # Prepare input
                u0 = out_full[0, :, :, 0]  # [H, W]
                H, W = inp_base.shape[1], inp_base.shape[2]
                time_channel = torch.full((1, H, W), t_norm, dtype=torch.float32)
                x = torch.cat([inp_base, u0.unsqueeze(0), time_channel], dim=0)
                x = x.unsqueeze(0).to(device)  # [1, C, H, W]
                
                # Get prediction
                with torch.no_grad():
                    pred = model(x)
                
                gt_i = out_full[0, :, :, t_idx].cpu().numpy()
                pd_i = pred[0, 0].cpu().numpy()
                err_i = np.abs(pd_i - gt_i)
                
                gt_list.append(gt_i * mk_i)
                pd_list.append(pd_i * mk_i)
                err_list.append(err_i * mk_i)
            
            # Get consistent color limits for GT and Pred
            all_vals = np.concatenate([np.array(gt_list).flatten(), np.array(pd_list).flatten()])
            vmin = np.min(all_vals[all_vals != 0]) if np.any(all_vals != 0) else 0
            vmax = np.max(all_vals)
            
            # Get color limit for error
            err_vmax = np.max(np.array(err_list))
            
            # Plot
            for t_plot_idx, t_idx in enumerate(time_steps):
                t_seconds = step_to_time(t_idx, args.n_steps)
                is_future = t_plot_idx >= (n_time_steps - n_future_frames)
                future_suffix = " (future)" if is_future else ""
                
                # Row 0: Ground Truth
                ax_gt = axes[0, t_plot_idx]
                im_gt = ax_gt.imshow(gt_list[t_plot_idx], vmin=vmin, vmax=vmax)
                ax_gt.set_title(f"GT t={t_seconds:.0f}s{future_suffix}")
                ax_gt.axis("off")
                
                # Row 1: Prediction
                ax_pd = axes[1, t_plot_idx]
                im_pd = ax_pd.imshow(pd_list[t_plot_idx], vmin=vmin, vmax=vmax)
                ax_pd.set_title(f"Pred t={t_seconds:.0f}s{future_suffix}")
                ax_pd.axis("off")
                
                # Row 2: Error
                ax_err = axes[2, t_plot_idx]
                im_err = ax_err.imshow(err_list[t_plot_idx], vmin=0, vmax=err_vmax, cmap='hot')
                ax_err.set_title(f"Error t={t_seconds:.0f}s{future_suffix}")
                ax_err.axis("off")
            
            # Add colorbars on the right side
            fig.subplots_adjust(right=0.9)
            
            # Colorbar for GT/Pred (shared)
            cbar_ax1 = fig.add_axes([0.92, 0.4, 0.02, 0.5])
            cbar1 = fig.colorbar(im_gt, cax=cbar_ax1)
            cbar1.set_label("u", fontsize=10)
            
            # Colorbar for Error
            cbar_ax2 = fig.add_axes([0.92, 0.1, 0.02, 0.2])
            cbar2 = fig.colorbar(im_err, cax=cbar_ax2)
            cbar2.set_label("|Error|", fontsize=10)
            
            fig.suptitle(f"Sample {sample_idx+1}", fontsize=14, y=0.98)
            fig.savefig(os.path.join(args.out_dir, f"Validation_{sample_idx+1:03d}.png"), dpi=200, bbox_inches='tight')
            plt.close(fig)

    # ============================================
    # OOD Test Set Visualization
    # ============================================
    if args.ood_pt_path is not None and os.path.isfile(args.ood_pt_path):
        import matplotlib.pyplot as plt
        
        # Time steps to visualize
        time_steps = [0, 3, 6, 9, 14, 19, 24]
        n_future_frames = 3
        
        # Time mapping: step 0 = 2s, step 49 = 100s
        T_START = 2.0
        T_END = 100.0
        
        def step_to_time(step, n_steps=50, t_start=T_START, t_end=T_END):
            """Convert step index to time in seconds."""
            return t_start + (step / (n_steps - 1)) * (t_end - t_start)
        
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
        n_time_steps = len(time_steps)
        print(f"Found {n_ood} OOD test samples")
        
        model.eval()
        
        for sample_idx in range(n_ood):
            inp_base = ood_inp[sample_idx]  # [C_in, H, W]
            out_full = ood_out[sample_idx]  # [C_out, H, W, T]
            
            # Figure layout: 3 rows (GT, Pred, Error) x n_time_steps columns
            fig, axes = plt.subplots(3, n_time_steps, figsize=(3 * n_time_steps, 9))
            
            # Get mask
            mk_i = (inp_base[0].numpy() > 0.5).astype(np.float32)
            
            # Collect all data first to get consistent color limits
            gt_list = []
            pd_list = []
            err_list = []
            
            for t_idx in time_steps:
                t_norm = t_idx / (args.n_steps - 1)
                
                u0 = out_full[0, :, :, 0]
                H, W = inp_base.shape[1], inp_base.shape[2]
                time_channel = torch.full((1, H, W), t_norm, dtype=torch.float32)
                x = torch.cat([inp_base, u0.unsqueeze(0), time_channel], dim=0)
                x = x.unsqueeze(0).to(device)
                
                with torch.no_grad():
                    pred = model(x)
                
                gt_i = out_full[0, :, :, t_idx].cpu().numpy()
                pd_i = pred[0, 0].cpu().numpy()
                err_i = np.abs(pd_i - gt_i)
                
                gt_list.append(gt_i * mk_i)
                pd_list.append(pd_i * mk_i)
                err_list.append(err_i * mk_i)
            
            # Get consistent color limits for GT and Pred
            all_vals = np.concatenate([np.array(gt_list).flatten(), np.array(pd_list).flatten()])
            vmin = np.min(all_vals[all_vals != 0]) if np.any(all_vals != 0) else 0
            vmax = np.max(all_vals)
            
            # Get color limit for error
            err_vmax = np.max(np.array(err_list))
            
            # Plot
            for t_plot_idx, t_idx in enumerate(time_steps):
                t_seconds = step_to_time(t_idx, args.n_steps)
                is_future = t_plot_idx >= (n_time_steps - n_future_frames)
                future_suffix = " (future)" if is_future else ""
                
                # Row 0: Ground Truth
                ax_gt = axes[0, t_plot_idx]
                im_gt = ax_gt.imshow(gt_list[t_plot_idx], vmin=vmin, vmax=vmax)
                ax_gt.set_title(f"GT t={t_seconds:.0f}s{future_suffix}")
                ax_gt.axis("off")
                
                # Row 1: Prediction
                ax_pd = axes[1, t_plot_idx]
                im_pd = ax_pd.imshow(pd_list[t_plot_idx], vmin=vmin, vmax=vmax)
                ax_pd.set_title(f"Pred t={t_seconds:.0f}s{future_suffix}")
                ax_pd.axis("off")
                
                # Row 2: Error
                ax_err = axes[2, t_plot_idx]
                im_err = ax_err.imshow(err_list[t_plot_idx], vmin=0, vmax=err_vmax, cmap='hot')
                ax_err.set_title(f"Error t={t_seconds:.0f}s{future_suffix}")
                ax_err.axis("off")
            
            # Add colorbars on the right side
            fig.subplots_adjust(right=0.9)
            
            # Colorbar for GT/Pred (shared)
            cbar_ax1 = fig.add_axes([0.92, 0.4, 0.02, 0.5])
            cbar1 = fig.colorbar(im_gt, cax=cbar_ax1)
            cbar1.set_label("u", fontsize=10)
            
            # Colorbar for Error
            cbar_ax2 = fig.add_axes([0.92, 0.1, 0.02, 0.2])
            cbar2 = fig.colorbar(im_err, cax=cbar_ax2)
            cbar2.set_label("|Error|", fontsize=10)
            
            fig.suptitle(f"OOD Sample {sample_idx+1}", fontsize=14, y=0.98)
            fig.savefig(os.path.join(args.out_dir, f"OOD_{sample_idx+1:03d}.png"), dpi=200, bbox_inches='tight')
            plt.close(fig)
        
        print(f"Saved {n_ood} OOD visualization plots")

    print("Test finished.")
    print("Saved to:", args.out_dir)


if __name__ == "__main__":
    main()