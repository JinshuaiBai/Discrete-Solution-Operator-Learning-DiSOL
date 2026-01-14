"""
DiSOL (Discrete Solution Operator Learning)

This module contains a reference PyTorch implementation of the DiSOL architecture used in the
paper "Discrete Solution Operator Learning for Geometry-dependent PDEs".

Design intent (paper ↔ code mapping):
  1) Local feature operator (learned discrete stencil blocks):
     - implemented by lightweight convolutional blocks (DoubleConv / ConvUnit variants),
       optionally enhanced by channel attention (SE2D / ChanAtten) and gated pointwise mixing.
  2) Multi-scale assembly operator (implicit domain decomposition):
     - implemented by the U-Net encoder–decoder backbone (UNet2D) with skip connections.
     - optional *gated skip* (GatedSkip2D) and mask-consistent routing can be enabled.
  3) Discrete solution readout (problem-solving operator):
     - implemented by a pointwise MLP readout (MLPHead2D) or linear heads.

Geometry feasibility projection:
  - The first input channel is assumed to be the binary geometry mask m(x) (1 inside Ω, 0 outside).
  - If `if_mask=True`, the output is multiplied by the mask to enforce zero outside the domain.

The architecture description and terminology are consistent with Supplementary Information Sec. C.2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional

# ------------------------------
# 1) Lightweight channel fusion: 1x1 expand → GN → GELU → gating → 1x1 squeeze → SE
# ------------------------------
class SE2D(nn.Module):
    def __init__(self, channels: int, r: int = 16):
        super().__init__()
        hidden = max(channels // r, 4)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv1d(hidden, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        s = self.avg(x).view(b, c, 1)     # [B,C,1]
        wts = self.fc(s).view(b, c, 1, 1) # [B,C,1,1]
        return x * wts


class GatedPointwise2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.pw = nn.Conv2d(channels, 2 * channels, kernel_size=1, bias=True)

    def forward(self, x):
        a, g = self.pw(x).chunk(2, dim=1)
        return a * torch.sigmoid(g)


class ChannelFusion2D(nn.Module):
    """
    1x1 expand (factor=expand) → GN → GELU → Gated 1x1 → 1x1 compress → SE → Residual
    """
    def __init__(self, in_channels: int, expand: int = 4, use_se: bool = True, groups_gn: int = 16):
        super().__init__()
        mid = in_channels * expand
        self.pw1 = nn.Conv2d(in_channels, mid, kernel_size=1, bias=False)
        # GroupNorm: more stable for small batches; if channels are too few it falls back to an LN-like behavior
        g = min(groups_gn, mid) if mid % groups_gn == 0 else 1
        self.norm = nn.GroupNorm(g, mid)
        self.act = nn.GELU()
        self.gate = GatedPointwise2D(mid)
        self.pw2 = nn.Conv2d(mid, in_channels, kernel_size=1, bias=False)
        self.se = SE2D(in_channels) if use_se else nn.Identity()

    def forward(self, x):
        y = self.pw1(x)
        y = self.act(self.norm(y))
        y = self.gate(y)
        y = self.pw2(y)
        y = self.se(y)
        return x + y

class InputProj2D(nn.Module):
    """Pure channel projection (no spatial feature extraction)."""
    def __init__(self, in_ch: int, out_ch: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias),
        )

    def forward(self, x):
        return self.proj(x)


# ------------------------------
# 1.1) Input-channel splitting + FiLM
# ------------------------------
class FiLM2D(nn.Module):
    """
    FiLM: x' = (1 + gamma) * x + beta
    gamma and beta are generated from cond_vec (cond_vec comes from global pooling of the geometry/boundary branch).
    """
    def __init__(self, cond_dim: int, feat_ch: int, hidden: int = 128, use_beta: bool = True):
        super().__init__()
        out_dim = feat_ch * (2 if use_beta else 1)
        self.use_beta = use_beta
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x_feat: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        """
        x_feat:  (B, C, H, W)
        cond_vec:(B, cond_dim)
        """
        B, C, _, _ = x_feat.shape
        p = self.mlp(cond_vec)  # (B, C) or (B, 2C)
        if self.use_beta:
            gamma, beta = p.chunk(2, dim=1)  # (B, C), (B, C)
        else:
            gamma, beta = p, None

        gamma = gamma.view(B, C, 1, 1)
        # "Safe" FiLM: use (1 + gamma) to avoid collapsing features to 0 at initialization
        y = (1.0 + gamma) * x_feat

        if beta is not None:
            beta = beta.view(B, C, 1, 1)
            y = y + beta
        return y


class SplitFiLMFusion2D(nn.Module):
    """
    Split input channels into geometry/BC group and physics group (forcing/coeff),
    encode them separately, apply FiLM modulation (conditioned on geometry) to physics features,
    then fuse and output a single feature map.

    Usage:
      fusion = SplitFiLMFusion2D(in_channels=K, geom_idx=[...], out_channels=base_ch)
      feat = fusion(x)  # (B, out_channels, H, W)
    """
    def __init__(
        self,
        in_channels: int,
        geom_idx: Sequence[int],
        out_channels: int,
        geom_ch: int = 32,
        phys_ch: int = 64,
        film_hidden: int = 128,
        norm: str = "gn",
        groups_gn: int = 8,
        use_beta: bool = True,
        fuse_expand: int = 2,
        fuse_use_se: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.geom_idx = sorted(list(geom_idx))
        self.phys_idx = [i for i in range(in_channels) if i not in self.geom_idx]

        assert len(self.geom_idx) > 0, "geom_idx must be non-empty."
        assert len(self.phys_idx) > 0, "physics channels must be non-empty."

        geom_in = len(self.geom_idx)
        phys_in = len(self.phys_idx)

        # ---------- norms ----------
        def make_norm(c):
            if norm == 'bn':
                return nn.BatchNorm2d(c)
            elif norm == 'ln':
                return nn.GroupNorm(1, c)
            else:
                g = min(groups_gn, c) if c % groups_gn == 0 else 1
                return nn.GroupNorm(g, c)

        # ---------- geometry stem ----------
        self.geom_stem = nn.Sequential(
            nn.Conv2d(geom_in, geom_ch, kernel_size=3, padding=1, bias=False),
            make_norm(geom_ch),
            nn.GELU(),
            nn.Conv2d(geom_ch, geom_ch, kernel_size=3, padding=1, bias=False),
            make_norm(geom_ch),
            nn.GELU(),
        )
        # geometry global vector
        self.geom_pool = nn.AdaptiveAvgPool2d(1)

        # ---------- physics stem ----------
        self.phys_stem = nn.Sequential(
            nn.Conv2d(phys_in, phys_ch, kernel_size=3, padding=1, bias=False),
            make_norm(phys_ch),
            nn.GELU(),
            nn.Conv2d(phys_ch, phys_ch, kernel_size=3, padding=1, bias=False),
            make_norm(phys_ch),
            nn.GELU(),
        )

        # ---------- FiLM ----------
        # cond_dim = geom_ch (after GAP)
        self.film = FiLM2D(cond_dim=geom_ch, feat_ch=phys_ch, hidden=film_hidden, use_beta=use_beta)

        # ---------- fuse (concat -> 1x1 compress) ----------
        self.fuse_pw = nn.Conv2d(geom_ch + phys_ch, out_channels, kernel_size=1, bias=False)

        # Optional: reuse your original ChannelFusion2D for an additional mixing stage
        self.fuse_refine = ChannelFusion2D(out_channels, expand=fuse_expand, use_se=fuse_use_se, groups_gn=16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_channels, H, W)
        return: (B, out_channels, H, W)
        """
        # split
        x_geom = x[:, self.geom_idx, :, :]
        x_phys = x[:, self.phys_idx, :, :]

        # stems
        g_feat = self.geom_stem(x_geom)  # (B, geom_ch, H, W)
        p_feat = self.phys_stem(x_phys)  # (B, phys_ch, H, W)

        # cond vec from geometry
        cond = self.geom_pool(g_feat).flatten(1)  # (B, geom_ch)

        # FiLM modulate physics features
        p_feat = self.film(p_feat, cond)

        # fuse + refine
        y = self.fuse_pw(torch.cat([g_feat, p_feat], dim=1))
        y = self.fuse_refine(y)
        return y


# ------------------------------
# 2) U-Net primitives with the newly added gated-skip module
# ------------------------------
class GatedSkip2D(nn.Module):
    """
    Gated skip: use decoder features (dec) to generate a gate map g ∈ [0,1],
    then scale encoder skip features (skip) elementwise: skip' = g ⊙ skip
    """
    def __init__(self, skip_ch: int, dec_ch: int, norm: str = 'gn', groups_gn: int = 8):
        super().__init__()

        def make_norm(c):
            if norm == 'bn':
                return nn.BatchNorm2d(c)
            elif norm == 'ln':
                return nn.GroupNorm(1, c)
            else:  # 'gn'
                g = min(groups_gn, c) if c % groups_gn == 0 else 1
                return nn.GroupNorm(g, c)

        # Generate the gate via a 1x1 conv on concatenated [skip, dec] (output channels = skip_ch)
        self.gate = nn.Sequential(
            nn.Conv2d(skip_ch + dec_ch, skip_ch, kernel_size=1, bias=True),
            make_norm(skip_ch),
            nn.GELU(),
            nn.Conv2d(skip_ch, skip_ch, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, skip: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        """
        skip: (B, C_skip, H, W)
        dec : (B, C_dec,  H, W)
        return: gated skip (B, C_skip, H, W)
        """
        g = self.gate(torch.cat([skip, dec], dim=1))
        return skip * g


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, norm='gn', groups_gn=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)

        def make_norm(c):
            if norm == 'bn':
                return nn.BatchNorm2d(c)
            elif norm == 'ln':
                return nn.GroupNorm(1, c)
            else:  # 'gn'
                g = min(groups_gn, c) if c % groups_gn == 0 else 1
                return nn.GroupNorm(g, c)

        self.n1 = make_norm(out_ch)
        self.n2 = make_norm(out_ch)
        self.a1 = nn.GELU()
        self.a2 = nn.GELU()

    def forward(self, x):
        x = self.a1(self.n1(self.conv1(x)))
        x = self.a2(self.n2(self.conv2(x)))
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = DoubleConv(in_ch, out_ch, **kwargs)

    def forward(self, x):
        return self.block(self.pool(x))


class Up(nn.Module):
    """
    Up with optional gated skip fusion.
    """
    def __init__(self, in_ch, skip_ch, out_ch, bilinear=True, use_gating=False, **kwargs):
        super().__init__()
        self.use_gating = use_gating

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.reduce = nn.Conv2d(in_ch, in_ch // 2, kernel_size=1)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.reduce = nn.Identity()

        dec_ch = in_ch // 2  # decoder channels after upsample + reduce

        # >>> New: gating module (optional)
        if self.use_gating:
            # Forward norm/groups kwargs (kept consistent with DoubleConv)
            norm = kwargs.get('norm', 'gn')
            groups_gn = kwargs.get('groups_gn', 8)
            self.gate = GatedSkip2D(skip_ch=skip_ch, dec_ch=dec_ch, norm=norm, groups_gn=groups_gn)
        else:
            self.gate = None

        self.block = DoubleConv(dec_ch + skip_ch, out_ch, **kwargs)

    def forward(self, x, skip):
        x = self.up(x)
        if isinstance(self.reduce, nn.Conv2d):
            x = self.reduce(x)

        # Spatial size alignment (handles odd/even shapes)
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        if dh != 0 or dw != 0:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])

        # >>> New: gated skip (before concatenation)
        if self.gate is not None:
            skip = self.gate(skip, x)

        x = torch.cat([skip, x], dim=1)
        return self.block(x)


class UNet2D(nn.Module):
    def __init__(self, in_ch, base_ch=64, depth=4, norm='gn', use_gating=False):
        super().__init__()
        chs = [base_ch * (2 ** i) for i in range(depth)]
        self.inc = DoubleConv(in_ch, chs[0], norm=norm)

        self.downs = nn.ModuleList()
        for i in range(1, depth):
            self.downs.append(Down(chs[i - 1], chs[i], norm=norm))

        self.ups = nn.ModuleList()
        for up_idx, i in enumerate(range(depth - 1, 0, -1)):
            dec_in = chs[i]
            skip_ch = chs[i - 1]
            out_ch = chs[i - 1]

            # Enable gate only for the deepest two up-fusion stages: up_idx = 0, 1; also set use_gating_2 below
            use_gating_2 = bool(use_gating and (up_idx < 2))
            self.ups.append(
                Up(dec_in, skip_ch, out_ch, bilinear=True, norm=norm, use_gating=use_gating_2)
            )

        self.out_channels = chs[0]

    def forward(self, x):
        skips = []
        x0 = self.inc(x)
        skips.append(x0)
        x_ = x0
        for d in self.downs:
            x_ = d(x_)
            skips.append(x_)
        y = x_
        for i, up in enumerate(self.ups):
            y = up(y, skips[-(i + 2)])
        return y


# ------------------------------
# 3) Spatial attention (CBAM Spatial Attention variant)
# ------------------------------
class SpatialAttention2D(nn.Module):
    """
    Channel-wise avg/max pooling → concat (2 channels) → 7x7 conv → sigmoid → pixel-wise multiply with input
    """
    def __init__(self, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        s = torch.cat([avg, mx], dim=1)
        w = self.act(self.conv(s))
        return x * w

# ------------------------------
# 4) Pixel-wise MLP reconstruction head (1x1 conv as a per-pixel MLP)
# ------------------------------
class MLPHead2D(nn.Module):
    def __init__(self, in_ch, hidden=128, out_ch=1, num_layers=2, dropout=0.0):
        super().__init__()
        layers = []
        ch = in_ch
        for i in range(num_layers - 1):
            layers += [nn.Conv2d(ch, hidden, 1, bias=True), nn.GELU()]
            if dropout > 0:
                layers += [nn.Dropout2d(dropout)]
            ch = hidden
        layers += [nn.Conv2d(ch, out_ch, 1, bias=True)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class LinearHead2D(nn.Module):
    """1x1 conv linear head: minimal-parameter readout."""
    def __init__(self, in_ch: int, out_ch: int = 1, bias: bool = True):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.proj(x)

class LinearParamMatchedHead2D(nn.Module):
    """
    Parameter-matched linear head (no activation):
    1x1 conv: in_ch -> hidden -> out_ch, but overall linear mapping.
    This is used to decouple 'capacity/params' from 'nonlinearity/structure'.
    """
    def __init__(self, in_ch: int, hidden: int = 128, out_ch: int = 1, bias: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=1, bias=bias),
            nn.Conv2d(hidden, out_ch, kernel_size=1, bias=bias),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------
# 5) Other non-critical layers
# ------------------------------
class ConvUnit(nn.Module):
    def __init__(self,channel):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channel, channel, 3,1,1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1),
        )

    def forward(self,x):
        return self.block(x)

class ConvUnits(nn.Module):
    def __init__(self,channel, num_repeat):
        super().__init__()
        self.block = nn.Sequential(
            *[ConvUnit(channel) for _ in range(num_repeat)]
        )

    def forward(self,x):
        return self.block(x)+x

class RCABs(nn.Module):
    def __init__(self,channel,num_rcab):
        super().__init__()
        self.block = nn.Sequential(
            *[RCAB(channel) for _ in range(num_rcab)],
        )
        self.conv = nn.Conv2d(channel, channel, 3,1,1)

    def forward(self,x):
        y = self.block(x)
        y = self.conv(y)
        return x + y

class RCAB(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel, channel, 3, padding=1),
            ChanAtten(channel)
        )

    def forward(self, x):
        return x + self.block(x)

class ChanAtten(nn.Module):
    """Channel attention used in RCAB.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat,
                 squeeze_factor=8):
        super(ChanAtten, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor,1,1,0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat,1,1,0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


# ------------------------------
# 6) Model configuration summary
# ------------------------------
@dataclass
class ModelCfg:
    # -----------------------------
    # Inputs / outputs
    # -----------------------------
    in_channels: int = 5          # e.g. [mask, bcD, bcN, f, kappa]
    out_channels: int = 1         # solution field (pattern)

    # -----------------------------
    # Feature fusion (Split + FiLM)
    # -----------------------------
    if_feature_fusion: bool = True

    # Which channels correspond to geometry / boundary conditions
    # Strongly recommended: mask + BC-type channels
    geom_idx: Tuple[int, ...] = (0, 1, 2)

    geom_ch: int = 32             # feature width for geometry branch (can be smaller)
    phys_ch: int = 64             # feature width for physics branch
    film_hidden: int = 128        # FiLM MLP hidden
    film_use_beta: bool = True    # whether to use beta (usually keep on)

    fusion_expand: int = 8        # ChannelFusion2D expand
    fusion_use_se: bool = True

    # -----------------------------
    # Backbone
    # -----------------------------
    if_U_net: bool = True
    base_ch: int = 32             # with gating, base_ch can be reduced significantly
    depth: int = 4                # UNet depth
    norm: str = "gn"              # 'gn' | 'bn' | 'ln'

    # gated skip (implemented as 'gate only at the two deepest levels')
    if_gate: bool = True

    # -----------------------------
    # Spatial attention (optional)
    # -----------------------------
    if_SA: bool = False           # recommended default False; optional robustness module
    SA_kernel: int = 5

    # -----------------------------
    # MLP Head
    # -----------------------------
    mlp_hidden: int = 128
    mlp_layers: int = 2
    mlp_dropout: float = 0.0

    # -----------------------------
    # RCAB / fallback backbone (mostly unused)
    # -----------------------------
    num_RCABs: int = 0
    num_RCAB: int = 0

    # -----------------------------
    # Other (extensible)
    # -----------------------------
    groups_gn: int = 8


# ------------------------------
# 7) Full model: ChannelFusion → UNet → SpatialAttention → MLPHead
# ------------------------------
class DiSOL(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Put channel fusion at the very beginning
        if cfg.if_feature_fusion:
            # geom_idx should at least include geometry mask + boundary type mask
            # e.g. x channels: [mask, bcD, bcN, f, kappa] -> geom_idx=[0,1,2]
            geom_idx = getattr(cfg, "geom_idx", [0, 1, 2])

            self.fusion = SplitFiLMFusion2D(
                in_channels=cfg.in_channels,
                geom_idx=geom_idx,
                out_channels=cfg.base_ch,
                geom_ch=getattr(cfg, "geom_ch", 32),
                phys_ch=getattr(cfg, "phys_ch", 64),
                film_hidden=getattr(cfg, "film_hidden", 128),
                norm=getattr(cfg, "norm", "gn"),
                groups_gn=getattr(cfg, "groups_gn", 8),
                use_beta=getattr(cfg, "film_use_beta", True),
                fuse_expand=getattr(cfg, "fusion_expand", 2),
                fuse_use_se=getattr(cfg, "fusion_use_se", True),
            )
            head_ch = cfg.base_ch
        else:
            self.fusion = ChannelFusion2D(
                cfg.in_channels, expand=cfg.fusion_expand, use_se=cfg.fusion_use_se
            )
            head_ch = cfg.in_channels

        # head
        self.head = nn.Sequential(
            nn.Conv2d(head_ch, cfg.base_ch, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(cfg.base_ch, cfg.base_ch, 3, 1, 1),
        )

        # U-Net
        if cfg.if_U_net:
            self.unet = UNet2D(cfg.base_ch, base_ch=cfg.base_ch, depth=cfg.depth, norm=cfg.norm, use_gating=cfg.if_gate)
        else:
            self.unet = nn.Sequential(
                *[ConvUnits(cfg.base_ch, cfg.num_RCAB * 4) for _ in range(cfg.num_RCABs * 7)]
            )
            self.unet.out_channels = cfg.base_ch

        # Spatial attention
        if cfg.if_SA:
            self.spatial_attn = SpatialAttention2D(kernel_size=cfg.SA_kernel)
        else:
            self.spatial_attn = nn.Sequential(
                nn.Conv2d(self.unet.out_channels, self.unet.out_channels, 3, 1, 1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(self.unet.out_channels, self.unet.out_channels, 3, 1, 1),
            )

        # Per-pixel MLP reconstruction
        self.mlp = MLPHead2D(
            self.unet.out_channels,
            hidden=cfg.mlp_hidden,
            out_ch=cfg.out_channels,
            num_layers=cfg.mlp_layers,
            dropout=cfg.mlp_dropout
        )

    def forward(self, x):
        """
        x: [B, in_channels, H, W]
        return: [B, out_channels, H, W]
        """
        mask = x[:, 0:1, :, :]  # geometry mask
        x = self.fusion(x)
        x = self.head(x)
        feats = self.unet(x)
        feats = self.spatial_attn(feats)
        out = self.mlp(feats)
        return out * mask