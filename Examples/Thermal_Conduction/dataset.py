"""
DiSOL PT dataloader for time-dependent problems.

This module provides a PyTorch Dataset for loading DiSOL training data
stored in a single `.pt` file for dynamic/time-dependent problems.

Expected `.pt` file format (a dict saved by `torch.save`):
    {
        "inp": Tensor [N, C_in, H, W],
        "out": Tensor [N, C_out, H, W, T],
        "lim": Tensor [N, K] or [N]  (optional; e.g., per-sample normalization bounds)
    }

Notes
-----
- For time-dependent problems, the output has an additional time dimension T.
- During training, a random time step is sampled for each sample.
- The input is augmented with initial condition (u0) and normalized time channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import os

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DiSOLPTKeys:
    """Key names used inside the saved `.pt` dict."""
    inp: str = "inp"
    out: str = "out"
    lim: str = "lim"


class DiSOLPTDataset(Dataset):
    """
    Dataset for loading DiSOL samples from a `.pt` file for time-dependent problems.

    Parameters
    ----------
    pt_file:
        Path to a `.pt` file saved via `torch.save`.
    split:
        One of {"train", "val", "all"}.
    train_ratio:
        Fraction of samples used for the training split (the remainder is val).
        Only used when split is "train" or "val".
    n_steps:
        Total number of time steps in the output sequence.
    keys:
        Key names used to read tensors from the loaded dict.

    Returns
    -------
    Each sample is a tuple (x, y, lim):
        x: float32 Tensor [C_in + 2, H, W]  (original input + u0 + time channel)
        y: float32 Tensor [C_out, H, W]     (output at sampled time step)
        lim: float32 Tensor [K] or an empty Tensor if lim is absent
    """

    def __init__(
        self,
        pt_file: Union[str, "os.PathLike[str]"],
        split: str = "train",
        train_ratio: float = 0.8,
        n_steps: int = 50,
        keys: DiSOLPTKeys = DiSOLPTKeys(),
    ) -> None:
        super().__init__()
        self.pt_file = str(pt_file)
        self.split = split.lower().strip()
        self.train_ratio = float(train_ratio)
        self.n_steps = n_steps
        self.keys = keys

        if self.split not in {"train", "val", "all"}:
            raise ValueError(f"split must be one of {{'train','val','all'}}, got: {split}")

        if not (0.0 < self.train_ratio < 1.0) and self.split in {"train", "val"}:
            raise ValueError(f"train_ratio must be in (0, 1) for split='{split}', got: {train_ratio}")

        # Load everything into memory (fastest & simplest for typical dataset sizes).
        data = torch.load(self.pt_file, map_location="cpu")
        if not isinstance(data, dict):
            raise TypeError(f"Expected a dict in '{self.pt_file}', got: {type(data)}")

        if self.keys.inp not in data or self.keys.out not in data:
            raise KeyError(
                f"Missing required keys in '{self.pt_file}'. "
                f"Required: '{self.keys.inp}', '{self.keys.out}'. "
                f"Found: {list(data.keys())}"
            )

        self.inp: Tensor = data[self.keys.inp]
        self.out: Tensor = data[self.keys.out]
        self.lim: Optional[Tensor] = data.get(self.keys.lim, None)

        self._validate_shapes()

        n = int(self.inp.shape[0])
        indices = torch.arange(n, dtype=torch.int64)

        if self.split == "all":
            self.indices = indices
        else:
            split_idx = int(round(n * self.train_ratio))
            if self.split == "train":
                self.indices = indices[:split_idx]
            else:
                self.indices = indices[split_idx:]

    def _validate_shapes(self) -> None:
        """Validate tensor ranks and leading dimensions."""
        if self.inp.ndim != 4:
            raise ValueError(f"'inp' must have shape [N, C_in, H, W], got {tuple(self.inp.shape)}")
        if self.out.ndim != 5:
            raise ValueError(f"'out' must have shape [N, C_out, H, W, T], got {tuple(self.out.shape)}")
        if self.inp.shape[0] != self.out.shape[0]:
            raise ValueError(
                f"Batch dimension mismatch: inp N={self.inp.shape[0]} vs out N={self.out.shape[0]}"
            )
        if self.lim is not None:
            if self.lim.ndim not in {1, 2}:
                raise ValueError(f"'lim' must have shape [N] or [N, K], got {tuple(self.lim.shape)}")
            if self.lim.shape[0] != self.inp.shape[0]:
                raise ValueError(
                    f"Batch dimension mismatch: lim N={self.lim.shape[0]} vs inp N={self.inp.shape[0]}"
                )

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        real_idx = int(self.indices[idx].item())

        inp_base = self.inp[real_idx]   # [C_in, H, W]
        out_full = self.out[real_idx]   # [C_out, H, W, T]

        # Random time step (from 1 to n_steps-1, avoiding t=0 which is initial condition)
        t_idx = np.random.randint(1, self.n_steps)
        t_norm = t_idx / (self.n_steps - 1)  # normalized time in [0, 1]

        # Initial condition (first time step)
        u0 = out_full[0, :, :, 0]  # [H, W]

        # Time channel (constant spatial field with normalized time value)
        H, W = inp_base.shape[1], inp_base.shape[2]
        time_channel = torch.full((1, H, W), t_norm, dtype=torch.float32)

        # Concatenate: [C_in, H, W] + [1, H, W] (u0) + [1, H, W] (time) -> [C_in + 2, H, W]
        x = torch.cat([inp_base, u0.unsqueeze(0), time_channel], dim=0)

        # Output at sampled time step
        y = out_full[:, :, :, t_idx]  # [C_out, H, W]

        x = x.to(torch.float32)
        y = y.to(torch.float32)

        if self.lim is None:
            lim = torch.empty(0, dtype=torch.float32)
        else:
            lim = self.lim[real_idx].to(torch.float32)

        return x, y, lim