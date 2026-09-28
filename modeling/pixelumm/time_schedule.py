# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Released PixelUMM noise-time sampling and spatial shift."""

from __future__ import annotations

import torch


def sample_logit_normal_noise_time(
    raw: torch.Tensor,
    *,
    logit_mean: float,
    logit_std: float,
) -> torch.Tensor:
    """Map N(0,1) variates to PixelUMM noise time; -inf means clean."""

    if logit_std <= 0:
        raise ValueError(f"logit_std must be positive, got {logit_std}")
    noise_time = torch.sigmoid(raw * float(logit_std) + float(logit_mean))
    return torch.where(torch.isneginf(raw), torch.zeros_like(noise_time), noise_time)


def pixelumm_resolution_shift(height: int, width: int) -> float:
    """Select the released spatial shift from the target short edge."""

    height = int(height)
    width = int(width)
    if height <= 0 or width <= 0:
        raise ValueError(f"Spatial dimensions must be positive, got {(height, width)}")
    short_edge = min(height, width)
    if short_edge <= 256:
        return 1.0
    if short_edge <= 640:
        return 2.0
    if short_edge <= 960:
        return 3.0
    raise ValueError(
        "PixelUMM resolution shift supports short edges up to 960 px; "
        f"got shape {(height, width)}"
    )


def flow_shift_noise_timestep(t: torch.Tensor, shift: float) -> torch.Tensor:
    """Shift PixelUMM noise time toward noise for shifts greater than one."""

    if shift <= 0:
        raise ValueError(f"shift must be positive, got {shift}")
    return shift * t / (1 + (shift - 1) * t)


__all__ = [
    "flow_shift_noise_timestep",
    "pixelumm_resolution_shift",
    "sample_logit_normal_noise_time",
]
