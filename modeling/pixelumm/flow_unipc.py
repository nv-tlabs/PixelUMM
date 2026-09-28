# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PixelUMM flow-UniPC inference loop with no PixelUMM runtime dependency."""

from __future__ import annotations

from typing import Callable

import torch

from .flow_unipc_scheduler import FlowUniPCScheduler


_NUM_TRAIN_TIMESTEPS = 1000


@torch.no_grad()
def sample_flow_unipc(
    velocity_fn: Callable[[torch.Tensor, float], torch.Tensor],
    initial_noise: torch.Tensor,
    *,
    num_steps: int = 35,
    shift: float = 1.0,
) -> torch.Tensor:
    """Integrate a PixelUMM flow-prediction trajectory with UniPC.

    The public callback receives normalized timesteps in ``[0, 1]``. The
    local scheduler retains the exact R07 discrete ``[0, 1000]`` convention
    internally and keeps the comparatively small solver state in FP32.
    """
    if not torch.is_floating_point(initial_noise):
        raise TypeError("UniPC initial_noise must be floating point")
    if num_steps < 1:
        raise ValueError(f"UniPC num_steps must be positive, got {num_steps}")
    if not torch.isfinite(torch.tensor(shift)) or shift <= 0:
        raise ValueError(f"UniPC shift must be finite and positive, got {shift}")

    solver_state = (
        initial_noise.float()
        if initial_noise.dtype in (torch.float16, torch.bfloat16)
        else initial_noise
    )
    scheduler = FlowUniPCScheduler(num_train_timesteps=_NUM_TRAIN_TIMESTEPS)
    scheduler.set_timesteps(
        int(num_steps),
        device=initial_noise.device,
        shift=float(shift),
    )

    latent = solver_state
    for timestep in scheduler.timesteps:
        normalized_timestep = float(timestep.item()) / _NUM_TRAIN_TIMESTEPS
        velocity = velocity_fn(latent, normalized_timestep)
        if velocity.shape != latent.shape:
            raise ValueError(
                "UniPC velocity_fn must preserve sample shape: "
                f"expected {tuple(latent.shape)}, got {tuple(velocity.shape)}"
            )
        latent = scheduler.step(
            model_output=velocity,
            timestep=timestep,
            sample=latent.unsqueeze(0),
        ).squeeze(0)

    return latent


__all__ = ["sample_flow_unipc"]
