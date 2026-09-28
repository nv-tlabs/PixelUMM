# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loss weights used by released PixelUMM training."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def sample_mean_token_weights(
    group_lens: torch.Tensor | Sequence[int],
    *,
    output_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Expand per-group ``1/N`` weights to aligned token weights.

    Summing one group's weighted token losses yields that media item's mean
    loss. Dividing by the global weight denominator therefore computes an
    exact global mean over generated media items, even when ranks pack
    different numbers of examples.
    """
    if not torch.is_tensor(group_lens):
        group_lens = torch.tensor(group_lens, dtype=torch.long, device=device)
    else:
        group_lens = group_lens.to(device=device, dtype=torch.long)
    group_lens = group_lens.reshape(-1)
    if group_lens.numel() == 0 and output_size != 0:
        raise ValueError("Non-empty token output requires at least one loss group")
    if group_lens.device.type == "cpu" and bool((group_lens <= 0).any()):
        raise ValueError(f"Loss group lengths must be positive, got {group_lens.tolist()}")
    token_weights = torch.repeat_interleave(
        torch.reciprocal(group_lens.to(dtype=torch.float32)),
        group_lens,
        output_size=int(output_size),
    )
    if token_weights.shape[0] != int(output_size):
        raise ValueError(
            "Loss group lengths do not match output size: "
            f"{token_weights.shape[0]} vs {output_size}"
        )
    return token_weights


def loss_weights_for_reduction(
    token_losses: torch.Tensor,
    reduction: str,
    *,
    non_token_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return aligned weights for an explicit loss-reduction contract."""
    if reduction == "token":
        return torch.ones(
            token_losses.numel(),
            dtype=torch.float32,
            device=token_losses.device,
        )
    if non_token_weights is None:
        raise ValueError(f"{reduction} reduction requires aligned loss weights")
    non_token_weights = non_token_weights.reshape(-1).to(
        device=token_losses.device,
        dtype=torch.float32,
    )
    if non_token_weights.numel() != token_losses.numel():
        raise ValueError(
            f"{reduction} weight length mismatch: "
            f"{non_token_weights.numel()} vs {token_losses.numel()}"
        )
    return non_token_weights
