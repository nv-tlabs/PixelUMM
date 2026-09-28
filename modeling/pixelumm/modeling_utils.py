# Copyright (c) 2022 Facebook, Inc. and its affiliates.
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: CC BY-NC 4.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under CC BY-NC 4.0, with the full license text
# available at https://github.com/facebookresearch/DiT/blob/main/LICENSE.txt.
#
# This modified file is released under the same license.

import math

import numpy as np
import torch
from torch import nn

# --------------------------------------------------------
# TimestepEmbedder
# Reference:
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
# --------------------------------------------------------
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


# ======================== Pixel-space modules ========================

def patchify_image(image, patch_size: int, max_size: int):
    """Patchify a PIL image into flat pixel patches in [-1, 1].

    Args:
        image: PIL.Image (any size)
        patch_size: patch size (e.g. 16)
        max_size: longest edge resize target
    Returns:
        patches: (num_patches, patch_size*patch_size*3)
        shape: (h_patches, w_patches)
    """
    from PIL import Image
    img = image.convert("RGB")
    orig_w, orig_h = img.size
    scale = max_size / max(orig_w, orig_h)
    new_w = max(patch_size, round(orig_w * scale / patch_size) * patch_size)
    new_h = max(patch_size, round(orig_h * scale / patch_size) * patch_size)
    img = img.resize((new_w, new_h), Image.BICUBIC)
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0 * 2.0 - 1.0
    H, W, C = img_tensor.shape
    h_patches = H // patch_size
    w_patches = W // patch_size
    patches = img_tensor.reshape(h_patches, patch_size, w_patches, patch_size, C)
    patches = patches.permute(0, 2, 1, 3, 4).reshape(-1, patch_size * patch_size * C)
    return patches, (h_patches, w_patches)




class RawPixelPatchLinearEmbed(nn.Module):
    """Single-linear image patch embedding: raw pixel patch -> hidden_size."""

    def __init__(self, patch_size=32, in_channels=3, hidden_size=3584):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.patch_dim = patch_size * patch_size * in_channels
        self.proj = nn.Linear(self.patch_dim, hidden_size, bias=True)

    def forward(self, pixel_patches):
        """pixel_patches: (num_patches, patch_size*patch_size*3) -> (num_patches, hidden_size)"""
        if pixel_patches.shape[-1] != self.patch_dim:
            raise ValueError(
                f"RawPixelPatchLinearEmbed expected last dim {self.patch_dim} "
                f"for {self.patch_size}x{self.patch_size}x{self.in_channels} patches, "
                f"got {pixel_patches.shape[-1]}"
            )
        return self.proj(pixel_patches)


class RawPixelVideoTubeLinearEmbed(nn.Module):
    """Single-linear video tube embedding: raw pixel tube -> hidden_size."""

    def __init__(self, patch_size=32, in_channels=3, temporal_patch_size=4, hidden_size=3584):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.temporal_patch_size = temporal_patch_size
        self.patch_dim = patch_size * patch_size * in_channels
        self.tube_dim = temporal_patch_size * self.patch_dim
        self.proj = nn.Linear(self.tube_dim, hidden_size, bias=True)

    def forward(self, pixel_video_tubes):
        """pixel_video_tubes: (num_tubes, temporal_patch_size*patch_size*patch_size*3) -> hidden."""
        if pixel_video_tubes.shape[-1] != self.tube_dim:
            raise ValueError(
                f"RawPixelVideoTubeLinearEmbed expected last dim {self.tube_dim} "
                f"for {self.temporal_patch_size}x{self.patch_size}x{self.patch_size}x{self.in_channels} tubes, "
                f"got {pixel_video_tubes.shape[-1]}"
            )
        return self.proj(pixel_video_tubes)



class UnpatchifyHead(nn.Module):
    """Predict pixel patches from hidden states. Zero-initialized output."""
    def __init__(self, hidden_size=3584, patch_size=16, out_channels=3):
        super().__init__()
        self.patch_dim = patch_size * patch_size * out_channels
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, self.patch_dim, bias=True)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, hidden_states):
        """Project hidden states to raw RGB patches."""
        return self.linear(self.norm(hidden_states))


class VideoUnpatchifyHead(nn.Module):
    """Predict raw video tube patches from hidden states."""

    def __init__(self, hidden_size=3584, patch_size=16, temporal_patch_size=4, out_channels=3):
        super().__init__()
        self.patch_dim = temporal_patch_size * patch_size * patch_size * out_channels
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, self.patch_dim, bias=True)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, hidden_states):
        return self.linear(self.norm(hidden_states))
