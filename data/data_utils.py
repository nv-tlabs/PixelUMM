# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0


import math

import torch
from torch.nn.attention.flex_attention import or_masks, and_masks


PIXELUMM_VIDEO_MROPE_BASE_FPS = 24.0


def temporal_rope_units_per_second(fps, temporal_compression_factor):
    """Return physical-time RoPE units per second for a temporal token grid."""
    fps = float(fps)
    temporal_compression_factor = float(temporal_compression_factor)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be finite and positive, got {fps}")
    if not math.isfinite(temporal_compression_factor) or temporal_compression_factor <= 0:
        raise ValueError(
            "temporal_compression_factor must be finite and positive, got "
            f"{temporal_compression_factor}"
        )
    return fps / temporal_compression_factor


def create_sparse_mask(document_lens, split_lens, attn_modes, device, total_length=None):
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def full_and_noise_mask(b, h, q_idx, kv_idx):
        return (full_and_noise_seq_id[q_idx] == full_and_noise_seq_id[kv_idx]) & (full_and_noise_seq_id[q_idx] >= 0)

    def remove_noise_mask(b, h, q_idx, kv_idx):
        return (~((noise_seq_id[kv_idx] >= 0) & (noise_seq_id[q_idx] != noise_seq_id[kv_idx])))

    def sample_mask(b, h, q_idx, kv_idx):
        return (document_id[q_idx] >= 0) & (document_id[q_idx] == document_id[kv_idx])

    full_and_noise_tmp = []
    noise_tmp = []

    for i, (length, model) in enumerate(zip(split_lens, attn_modes)):
        value = i if model in ['full', 'noise'] else -1
        full_and_noise_tmp.extend([value] * length)
        value_noise = i if model == 'noise' else -1
        noise_tmp.extend([value_noise] * length)

    logical_length = sum(document_lens)
    total_length = logical_length if total_length is None else int(total_length)
    if total_length < logical_length:
        raise ValueError((total_length, logical_length))
    trailing_padding = total_length - logical_length
    full_and_noise_tmp.extend([-1] * trailing_padding)
    noise_tmp.extend([-1] * trailing_padding)
    full_and_noise_seq_id = torch.Tensor(full_and_noise_tmp).to(device)
    noise_seq_id = torch.Tensor(noise_tmp).to(device)

    document_id = torch.cat(
        [torch.full((l,), i) for i, l in enumerate(document_lens, start=1)]
        + ([torch.full((trailing_padding,), -1)] if trailing_padding else [])
    ).to(device)

    return and_masks(or_masks(causal_mask, full_and_noise_mask), remove_noise_mask, sample_mask)


def tubeify(video, patch_size, temporal_patch_size):
    """Patchify a fixed-length video clip into spatio-temporal tube patches.

    Args:
        video: (T, C, H, W) tensor.
        patch_size: spatial patch size.
        temporal_patch_size: number of frames per tube token.

    Returns:
        (num_tubes, temporal_patch_size * patch_size * patch_size * C)
    """
    t, c, h, w = video.shape
    p = patch_size
    assert t % temporal_patch_size == 0
    assert h % p == 0 and w % p == 0
    num_tubes_t = t // temporal_patch_size
    video = video.reshape(num_tubes_t, temporal_patch_size, c, h // p, p, w // p, p)
    # Canonical video token order is THW:
    # temporal group first, then the full spatial grid for that group.
    video = torch.einsum("utchpwq->uhwtpqc", video)
    video = video.reshape(-1, temporal_patch_size * p ** 2 * c)
    return video


def build_pixelumm_vision_mrope_position_ids(
    temporal_offset,
    t_tokens,
    h_tokens,
    w_tokens,
    device=None,
    fps=None,
    base_fps=PIXELUMM_VIDEO_MROPE_BASE_FPS,
    temporal_compression_factor=4,
):
    """Build PixelUMM VFM unified_3d_mrope ids for one THW-flattened diffusion/pixel grid."""
    if fps is not None:
        tps = temporal_rope_units_per_second(fps, temporal_compression_factor)
        base_tps = temporal_rope_units_per_second(base_fps, temporal_compression_factor)
        t = torch.arange(t_tokens, dtype=torch.float32, device=device)
        t = t / tps * base_tps + float(temporal_offset)
        t = t.view(-1, 1).expand(-1, h_tokens * w_tokens)
    else:
        t = torch.arange(t_tokens, dtype=torch.long, device=device).view(-1, 1).expand(-1, h_tokens * w_tokens)
        t = t + int(temporal_offset)
    h = torch.arange(h_tokens, dtype=torch.long, device=device).view(1, -1, 1).expand(t_tokens, -1, w_tokens)
    w = torch.arange(w_tokens, dtype=torch.long, device=device).view(1, 1, -1).expand(t_tokens, h_tokens, -1)
    if fps is not None:
        ids = torch.stack([t.flatten(), h.flatten().to(torch.float32), w.flatten().to(torch.float32)], dim=0)
    else:
        ids = torch.stack([t.flatten(), h.flatten(), w.flatten()], dim=0)
    next_temporal_offset = math.ceil(ids.max().item()) + 1 if ids.numel() else int(temporal_offset)
    return ids, next_temporal_offset


def untubeify(video_tubes, image_size, num_frames, patch_size):
    """Reconstruct video frames from THW-ordered spatio-temporal tube tokens."""
    H, W = image_size
    h_patches, w_patches = H // patch_size, W // patch_size
    tube_frames = video_tubes.shape[-1] // (patch_size * patch_size * 3)
    if num_frames % tube_frames != 0:
        raise ValueError(
            f"Expected num_frames to be divisible by tube_frames={tube_frames}, got {num_frames}"
        )
    temporal_groups = num_frames // tube_frames
    frames = video_tubes.float().cpu().reshape(
        temporal_groups, h_patches, w_patches, tube_frames, patch_size, patch_size, 3
    )
    frames = frames.permute(0, 3, 1, 4, 2, 5, 6).contiguous().reshape(num_frames, H, W, 3)
    return frames


def add_special_tokens(tokenizer):
    all_special_tokens = []
    for k, v in tokenizer.special_tokens_map.items():
        if isinstance(v, str):
            all_special_tokens.append(v)
        elif isinstance(v, list):
            all_special_tokens += v

    new_tokens = []

    if '<|im_start|>' not in all_special_tokens:
        new_tokens.append('<|im_start|>')

    if '<|im_end|>' not in all_special_tokens:
        new_tokens.append('<|im_end|>')

    if '<|vision_start|>' not in all_special_tokens:
        new_tokens.append('<|vision_start|>')

    if '<|vision_end|>' not in all_special_tokens:
        new_tokens.append('<|vision_end|>')

    num_new_tokens = tokenizer.add_tokens(new_tokens)
    bos_token_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
    eos_token_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    start_of_image = tokenizer.convert_tokens_to_ids('<|vision_start|>')
    end_of_image = tokenizer.convert_tokens_to_ids('<|vision_end|>')

    new_token_ids = dict(
        bos_token_id=bos_token_id, 
        eos_token_id=eos_token_id, 
        start_of_image=start_of_image, 
        end_of_image=end_of_image, 
    )

    return tokenizer, new_token_ids, num_new_tokens


def len2weight(x, loss_reduction='square'):
    if x == 0:
        return x
    if loss_reduction == 'token':
        return 1
    if loss_reduction == 'sample':
        return 1 / x
    if loss_reduction == 'square':
        return 1 / (x ** 0.5)
    raise NotImplementedError(loss_reduction)
