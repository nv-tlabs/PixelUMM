# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 The Qwen Team and The HuggingFace Inc. team.
# SPDX-FileCopyrightText: Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# Qwen3 MoT + NaViT runtime for PixelUMM
# Based on Bagel/modeling/bagel/qwen2_navit.py (ByteDance, Apache-2.0).
# PixelUMM keeps only the released SenseNova T/H/W MRoPE topology:
# upstream FlexAttention for packed training and public FlashAttention varlen
# for inference.

import copy
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn
from torch.nn.attention.flex_attention import flex_attention
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3MLP,
    Qwen3RMSNorm,
)
from transformers.utils import ModelOutput

from modeling.qwen3.configuration_qwen3 import Qwen3Config

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None


torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 4096
from torch._inductor.runtime.hints import TRITON_MAX_BLOCK

TRITON_MAX_BLOCK["X"] = 65536
flex_attention = torch.compile(
    flex_attention,
    fullgraph=True,
    dynamic=True,
    mode="default",
)

def _apply_sequence_chunked(
    fn,
    seq: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Apply a token-wise module in bounded sequence chunks.

    Qwen3MLP is independent along the sequence dimension, so splitting that
    dimension preserves its mathematical contract while bounding the transient
    SwiGLU intermediate from ``sequence_length * intermediate_size`` to
    ``chunk_size * intermediate_size``. A non-positive chunk size keeps the
    original path. Empty tensors still call ``fn`` so pixelumm-zero branches keep
    their zero-gradient autograd edges.
    """
    if chunk_size <= 0 or seq.shape[0] <= chunk_size:
        return fn(seq)
    return torch.cat(
        tuple(fn(chunk) for chunk in seq.split(chunk_size, dim=0)),
        dim=0,
    )

def _flash_varlen_inference_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    query_lens,
    key_lens,
    *,
    causal: bool,
) -> torch.Tensor:
    """Run the public FlashAttention varlen API with packed Q/K/V tensors."""

    if flash_attn_varlen_func is None:
        raise RuntimeError(
            "PixelUMM inference requires flash_attn_varlen_func; "
            "install requirements-flash-attn.txt"
        )

    def _as_int_list(lengths) -> list[int]:
        if isinstance(lengths, torch.Tensor):
            lengths = lengths.detach().cpu().tolist()
        return [int(length) for length in lengths]

    query_lengths = _as_int_list(query_lens)
    key_lengths = _as_int_list(key_lens)
    if len(query_lengths) != len(key_lengths):
        raise ValueError("query_lens and key_lens must contain the same number of samples")
    if not query_lengths or any(length <= 0 for length in query_lengths + key_lengths):
        raise ValueError("inference attention segments must be non-empty")
    if sum(query_lengths) != query_states.shape[0]:
        raise ValueError("query_lens do not match the flattened query tensor")
    if sum(key_lengths) != key_states.shape[0] or key_states.shape[0] != value_states.shape[0]:
        raise ValueError("key_lens do not match the flattened key/value tensors")
    if causal and any(
        key_length < query_length
        for query_length, key_length in zip(query_lengths, key_lengths)
    ):
        raise ValueError("causal inference requires key_length >= query_length")
    if query_states.shape[1] % key_states.shape[1] != 0:
        raise ValueError(
            "FlashAttention GQA requires query heads divisible by KV heads: "
            f"query_heads={query_states.shape[1]} kv_heads={key_states.shape[1]}"
        )

    def _cumulative_lengths(lengths: list[int]) -> torch.Tensor:
        cumulative = torch.zeros(
            len(lengths) + 1,
            dtype=torch.int32,
            device=query_states.device,
        )
        cumulative[1:] = torch.tensor(
            lengths,
            dtype=torch.int32,
            device=query_states.device,
        ).cumsum(dim=0)
        return cumulative

    result = flash_attn_varlen_func(
        q=query_states.contiguous(),
        k=key_states.contiguous(),
        v=value_states.contiguous(),
        cu_seqlens_q=_cumulative_lengths(query_lengths),
        cu_seqlens_k=_cumulative_lengths(key_lengths),
        max_seqlen_q=max(query_lengths),
        max_seqlen_k=max(key_lengths),
        causal=causal,
    )
    return result[0] if isinstance(result, tuple) else result

class NaiveCache:
    def __init__(self, num_layers):
        self.key_cache = {k: None for k in range(num_layers)}
        self.value_cache = {k: None for k in range(num_layers)}

    @property
    def num_layers(self):
        return len(self.key_cache)

@dataclass
class BaseNavitOutputWithPast(ModelOutput):
    packed_query_sequence: torch.FloatTensor = None
    past_key_values: Optional[NaiveCache] = None

def _compute_default_rope_parameters(config, device=None, **_kwargs):
    """Default RoPE frequencies, copied from SenseNova's Qwen3 fork."""
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    attention_factor = 1.0
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
    )
    return inv_freq, attention_factor

def rotate_half(x):
    """Rotates half the hidden dims of the input, copied from SenseNova's Qwen3 fork."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Apply RoPE exactly like SenseNova's Qwen3 fork.

    ``position_ids`` is kept for API parity with the fork; it is unused.
    """
    del position_ids
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class SenseNovaQwen3RotaryEmbedding(nn.Module):
    """Qwen3 rotary embedding copied from SenseNova's fork.

    This mirrors ``references/SenseNova-U1/.../modeling_qwen3.py``. SenseNova
    computes RoPE frequencies for a 2x wider axis and keeps every other
    frequency, instead of recomputing a fresh reduced-dim frequency range.
    """

    inv_freq: torch.Tensor

    @staticmethod
    def compute_default_rope_parameters(config, device=None, seq_len=None):
        del seq_len
        inv_freq, attention_scaling = _compute_default_rope_parameters(config, device)

        cfg2 = copy.deepcopy(config)
        head_dim = getattr(cfg2, "head_dim", None)
        if head_dim is None:
            head_dim = cfg2.hidden_size // cfg2.num_attention_heads
            setattr(cfg2, "head_dim", head_dim)
        cfg2.head_dim = int(head_dim) * 2

        inv_freq_full, _ = _compute_default_rope_parameters(cfg2, device)
        return inv_freq_full[::2], attention_scaling

    def __init__(self, config, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.config = config

        if self.rope_type == "default" or self.rope_type is None:
            base_rope_init_fn = self.compute_default_rope_parameters
        else:
            raise ValueError(
                f"SenseNovaQwen3RotaryEmbedding only supports the default release RoPE, got {self.rope_type}"
            )

        def _rope_init_fn_keep_freq_range(cfg, dev=None):
            return base_rope_init_fn(cfg, dev)

        self.rope_init_fn = _rope_init_fn_keep_freq_range

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def pad_sequence(tensor, pad_size):
    H, L, D = tensor.shape
    pad_tensor = tensor.new_zeros((H, pad_size, D))
    return torch.cat([tensor, pad_tensor], dim=1)

class SenseNovaMRoPEMixin:
    def _init_sensenova_mrope(self, config):
        if config.qwen3_mrope_type != "sensenova":
            raise ValueError("PixelUMM requires qwen3_mrope_type='sensenova'")
        if self.head_dim % 4 != 0:
            raise ValueError(
                "SenseNova split Q/K norm requires head_dim divisible by 4, "
                f"got {self.head_dim}"
            )
        self.qwen3_mrope_type = "sensenova"

        t_config = copy.deepcopy(config)
        t_config.head_dim = self.head_dim // 2
        t_config.rope_parameters = copy.deepcopy(config.rope_parameters)
        self.rotary_emb_t = SenseNovaQwen3RotaryEmbedding(config=t_config)

        hw_config = copy.deepcopy(config)
        hw_config.head_dim = self.head_dim // 4
        hw_config.rope_theta = getattr(config, "rope_theta_hw", 10000.0)
        hw_config.max_position_embeddings = getattr(
            config, "max_position_embeddings_hw", 10000
        )
        hw_config.rope_parameters = copy.deepcopy(config.rope_parameters)
        hw_config.rope_parameters["rope_theta"] = hw_config.rope_theta
        self.rotary_emb_hw = SenseNovaQwen3RotaryEmbedding(config=hw_config)

        norm_dim = self.head_dim // 2
        self.q_norm_t = Qwen3RMSNorm(norm_dim, eps=config.rms_norm_eps)
        self.q_norm_hw = Qwen3RMSNorm(norm_dim, eps=config.rms_norm_eps)
        self.k_norm_t = Qwen3RMSNorm(norm_dim, eps=config.rms_norm_eps)
        self.k_norm_hw = Qwen3RMSNorm(norm_dim, eps=config.rms_norm_eps)

    def _remove_unused_base_qk_norms(self):
        num_removed = int(self.q_norm is not None) + int(self.k_norm is not None)
        self.q_norm = None
        self.k_norm = None
        return num_removed

    def _init_sensenova_mot_norms(self, config):
        norm_dim = self.head_dim // 2
        self.q_norm_t_moe_gen = Qwen3RMSNorm(norm_dim, eps=config.rms_norm_eps)
        self.q_norm_hw_moe_gen = Qwen3RMSNorm(norm_dim, eps=config.rms_norm_eps)
        self.k_norm_t_moe_gen = Qwen3RMSNorm(norm_dim, eps=config.rms_norm_eps)
        self.k_norm_hw_moe_gen = Qwen3RMSNorm(norm_dim, eps=config.rms_norm_eps)

    @staticmethod
    def _apply_sensenova_qk_norm(
        query_states, key_states, q_norm_t, q_norm_hw, k_norm_t, k_norm_hw
    ):
        query_states_t, query_states_hw = query_states.chunk(2, dim=-1)
        key_states_t, key_states_hw = key_states.chunk(2, dim=-1)
        return (
            torch.cat(
                [q_norm_t(query_states_t), q_norm_hw(query_states_hw)], dim=-1
            ),
            torch.cat(
                [k_norm_t(key_states_t), k_norm_hw(key_states_hw)], dim=-1
            ),
        )

    def _apply_axis_rope(
        self, query_states, key_states, rotary_emb, position_ids
    ):
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        cos, sin = rotary_emb(query_states, position_ids.unsqueeze(0))
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        return (
            query_states.squeeze(0).transpose(0, 1),
            key_states.squeeze(0).transpose(0, 1),
        )

    def _apply_sensenova_mrope(
        self, query_states, key_states, packed_position_ids
    ):
        packed_position_ids = packed_position_ids[:, : query_states.shape[0]]
        if packed_position_ids.ndim != 2 or packed_position_ids.shape[0] != 3:
            raise ValueError(
                "SenseNova MRoPE expects T/H/W position IDs shaped (3, seq), "
                f"got {tuple(packed_position_ids.shape)}"
            )

        query_t, query_hw = query_states.chunk(2, dim=-1)
        query_h, query_w = query_hw.chunk(2, dim=-1)
        key_t, key_hw = key_states.chunk(2, dim=-1)
        key_h, key_w = key_hw.chunk(2, dim=-1)

        query_t, key_t = self._apply_axis_rope(
            query_t, key_t, self.rotary_emb_t, packed_position_ids[0]
        )
        query_h, key_h = self._apply_axis_rope(
            query_h, key_h, self.rotary_emb_hw, packed_position_ids[1]
        )
        query_w, key_w = self._apply_axis_rope(
            query_w, key_w, self.rotary_emb_hw, packed_position_ids[2]
        )
        return (
            torch.cat([query_t, query_h, query_w], dim=-1),
            torch.cat([key_t, key_h, key_w], dim=-1),
        )

class PackedAttentionMoT(SenseNovaMRoPEMixin, Qwen3Attention):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self._init_sensenova_mrope(config)
        self._init_sensenova_mot_norms(config)

        self.q_proj_moe_gen = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj_moe_gen = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj_moe_gen = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj_moe_gen = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
        )

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: Optional[List[int]],
        attention_mask,
        packed_position_embeddings: torch.Tensor,
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
    ) -> torch.Tensor:
        del sample_lens
        if packed_und_token_indexes is None or packed_gen_token_indexes is None:
            raise ValueError("MoT training requires UND and GEN token indexes")

        query = packed_sequence.new_zeros(
            packed_sequence.shape[0], self.num_heads * self.head_dim
        )
        key = packed_sequence.new_zeros(
            packed_sequence.shape[0],
            self.num_key_value_heads * self.head_dim,
        )
        value = packed_sequence.new_zeros(
            packed_sequence.shape[0],
            self.num_key_value_heads * self.head_dim,
        )
        und = packed_sequence[packed_und_token_indexes]
        gen = packed_sequence[packed_gen_token_indexes]
        query[packed_und_token_indexes] = self.q_proj(und)
        query[packed_gen_token_indexes] = self.q_proj_moe_gen(gen)
        key[packed_und_token_indexes] = self.k_proj(und)
        key[packed_gen_token_indexes] = self.k_proj_moe_gen(gen)
        value[packed_und_token_indexes] = self.v_proj(und)
        value[packed_gen_token_indexes] = self.v_proj_moe_gen(gen)

        query = query.view(-1, self.num_heads, self.head_dim)
        key = key.view(-1, self.num_key_value_heads, self.head_dim)
        value = value.view(-1, self.num_key_value_heads, self.head_dim)
        normalized_query = torch.zeros_like(query)
        normalized_key = torch.zeros_like(key)
        normalized_query[packed_und_token_indexes], normalized_key[
            packed_und_token_indexes
        ] = self._apply_sensenova_qk_norm(
            query[packed_und_token_indexes],
            key[packed_und_token_indexes],
            self.q_norm_t,
            self.q_norm_hw,
            self.k_norm_t,
            self.k_norm_hw,
        )
        normalized_query[packed_gen_token_indexes], normalized_key[
            packed_gen_token_indexes
        ] = self._apply_sensenova_qk_norm(
            query[packed_gen_token_indexes],
            key[packed_gen_token_indexes],
            self.q_norm_t_moe_gen,
            self.q_norm_hw_moe_gen,
            self.k_norm_t_moe_gen,
            self.k_norm_hw_moe_gen,
        )
        query, key = self._apply_sensenova_mrope(
            normalized_query,
            normalized_key,
            packed_position_embeddings,
        )

        if not hasattr(attention_mask, "shape"):
            raise TypeError(
                "PixelUMM training requires a FlexAttention BlockMask"
            )
        pad_size = attention_mask.shape[-2] - query.shape[0]
        if pad_size < 0:
            raise ValueError("attention mask is shorter than the packed sequence")
        query = pad_sequence(query.permute(1, 0, 2), pad_size)
        key = pad_sequence(key.permute(1, 0, 2), pad_size)
        value = pad_sequence(value.permute(1, 0, 2), pad_size)
        output = flex_attention(
            query.unsqueeze(0),
            key.unsqueeze(0),
            value.unsqueeze(0),
            enable_gqa=True,
            block_mask=attention_mask,
        )
        end_index = output.shape[2] - pad_size if pad_size else output.shape[2]
        output = output[0, :, :end_index, :]
        output = output.transpose(0, 1).reshape(
            -1, self.num_heads * self.head_dim
        )

        projected = output.new_zeros(output.shape[0], self.hidden_size)
        projected[packed_und_token_indexes] = self.o_proj(
            output[packed_und_token_indexes]
        )
        projected[packed_gen_token_indexes] = self.o_proj_moe_gen(
            output[packed_gen_token_indexes]
        )
        return projected

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values: bool = True,
        is_causal: bool = True,
        mode: str = "und",
        packed_gen_token_indexes=None,
        packed_und_token_indexes=None,
    ):
        if mode not in {"und", "gen"}:
            raise ValueError(f"Unsupported inference mode: {mode!r}")

        if mode == "und":
            query = self.q_proj(packed_query_sequence).view(
                -1, self.num_heads, self.head_dim
            )
            key = self.k_proj(packed_query_sequence).view(
                -1, self.num_key_value_heads, self.head_dim
            )
            value = self.v_proj(packed_query_sequence).view(
                -1, self.num_key_value_heads, self.head_dim
            )
            query, key = self._apply_sensenova_qk_norm(
                query,
                key,
                self.q_norm_t,
                self.q_norm_hw,
                self.k_norm_t,
                self.k_norm_hw,
            )
        else:
            if packed_gen_token_indexes is None or packed_und_token_indexes is None:
                raise ValueError("GEN inference requires UND and GEN token indexes")
            sequence = packed_query_sequence.to(torch.bfloat16)
            query = sequence.new_zeros(
                sequence.shape[0], self.num_heads * self.head_dim
            )
            key = sequence.new_zeros(
                sequence.shape[0],
                self.num_key_value_heads * self.head_dim,
            )
            value = sequence.new_zeros(
                sequence.shape[0],
                self.num_key_value_heads * self.head_dim,
            )
            und = sequence[packed_und_token_indexes]
            gen = sequence[packed_gen_token_indexes]
            query[packed_und_token_indexes] = self.q_proj(und)
            query[packed_gen_token_indexes] = self.q_proj_moe_gen(gen)
            key[packed_und_token_indexes] = self.k_proj(und)
            key[packed_gen_token_indexes] = self.k_proj_moe_gen(gen)
            value[packed_und_token_indexes] = self.v_proj(und)
            value[packed_gen_token_indexes] = self.v_proj_moe_gen(gen)
            query = query.view(-1, self.num_heads, self.head_dim).float()
            key = key.view(-1, self.num_key_value_heads, self.head_dim).float()
            value = value.view(-1, self.num_key_value_heads, self.head_dim)
            query[packed_und_token_indexes], key[
                packed_und_token_indexes
            ] = self._apply_sensenova_qk_norm(
                query[packed_und_token_indexes],
                key[packed_und_token_indexes],
                self.q_norm_t,
                self.q_norm_hw,
                self.k_norm_t,
                self.k_norm_hw,
            )
            query[packed_gen_token_indexes], key[
                packed_gen_token_indexes
            ] = self._apply_sensenova_qk_norm(
                query[packed_gen_token_indexes],
                key[packed_gen_token_indexes],
                self.q_norm_t_moe_gen,
                self.q_norm_hw_moe_gen,
                self.k_norm_t_moe_gen,
                self.k_norm_hw_moe_gen,
            )

        query, key = self._apply_sensenova_mrope(
            query,
            key,
            packed_query_position_embeddings,
        )
        query = query.to(torch.bfloat16)
        key = key.to(torch.bfloat16)
        value = value.to(torch.bfloat16)

        if past_key_values is not None and past_key_values.key_cache[
            self.layer_idx
        ] is not None:
            if key_values_lens is None or packed_key_value_indexes is None:
                raise ValueError(
                    "cached inference requires key lengths and KV indexes"
                )
            past_key = past_key_values.key_cache[self.layer_idx]
            past_value = past_key_values.value_cache[self.layer_idx]
            sequence_length = sum(query_lens) + sum(key_values_lens)
            merged_key = past_key.new_zeros(
                sequence_length,
                self.num_key_value_heads,
                self.head_dim,
            )
            merged_value = past_value.new_zeros(
                sequence_length,
                self.num_key_value_heads,
                self.head_dim,
            )
            merged_key[packed_query_indexes] = key
            merged_key[packed_key_value_indexes] = past_key
            merged_value[packed_query_indexes] = value
            merged_value[packed_key_value_indexes] = past_value
            effective_key_lens = key_values_lens + query_lens
        else:
            merged_key = key
            merged_value = value
            effective_key_lens = query_lens

        output = _flash_varlen_inference_attention(
            query,
            merged_key,
            merged_value,
            query_lens,
            effective_key_lens,
            causal=is_causal,
        ).reshape(-1, self.num_heads * self.head_dim)
        if mode == "und":
            output = self.o_proj(output)
        else:
            projected = output.new_zeros(output.shape[0], self.hidden_size)
            projected[packed_und_token_indexes] = self.o_proj(
                output[packed_und_token_indexes]
            )
            projected[packed_gen_token_indexes] = self.o_proj_moe_gen(
                output[packed_gen_token_indexes]
            )
            output = projected

        if update_past_key_values:
            if past_key_values is None:
                raise ValueError("update_past_key_values requires a NaiveCache")
            past_key_values.key_cache[self.layer_idx] = merged_key
            past_key_values.value_cache[self.layer_idx] = merged_value
        return output, past_key_values

class Qwen3MoTDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.qwen_mlp_sequence_chunk_size = int(
            getattr(config, "qwen_mlp_sequence_chunk_size", 0)
        )
        if self.qwen_mlp_sequence_chunk_size < 0:
            raise ValueError("qwen_mlp_sequence_chunk_size must be >= 0")
        self.self_attn = PackedAttentionMoT(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.mlp_moe_gen = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.input_layernorm_moe_gen = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm_moe_gen = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: Optional[List[int]],
        attention_mask,
        packed_position_embeddings: torch.Tensor,
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
    ) -> torch.Tensor:
        if packed_und_token_indexes is None or packed_gen_token_indexes is None:
            raise ValueError("MoT training requires UND and GEN token indexes")

        residual = packed_sequence
        normalized = torch.zeros_like(packed_sequence)
        und_normalized = self.input_layernorm(
            packed_sequence[packed_und_token_indexes]
        )
        gen_normalized = self.input_layernorm_moe_gen(
            packed_sequence[packed_gen_token_indexes]
        )
        normalized[packed_und_token_indexes] = und_normalized
        normalized[packed_gen_token_indexes] = gen_normalized
        packed_sequence = residual + self.self_attn(
            packed_sequence=normalized,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_embeddings=packed_position_embeddings,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
        )

        residual = packed_sequence
        mlp_output = torch.zeros_like(packed_sequence)
        und_mlp = _apply_sequence_chunked(
            self.mlp,
            self.post_attention_layernorm(
                packed_sequence[packed_und_token_indexes]
            ),
            self.qwen_mlp_sequence_chunk_size,
        )
        gen_mlp = _apply_sequence_chunked(
            self.mlp_moe_gen,
            self.post_attention_layernorm_moe_gen(
                packed_sequence[packed_gen_token_indexes]
            ),
            self.qwen_mlp_sequence_chunk_size,
        )
        mlp_output[packed_und_token_indexes] = und_mlp
        mlp_output[packed_gen_token_indexes] = gen_mlp
        return residual + mlp_output

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values: bool = True,
        is_causal: bool = True,
        mode: str = "und",
        packed_gen_token_indexes=None,
        packed_und_token_indexes=None,
    ):
        if mode == "und":
            normalized = self.input_layernorm(packed_query_sequence)
        elif mode == "gen":
            if packed_gen_token_indexes is None or packed_und_token_indexes is None:
                raise ValueError("GEN inference requires UND and GEN token indexes")
            normalized = torch.zeros_like(packed_query_sequence)
            normalized[packed_und_token_indexes] = self.input_layernorm(
                packed_query_sequence[packed_und_token_indexes]
            )
            normalized[packed_gen_token_indexes] = self.input_layernorm_moe_gen(
                packed_query_sequence[packed_gen_token_indexes]
            )
        else:
            raise ValueError(f"Unsupported inference mode: {mode!r}")

        residual = packed_query_sequence
        attention_output, past_key_values = self.self_attn(
            packed_query_sequence=normalized,
            query_lens=query_lens,
            packed_query_position_embeddings=packed_query_position_embeddings,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
            mode=mode,
            packed_gen_token_indexes=packed_gen_token_indexes,
            packed_und_token_indexes=packed_und_token_indexes,
        )
        packed_query_sequence = residual + attention_output

        residual = packed_query_sequence
        if mode == "und":
            mlp_output = self.mlp(
                self.post_attention_layernorm(packed_query_sequence)
            )
        else:
            mlp_output = torch.zeros_like(packed_query_sequence).to(
                torch.bfloat16
            )
            mlp_output[packed_und_token_indexes] = self.mlp(
                self.post_attention_layernorm(
                    packed_query_sequence[packed_und_token_indexes]
                ).to(torch.bfloat16)
            )
            mlp_output[packed_gen_token_indexes] = self.mlp_moe_gen(
                self.post_attention_layernorm_moe_gen(
                    packed_query_sequence[packed_gen_token_indexes]
                ).to(torch.bfloat16)
            )
        return residual + mlp_output, past_key_values

class Qwen3PreTrainedModel(PreTrainedModel):
    config_class = Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3MoTDecoderLayer"]


class Qwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        if config.layer_module != "Qwen3MoTDecoderLayer":
            raise ValueError("PixelUMM requires Qwen3MoTDecoderLayer")
        if config.qwen3_mrope_type != "sensenova":
            raise ValueError("PixelUMM requires qwen3_mrope_type='sensenova'")
        if config.freeze_und:
            raise ValueError("PixelUMM release does not support freeze_und")
        if config.conditional_branch_gradient_contract != "pixelumm_zero_v1":
            raise ValueError(
                "PixelUMM requires conditional_branch_gradient_contract="
                "'pixelumm_zero_v1'"
            )

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                Qwen3MoTDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.norm_moe_gen = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_init()

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_ids: torch.Tensor,
        packed_und_token_indexes: Optional[torch.LongTensor] = None,
        packed_gen_token_indexes: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if packed_position_ids.ndim != 2 or packed_position_ids.shape[0] != 3:
            raise ValueError(
                "PixelUMM requires T/H/W position IDs shaped (3, seq), "
                f"got {tuple(packed_position_ids.shape)}"
            )
        if packed_und_token_indexes is None:
            raise ValueError("packed_und_token_indexes is required")
        if packed_gen_token_indexes is None:
            packed_gen_token_indexes = packed_und_token_indexes.new_empty((0,))
        position_ids = packed_position_ids[
            :, : packed_sequence.shape[0]
        ].contiguous()

        for layer in self.layers:
            packed_sequence = layer(
                packed_sequence=packed_sequence,
                sample_lens=None,
                attention_mask=attention_mask,
                packed_position_embeddings=position_ids,
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_gen_token_indexes,
            )

        output = torch.zeros_like(packed_sequence)
        output[packed_und_token_indexes] = self.norm(
            packed_sequence[packed_und_token_indexes]
        )
        output[packed_gen_token_indexes] = self.norm_moe_gen(
            packed_sequence[packed_gen_token_indexes]
        )
        return output

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values: bool = True,
        is_causal: bool = True,
        mode: str = "und",
        packed_gen_token_indexes=None,
        packed_und_token_indexes=None,
    ) -> BaseNavitOutputWithPast:
        if (
            packed_query_position_ids.ndim != 2
            or packed_query_position_ids.shape[0] != 3
        ):
            raise ValueError(
                "PixelUMM requires T/H/W query position IDs shaped (3, seq), "
                f"got {tuple(packed_query_position_ids.shape)}"
            )
        position_ids = packed_query_position_ids[
            :, : packed_query_sequence.shape[0]
        ].contiguous()
        extra_inputs = {"mode": mode}
        if mode == "gen":
            if packed_gen_token_indexes is None or packed_und_token_indexes is None:
                raise ValueError("GEN inference requires UND and GEN token indexes")
            extra_inputs.update(
                packed_gen_token_indexes=packed_gen_token_indexes,
                packed_und_token_indexes=packed_und_token_indexes,
            )
        elif mode != "und":
            raise ValueError(f"Unsupported inference mode: {mode!r}")

        for layer in self.layers:
            packed_query_sequence, past_key_values = layer(
                packed_query_sequence=packed_query_sequence,
                query_lens=query_lens,
                packed_query_position_embeddings=position_ids,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=update_past_key_values,
                is_causal=is_causal,
                **extra_inputs,
            )

        if mode == "und":
            packed_query_sequence = self.norm(packed_query_sequence)
        else:
            output = torch.zeros_like(packed_query_sequence)
            output[packed_und_token_indexes] = self.norm(
                packed_query_sequence[packed_und_token_indexes]
            )
            output[packed_gen_token_indexes] = self.norm_moe_gen(
                packed_query_sequence[packed_gen_token_indexes]
            )
            packed_query_sequence = output
        return BaseNavitOutputWithPast(
            packed_query_sequence=packed_query_sequence,
            past_key_values=past_key_values,
        )


class Qwen3ForCausalLM(Qwen3PreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        self.post_init()

    def remove_unused_sensenova_base_qk_norms(self):
        num_removed = 0
        for layer in self.model.layers:
            num_removed += layer.self_attn._remove_unused_base_qk_norms()
        return num_removed

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_ids: torch.Tensor,
        packed_und_token_indexes: Optional[torch.LongTensor] = None,
        packed_gen_token_indexes: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        return self.model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            packed_position_ids=packed_position_ids,
            attention_mask=attention_mask,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
        )

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values: bool = True,
        is_causal: bool = True,
        mode: str = "und",
        packed_gen_token_indexes=None,
        packed_und_token_indexes=None,
    ) -> BaseNavitOutputWithPast:
        return self.model(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_ids=packed_query_position_ids,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
            mode=mode,
            packed_gen_token_indexes=packed_gen_token_indexes,
            packed_und_token_indexes=packed_und_token_indexes,
        )
