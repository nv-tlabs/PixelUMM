# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3 configuration for PixelUMM — standalone, no HF transformers dependency."""

from transformers.configuration_utils import PretrainedConfig


class Qwen3Config(PretrainedConfig):
    """
    Qwen3-0.6B config with MoT extensions.

    Qwen3-0.6B reference values:
        hidden_size=1024, intermediate_size=3072, num_hidden_layers=28,
        num_attention_heads=16, num_key_value_heads=8, head_dim=128,
        attention_bias=False, rope_theta=1e6, vocab_size=151936
    """

    model_type = "qwen3"

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=40960,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=True,
        rope_theta=1000000.0,
        rope_parameters=None,  # dict with rope_type etc; None → default
        attention_bias=False,
        attention_dropout=0.0,
        # Hugging Face Qwen compatibility fields.
        # use_sliding_window，sliding_window，max_window_layers，layer_types
        # PixelUMM's qwen3_navit path currently keeps full attention and does not consume
        # sliding-window attention, but newer transformer configs may define them.
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        layer_types=None,
        pad_token_id=None,
        bos_token_id=151643,
        eos_token_id=151645,
        # MoT extensions (same as Qwen2 version)
        layer_module="Qwen3MoTDecoderLayer",
        freeze_und=False,
        qk_norm=True,  # always True for Qwen3
        # PixelUMM's released SenseNova T/H/W MRoPE topology.
        qwen3_mrope_type="sensenova",
        mrope_section=None,
        rope_theta_hw=10000.0,
        max_position_embeddings_hw=10000,
        inference_attention_backend="flash_attn_varlen",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        # MRoPE section (set before rope_parameters which needs it)
        if mrope_section is None:
            self.mrope_section = [24, 20, 20]  # default for head_dim=128
        else:
            self.mrope_section = mrope_section
        self.qwen3_mrope_type = qwen3_mrope_type
        self.rope_theta_hw = rope_theta_hw
        self.max_position_embeddings_hw = max_position_embeddings_hw
        # rope_parameters: ensure all fields needed by Qwen3 rotary and local interleaved MRoPE.
        if rope_parameters is None:
            rope_parameters = {"rope_type": "default"}
        rope_parameters.setdefault("rope_theta", rope_theta)
        rope_parameters.setdefault("mrope_section", self.mrope_section)
        self.rope_parameters = rope_parameters
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if self.use_sliding_window else None
        self.max_window_layers = max_window_layers
        if layer_types is None:
            layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        self.layer_types = layer_types
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.layer_module = layer_module
        self.freeze_und = freeze_und
        self.qk_norm = qk_norm
        self.inference_attention_backend = str(inference_attention_backend)
        if self.inference_attention_backend != "flash_attn_varlen":
            raise ValueError(
                "inference_attention_backend must be 'flash_attn_varlen'; "
                f"got {self.inference_attention_backend!r}"
            )
