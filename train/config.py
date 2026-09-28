# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration for the released F18 and F22 PixelUMM profiles."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Optional, TypeVar

from train.scientific_config import load_scientific_config


@dataclass(frozen=True)
class ModelArguments:
    llm_path: str = "Qwen/Qwen3-8B"
    llm_qk_norm: bool = True
    llm_backend: str = "qwen3"
    qwen3_mrope_type: str = "sensenova"
    qwen3_mrope_section: str = ""
    qwen3_rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = False
    layer_module: str = "Qwen3MoTDecoderLayer"
    pixel_token_patch_size: int = 16
    pixel_embedder_type: str = "image_raw_patch_linear"
    pixel_image_additive_pos_embed_type: str = "none"
    pixel_video_additive_pos_embed_type: str = "none"
    sensenova_vision_rope_theta: float = 10_000.0
    sensenova_vision_max_position_embeddings: int = 10_000
    add_timestep_embedding: bool = False
    add_noise_scale_embedding: bool = False
    noise_scale: float = 1.0
    noise_scale_mode: str = "constant"
    enable_pixel_video: bool = True
    pixel_video_temporal_patch_size: int = 4
    pixel_video_embedder_type: str = "video_raw_tube_linear"
    pixel_separate_und_gen_embedder: bool = True
    pixel_video_separate_und_gen_embedder: bool = True
    packed_attention_impl: str = "flex"
    inference_attention_backend: str = "flash_attn_varlen"
    packed_expert_routing: str = "pixelumm_mot"
    pixel_head_type: str = "minit2i_linear"
    pixel_video_head_type: str = "jit_style"
    qwen_mlp_sequence_chunk_size: int = 32_768
    ce_loss_checkpoint_chunk_size: int = 4096
    text_cond_dropout_prob: float = 0.1
    pixel_gen_cond_dropout_prob: float = 0.1
    pixel_und_cond_dropout_prob: float = 0.0


@dataclass(frozen=True)
class DataArguments:
    packed_sequence_layout: str = "pixelumm_mot"


@dataclass(frozen=True)
class TrainingArguments:
    gen_space: str = "pixel"
    und_space: str = "pixel"
    prediction_type: str = "x"
    loss_type: str = "v_loss"
    x_pred_t_min: float = 0.05
    visual_gen: bool = True
    visual_und: bool = True
    freeze_und: bool = False
    conditional_branch_gradient_contract: str = "pixelumm_zero_v1"
    noise_time_logit_mean: float = 0.0
    noise_time_logit_std: float = 1.0
    time_schedule: str = "pixelumm_resolution"
    mse_loss_reduction: str = "sample"
    eval_num_timesteps: int = 50
    eval_sampler: str = "dpm-solver"
    eval_timestep_shift: float = 3.0
    eval_cfg_text_scale: float = 3.5
    eval_cfg_renorm_type: str = "none"
    eval_video_sampler: str = "unipc"
    eval_video_timestep_shift: float = 10.0
    eval_video_cfg_text_scale: float = 6.0


T = TypeVar("T")


def _instantiate(cls: type[T], values: dict, *, section: str) -> T:
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise RuntimeError(
            f"release config has unsupported {section} fields: {unknown}"
        )
    return cls(**values)


def load_release_config(
    path: str | Path,
    *,
    llm_path: str | None = None,
) -> tuple[ModelArguments, DataArguments, TrainingArguments]:
    sections = load_scientific_config(Path(path))
    model_args = _instantiate(ModelArguments, sections["model"], section="model")
    data_args = _instantiate(DataArguments, sections["data"], section="data")
    training_args = _instantiate(
        TrainingArguments, sections["training"], section="training"
    )
    if llm_path is not None:
        model_args = replace(model_args, llm_path=str(llm_path))
    return model_args, data_args, training_args


def parse_mrope_section(value: Optional[str]) -> Optional[list[int]]:
    if value is None or not value.strip():
        return None
    section = [int(item.strip()) for item in value.strip("[]()").split(",") if item.strip()]
    if len(section) != 3:
        raise ValueError(f"Expected qwen3_mrope_section T,H,W, got {value!r}")
    return section


def configure_qwen3_mrope_config(llm_config, model_args: ModelArguments) -> None:
    llm_config.qwen3_mrope_type = model_args.qwen3_mrope_type
    if not getattr(llm_config, "rope_parameters", None):
        llm_config.rope_parameters = {}
    if model_args.qwen3_rope_theta > 0:
        theta = float(model_args.qwen3_rope_theta)
        llm_config.rope_theta = theta
        llm_config.rope_parameters["rope_theta"] = theta
    section = parse_mrope_section(model_args.qwen3_mrope_section)
    if section is None:
        section = getattr(llm_config, "mrope_section", None)
    if section is None:
        section = llm_config.rope_parameters.get("mrope_section")
    if section is not None:
        llm_config.mrope_section = section
        llm_config.rope_parameters["mrope_section"] = section
        head_dim = getattr(
            llm_config,
            "head_dim",
            llm_config.hidden_size // llm_config.num_attention_heads,
        )
        rotary_dim = 2 * sum(section)
        if rotary_dim > head_dim:
            raise ValueError(
                f"mrope section {section} gives rotary_dim={rotary_dim} > head_dim={head_dim}"
            )
        llm_config.rope_parameters["partial_rotary_factor"] = rotary_dim / head_dim
