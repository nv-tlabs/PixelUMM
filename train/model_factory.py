# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construct a PixelUMM model from the selected architecture profile."""

from __future__ import annotations

from transformers import AutoTokenizer
from transformers.modeling_utils import no_init_weights

from data.data_utils import add_special_tokens
from modeling.pixelumm.pixelumm import PixelUMM, PixelUMMConfig
from modeling.pixelumm.qwen3_navit import Qwen3ForCausalLM
from modeling.qwen3.configuration_qwen3 import Qwen3Config
from train.config import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
    configure_qwen3_mrope_config,
)


def _count_parameters(module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def build_pixelumm_model(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    logger,
):
    """Build the selected topology and tokenizer without loading checkpoint weights."""
    if model_args.llm_backend != "qwen3":
        raise ValueError(
            "PixelUMM-release supports only the R07 Qwen3 backend; "
            f"got {model_args.llm_backend!r}"
        )

    llm_config = Qwen3Config.from_pretrained(model_args.llm_path)
    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = training_args.freeze_und
    llm_config.qwen_mlp_sequence_chunk_size = int(
        model_args.qwen_mlp_sequence_chunk_size
    )
    llm_config.inference_attention_backend = model_args.inference_attention_backend
    llm_config.conditional_branch_gradient_contract = (
        training_args.conditional_branch_gradient_contract
    )

    configure_qwen3_mrope_config(llm_config, model_args)
    llm_config.rope_theta_hw = model_args.sensenova_vision_rope_theta
    llm_config.max_position_embeddings_hw = (
        model_args.sensenova_vision_max_position_embeddings
    )

    with no_init_weights():
        language_model = Qwen3ForCausalLM(llm_config)
    removed = language_model.remove_unused_sensenova_base_qk_norms()
    logger.info(
        "Removed %d unused inherited Q/K norm tensors.",
        removed,
    )

    config = PixelUMMConfig(
        visual_gen=training_args.visual_gen,
        visual_und=training_args.visual_und,
        llm_config=llm_config,
        llm_backend=model_args.llm_backend,
        gen_space=training_args.gen_space,
        und_space=training_args.und_space,
        patch_size=model_args.pixel_token_patch_size,
        pixel_embedder_type=model_args.pixel_embedder_type,
        pixel_image_additive_pos_embed_type=model_args.pixel_image_additive_pos_embed_type,
        pixel_video_additive_pos_embed_type=model_args.pixel_video_additive_pos_embed_type,
        add_timestep_embedding=model_args.add_timestep_embedding,
        add_noise_scale_embedding=model_args.add_noise_scale_embedding,
        noise_scale=model_args.noise_scale,
        noise_scale_mode=model_args.noise_scale_mode,
        enable_pixel_video=model_args.enable_pixel_video,
        pixel_video_embedder_type=model_args.pixel_video_embedder_type,
        pixel_separate_und_gen_embedder=model_args.pixel_separate_und_gen_embedder,
        pixel_video_separate_und_gen_embedder=model_args.pixel_video_separate_und_gen_embedder,
        pixel_video_temporal_patch_size=model_args.pixel_video_temporal_patch_size,
        noise_time_logit_mean=training_args.noise_time_logit_mean,
        noise_time_logit_std=training_args.noise_time_logit_std,
        time_schedule=training_args.time_schedule,
        prediction_type=training_args.prediction_type,
        loss_type=training_args.loss_type,
        x_pred_t_min=training_args.x_pred_t_min,
        pixel_head_type=model_args.pixel_head_type,
        pixel_video_head_type=model_args.pixel_video_head_type,
        ce_loss_checkpoint_chunk_size=model_args.ce_loss_checkpoint_chunk_size,
        mse_loss_reduction=training_args.mse_loss_reduction,
        packed_attention_impl=model_args.packed_attention_impl,
        packed_expert_routing=model_args.packed_expert_routing,
        packed_sequence_layout=data_args.packed_sequence_layout,
        conditional_branch_gradient_contract=(
            training_args.conditional_branch_gradient_contract
        ),
    )
    model = PixelUMM(language_model, config)

    tokenizer = AutoTokenizer.from_pretrained(model_args.llm_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    current_vocab_size = model.language_model.model.embed_tokens.num_embeddings
    if num_new_tokens > 0 or current_vocab_size != len(tokenizer):
        logger.info(
            "Resizing token embeddings: %d -> %d", current_vocab_size, len(tokenizer)
        )
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    total_param_count = _count_parameters(model)
    lm_param_count = _count_parameters(model.language_model)
    logger.info(
        "Model parameter count: %.2fB (LM-only: %.2fB)",
        total_param_count / 1e9,
        lm_param_count / 1e9,
    )
    return model, tokenizer, new_token_ids
