# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from torch.utils.checkpoint import checkpoint
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from data.data_utils import PIXELUMM_VIDEO_MROPE_BASE_FPS, build_pixelumm_vision_mrope_position_ids, create_sparse_mask, tubeify
from .qwen3_navit import NaiveCache
from .loss_reduction import sample_mean_token_weights
from .modeling_utils import TimestepEmbedder, RawPixelPatchLinearEmbed, RawPixelVideoTubeLinearEmbed, UnpatchifyHead, VideoUnpatchifyHead, patchify_image
from .dpm_solver import sample_flow_dpm
from .flow_unipc import sample_flow_unipc
from .time_schedule import flow_shift_noise_timestep, pixelumm_resolution_shift, sample_logit_normal_noise_time
_CHATML_USER_START = '<|im_start|>user\n'
_CHATML_ASSISTANT_START = '<|im_start|>assistant\n'
_CHATML_SYSTEM_START = '<|im_start|>system\n'
_CHATML_END = '<|im_end|>'
_CHATML_SEP = '\n'
_DEFAULT_SYSTEM_PROMPT = 'You are a helpful assistant.'
_PIXELUMM_TEXT_TOKEN_LIMIT = 4096

def _checkpointed_chunked_lm_head_cross_entropy(lm_head: nn.Module, hidden_states: torch.Tensor, labels: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Compute token CE without retaining one full ``tokens x vocab`` tensor.

    Each chunk is activation-checkpointed, so its LM-head logits are discarded
    after the forward pass and recomputed during backward.  This preserves the
    exact per-token CE contract used by task logging/reweighting while bounding
    peak logits/workspace memory by ``chunk_size x vocab_size``.
    """
    chunk_size = int(chunk_size)
    if chunk_size <= 0 or hidden_states.shape[0] <= chunk_size:
        logits = lm_head(hidden_states)
        return F.cross_entropy(logits, labels, reduction='none')

    def chunk_loss(chunk_hidden: torch.Tensor, chunk_labels: torch.Tensor) -> torch.Tensor:
        logits = lm_head(chunk_hidden)
        return F.cross_entropy(logits, chunk_labels, reduction='none')
    losses = []
    for start in range(0, hidden_states.shape[0], chunk_size):
        end = min(start + chunk_size, hidden_states.shape[0])
        chunk_hidden = hidden_states[start:end]
        chunk_labels = labels[start:end]
        if torch.is_grad_enabled():
            loss = checkpoint(chunk_loss, chunk_hidden, chunk_labels, use_reentrant=False)
        else:
            loss = chunk_loss(chunk_hidden, chunk_labels)
        losses.append(loss)
    return torch.cat(losses, dim=0)

class PixelUMMConfig(PretrainedConfig):

    def __init__(
        self,
        visual_gen=True,
        visual_und=True,
        llm_config=None,
        llm_backend='qwen3',
        gen_space='pixel',
        und_space='pixel',
        patch_size=16,
        pixel_embedder_type='image_raw_patch_linear',
        pixel_image_additive_pos_embed_type='none',
        pixel_video_additive_pos_embed_type='none',
        add_timestep_embedding=False,
        add_noise_scale_embedding=False,
        noise_scale=1.0,
        noise_scale_mode='constant',
        enable_pixel_video=True,
        pixel_video_embedder_type='video_raw_tube_linear',
        pixel_separate_und_gen_embedder=True,
        pixel_video_separate_und_gen_embedder=True,
        pixel_video_temporal_patch_size=4,
        noise_time_logit_mean=0.0,
        noise_time_logit_std=1.0,
        time_schedule='pixelumm_resolution',
        prediction_type='x',
        loss_type='v_loss',
        x_pred_t_min=0.05,
        pixel_head_type='minit2i_linear',
        pixel_video_head_type='jit_style',
        ce_loss_checkpoint_chunk_size=0,
        mse_loss_reduction='sample',
        packed_attention_impl='flex',
        packed_expert_routing='pixelumm_mot',
        packed_sequence_layout='pixelumm_mot',
        conditional_branch_gradient_contract='pixelumm_zero_v1',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.visual_gen = visual_gen
        self.visual_und = visual_und
        if not self.visual_gen or not self.visual_und:
            raise ValueError('PixelUMM-release requires both R07 visual GEN and visual UND branches')
        self.llm_config = llm_config
        self.llm_backend = llm_backend
        if self.llm_backend != 'qwen3':
            raise ValueError(f'PixelUMM-release requires the R07 Qwen3 backend; got llm_backend={self.llm_backend!r}')
        self.gen_space = gen_space
        self.und_space = und_space
        if self.gen_space != 'pixel' or self.und_space != 'pixel':
            raise ValueError(f'PixelUMM-release requires the R07 raw-pixel GEN/UND spaces; got gen_space={self.gen_space!r}, und_space={self.und_space!r}')
        self.conditional_branch_gradient_contract = conditional_branch_gradient_contract
        if self.conditional_branch_gradient_contract != 'pixelumm_zero_v1':
            raise ValueError(f"PixelUMM-release requires the R07 conditional gradient contract 'pixelumm_zero_v1', got {self.conditional_branch_gradient_contract!r}")
        self.patch_size = patch_size
        self.pixel_embedder_type = pixel_embedder_type
        self.pixel_image_additive_pos_embed_type = pixel_image_additive_pos_embed_type
        self.pixel_video_additive_pos_embed_type = pixel_video_additive_pos_embed_type
        if self.pixel_embedder_type != 'image_raw_patch_linear':
            raise ValueError(f"PixelUMM-release only supports the R07 image embedder pixel_embedder_type='image_raw_patch_linear', got {self.pixel_embedder_type!r}")
        if self.pixel_image_additive_pos_embed_type != 'none':
            raise ValueError("PixelUMM-release requires the R07 image positional policy pixel_image_additive_pos_embed_type='none'")
        if self.pixel_video_additive_pos_embed_type != 'none':
            raise ValueError("PixelUMM-release requires the R07 video positional policy pixel_video_additive_pos_embed_type='none'")
        if add_noise_scale_embedding and (not add_timestep_embedding):
            raise ValueError('add_noise_scale_embedding=True requires add_timestep_embedding=True')
        if getattr(llm_config, 'qwen3_mrope_type', None) != 'sensenova':
            raise ValueError("PixelUMM-release requires qwen3_mrope_type='sensenova'")
        self.add_timestep_embedding = add_timestep_embedding
        self.add_noise_scale_embedding = add_noise_scale_embedding
        if self.add_timestep_embedding or self.add_noise_scale_embedding:
            raise ValueError('PixelUMM-release requires the R07 embedding policy: add_timestep_embedding=False and add_noise_scale_embedding=False')
        self.noise_scale = noise_scale
        self.noise_scale_mode = noise_scale_mode
        if self.noise_scale_mode != 'constant' or float(self.noise_scale) != 1.0:
            raise ValueError("PixelUMM-release requires the R07 unit Gaussian noise scale: noise_scale_mode='constant' and noise_scale=1.0")
        self.enable_pixel_video = enable_pixel_video
        if not self.enable_pixel_video:
            raise ValueError('PixelUMM-release requires the R07 pixel-video path')
        self.pixel_video_embedder_type = pixel_video_embedder_type
        self.pixel_separate_und_gen_embedder = bool(pixel_separate_und_gen_embedder)
        if not isinstance(pixel_video_separate_und_gen_embedder, bool):
            raise ValueError('pixel_video_separate_und_gen_embedder must be a boolean')
        self.pixel_video_separate_und_gen_embedder = pixel_video_separate_und_gen_embedder
        if not self.pixel_separate_und_gen_embedder:
            raise ValueError('PixelUMM-release requires separate UND/GEN image embedders')
        if self.pixel_video_embedder_type != 'video_raw_tube_linear':
            raise ValueError(f"PixelUMM-release only supports the R07 video embedder pixel_video_embedder_type='video_raw_tube_linear', got {self.pixel_video_embedder_type!r}")
        self.pixel_video_temporal_patch_size = pixel_video_temporal_patch_size
        self.noise_time_logit_mean = noise_time_logit_mean
        self.noise_time_logit_std = noise_time_logit_std
        self.time_schedule = time_schedule
        if self.time_schedule != 'pixelumm_resolution':
            raise ValueError("PixelUMM-release requires time_schedule='pixelumm_resolution'")
        self.prediction_type = prediction_type
        self.loss_type = loss_type
        if self.prediction_type != 'x' or self.loss_type != 'v_loss':
            raise ValueError("PixelUMM-release requires prediction_type='x' and loss_type='v_loss'")
        self.x_pred_t_min = x_pred_t_min
        self.pixel_head_type = pixel_head_type
        self.pixel_video_head_type = pixel_video_head_type
        if self.pixel_head_type != 'minit2i_linear':
            raise ValueError("PixelUMM-release requires pixel_head_type='minit2i_linear'")
        if self.pixel_video_head_type != 'jit_style':
            raise ValueError("PixelUMM-release requires pixel_video_head_type='jit_style'")
        self.ce_loss_checkpoint_chunk_size = int(ce_loss_checkpoint_chunk_size)
        if self.ce_loss_checkpoint_chunk_size < 0:
            raise ValueError(f'ce_loss_checkpoint_chunk_size must be non-negative; got {self.ce_loss_checkpoint_chunk_size}.')
        self.mse_loss_reduction = mse_loss_reduction
        if self.mse_loss_reduction != 'sample':
            raise ValueError('PixelUMM-release requires the R07 sample-mean MSE reduction')
        self.packed_attention_impl = packed_attention_impl
        if self.packed_attention_impl != 'flex':
            raise ValueError('PixelUMM-release requires the R07 FlexAttention backend')
        self.packed_expert_routing = packed_expert_routing
        self.packed_sequence_layout = packed_sequence_layout
        if self.packed_expert_routing != 'pixelumm_mot':
            raise ValueError("PixelUMM-release requires packed_expert_routing='pixelumm_mot'")
        if self.packed_sequence_layout != 'pixelumm_mot':
            raise ValueError("PixelUMM-release requires packed_sequence_layout='pixelumm_mot'")

class PixelUMM(PreTrainedModel):
    config_class = PixelUMMConfig
    base_model_prefix = 'pixelumm'

    def __init__(self, language_model, config: PixelUMMConfig):
        super().__init__(config)
        self.language_model = language_model
        self.hidden_size = config.llm_config.hidden_size
        if getattr(config.llm_config, 'layer_module', None) != 'Qwen3MoTDecoderLayer':
            raise ValueError("PixelUMM-release requires layer_module='Qwen3MoTDecoderLayer'")
        self.num_heads = config.llm_config.num_attention_heads
        self.time_embedder = TimestepEmbedder(self.hidden_size)
        self.time_embedder.requires_grad_(False)

        def build_pixel_image_embedder():
            return RawPixelPatchLinearEmbed(patch_size=config.patch_size, in_channels=3, hidden_size=self.hidden_size)
        self.pixel_patch_embed = build_pixel_image_embedder()
        self.pixel_patch_embed_mot_gen = build_pixel_image_embedder()
        self.pixel_video_patch_embed_mot_gen = RawPixelVideoTubeLinearEmbed(patch_size=config.patch_size, in_channels=3, temporal_patch_size=config.pixel_video_temporal_patch_size, hidden_size=self.hidden_size)
        if config.pixel_video_separate_und_gen_embedder:
            self.pixel_video_patch_embed = RawPixelVideoTubeLinearEmbed(patch_size=config.patch_size, in_channels=3, temporal_patch_size=config.pixel_video_temporal_patch_size, hidden_size=self.hidden_size)
        self.unpatchify_head = UnpatchifyHead(hidden_size=self.hidden_size, patch_size=config.patch_size, out_channels=3)
        self.video_unpatchify_head = VideoUnpatchifyHead(hidden_size=self.hidden_size, patch_size=config.patch_size, temporal_patch_size=config.pixel_video_temporal_patch_size, out_channels=3)
        self.config = config

    def _image_patch_embedder(self, gen_model=False):
        return self.pixel_patch_embed_mot_gen if gen_model else self.pixel_patch_embed

    def embed_pixel_video_tubes(self, pixel_video_tubes: torch.Tensor, gen_model: bool=True) -> torch.Tensor:
        if not gen_model:
            raise RuntimeError('Pixel video embedding is generation-only; video understanding uses image UND frames.')
        return self.pixel_video_patch_embed_mot_gen(pixel_video_tubes)

    def _sample_train_timesteps(self, raw_timesteps: torch.Tensor, modality: Optional[str]=None) -> torch.Tensor:
        if modality not in {None, 'image', 'video'}:
            raise ValueError(f'Unsupported timestep modality={modality!r}')
        return sample_logit_normal_noise_time(
            raw_timesteps,
            logit_mean=float(self.config.noise_time_logit_mean),
            logit_std=float(self.config.noise_time_logit_std),
        )

    def _apply_time_schedule(self, t: torch.Tensor, spatial_hw: Optional[Tuple[int, int]]=None) -> torch.Tensor:
        """Shift PixelUMM noise-time ``t`` toward noise when shift > 1.

        Unlike the upstream SenseNova implementation, PixelUMM defines
        ``z_t = (1 - t) * clean + t * noise``.  ``t`` is therefore already
        the noise fraction and must not be complemented before shifting.
        """
        if spatial_hw is None:
            raise ValueError('time_schedule=pixelumm_resolution requires spatial_hw')
        return flow_shift_noise_timestep(t, pixelumm_resolution_shift(*spatial_hw))

    def _apply_time_schedule_per_sample(self, timesteps: torch.Tensor, token_lens, grid_hw, *, patch_size: int) -> torch.Tensor:
        """Apply a spatially selected schedule without mixing packed samples."""
        token_lens = self._tensor_to_int_list(token_lens)
        if torch.is_tensor(grid_hw):
            grid_hw = [(int(h), int(w)) for (h, w) in grid_hw.tolist()]
        else:
            grid_hw = [(int(h), int(w)) for (h, w) in grid_hw]
        if len(token_lens) != len(grid_hw):
            raise ValueError(f'Per-sample timestep metadata length mismatch: token_lens={len(token_lens)} grid_hw={len(grid_hw)}')
        if sum(token_lens) != int(timesteps.numel()):
            raise ValueError(f'Per-sample timestep token lengths do not match packed timesteps: {sum(token_lens)} vs {int(timesteps.numel())}')
        chunks = []
        cursor = 0
        for (token_len, (grid_h, grid_w)) in zip(token_lens, grid_hw):
            chunk = timesteps[cursor:cursor + token_len]
            chunks.append(self._apply_time_schedule(chunk, spatial_hw=(grid_h * int(patch_size), grid_w * int(patch_size))))
            cursor += token_len
        return torch.cat(chunks, dim=0) if chunks else timesteps

    def _text_mrope_position_ids(self, start_position, length, device=None):
        pos = torch.arange(start_position, start_position + length, dtype=torch.long, device=device)
        zeros = torch.zeros_like(pos)
        return torch.stack([pos, zeros, zeros], dim=0)

    @staticmethod
    def _pixelumm_vfm_vision_mrope_position_ids(temporal_offset, t_tokens, h_tokens, w_tokens, device=None, fps=None, base_fps=PIXELUMM_VIDEO_MROPE_BASE_FPS, temporal_compression_factor=4):
        return build_pixelumm_vision_mrope_position_ids(temporal_offset, t_tokens, h_tokens, w_tokens, device=device, fps=fps, base_fps=base_fps, temporal_compression_factor=temporal_compression_factor)

    def _image_mrope_position_ids(self, start_position, h_tokens, w_tokens, device=None, include_start=True, include_end=True):
        grid_t = start_position + int(include_start)
        rows = torch.arange(h_tokens, dtype=torch.long, device=device).repeat_interleave(w_tokens)
        cols = torch.arange(w_tokens, dtype=torch.long, device=device).repeat(h_tokens)
        t = torch.full((h_tokens * w_tokens,), grid_t, dtype=torch.long, device=device)
        (t_chunks, h_chunks, w_chunks) = ([], [], [])
        if include_start:
            start = torch.full((1,), start_position, dtype=torch.long, device=device)
            t_chunks.append(start)
            h_chunks.append(start.new_zeros(1))
            w_chunks.append(start.new_zeros(1))
        t_chunks.append(t)
        h_chunks.append(rows)
        w_chunks.append(cols)
        next_position = grid_t + 1
        if include_end:
            end = torch.full((1,), next_position, dtype=torch.long, device=device)
            t_chunks.append(end)
            h_chunks.append(end.new_zeros(1))
            w_chunks.append(end.new_zeros(1))
            next_position += 1
        legacy_t = torch.cat(t_chunks)
        axes = [legacy_t, torch.cat(h_chunks), torch.cat(w_chunks)]
        return (torch.stack(axes, dim=0), next_position)

    def _video_mrope_position_ids(self, start_position, temporal_groups, h_tokens, w_tokens, device=None, fps=None, temporal_compression_factor=None, include_start=True, include_end=True):
        grid_start = start_position + int(include_start)
        if fps is not None:
            (vision_ids, next_position) = self._pixelumm_vfm_vision_mrope_position_ids(grid_start, temporal_groups, h_tokens, w_tokens, device=device, fps=fps, base_fps=PIXELUMM_VIDEO_MROPE_BASE_FPS, temporal_compression_factor=temporal_compression_factor if temporal_compression_factor is not None else self.config.pixel_video_temporal_patch_size)
            t = vision_ids[0]
            h = vision_ids[1]
            w = vision_ids[2]
        else:
            t = torch.arange(temporal_groups, dtype=torch.long, device=device)[:, None, None]
            h = torch.arange(h_tokens, dtype=torch.long, device=device)[None, :, None]
            w = torch.arange(w_tokens, dtype=torch.long, device=device)[None, None, :]
            t = t.expand(temporal_groups, h_tokens, w_tokens).reshape(-1) + grid_start
            h = h.expand(temporal_groups, h_tokens, w_tokens).reshape(-1)
            w = w.expand(temporal_groups, h_tokens, w_tokens).reshape(-1)
            next_position = grid_start + max(1, temporal_groups)
        (t_chunks, h_chunks, w_chunks) = ([], [], [])
        if include_start:
            start = torch.full((1,), start_position, dtype=torch.long, device=device)
            t_chunks.append(start.to(dtype=t.dtype))
            h_chunks.append(torch.zeros(1, dtype=h.dtype, device=device))
            w_chunks.append(torch.zeros(1, dtype=w.dtype, device=device))
        t_chunks.append(t)
        h_chunks.append(h)
        w_chunks.append(w)
        if include_end:
            end = torch.full((1,), next_position, dtype=t.dtype, device=device)
            t_chunks.append(end)
            h_chunks.append(torch.zeros(1, dtype=h.dtype, device=device))
            w_chunks.append(torch.zeros(1, dtype=w.dtype, device=device))
            next_position += 1
        legacy_t = torch.cat(t_chunks)
        axes = [legacy_t, torch.cat(h_chunks), torch.cat(w_chunks)]
        return (torch.stack(axes, dim=0), next_position)

    def _advance_text_query_position_ids(self, packed_query_position_ids):
        next_position_ids = packed_query_position_ids.clone()
        next_position_ids[0] = next_position_ids[0] + 1
        return next_position_ids

    def _pixel_video_token_lens(self, packed_pixel_video_patches, pixel_video_patch_seqlens=None):
        if pixel_video_patch_seqlens is not None:
            return [int(x) for x in pixel_video_patch_seqlens.tolist()]
        return [int(packed_pixel_video_patches.shape[0])]

    @staticmethod
    def _tensor_to_int_list(values):
        if values is None:
            return None
        if torch.is_tensor(values):
            values = values.detach().cpu().flatten().tolist()
        return [int(x) for x in values]

    @staticmethod
    def _has_trainable_parameters(module: Optional[nn.Module]) -> bool:
        return module is not None and any((parameter.requires_grad for parameter in module.parameters()))

    @staticmethod
    def _zero_graph_anchor(value: torch.Tensor) -> torch.Tensor:
        return value.sum() * 0.0

    def _dummy_pixel_image_embedder_anchor(self, module: nn.Module, reference: torch.Tensor) -> torch.Tensor:
        patch_dim = getattr(module, 'patch_dim', None)
        if patch_dim is None:
            raise RuntimeError(f'pixelumm_zero_v1 cannot build an image-embedder dummy for {type(module).__name__}: missing patch_dim')
        dummy = reference.new_zeros((1, int(patch_dim)))
        output = module(dummy)
        return self._zero_graph_anchor(output)

    def _dummy_pixel_video_embedder_anchor(self, module: nn.Module, reference: torch.Tensor, *, gen_model: bool) -> torch.Tensor:
        tube_dim = getattr(module, 'tube_dim', None)
        if tube_dim is None:
            tube_dim = getattr(module, 'video_tube_dim', None)
        if tube_dim is None:
            tube_dim = getattr(module, 'patch_dim', None)
        if tube_dim is None:
            raise RuntimeError(f'pixelumm_zero_v1 cannot build a video-embedder dummy for {type(module).__name__}: missing tube_dim/video_tube_dim/patch_dim')
        dummy = reference.new_zeros((1, int(tube_dim)))
        position_ids = torch.zeros((1, 3), dtype=torch.long, device=reference.device)
        if gen_model:
            output = self.embed_pixel_video_tubes(dummy, gen_model=True)
        else:
            try:
                output = module(dummy, position_ids=position_ids)
            except TypeError:
                output = module(dummy)
        return self._zero_graph_anchor(output)

    def _dummy_pixel_image_head_anchor(self, reference: torch.Tensor) -> torch.Tensor:
        head = self.unpatchify_head
        dummy_hidden = reference.new_zeros((1, self.hidden_size))
        output = head(dummy_hidden)
        return self._zero_graph_anchor(output)

    def _dummy_pixel_video_head_anchor(self, reference: torch.Tensor) -> torch.Tensor:
        dummy_hidden = reference.new_zeros((1, self.hidden_size))
        output = self.video_unpatchify_head(dummy_hidden)
        return self._zero_graph_anchor(output)

    def _pixelumm_zero_root_graph_anchor(self, last_hidden_state: torch.Tensor, *, has_und_image: bool, has_und_video: bool, has_gen_image: bool, has_gen_video: bool, has_text_output: bool) -> torch.Tensor:
        """Keep every inactive *trainable* root branch in the autograd graph.

        This mirrors PixelUMM VFM's dummy-forward contract.  Frozen coarse
        branches remain excluded because their parameters have
        ``requires_grad=False``; the helper therefore cannot create optimizer
        state or parameter updates for F18/F19's disabled branch.
        """
        anchor = last_hidden_state.new_zeros(())
        if not has_text_output and self._has_trainable_parameters(self.language_model.lm_head):
            anchor = anchor + self._zero_graph_anchor(self.language_model.lm_head(last_hidden_state[:0]))
        image_und_embedder = getattr(self, 'pixel_patch_embed', None)
        if not has_und_image and self._has_trainable_parameters(image_und_embedder):
            anchor = anchor + self._dummy_pixel_image_embedder_anchor(image_und_embedder, last_hidden_state)
        video_und_embedder = getattr(self, 'pixel_video_patch_embed', None)
        if not has_und_video and self._has_trainable_parameters(video_und_embedder):
            anchor = anchor + self._dummy_pixel_video_embedder_anchor(video_und_embedder, last_hidden_state, gen_model=False)
        image_gen_embedder = self._image_patch_embedder(gen_model=True)
        if not has_gen_image and self._has_trainable_parameters(image_gen_embedder):
            anchor = anchor + self._dummy_pixel_image_embedder_anchor(image_gen_embedder, last_hidden_state)
        if not has_gen_image and self._has_trainable_parameters(getattr(self, 'unpatchify_head', None)):
            anchor = anchor + self._dummy_pixel_image_head_anchor(last_hidden_state)
        video_gen_embedder = getattr(self, 'pixel_video_patch_embed_mot_gen', None)
        if not has_gen_video and self._has_trainable_parameters(video_gen_embedder):
            anchor = anchor + self._dummy_pixel_video_embedder_anchor(video_gen_embedder, last_hidden_state, gen_model=True)
        video_head = getattr(self, 'video_unpatchify_head', None)
        if not has_gen_video and self._has_trainable_parameters(video_head):
            anchor = anchor + self._dummy_pixel_video_head_anchor(last_hidden_state)
        return anchor

    def forward(
        self,
        sequence_length: int,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        sample_lens: List[int],
        packed_position_ids: torch.LongTensor,
        packed_mrope_position_ids: Optional[torch.LongTensor] = None,
        split_lens: List[int] = None,
        attn_modes: List[str] = None,
        ce_loss_indexes: Optional[torch.BoolTensor] = None,
        packed_label_ids: Optional[torch.LongTensor] = None,
        packed_pixel_patches_und: Optional[torch.Tensor] = None,
        packed_pixel_patch_indexes_und: Optional[torch.LongTensor] = None,
        packed_pixel_video_patches_und: Optional[torch.Tensor] = None,
        packed_pixel_video_patch_indexes_und: Optional[torch.LongTensor] = None,
        packed_pixel_patches_gen: Optional[torch.Tensor] = None,
        packed_pixel_patch_indexes_gen: Optional[torch.LongTensor] = None,
        packed_pixel_grid_hw_gen: Optional[torch.LongTensor] = None,
        packed_gen_image_lens: Optional[torch.LongTensor] = None,
        packed_pixel_video_patches_gen: Optional[torch.Tensor] = None,
        packed_pixel_video_patch_indexes_gen: Optional[torch.LongTensor] = None,
        pixel_video_patch_seqlens_gen: Optional[torch.IntTensor] = None,
        pixel_video_grid_hw_gen: Optional[torch.LongTensor] = None,
        packed_gen_video_lens: Optional[torch.LongTensor] = None,
        packed_timesteps: Optional[torch.LongTensor] = None,
    ) -> dict:
        """
        Args:
            sequence_length: length of sequence.
            packed_text_ids: 1-D int tensor, packed text token ids.
            packed_text_indexes: 1-D int tensor, packed text token indexes in sequence.
            sample_lens: A list of N ints, length of each sample in packed_sequence.
            packed_position_ids: packed 1-D positions, an image has only one global position shared
                by all latent tokens.

            packed_label_ids: 1-D int tensor, packed label token ids.
            ce_loss_indexes: 1-D bool tensor, where to compute ce loss.
            packed_timesteps: 1-D float tensor, flow timesteps. 0 indicates use clean image.
        """
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros(size=(sequence_length, self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding
        logical_seqlen = sum(sample_lens)
        flex_seqlen = (int(sequence_length) + 127) // 128 * 128
        if flex_seqlen < logical_seqlen:
            raise ValueError((logical_seqlen, flex_seqlen))
        sparse_mask = create_sparse_mask(
            sample_lens,
            split_lens,
            attn_modes,
            packed_text_embedding.device,
            total_length=flex_seqlen,
        )
        attention_mask = create_block_mask(
            sparse_mask,
            B=1,
            H=1,
            Q_LEN=flex_seqlen,
            KV_LEN=flex_seqlen,
            device=packed_text_embedding.device,
            BLOCK_SIZE=128,
            _compile=True,
        )
        packed_und_image_indexes = None
        if packed_pixel_patches_und is not None:
            und_embed = self.pixel_patch_embed(packed_pixel_patches_und)
            packed_sequence[packed_pixel_patch_indexes_und] = und_embed
            packed_und_image_indexes = packed_pixel_patch_indexes_und
        if packed_pixel_video_patches_und is not None:
            if not hasattr(self, 'pixel_video_patch_embed'):
                raise ValueError('R07 video UND patches require the dedicated video embedder')
            video_und_embed = self.pixel_video_patch_embed(packed_pixel_video_patches_und)
            packed_sequence[packed_pixel_video_patch_indexes_und] = video_und_embed
            if packed_und_image_indexes is None:
                packed_und_image_indexes = packed_pixel_video_patch_indexes_und
            else:
                packed_und_image_indexes = torch.cat([packed_und_image_indexes, packed_pixel_video_patch_indexes_und], dim=0).sort().values
        packed_gen_token_indexes = None
        image_clean_target = None
        image_noisy_patches = None
        image_timesteps = None
        video_clean_target = None
        video_noisy_patches = None
        video_timesteps = None
        has_gen_tokens = packed_pixel_patches_gen is not None or packed_pixel_video_patches_gen is not None
        if has_gen_tokens and packed_timesteps is None:
            raise ValueError('Generation tokens are present but packed_timesteps is missing')
        raw_timesteps = packed_timesteps if has_gen_tokens else None
        if not has_gen_tokens and raw_timesteps is not None:
            raise ValueError('PixelUMM-release received timesteps without R07 pixel GEN tokens')
        all_gen_indexes = []
        if packed_pixel_patches_gen is not None:
            all_gen_indexes.append(packed_pixel_patch_indexes_gen)
        if packed_pixel_video_patches_gen is not None:
            all_gen_indexes.append(packed_pixel_video_patch_indexes_gen)
        if all_gen_indexes:
            sorted_gen_indexes = torch.cat(all_gen_indexes, dim=0).sort().values
            if sorted_gen_indexes.numel() != raw_timesteps.numel():
                raise ValueError(f'packed_timesteps length does not match image/video generation token count: {raw_timesteps.numel()} vs {sorted_gen_indexes.numel()}')
            timestep_by_seq = raw_timesteps.new_zeros(sequence_length)
            timestep_by_seq[sorted_gen_indexes] = raw_timesteps
        if packed_pixel_video_patches_gen is not None:
            clean_patches = packed_pixel_video_patches_gen
            video_timesteps = timestep_by_seq[packed_pixel_video_patch_indexes_gen]
            video_timesteps = self._sample_train_timesteps(video_timesteps, modality='video')
            video_token_lens = self._pixel_video_token_lens(packed_pixel_video_patches_gen, pixel_video_patch_seqlens_gen)
            if sum(video_token_lens) != int(video_timesteps.shape[0]):
                raise ValueError(f'pixel_video_patch_seqlens_gen does not match packed video gen token count: {sum(video_token_lens)} vs {int(video_timesteps.shape[0])}')
            if pixel_video_grid_hw_gen is None:
                raise ValueError('time_schedule=pixelumm_resolution requires pixel_video_grid_hw_gen')
            video_timesteps = self._apply_time_schedule_per_sample(video_timesteps, video_token_lens, pixel_video_grid_hw_gen, patch_size=self.config.patch_size)
            noise = torch.randn_like(clean_patches)
            noisy_patches = (1 - video_timesteps[:, None]) * clean_patches + video_timesteps[:, None] * noise
            gen_embed = self.embed_pixel_video_tubes(noisy_patches)
            packed_sequence[packed_pixel_video_patch_indexes_gen] = gen_embed
            packed_gen_token_indexes = packed_pixel_video_patch_indexes_gen
            video_clean_target = clean_patches
            video_noisy_patches = noisy_patches
        if packed_pixel_patches_gen is not None:
            clean_patches = packed_pixel_patches_gen
            image_timesteps = timestep_by_seq[packed_pixel_patch_indexes_gen]
            image_timesteps = self._sample_train_timesteps(image_timesteps, modality='image')
            if packed_pixel_grid_hw_gen is None:
                raise ValueError('time_schedule=pixelumm_resolution requires packed_pixel_grid_hw_gen')
            schedule_token_lens = [int(h) * int(w) for (h, w) in packed_pixel_grid_hw_gen.tolist()]
            image_timesteps = self._apply_time_schedule_per_sample(image_timesteps, schedule_token_lens, packed_pixel_grid_hw_gen, patch_size=self.config.patch_size)
            noise = torch.randn_like(clean_patches)
            noisy_patches = (1 - image_timesteps[:, None]) * clean_patches + image_timesteps[:, None] * noise
            gen_embed = self._image_patch_embedder(gen_model=True)(noisy_patches)
            packed_sequence[packed_pixel_patch_indexes_gen] = gen_embed
            if packed_gen_token_indexes is None:
                packed_gen_token_indexes = packed_pixel_patch_indexes_gen
            else:
                packed_gen_token_indexes = torch.cat([packed_gen_token_indexes, packed_pixel_patch_indexes_gen], dim=0).sort().values
            image_clean_target = clean_patches
            image_noisy_patches = noisy_patches
        packed_und_token_indexes = packed_text_indexes
        if packed_und_image_indexes is not None:
            packed_und_token_indexes = torch.cat([packed_und_token_indexes, packed_und_image_indexes], dim=0)
        extra_inputs = {'packed_und_token_indexes': packed_und_token_indexes, 'packed_gen_token_indexes': packed_gen_token_indexes}
        llm_position_ids = packed_mrope_position_ids if packed_mrope_position_ids is not None else packed_position_ids
        if llm_position_ids is not None:
            llm_position_ids = llm_position_ids.contiguous()
        last_hidden_state = self.language_model(packed_sequence=packed_sequence, sample_lens=sample_lens, attention_mask=attention_mask, packed_position_ids=llm_position_ids, **extra_inputs)
        mse = None
        mse_loss_reduction_weights = None
        if image_clean_target is not None or video_clean_target is not None:
            mse_parts = []
            mse_weight_parts = []

            def per_token_mse(pred, target, loss_mask=None):
                sq = (pred.float() - target.float()) ** 2
                if loss_mask is None:
                    return sq.mean(dim=-1, keepdim=True)
                mask = loss_mask.to(device=sq.device, dtype=sq.dtype)
                denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
                return (sq * mask).sum(dim=-1, keepdim=True) / denom
            if image_clean_target is not None:
                image_has_mse = image_timesteps > 0
                image_hidden = last_hidden_state[packed_pixel_patch_indexes_gen[image_has_mse]]
                image_model_out = self.unpatchify_head(image_hidden)
                t = image_timesteps[:, None]
                t_f32 = t[image_has_mse].float().clamp(min=self.config.x_pred_t_min)
                v_pred = (image_noisy_patches[image_has_mse].float() - image_model_out.float()) / t_f32
                v_target = (image_noisy_patches[image_has_mse].float() - image_clean_target[image_has_mse].float()) / t_f32
                image_mse = per_token_mse(v_pred, v_target)
                mse_parts.append(image_mse)
                if packed_gen_image_lens is None:
                    raise ValueError('R07 sample-mean MSE requires packed_gen_image_lens')
                mse_weight_parts.append(sample_mean_token_weights(packed_gen_image_lens, output_size=image_mse.shape[0], device=image_mse.device))
            if video_clean_target is not None:
                video_has_mse = video_timesteps > 0
                video_hidden = last_hidden_state[packed_pixel_video_patch_indexes_gen[video_has_mse]]
                video_model_out = self.video_unpatchify_head(video_hidden)
                t = video_timesteps[:, None]
                t_f32 = t[video_has_mse].float().clamp(min=self.config.x_pred_t_min)
                v_pred = (video_noisy_patches[video_has_mse].float() - video_model_out.float()) / t_f32
                v_target = (video_noisy_patches[video_has_mse].float() - video_clean_target[video_has_mse].float()) / t_f32
                video_mse = per_token_mse(v_pred, v_target)
                mse_parts.append(video_mse)
                if packed_gen_video_lens is None:
                    raise ValueError('R07 sample-mean MSE requires packed_gen_video_lens')
                mse_weight_parts.append(sample_mean_token_weights(packed_gen_video_lens, output_size=video_mse.shape[0], device=video_mse.device))
            mse = torch.cat(mse_parts, dim=0)
            if mse_weight_parts:
                mse_loss_reduction_weights = torch.cat(mse_weight_parts)
        ce = None
        if ce_loss_indexes is not None:
            ce = _checkpointed_chunked_lm_head_cross_entropy(self.language_model.lm_head, last_hidden_state[ce_loss_indexes], packed_label_ids, self.config.ce_loss_checkpoint_chunk_size)

        def _has_elements(value: Optional[torch.Tensor]) -> bool:
            return value is not None and int(value.numel()) > 0
        conditional_graph_anchor = self._pixelumm_zero_root_graph_anchor(last_hidden_state, has_und_image=_has_elements(packed_pixel_patches_und), has_und_video=_has_elements(packed_pixel_video_patches_und), has_gen_image=_has_elements(packed_pixel_patches_gen), has_gen_video=_has_elements(packed_pixel_video_patches_gen), has_text_output=ce is not None and int(ce.numel()) > 0)
        return dict(mse=mse, ce=ce, mse_loss_reduction_weights=mse_loss_reduction_weights, conditional_graph_anchor=conditional_graph_anchor)

    def _encode_prompt_text(self, tokenizer, prompt: str) -> list[int]:
        try:
            return tokenizer.encode(prompt, add_special_tokens=False)
        except TypeError:
            return tokenizer.encode(prompt)

    def _text_to_visual_chatml_prompt_ids(self, tokenizer, prompt: str) -> list[int]:
        caption_ids = self._encode_prompt_text(tokenizer, prompt)[:_PIXELUMM_TEXT_TOKEN_LIMIT]
        return self._encode_prompt_text(tokenizer, _CHATML_USER_START) + caption_ids + self._encode_prompt_text(tokenizer, f'{_CHATML_END}{_CHATML_SEP}{_CHATML_ASSISTANT_START}')

    def _image_to_text_chatml_prefix_ids(self, tokenizer) -> list[int]:
        return self._encode_prompt_text(tokenizer, f'{_CHATML_SYSTEM_START}{_DEFAULT_SYSTEM_PROMPT}{_CHATML_END}{_CHATML_SEP}{_CHATML_USER_START}')

    def _image_to_text_chatml_suffix_ids(self, tokenizer, instruction: str) -> list[int]:
        return self._encode_prompt_text(tokenizer, f'{instruction}{_CHATML_END}{_CHATML_SEP}{_CHATML_ASSISTANT_START}')

    def _prepare_token_ids_as_text_cache_input(self, curr_kvlens, curr_rope, prompt_ids_list):
        packed_text_ids = list()
        packed_text_position_ids = list()
        text_token_lens = list()
        packed_text_indexes = list()
        packed_key_value_indexes = list()
        curr = 0
        (newlens, new_rope) = (list(), list())
        for (text_ids, curr_kvlen, curr_position_id) in zip(prompt_ids_list, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.append(self._text_mrope_position_ids(curr_position_id, len(text_ids)))
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            curr += len(text_ids)
        packed_text_position_ids = torch.cat(packed_text_position_ids, dim=1)
        generation_input = {'text_token_lens': torch.tensor(text_token_lens, dtype=torch.int), 'packed_text_ids': torch.tensor(packed_text_ids, dtype=torch.long), 'packed_text_position_ids': packed_text_position_ids, 'packed_text_indexes': torch.tensor(packed_text_indexes, dtype=torch.long), 'packed_key_value_indexes': torch.tensor(packed_key_value_indexes, dtype=torch.long), 'key_values_lens': torch.tensor(curr_kvlens, dtype=torch.int)}
        return (generation_input, newlens, new_rope)

    def prepare_text_to_visual_chatml_prompts(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
        """T2I/T2V inference prefill matching the ChatML training prompt."""
        prompt_ids_list = [self._text_to_visual_chatml_prompt_ids(tokenizer, prompt) + [new_token_ids['eos_token_id']] for prompt in prompts]
        return self._prepare_token_ids_as_text_cache_input(curr_kvlens, curr_rope, prompt_ids_list)

    def prepare_image_to_text_chatml_prefixes(self, curr_kvlens, curr_rope, tokenizer):
        """I2T inference prefill before the image tokens."""
        prompt_ids = self._image_to_text_chatml_prefix_ids(tokenizer)
        prompt_ids_list = [prompt_ids for _ in curr_kvlens]
        return self._prepare_token_ids_as_text_cache_input(curr_kvlens, curr_rope, prompt_ids_list)

    def prepare_image_to_text_chatml_suffix_contexts_for_decode(self, curr_kvlens, curr_rope, instructions, tokenizer):
        """I2T inference prefill after image tokens, aligned with train-time CE shift."""
        suffix_ids_list = [self._image_to_text_chatml_suffix_ids(tokenizer, instruction) for instruction in instructions]
        if any((len(ids) == 0 for ids in suffix_ids_list)):
            raise ValueError('image-to-text ChatML suffix cannot be empty')
        context_ids_list = [ids[:-1] for ids in suffix_ids_list]
        start_token_ids = [ids[-1] for ids in suffix_ids_list]
        (generation_input, newlens, new_rope) = self._prepare_token_ids_as_text_cache_input(curr_kvlens, curr_rope, context_ids_list)
        return (generation_input, newlens, new_rope, start_token_ids)

    @torch.no_grad
    def forward_cache_update_text(self, past_key_values: NaiveCache, packed_text_ids: torch.IntTensor, packed_text_position_ids: torch.LongTensor, text_token_lens: torch.LongTensor, packed_text_indexes: torch.LongTensor, packed_key_value_indexes: torch.LongTensor, key_values_lens: torch.IntTensor):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        output = self.language_model.forward_inference(packed_query_sequence=packed_text_embedding, query_lens=text_token_lens, packed_query_position_ids=packed_text_position_ids, packed_query_indexes=packed_text_indexes, past_key_values=past_key_values, packed_key_value_indexes=packed_key_value_indexes, key_values_lens=key_values_lens, update_past_key_values=True, is_causal=True, mode='und')
        past_key_values = output.past_key_values
        return past_key_values

    def _validate_pixel_generation_prefix_contract(self, visual_start_prefilled: bool) -> None:
        if not visual_start_prefilled:
            raise ValueError('pixelumm_mot pixel generation requires one causal <vision_start> in the KV cache')

    def prepare_pixel_noise(self, curr_kvlens, curr_rope, image_sizes, *, visual_start_prefilled: bool=False):
        """Generate image noise and construct the denoising query.

        For ``pixelumm_mot``, callers must first cache exactly one causal
        ``<vision_start>`` with :meth:`prepare_visual_delimiter_tokens`.  This
        method then returns a patch-only, bidirectional noise query, matching
        the train-time ``causal start -> full/noise patches`` mask exactly.
        """
        self._validate_pixel_generation_prefix_contract(visual_start_prefilled)
        (packed_text_ids, packed_text_indexes) = (list(), list())
        (packed_pixel_token_indexes, packed_init_noises) = (list(), list())
        (packed_position_ids, packed_seqlens, packed_indexes) = (list(), list(), list())
        packed_key_value_indexes = list()
        query_curr = curr = 0
        for ((H, W), curr_kvlen, curr_position_id) in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen
            (h, w) = (H // self.config.patch_size, W // self.config.patch_size)
            num_image_tokens = h * w
            patch_dim = self.config.patch_size * self.config.patch_size * 3
            packed_init_noises.append(torch.randn(num_image_tokens, patch_dim))
            packed_pixel_token_indexes.extend(range(query_curr, query_curr + num_image_tokens))
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens
            (mrope_ids, _) = self._image_mrope_position_ids(curr_position_id, h, w, include_start=False, include_end=False)
            packed_position_ids.append(mrope_ids)
            packed_seqlens.append(num_image_tokens + 0)
        packed_position_ids = torch.cat(packed_position_ids, dim=1)
        generation_input = {'packed_text_ids': torch.tensor(packed_text_ids, dtype=torch.long), 'packed_text_indexes': torch.tensor(packed_text_indexes, dtype=torch.long), 'packed_init_noises': torch.cat(packed_init_noises, dim=0), 'packed_pixel_token_indexes': torch.tensor(packed_pixel_token_indexes, dtype=torch.long), 'packed_seqlens': torch.tensor(packed_seqlens, dtype=torch.int), 'packed_position_ids': packed_position_ids, 'key_values_lens': torch.tensor(curr_kvlens, dtype=torch.int), 'packed_indexes': torch.tensor(packed_indexes, dtype=torch.long), 'packed_key_value_indexes': torch.tensor(packed_key_value_indexes, dtype=torch.long)}
        return generation_input

    def prepare_pixel_video_noise(self, curr_kvlens, curr_rope, video_sizes, fps=None, *, visual_start_prefilled: bool=False):
        """Pixel-video version of :meth:`prepare_pixel_noise`."""
        self._validate_pixel_generation_prefix_contract(visual_start_prefilled)
        (packed_text_ids, packed_text_indexes) = (list(), list())
        (packed_pixel_video_token_indexes, packed_init_noises) = (list(), list())
        (packed_position_ids, packed_seqlens, packed_indexes) = (list(), list(), list())
        packed_key_value_indexes = list()
        query_curr = curr = 0
        tube_frames = self.config.pixel_video_temporal_patch_size
        patch_dim = tube_frames * self.config.patch_size * self.config.patch_size * 3
        for ((num_frames, H, W), curr_kvlen, curr_position_id) in zip(video_sizes, curr_kvlens, curr_rope):
            if num_frames % tube_frames != 0:
                raise ValueError(f'Expected num_frames to be divisible by tube_frames={tube_frames}, got {num_frames}')
            else:
                temporal_groups = num_frames // tube_frames
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen
            (h, w) = (H // self.config.patch_size, W // self.config.patch_size)
            num_video_tokens = h * w * temporal_groups
            packed_init_noises.append(torch.randn(num_video_tokens, patch_dim))
            packed_pixel_video_token_indexes.extend(range(query_curr, query_curr + num_video_tokens))
            packed_indexes.extend(range(curr, curr + num_video_tokens))
            curr += num_video_tokens
            query_curr += num_video_tokens
            (mrope_ids, _) = self._video_mrope_position_ids(curr_position_id, temporal_groups, h, w, fps=fps, temporal_compression_factor=tube_frames, include_start=False, include_end=False)
            packed_position_ids.append(mrope_ids)
            packed_seqlens.append(num_video_tokens + 0)
        packed_position_ids = torch.cat(packed_position_ids, dim=1)
        generation_input = {'packed_text_ids': torch.tensor(packed_text_ids, dtype=torch.long), 'packed_text_indexes': torch.tensor(packed_text_indexes, dtype=torch.long), 'packed_init_noises': torch.cat(packed_init_noises, dim=0), 'packed_pixel_token_indexes': torch.tensor(packed_pixel_video_token_indexes, dtype=torch.long), 'packed_seqlens': torch.tensor(packed_seqlens, dtype=torch.int), 'packed_position_ids': packed_position_ids, 'key_values_lens': torch.tensor(curr_kvlens, dtype=torch.int), 'packed_indexes': torch.tensor(packed_indexes, dtype=torch.long), 'packed_key_value_indexes': torch.tensor(packed_key_value_indexes, dtype=torch.long)}
        return generation_input

    def prepare_pixel_images(self, curr_kvlens, curr_rope, images):
        """Prepare clean pixel patches for understanding (KV cache prefill).

        Pixel-space version of prepare_vit_images. No ViT — PIL image → pixel patchify.
        Uses 1D RoPE (same as rest of unified PixelUMM).
        """
        packed_pixel_token_indexes = list()
        packed_pixel_patches = []
        (packed_text_ids, packed_text_indexes) = (list(), list())
        (packed_seqlens, packed_position_ids, packed_indexes) = (list(), list(), list())
        packed_key_value_indexes = list()
        _curr = curr = 0
        (newlens, new_rope) = (list(), list())
        for (image, curr_kvlen, curr_position_id) in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen
            ps = self.config.patch_size
            (patches, (h_patches, w_patches)) = patchify_image(image, ps, max_size=max(image.size))
            num_img_tokens = patches.shape[0]
            (mrope_h, mrope_w) = (h_patches, w_patches)
            packed_pixel_patches.append(patches)
            packed_pixel_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens
            (mrope_ids, next_position_id) = self._image_mrope_position_ids(curr_position_id, mrope_h, mrope_w, include_start=False, include_end=False)
            packed_position_ids.append(mrope_ids)
            new_rope.append(next_position_id)
            packed_seqlens.append(num_img_tokens)
            newlens.append(curr_kvlen + num_img_tokens)
        packed_position_ids = torch.cat(packed_position_ids, dim=1)
        generation_input = {'packed_text_ids': torch.tensor(packed_text_ids, dtype=torch.long), 'packed_text_indexes': torch.tensor(packed_text_indexes, dtype=torch.long), 'packed_pixel_patches': torch.cat(packed_pixel_patches, dim=0), 'packed_pixel_token_indexes': torch.tensor(packed_pixel_token_indexes, dtype=torch.long), 'packed_position_ids': packed_position_ids, 'packed_seqlens': torch.tensor(packed_seqlens, dtype=torch.int), 'packed_indexes': torch.tensor(packed_indexes, dtype=torch.long), 'packed_key_value_indexes': torch.tensor(packed_key_value_indexes, dtype=torch.long), 'key_values_lens': torch.tensor(curr_kvlens, dtype=torch.int)}
        return (generation_input, newlens, new_rope)

    def prepare_text_spans_for_cache(self, curr_kvlens, curr_rope, texts, tokenizer):
        """Causal mid-sequence text spans (e.g. V2T tube timestamp markers)."""
        ids_list = [self._encode_prompt_text(tokenizer, text) for text in texts]
        if any((len(ids) == 0 for ids in ids_list)):
            raise ValueError('Cache text span cannot be empty')
        return self._prepare_token_ids_as_text_cache_input(curr_kvlens, curr_rope, ids_list)

    def prepare_pixel_video_und_tubes_for_text(self, curr_kvlens, curr_rope, tube_tensors):
        """Pack video-UND tubes with the released 3D sequence/H/W coordinates."""
        if not self.config.pixel_video_separate_und_gen_embedder:
            raise ValueError('This checkpoint profile has no video UND embedder; video-VLM tubes are unsupported')
        if not len(curr_kvlens) == len(curr_rope) == len(tube_tensors):
            raise ValueError('V2T tube batches must match cache batch size')
        temporal_patch_size = int(self.config.pixel_video_temporal_patch_size)
        ps = self.config.patch_size
        packed_pixel_video_patches = []
        packed_pixel_video_patch_indexes = []
        packed_position_ids = []
        packed_seqlens = []
        packed_indexes = []
        packed_key_value_indexes = []
        newlens = []
        new_rope = []
        query_curr = 0
        combined_curr = 0
        for (curr_kvlen, curr_position_id, tube) in zip(curr_kvlens, curr_rope, tube_tensors):
            if tube.dim() != 4 or tube.shape[0] != temporal_patch_size:
                raise ValueError(f'V2T tube must be (temporal_patch_size, C, H, W) with temporal_patch_size={temporal_patch_size}, got {tuple(tube.shape)}')
            (_, _, H, W) = tube.shape
            if H % ps or W % ps:
                raise ValueError(f'V2T tube spatial size {(H, W)} not divisible by patch={ps}')
            h = H // ps
            w = W // ps
            num_tokens = h * w
            packed_key_value_indexes.extend(range(combined_curr, combined_curr + curr_kvlen))
            combined_curr += curr_kvlen
            tokens = tubeify(tube, ps, temporal_patch_size)
            packed_pixel_video_patches.append(tokens)
            packed_pixel_video_patch_indexes.extend(range(query_curr, query_curr + num_tokens))
            packed_indexes.extend(range(combined_curr, combined_curr + num_tokens))
            query_curr += num_tokens
            combined_curr += num_tokens
            grid_s = float(curr_position_id)
            position_axes = [torch.full((num_tokens,), grid_s, dtype=torch.float32)]
            position_axes.extend([torch.arange(h, dtype=torch.float32).repeat_interleave(w), torch.arange(w, dtype=torch.float32).repeat(h)])
            packed_position_ids.append(torch.stack(position_axes, dim=0))
            packed_seqlens.append(num_tokens)
            newlens.append(curr_kvlen + num_tokens)
            new_rope.append(int(curr_position_id) + 1)
        generation_input = {'packed_text_ids': torch.zeros(0, dtype=torch.long), 'packed_text_indexes': torch.zeros(0, dtype=torch.long), 'packed_pixel_video_patches': torch.cat(packed_pixel_video_patches, dim=0), 'packed_pixel_video_patch_indexes': torch.tensor(packed_pixel_video_patch_indexes, dtype=torch.long), 'packed_position_ids': torch.cat(packed_position_ids, dim=1), 'packed_seqlens': torch.tensor(packed_seqlens, dtype=torch.int), 'packed_indexes': torch.tensor(packed_indexes, dtype=torch.long), 'packed_key_value_indexes': torch.tensor(packed_key_value_indexes, dtype=torch.long), 'key_values_lens': torch.tensor(curr_kvlens, dtype=torch.int)}
        return (generation_input, newlens, new_rope)

    @torch.no_grad
    def forward_cache_update_pixel(self, past_key_values: NaiveCache, packed_text_ids: torch.LongTensor, packed_text_indexes: torch.LongTensor, packed_pixel_patches: Optional[torch.Tensor]=None, packed_pixel_token_indexes: Optional[torch.LongTensor]=None, packed_position_ids: Optional[torch.LongTensor]=None, packed_seqlens: Optional[torch.IntTensor]=None, packed_indexes: Optional[torch.LongTensor]=None, packed_key_value_indexes: Optional[torch.LongTensor]=None, key_values_lens: Optional[torch.IntTensor]=None, packed_pixel_video_patches: Optional[torch.Tensor]=None, packed_pixel_video_patch_indexes: Optional[torch.LongTensor]=None):
        """Prefill clean pixel patches into KV cache (understanding)."""
        if packed_pixel_video_patches is not None and not self.config.pixel_video_separate_und_gen_embedder:
            raise ValueError('This checkpoint profile has no video UND embedder; video-VLM tubes are unsupported')
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding
        model_dtype = packed_text_embedding.dtype
        if packed_pixel_video_patches is not None:
            if packed_pixel_video_patch_indexes is None:
                raise ValueError('Video pixel prefill requires video patch indexes')
            pixel_embed = self.pixel_video_patch_embed(packed_pixel_video_patches.to(model_dtype))
            pixel_token_indexes = packed_pixel_video_patch_indexes
        else:
            if packed_pixel_patches is None or packed_pixel_token_indexes is None:
                raise ValueError('Pixel prefill requires image or video pixel patches')
            pixel_embed = self.pixel_patch_embed(packed_pixel_patches.to(model_dtype))
            pixel_token_indexes = packed_pixel_token_indexes
        if pixel_embed.dtype != packed_sequence.dtype:
            pixel_embed = pixel_embed.to(packed_sequence.dtype)
        packed_sequence[pixel_token_indexes] = pixel_embed
        output = self.language_model.forward_inference(packed_query_sequence=packed_sequence, query_lens=packed_seqlens, packed_query_position_ids=packed_position_ids, packed_query_indexes=packed_indexes, past_key_values=past_key_values, packed_key_value_indexes=packed_key_value_indexes, key_values_lens=key_values_lens, update_past_key_values=True, is_causal=False, mode='und')
        return output.past_key_values

    def generate_image(self, packed_text_ids: torch.LongTensor, packed_text_indexes: torch.LongTensor, packed_init_noises: torch.Tensor, packed_pixel_token_indexes: torch.LongTensor, packed_seqlens: torch.IntTensor, packed_position_ids: torch.LongTensor, packed_indexes: torch.LongTensor, past_key_values: NaiveCache, key_values_lens: torch.IntTensor, packed_key_value_indexes: torch.LongTensor, num_timesteps: int=24, timestep_shift: float=1.0, sampler: str='dpm-solver', cfg_renorm_min: float=0.0, cfg_renorm_type: str='global', cfg_interval: Optional[Tuple[float, float]]=[0, 1], cfg_text_scale: float=1.0, cfg_text_packed_query_indexes: Optional[torch.LongTensor]=None, cfg_text_packed_position_ids: Optional[torch.LongTensor]=None, cfg_text_past_key_values: Optional[NaiveCache]=None, cfg_text_key_values_lens: Optional[torch.IntTensor]=None, cfg_text_packed_key_value_indexes: Optional[torch.LongTensor]=None, cfg_img_scale: float=1.0, cfg_img_packed_query_indexes: Optional[torch.LongTensor]=None, cfg_img_packed_position_ids: Optional[torch.LongTensor]=None, cfg_img_past_key_values: Optional[NaiveCache]=None, cfg_img_key_values_lens: Optional[torch.IntTensor]=None, cfg_img_packed_key_value_indexes: Optional[torch.LongTensor]=None):
        x_t = packed_init_noises

        def _v_at(x, t_scalar):
            """Compute velocity at state x and timestep t_scalar (with CFG gating)."""
            ts = torch.tensor([t_scalar] * x.shape[0], device=x.device)
            if t_scalar > cfg_interval[0] and t_scalar <= cfg_interval[1]:
                (cts, cis) = (cfg_text_scale, cfg_img_scale)
            else:
                (cts, cis) = (1.0, 1.0)
            return self._forward_flow(x_t=x, timestep=ts, packed_pixel_token_indexes=packed_pixel_token_indexes, packed_text_ids=packed_text_ids, packed_text_indexes=packed_text_indexes, packed_position_ids=packed_position_ids, packed_indexes=packed_indexes, packed_seqlens=packed_seqlens, key_values_lens=key_values_lens, past_key_values=past_key_values, packed_key_value_indexes=packed_key_value_indexes, cfg_renorm_min=cfg_renorm_min, cfg_renorm_type=cfg_renorm_type, cfg_text_scale=cts, cfg_text_packed_position_ids=cfg_text_packed_position_ids, cfg_text_packed_query_indexes=cfg_text_packed_query_indexes, cfg_text_key_values_lens=cfg_text_key_values_lens, cfg_text_past_key_values=cfg_text_past_key_values, cfg_text_packed_key_value_indexes=cfg_text_packed_key_value_indexes, cfg_img_scale=cis, cfg_img_packed_position_ids=cfg_img_packed_position_ids, cfg_img_packed_query_indexes=cfg_img_packed_query_indexes, cfg_img_key_values_lens=cfg_img_key_values_lens, cfg_img_past_key_values=cfg_img_past_key_values, cfg_img_packed_key_value_indexes=cfg_img_packed_key_value_indexes)
        if sampler != 'dpm-solver':
            raise ValueError(f"PixelUMM-release T2I requires sampler='dpm-solver', got {sampler!r}")
        x_t = sample_flow_dpm(_v_at, x_t, num_steps=num_timesteps, shift=timestep_shift)
        delimiter_tokens = 0
        generated_lens = packed_seqlens - delimiter_tokens
        unpacked_latent = x_t.split(generated_lens.tolist())
        return unpacked_latent

    @torch.no_grad
    def generate_video(self, packed_text_ids: torch.LongTensor, packed_text_indexes: torch.LongTensor, packed_init_noises: torch.Tensor, packed_pixel_token_indexes: torch.LongTensor, packed_seqlens: torch.IntTensor, packed_position_ids: torch.LongTensor, packed_indexes: torch.LongTensor, past_key_values: NaiveCache, key_values_lens: torch.IntTensor, packed_key_value_indexes: torch.LongTensor, num_timesteps: int=24, timestep_shift: float=1.0, sampler: str='unipc', cfg_renorm_min: float=0.0, cfg_renorm_type: str='global', cfg_interval: Optional[Tuple[float, float]]=[0, 1], cfg_text_scale: float=1.0, cfg_text_packed_query_indexes: Optional[torch.LongTensor]=None, cfg_text_packed_position_ids: Optional[torch.LongTensor]=None, cfg_text_past_key_values: Optional[NaiveCache]=None, cfg_text_key_values_lens: Optional[torch.IntTensor]=None, cfg_text_packed_key_value_indexes: Optional[torch.LongTensor]=None, cfg_img_scale: float=1.0, cfg_img_packed_query_indexes: Optional[torch.LongTensor]=None, cfg_img_packed_position_ids: Optional[torch.LongTensor]=None, cfg_img_past_key_values: Optional[NaiveCache]=None, cfg_img_key_values_lens: Optional[torch.IntTensor]=None, cfg_img_packed_key_value_indexes: Optional[torch.LongTensor]=None):
        x_t = packed_init_noises

        def _v_at(x, t_scalar):
            ts = torch.tensor([t_scalar] * x.shape[0], device=x.device)
            if t_scalar > cfg_interval[0] and t_scalar <= cfg_interval[1]:
                (cts, cis) = (cfg_text_scale, cfg_img_scale)
            else:
                (cts, cis) = (1.0, 1.0)
            return self._forward_flow(x_t=x, timestep=ts, packed_pixel_token_indexes=packed_pixel_token_indexes, packed_text_ids=packed_text_ids, packed_text_indexes=packed_text_indexes, packed_position_ids=packed_position_ids, packed_indexes=packed_indexes, packed_seqlens=packed_seqlens, key_values_lens=key_values_lens, past_key_values=past_key_values, packed_key_value_indexes=packed_key_value_indexes, cfg_renorm_min=cfg_renorm_min, cfg_renorm_type=cfg_renorm_type, cfg_text_scale=cts, cfg_text_packed_position_ids=cfg_text_packed_position_ids, cfg_text_packed_query_indexes=cfg_text_packed_query_indexes, cfg_text_key_values_lens=cfg_text_key_values_lens, cfg_text_past_key_values=cfg_text_past_key_values, cfg_text_packed_key_value_indexes=cfg_text_packed_key_value_indexes, cfg_img_scale=cis, cfg_img_packed_position_ids=cfg_img_packed_position_ids, cfg_img_packed_query_indexes=cfg_img_packed_query_indexes, cfg_img_key_values_lens=cfg_img_key_values_lens, cfg_img_past_key_values=cfg_img_past_key_values, cfg_img_packed_key_value_indexes=cfg_img_packed_key_value_indexes, use_pixel_video=True)
        if sampler == 'unipc':
            x_t = sample_flow_unipc(_v_at, x_t, num_steps=num_timesteps, shift=timestep_shift)
        elif sampler == 'dpm-solver':
            if num_timesteps < 3:
                raise ValueError('T2V DPM-Solver requires at least 3 steps')
            x_t = sample_flow_dpm(_v_at, x_t, num_steps=num_timesteps, shift=timestep_shift)
        else:
            raise ValueError(f"PixelUMM T2V supports 'unipc' or 'dpm-solver', got {sampler!r}")
        delimiter_tokens = 0
        generated_lens = packed_seqlens - delimiter_tokens
        return x_t.split(generated_lens.tolist())

    @torch.no_grad
    def _forward_flow(self, x_t: torch.Tensor, timestep: torch.LongTensor, packed_pixel_token_indexes: torch.LongTensor, packed_text_ids: torch.LongTensor, packed_text_indexes: torch.LongTensor, packed_indexes: torch.LongTensor, packed_position_ids: torch.LongTensor, packed_seqlens: torch.IntTensor, key_values_lens: torch.IntTensor, past_key_values: NaiveCache, packed_key_value_indexes: torch.LongTensor, cfg_renorm_min: float=0.0, cfg_renorm_type: str='global', cfg_text_scale: float=1.0, cfg_text_packed_position_ids: Optional[torch.LongTensor]=None, cfg_text_packed_query_indexes: Optional[torch.LongTensor]=None, cfg_text_key_values_lens: Optional[torch.Tensor]=None, cfg_text_past_key_values: Optional[NaiveCache]=None, cfg_text_packed_key_value_indexes: Optional[torch.LongTensor]=None, cfg_img_scale: float=1.0, cfg_img_packed_position_ids: Optional[torch.LongTensor]=None, cfg_img_packed_query_indexes: Optional[torch.LongTensor]=None, cfg_img_key_values_lens: Optional[torch.Tensor]=None, cfg_img_past_key_values: Optional[NaiveCache]=None, cfg_img_packed_key_value_indexes: Optional[torch.LongTensor]=None, use_pixel_video: bool=False):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding
        assert timestep.unique().shape[0] == 1
        t_val = timestep[0].item()
        x_t_raw = x_t
        if use_pixel_video:
            x_t = self.embed_pixel_video_tubes(x_t)
        else:
            image_embedder = self._image_patch_embedder(gen_model=True)
            x_t = image_embedder(x_t)
        if x_t.dtype != packed_sequence.dtype:
            x_t = x_t.to(packed_sequence.dtype)
        packed_generation_token_indexes = packed_pixel_token_indexes
        packed_sequence[packed_generation_token_indexes] = x_t
        extra_inputs = {'mode': 'gen', 'packed_gen_token_indexes': packed_generation_token_indexes, 'packed_und_token_indexes': packed_text_indexes}
        output = self.language_model.forward_inference(packed_query_sequence=packed_sequence, query_lens=packed_seqlens, packed_query_position_ids=packed_position_ids, packed_query_indexes=packed_indexes, past_key_values=past_key_values, key_values_lens=key_values_lens, packed_key_value_indexes=packed_key_value_indexes, update_past_key_values=False, is_causal=False, **extra_inputs)

        def _pixel_head(hidden_seq):
            hidden_sub = hidden_seq[packed_pixel_token_indexes]
            if use_pixel_video:
                return self.video_unpatchify_head(hidden_sub)
            return self.unpatchify_head(hidden_sub)
        model_out = _pixel_head(output.packed_query_sequence)
        v_t = (x_t_raw.float() - model_out.float()) / max(t_val, self.config.x_pred_t_min)
        if cfg_text_scale > 1.0:
            cfg_text_output = self.language_model.forward_inference(packed_query_sequence=packed_sequence, query_lens=packed_seqlens, packed_query_position_ids=cfg_text_packed_position_ids, packed_query_indexes=cfg_text_packed_query_indexes, past_key_values=cfg_text_past_key_values, key_values_lens=cfg_text_key_values_lens, packed_key_value_indexes=cfg_text_packed_key_value_indexes, update_past_key_values=False, is_causal=False, **extra_inputs)
            cfg_text_out = _pixel_head(cfg_text_output.packed_query_sequence)
            cfg_text_v_t = (x_t_raw.float() - cfg_text_out.float()) / max(t_val, self.config.x_pred_t_min)
        if cfg_img_scale > 1.0:
            cfg_img_output = self.language_model.forward_inference(packed_query_sequence=packed_sequence, query_lens=packed_seqlens, packed_query_position_ids=cfg_img_packed_position_ids, packed_query_indexes=cfg_img_packed_query_indexes, past_key_values=cfg_img_past_key_values, key_values_lens=cfg_img_key_values_lens, packed_key_value_indexes=cfg_img_packed_key_value_indexes, update_past_key_values=False, is_causal=False, **extra_inputs)
            cfg_img_out = _pixel_head(cfg_img_output.packed_query_sequence)
            cfg_img_v_t = (x_t_raw.float() - cfg_img_out.float()) / max(t_val, self.config.x_pred_t_min)
        if cfg_text_scale > 1.0:
            if cfg_renorm_type == 'text_channel':
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)
                scale = (norm_v_t / (norm_v_t_text_ + 1e-08)).clamp(min=cfg_renorm_min, max=1.0)
                v_t_text = v_t_text_ * scale
                if cfg_img_scale > 1.0:
                    v_t = cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
                else:
                    v_t = v_t_text
            else:
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                if cfg_img_scale > 1.0:
                    v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
                else:
                    v_t_ = v_t_text_
                if cfg_renorm_type == 'none':
                    v_t = v_t_
                else:
                    if cfg_renorm_type == 'global':
                        norm_v_t = torch.norm(v_t)
                        norm_v_t_ = torch.norm(v_t_)
                    elif cfg_renorm_type == 'channel':
                        norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                        norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
                    else:
                        raise NotImplementedError(f'{cfg_renorm_type} is not supported')
                    scale = (norm_v_t / (norm_v_t_ + 1e-08)).clamp(min=cfg_renorm_min, max=1.0)
                    v_t = v_t_ * scale
        return v_t

    def prepare_text_start_tokens(self, curr_kvlens, curr_rope, start_token_ids):
        (packed_start_tokens, packed_key_value_indexes) = (list(), list())
        packed_query_position_ids = list()
        curr = 0
        for (curr_kvlen, curr_position_id, token_id) in zip(curr_kvlens, curr_rope, start_token_ids):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            packed_start_tokens.append(int(token_id))
            packed_query_position_ids.append(self._text_mrope_position_ids(curr_position_id, 1))
            curr += curr_kvlen
        packed_query_position_ids = torch.cat(packed_query_position_ids, dim=1)
        generation_input = {'packed_start_tokens': torch.tensor(packed_start_tokens, dtype=torch.long), 'packed_query_position_ids': packed_query_position_ids, 'key_values_lens': torch.tensor(curr_kvlens, dtype=torch.int), 'packed_key_value_indexes': torch.tensor(packed_key_value_indexes, dtype=torch.long)}
        return generation_input

    def prepare_visual_delimiter_tokens(self, curr_kvlens, curr_rope, new_token_ids, delimiter: str):
        """Prepare a single causal visual delimiter per sample for pixelumm_mot inference.

        ``<soi>`` and ``<eoi>`` are text embeddings routed through the und expert,
        but their MRoPE coordinates are the visual marker coordinates. For both
        SenseNova and Qwen-VL MRoPE, those marker coordinates match
        ``_text_mrope_position_ids(curr_position_id, 1)``.
        """
        if delimiter == 'start':
            token_id = new_token_ids['start_of_image']
        elif delimiter == 'end':
            token_id = new_token_ids['end_of_image']
        else:
            raise ValueError(f"Unknown visual delimiter {delimiter!r}; expected 'start' or 'end'")
        (packed_text_ids, packed_text_indexes) = (list(), list())
        (packed_text_position_ids, packed_key_value_indexes) = (list(), list())
        (newlens, new_rope) = (list(), list())
        curr = 0
        for (curr_kvlen, curr_position_id) in zip(curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen
            packed_text_ids.append(token_id)
            packed_text_indexes.append(curr)
            packed_text_position_ids.append(self._text_mrope_position_ids(curr_position_id, 1))
            curr += 1
            newlens.append(curr_kvlen + 1)
            new_rope.append(curr_position_id + 1)
        packed_text_position_ids = torch.cat(packed_text_position_ids, dim=1)
        generation_input = {'text_token_lens': torch.ones(len(curr_kvlens), dtype=torch.int), 'packed_text_ids': torch.tensor(packed_text_ids, dtype=torch.long), 'packed_text_position_ids': packed_text_position_ids, 'packed_text_indexes': torch.tensor(packed_text_indexes, dtype=torch.long), 'packed_key_value_indexes': torch.tensor(packed_key_value_indexes, dtype=torch.long), 'key_values_lens': torch.tensor(curr_kvlens, dtype=torch.int)}
        return (generation_input, newlens, new_rope)

    @torch.no_grad
    def generate_text(self, past_key_values: NaiveCache, packed_key_value_indexes: torch.LongTensor, key_values_lens: torch.IntTensor, packed_start_tokens: torch.LongTensor, packed_query_position_ids: torch.LongTensor, max_length: int, do_sample: bool=False, temperature: float=1.0, end_token_id: int=None, return_generation_metadata: bool=False):
        step = 0
        generated_sequence = []
        curr_tokens = packed_start_tokens
        hit_end_token = False
        while step < max_length:
            generated_sequence.append(curr_tokens)
            packed_text_embedding = self.language_model.model.embed_tokens(curr_tokens)
            query_lens = torch.ones_like(curr_tokens)
            packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(0, len(key_values_lens), device=key_values_lens.device, dtype=key_values_lens.dtype)
            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] += i
            packed_key_value_indexes = torch.cat(uppacked, dim=0)
            output = self.language_model.forward_inference(packed_query_sequence=packed_text_embedding, query_lens=query_lens, packed_query_position_ids=packed_query_position_ids, packed_query_indexes=packed_query_indexes, past_key_values=past_key_values, key_values_lens=key_values_lens, packed_key_value_indexes=packed_key_value_indexes, update_past_key_values=True, is_causal=True, mode='und')
            past_key_values = output.past_key_values
            packed_query_sequence = output.packed_query_sequence
            pred_logits = self.language_model.lm_head(packed_query_sequence)
            if do_sample:
                probs = nn.functional.softmax(pred_logits / temperature, dim=-1)
                curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                curr_tokens = torch.argmax(pred_logits, dim=-1)
            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] = torch.cat([uppacked[i], torch.tensor([uppacked[i][-1] + 1], device=uppacked[i].device)], dim=0)
            packed_key_value_indexes = torch.cat(uppacked, dim=0)
            key_values_lens = key_values_lens + 1
            packed_query_position_ids = self._advance_text_query_position_ids(packed_query_position_ids)
            step += 1
            if end_token_id is not None and curr_tokens[0] == end_token_id:
                hit_end_token = True
                break
        output_device = generated_sequence[0].device
        generated_sequence = torch.stack([i.to(output_device) for i in generated_sequence], dim=0)
        if return_generation_metadata:
            return (generated_sequence, {'hit_end_token': hit_end_token})
        return generated_sequence
