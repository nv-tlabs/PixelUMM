# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0


import math
import random

import numpy as np
import torch

from .data_utils import len2weight
class DataConfig:
    def __init__(
        self,
        grouped_datasets,
        text_cond_dropout_prob=0.1,
        pixel_und_cond_dropout_prob=0.4,
        pixel_gen_cond_dropout_prob=0.1,
        pixel_token_patch_size=16,
        pixel_video_temporal_patch_size=4,
    ):
        self.grouped_datasets = grouped_datasets
        self.text_cond_dropout_prob = text_cond_dropout_prob
        self.pixel_und_cond_dropout_prob = pixel_und_cond_dropout_prob
        self.pixel_gen_cond_dropout_prob = pixel_gen_cond_dropout_prob
        self.pixel_token_patch_size = pixel_token_patch_size
        self.pixel_video_temporal_patch_size = pixel_video_temporal_patch_size


class PackedDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, 
        data_config, 
        tokenizer, 
        special_tokens,
        local_rank, 
        world_size, 
        expected_num_tokens=32768, 
        max_num_tokens_per_sample=16384,
        max_num_tokens=36864,
    ):
        super().__init__()
        self.expected_num_tokens = expected_num_tokens
        self.max_num_tokens_per_sample = max_num_tokens_per_sample
        self.max_num_tokens = max_num_tokens
        if not 0 < self.expected_num_tokens <= self.max_num_tokens:
            raise ValueError("expected_num_tokens must be in (0, max_num_tokens]")
        if not 0 < self.max_num_tokens_per_sample <= self.max_num_tokens:
            raise ValueError("max_num_tokens_per_sample must be in (0, max_num_tokens]")
        self.tokenizer = tokenizer
        self.local_rank = local_rank
        self.world_size = world_size
        for k, v in special_tokens.items():
            setattr(self, k, v)
        self.data_config = data_config

        grouped_dataset_names, grouped_datasets, is_mandatory, grouped_weights = self.build_datasets(
            data_config.grouped_datasets
        )
        self.grouped_dataset_names = grouped_dataset_names
        self.grouped_datasets = grouped_datasets
        self.dataset_iters = [iter(dataset) for dataset in grouped_datasets]
        self.is_mandatory = is_mandatory
        self.grouped_weights = grouped_weights
        if (
            not grouped_weights
            or not all(math.isfinite(weight) and weight >= 0 for weight in grouped_weights)
            or sum(grouped_weights) <= 0
        ):
            raise ValueError("Local JSONL group weights must be finite, nonnegative and nonzero")
        if any(mandatory and weight <= 0 for mandatory, weight in zip(is_mandatory, grouped_weights)):
            raise ValueError("Mandatory local JSONL groups require positive weight")
        self.print_sampling_weights()

    def build_datasets(self, datasets_metainfo):
        grouped_dataset_names = []
        datasets = []
        is_mandatory = []
        grouped_weights = []
        for grouped_dataset_name, raw_dataset_args in datasets_metainfo.items():
            # Never mutate the parsed YAML. Reusing one DataConfig in a CPU
            # preflight and then in training must construct the same datasets.
            dataset_args = dict(raw_dataset_args)
            grouped_dataset_names.append(grouped_dataset_name)
            is_mandatory.append(dataset_args.pop('is_mandatory', False))
            grouped_weights.append(dataset_args.pop('weight', 0.0))
            registry_name = dataset_args.pop('registry_name', grouped_dataset_name)

            if registry_name != "pixelumm_local_jsonl":
                raise ValueError(
                    "PixelUMM-release supports only registry_name="
                    f"'pixelumm_local_jsonl', got {registry_name!r} for "
                    f"{grouped_dataset_name!r}"
                )
            if "manifest_paths" not in dataset_args:
                raise ValueError(f"{grouped_dataset_name} requires manifest_paths")

            from .local_jsonl_dataset import PixelUMMLocalJSONLDataset

            dataset = PixelUMMLocalJSONLDataset(
                tokenizer=self.tokenizer,
                local_rank=self.local_rank,
                world_size=self.world_size,
                **dataset_args
            )
            datasets.append(dataset)

        return grouped_dataset_names, datasets, is_mandatory, grouped_weights

    def print_sampling_weights(self):
        if self.local_rank != 0:
            return
        total_weight = sum(self.grouped_weights)
        print("[DATA MIX] Effective sampling weights:")
        for group_name, group_weight in zip(self.grouped_dataset_names, self.grouped_weights):
            group_prob = float(group_weight) / float(total_weight)
            print(f"  {group_name}: effective={group_prob * 100.0:.4f}% raw_weight={group_weight:g}")

    @staticmethod
    def _sample_num_tokens(sample):
        return int(sample['num_tokens']) + sum(
            0 if item.get('type') == 'text_token_span' else 2
            for item in sample['sequence_plan']
        )

    def set_epoch(self, seed):
        for dataset in self.grouped_datasets:
            dataset.set_epoch(seed)

    def set_sequence_status(self):
        sequence_status = dict(
            curr                            = 0,
            sample_lens                     = list(),
            packed_position_ids             = list(),
            packed_mrope_position_ids        = list(),
            split_lens                      = list(),
            attn_modes                      = list(),
            # text
            packed_text_ids                 = list(),
            packed_text_indexes             = list(),
            packed_label_ids                = list(),
            ce_loss_indexes                 = list(),
            ce_loss_weights                 = list(),
            # pixel understanding (clean patches, no timestep)
            raw_pixel_images_und            = list(),   # list of (C,H,W) tensors, GPU-patchified before forward
            raw_pixel_image_ranges_und      = list(),
            packed_pixel_patch_indexes_und  = list(),
            raw_pixel_videos_und            = list(),   # list of (T,C,H,W) t4 tubes
            raw_pixel_video_ranges_und      = list(),
            raw_pixel_video_temporal_patch_sizes_und = list(),
            packed_pixel_video_patch_indexes_und = list(),
            # pixel generation (noisy patches, with timestep)
            raw_pixel_images_gen            = list(),   # list of (C,H,W) tensors, GPU-patchified before forward
            raw_pixel_image_ranges_gen      = list(),
            pixel_grid_hw_gen               = list(),   # one (h, w) token grid per pixel_gen segment
            gen_image_grid_hw               = list(),   # list of (h_patches, w_patches) per gen image
            packed_pixel_patch_indexes_gen  = list(),
            raw_pixel_videos_gen            = list(),   # list of (T,C,H,W) tensors, GPU-tubeified before forward
            raw_pixel_video_ranges_gen      = list(),
            raw_pixel_video_temporal_patch_sizes_gen = list(),
            packed_pixel_video_patch_indexes_gen = list(),
            pixel_video_patch_seqlens_gen   = list(),
            pixel_video_grid_hw_gen         = list(),  # one spatial patch grid per generated video
            gen_video_lens                  = list(),  # loss=1 video lengths in model-output order
            packed_timesteps                = list(),
        )
        return sequence_status

    def to_tensor(self, sequence_status):
        data = dict(
            sequence_length=sum(sequence_status['sample_lens']),
            sample_lens=list(sequence_status['sample_lens']),
            packed_text_ids=torch.tensor(sequence_status['packed_text_ids']),
            packed_text_indexes=torch.tensor(sequence_status['packed_text_indexes']),
            packed_position_ids=torch.tensor(sequence_status['packed_position_ids']),
        )
        if len(sequence_status['packed_mrope_position_ids']) > 0:
            data['packed_mrope_position_ids'] = torch.cat(
                sequence_status['packed_mrope_position_ids'], dim=1
            )

        sequence_len = data['sequence_length']
        if sequence_len > self.max_num_tokens:
            raise ValueError(
                f"Packed sequence length {sequence_len} exceeds max_num_tokens={self.max_num_tokens}. "
                "This usually means sample['num_tokens'] underestimated the true packed length."
            )
        # The capacity is a packing limit, not an extra document. The model
        # pads Q/K/V and the attention mask to the next 128-token boundary.
        data['split_lens'] = list(sequence_status['split_lens'])
        data['attn_modes'] = list(sequence_status['attn_modes'])

        if sequence_status['raw_pixel_images_und']:
            data['raw_pixel_images_und'] = sequence_status['raw_pixel_images_und']
            data['raw_pixel_image_ranges_und'] = sequence_status['raw_pixel_image_ranges_und']
            data['packed_pixel_patch_indexes_und'] = torch.tensor(sequence_status['packed_pixel_patch_indexes_und'])
        if sequence_status['raw_pixel_videos_und']:
            data['raw_pixel_videos_und'] = sequence_status['raw_pixel_videos_und']
            data['raw_pixel_video_ranges_und'] = sequence_status['raw_pixel_video_ranges_und']
            data['raw_pixel_video_temporal_patch_sizes_und'] = sequence_status[
                'raw_pixel_video_temporal_patch_sizes_und'
            ]
            data['packed_pixel_video_patch_indexes_und'] = torch.tensor(
                sequence_status['packed_pixel_video_patch_indexes_und']
            )
        if sequence_status['raw_pixel_images_gen']:
            data['raw_pixel_images_gen'] = sequence_status['raw_pixel_images_gen']
            data['raw_pixel_image_ranges_gen'] = sequence_status['raw_pixel_image_ranges_gen']
            data['packed_pixel_patch_indexes_gen'] = torch.tensor(
                sequence_status['packed_pixel_patch_indexes_gen'], dtype=torch.long
            )
            data['packed_pixel_grid_hw_gen'] = torch.tensor(
                sequence_status['pixel_grid_hw_gen'], dtype=torch.long
            )
            # Per-loss=1-image patch counts + grid (h, w) for packed MSE targets.
            if len(sequence_status['gen_image_grid_hw']) > 0:
                loss_image_lens = [h * w for (h, w) in sequence_status['gen_image_grid_hw']]
                data['packed_gen_image_lens'] = torch.tensor(loss_image_lens, dtype=torch.long)
        if sequence_status['raw_pixel_videos_gen']:
            data['raw_pixel_videos_gen'] = sequence_status['raw_pixel_videos_gen']
            data['raw_pixel_video_ranges_gen'] = sequence_status['raw_pixel_video_ranges_gen']
            data['raw_pixel_video_temporal_patch_sizes_gen'] = sequence_status[
                'raw_pixel_video_temporal_patch_sizes_gen'
            ]
            data['packed_pixel_video_patch_indexes_gen'] = torch.tensor(sequence_status['packed_pixel_video_patch_indexes_gen'])
            data['pixel_video_patch_seqlens_gen'] = torch.tensor(sequence_status['pixel_video_patch_seqlens_gen'])
            data['pixel_video_grid_hw_gen'] = torch.tensor(
                sequence_status['pixel_video_grid_hw_gen'], dtype=torch.long
            )
            if len(sequence_status['gen_video_lens']) > 0:
                data['packed_gen_video_lens'] = torch.tensor(
                    sequence_status['gen_video_lens'], dtype=torch.long
                )
        # Timesteps (generation)
        if len(sequence_status['packed_timesteps']) > 0:
            data['packed_timesteps'] = torch.tensor(sequence_status['packed_timesteps'])

        # CE loss (understanding / text generation)
        if len(sequence_status['packed_label_ids']) > 0:
            data['packed_label_ids'] = torch.tensor(sequence_status['packed_label_ids'])
            data['ce_loss_indexes'] = torch.tensor(sequence_status['ce_loss_indexes'])
            data['ce_loss_weights'] = torch.tensor(sequence_status['ce_loss_weights'])

        return data

    def __iter__(self):
        mandatory_groups = [
            index for index, mandatory in enumerate(self.is_mandatory)
            if mandatory
        ]
        mandatory_cursor = 0

        def next_from_group(group_index):
            try:
                return next(self.dataset_iters[group_index])
            except StopIteration as exc:
                raise RuntimeError(
                    f"Local JSONL dataset {self.grouped_dataset_names[group_index]!r} "
                    "unexpectedly stopped cycling"
                ) from exc

        def sample_group_index():
            n = random.random() * sum(self.grouped_weights)
            accum = 0.0
            for i, weight in enumerate(self.grouped_weights):
                if weight <= 0:
                    continue
                accum += weight
                if n < accum:
                    return i
            return max(i for i, weight in enumerate(self.grouped_weights) if weight > 0)

        sequence_status = self.set_sequence_status()
        while True:
            if sequence_status['curr'] == 0 and mandatory_groups:
                # One rotating anchor per batch covers every toy modality
                # without assuming four unrelated media samples fit together.
                group_index = mandatory_groups[mandatory_cursor % len(mandatory_groups)]
                mandatory_cursor += 1
                sample = next_from_group(group_index)
                num_tokens = self._sample_num_tokens(sample)
                if num_tokens > self.max_num_tokens_per_sample:
                    raise ValueError(
                        f"Mandatory dataset group {self.grouped_dataset_names[group_index]!r} "
                        f"has {num_tokens} tokens, exceeding "
                        f"max_num_tokens_per_sample={self.max_num_tokens_per_sample}"
                    )
                sequence_status = self.pack_sequence(sample, sequence_status)
                if sequence_status['curr'] >= self.expected_num_tokens:
                    yield self.to_tensor(sequence_status)
                    sequence_status = self.set_sequence_status()
                    continue

            group_index = sample_group_index()
            sample = next_from_group(group_index)

            num_tokens = self._sample_num_tokens(sample)
            if num_tokens > self.max_num_tokens_per_sample:
                raise ValueError(
                    f"Local JSONL sample has {num_tokens} tokens, exceeding "
                    f"max_num_tokens_per_sample={self.max_num_tokens_per_sample}"
                )

            if sequence_status['curr'] + num_tokens > self.max_num_tokens:
                # Yield the current sequence before reading another example.
                yield self.to_tensor(sequence_status)
                sequence_status = self.set_sequence_status()
                continue

            sequence_status = self.pack_sequence(sample, sequence_status)

            if sequence_status['curr'] >= self.expected_num_tokens:
                data = self.to_tensor(sequence_status)
                yield data
                sequence_status = self.set_sequence_status()

    def pack_sequence(self, sample, sequence_status):
        image_tensor_list = sample.get('image_tensor_list', [])
        video_tensor_list = sample.get('video_tensor_list', [])
        image_value_range_list = list(
            sample.get('image_value_range_list', ['float_neg1_pos1'] * len(image_tensor_list))
        )
        video_value_range_list = list(
            sample.get('video_value_range_list', ['float_neg1_pos1'] * len(video_tensor_list))
        )
        if len(image_value_range_list) != len(image_tensor_list):
            raise ValueError(
                "image_value_range_list must have one entry per image_tensor_list item "
                f"({len(image_value_range_list)} vs {len(image_tensor_list)})"
            )
        if len(video_value_range_list) != len(video_tensor_list):
            raise ValueError(
                "video_value_range_list must have one entry per video_tensor_list item "
                f"({len(video_value_range_list)} vs {len(video_tensor_list)})"
            )
        text_ids_list = sample['text_ids_list']
        sequence_plan = sample['sequence_plan']

        split_lens, attn_modes = list(), list()
        curr = sequence_status['curr']
        curr_rope_id = 0
        curr_mrope_id = 0
        current_gen_timestep = None
        sample_lens = 0
        cfg_group_decisions = {}
        cfg_group_probabilities = {}
        def cfg_dropout_decision(item, default_probability):
            """Resolve the released local JSONL's binary CFG dropout groups."""
            probability = float(default_probability)
            probability_source = item.get('cfg_dropout_prob_source')
            if probability_source is not None:
                if probability_source != 'text':
                    raise ValueError(
                        "cfg_dropout_prob_source must be 'text' when supplied, "
                        f"got {probability_source!r}"
                    )
                probability = float(self.data_config.text_cond_dropout_prob)
            if item.get('enable_cfg', 0) != 1:
                return False

            group = item.get('cfg_dropout_group')
            if group is None:
                return random.random() < probability
            group = str(group)
            if not group:
                raise ValueError('cfg_dropout_group cannot be empty')
            previous_probability = cfg_group_probabilities.setdefault(
                group, probability
            )
            if previous_probability != probability:
                raise ValueError(
                    "Every item in one CFG dropout group must use the same "
                    f"probability: group={group!r} "
                    f"first={previous_probability} current={probability}"
                )
            if group not in cfg_group_decisions:
                cfg_group_decisions[group] = random.random() < probability
            return bool(cfg_group_decisions[group])

        def append_text_mrope(length):
            nonlocal curr_mrope_id
            pos = torch.arange(curr_mrope_id, curr_mrope_id + length, dtype=torch.long)
            zeros = torch.zeros_like(pos)
            sequence_status['packed_mrope_position_ids'].append(
                torch.stack([pos, zeros, zeros], dim=0)
            )
            curr_mrope_id += length

        def append_image_mrope(h_tokens, w_tokens, include_start=True, include_end=True):
            nonlocal curr_mrope_id
            start = torch.full((1,), curr_mrope_id, dtype=torch.long)
            grid_t = curr_mrope_id + int(include_start)
            end = torch.full((1,), grid_t + 1, dtype=torch.long)
            rows = torch.arange(h_tokens, dtype=torch.long).repeat_interleave(w_tokens)
            cols = torch.arange(w_tokens, dtype=torch.long).repeat(h_tokens)
            t = torch.full((h_tokens * w_tokens,), grid_t, dtype=torch.long)
            t_chunks, h_chunks, w_chunks = [], [], []
            if include_start:
                t_chunks.append(start)
                h_chunks.append(start.new_zeros(1))
                w_chunks.append(start.new_zeros(1))
            t_chunks.append(t)
            h_chunks.append(rows)
            w_chunks.append(cols)
            if include_end:
                t_chunks.append(end)
                h_chunks.append(start.new_zeros(1))
                w_chunks.append(start.new_zeros(1))
            sequence_status['packed_mrope_position_ids'].append(
                torch.stack(
                    [torch.cat(t_chunks), torch.cat(h_chunks), torch.cat(w_chunks)],
                    dim=0,
                )
            )
            curr_mrope_id += 1 + int(include_start) + int(include_end)

        def append_video_mrope(
            temporal_groups,
            h_tokens,
            w_tokens,
            include_start=True,
            include_end=True,
        ):
            nonlocal curr_mrope_id
            start = torch.full((1,), curr_mrope_id, dtype=torch.long)
            grid_start = curr_mrope_id + int(include_start)
            end = torch.full((1,), grid_start + max(1, temporal_groups), dtype=torch.long)
            t = torch.arange(temporal_groups, dtype=torch.long)[:, None, None]
            h = torch.arange(h_tokens, dtype=torch.long)[None, :, None]
            w = torch.arange(w_tokens, dtype=torch.long)[None, None, :]
            t = t.expand(temporal_groups, h_tokens, w_tokens).reshape(-1) + grid_start
            h = h.expand(temporal_groups, h_tokens, w_tokens).reshape(-1)
            w = w.expand(temporal_groups, h_tokens, w_tokens).reshape(-1)
            t_chunks, h_chunks, w_chunks = [], [], []
            if include_start:
                t_chunks.append(start)
                h_chunks.append(torch.zeros(1, dtype=torch.long))
                w_chunks.append(torch.zeros(1, dtype=torch.long))
            t_chunks.append(t)
            h_chunks.append(h)
            w_chunks.append(w)
            if include_end:
                t_chunks.append(end)
                h_chunks.append(torch.zeros(1, dtype=torch.long))
                w_chunks.append(torch.zeros(1, dtype=torch.long))
            sequence_status['packed_mrope_position_ids'].append(
                torch.stack(
                    [torch.cat(t_chunks), torch.cat(h_chunks), torch.cat(w_chunks)],
                    dim=0,
                )
            )
            curr_mrope_id += max(1, temporal_groups) + int(include_start) + int(include_end)

        def append_special_token_loss(token_index, item):
            if item is None or item.get('special_token_loss', 0) != 1:
                return
            sequence_status['ce_loss_indexes'].append(token_index)
            sequence_status['ce_loss_weights'].append(1.0)
            sequence_status['packed_label_ids'].append(item['special_token_label'])

        def append_start_of_image_delimiter(append_mrope=True):
            nonlocal curr, curr_split_len, curr_rope_id, sample_lens
            sequence_status['packed_text_ids'].append(self.start_of_image)
            sequence_status['packed_text_indexes'].append(curr)
            curr += 1
            curr_split_len += 1
            attn_modes.append("causal")
            sequence_status['packed_position_ids'].extend(
                range(curr_rope_id, curr_rope_id + curr_split_len)
            )
            curr_rope_id += curr_split_len
            if append_mrope:
                append_text_mrope(curr_split_len)
            split_lens.append(curr_split_len)
            sample_lens += curr_split_len
            curr_split_len = 0

        def append_causal_end_of_image_delimiter(append_mrope=True, item=None):
            nonlocal curr, curr_split_len, curr_rope_id, sample_lens
            sequence_status['packed_text_ids'].append(self.end_of_image)
            sequence_status['packed_text_indexes'].append(curr)
            append_special_token_loss(curr, item)
            curr += 1
            curr_split_len += 1
            attn_modes.append("causal")
            sequence_status['packed_position_ids'].extend(
                range(curr_rope_id, curr_rope_id + curr_split_len)
            )
            curr_rope_id += curr_split_len
            if append_mrope:
                append_text_mrope(curr_split_len)
            split_lens.append(curr_split_len)
            sample_lens += curr_split_len
            curr_split_len = 0

        def next_item_is_target_text(item_index):
            """pixelumm_mot keeps <eoi> only before i2t/v2t target text."""
            if item_index + 1 >= len(sequence_plan):
                return False
            next_item = sequence_plan[item_index + 1]
            return next_item.get('type') in {'text', 'text_token_span'} and next_item.get('loss', 0) == 1

        def append_text_token_span(item, item_index):
            """Append exact pre-tokenized text with per-token CE supervision.

            ``text`` items are a high-level convenience that add
            <|im_start|>/<|im_end|> and supervise whole spans. ``text_token_span``
            is the lower-level primitive for chat templates: callers provide
            final input ids, next-token labels, and a per-position loss mask.
            """
            nonlocal curr, curr_split_len, curr_rope_id
            span = text_ids_list.pop(0)
            if not isinstance(span, dict):
                raise TypeError(f"text_token_span expects a dict entry in text_ids_list, got {type(span)!r}")

            cfg_dropped = cfg_dropout_decision(
                item, self.data_config.text_cond_dropout_prob
            )
            if cfg_dropped:
                dropout_ids = span.get('cfg_dropout_input_ids')
                if dropout_ids is None:
                    return
                input_ids = [int(token) for token in dropout_ids]
                label_ids = input_ids[1:] + ([input_ids[-1]] if input_ids else [])
                loss_mask = [False] * len(input_ids)
            else:
                input_ids = [int(token) for token in span.get('input_ids', [])]
                label_ids = span.get('label_ids')
                if label_ids is None:
                    label_ids = input_ids[1:] + [self.eos_token_id]
                label_ids = [int(token) for token in label_ids]
                loss_mask = [bool(value) for value in span.get('loss_mask', [False] * len(input_ids))]
            if not input_ids:
                return

            if len(label_ids) != len(input_ids) or len(loss_mask) != len(input_ids):
                raise ValueError(
                    "text_token_span input_ids, label_ids, and loss_mask must have matching lengths "
                    f"({len(input_ids)}, {len(label_ids)}, {len(loss_mask)})"
                )

            sequence_status['packed_text_ids'].extend(input_ids)
            sequence_status['packed_text_indexes'].extend(range(curr, curr + len(input_ids)))

            ce_positions = [offset for offset, enabled in enumerate(loss_mask) if enabled]
            if ce_positions:
                loss_weights = span.get('loss_weights')
                if loss_weights is not None:
                    loss_weights = [float(value) for value in loss_weights]
                    if len(loss_weights) != len(input_ids):
                        raise ValueError(
                            "text_token_span loss_weights must match input_ids length "
                            f"({len(loss_weights)} vs {len(input_ids)})"
                        )
                else:
                    span_weight = float(span.get('loss_weight', len2weight(len(ce_positions))))
                    loss_weights = [span_weight] * len(input_ids)

                sequence_status['ce_loss_indexes'].extend(curr + offset for offset in ce_positions)
                sequence_status['ce_loss_weights'].extend(loss_weights[offset] for offset in ce_positions)
                sequence_status['packed_label_ids'].extend(label_ids[offset] for offset in ce_positions)

            span_len = len(input_ids)
            curr += span_len
            curr_split_len += span_len
            if split_start:
                attn_modes.append(str(item.get("attention_mode", "causal")))
            sequence_status['packed_position_ids'].extend(
                range(curr_rope_id, curr_rope_id + span_len)
            )
            curr_rope_id += span_len
            append_text_mrope(span_len)

        for item_index, item in enumerate(sequence_plan):
            split_start = item.get('split_start', True)
            if split_start:
                curr_split_len = 0

            if item['type'] == 'text_token_span':
                append_text_token_span(item, item_index)

            elif item['type'] == 'text':
                text_ids = text_ids_list.pop(0)
                cfg_dropped = cfg_dropout_decision(
                    item, self.data_config.text_cond_dropout_prob
                )
                if cfg_dropped:
                    continue

                shifted_text_ids = [self.bos_token_id] + text_ids
                sequence_status['packed_text_ids'].extend(shifted_text_ids)
                sequence_status['packed_text_indexes'].extend(range(curr, curr + len(shifted_text_ids)))
                if item['loss'] == 1:
                    sequence_status['ce_loss_indexes'].extend(range(curr, curr + len(shifted_text_ids)))
                    sequence_status['ce_loss_weights'].extend(
                        [len2weight(len(shifted_text_ids))] * len(shifted_text_ids)
                    )
                    sequence_status['packed_label_ids'].extend(text_ids + [self.eos_token_id])
                curr += len(shifted_text_ids)
                curr_split_len += len(shifted_text_ids)

                # add a <|im_end|> token
                sequence_status['packed_text_ids'].append(self.eos_token_id)
                sequence_status['packed_text_indexes'].append(curr)
                if item['special_token_loss'] == 1:
                    sequence_status['ce_loss_indexes'].append(curr)
                    sequence_status['ce_loss_weights'].append(1.0)
                    sequence_status['packed_label_ids'].append(item['special_token_label'])
                curr += 1
                curr_split_len += 1

                # 1D RoPE: text segment — sequential position IDs
                attn_modes.append("causal")
                sequence_status['packed_position_ids'].extend(range(curr_rope_id, curr_rope_id + curr_split_len))
                curr_rope_id += curr_split_len
                append_text_mrope(len(shifted_text_ids) + 1)

            elif item['type'] == 'pixel_video_und':
                if item.get('video_und_mrope_mode', 'sequence_hw') != 'sequence_hw':
                    raise ValueError("PixelUMM-release supports only video_und_mrope_mode='sequence_hw'")
                video_tensor = video_tensor_list.pop(0)
                video_value_range = video_value_range_list.pop(0)
                cfg_dropped = cfg_dropout_decision(
                    item, self.data_config.pixel_und_cond_dropout_prob
                )
                if cfg_dropped:
                    continue

                append_start_delimiter = bool(item.get('append_start_delimiter', True))
                if append_start_delimiter:
                    append_start_of_image_delimiter(append_mrope=False)

                num_frames, _, H, W = video_tensor.shape
                temporal_patch_size = int(item.get(
                    'temporal_patch_size', self.data_config.pixel_video_temporal_patch_size
                ))
                if num_frames != temporal_patch_size:
                    raise ValueError(
                        "pixel_video_und expects exactly one temporal tube per plan item: "
                        f"frames={num_frames} temporal_patch_size={temporal_patch_size}"
                    )
                if H % self.data_config.pixel_token_patch_size or W % self.data_config.pixel_token_patch_size:
                    raise ValueError(
                        f"pixel_video_und shape {(num_frames, H, W)} is not divisible by "
                        f"patch={self.data_config.pixel_token_patch_size}"
                    )
                h = H // self.data_config.pixel_token_patch_size
                w = W // self.data_config.pixel_token_patch_size
                num_video_tokens = h * w
                sequence_status['raw_pixel_videos_und'].append(video_tensor)
                sequence_status['raw_pixel_video_ranges_und'].append(video_value_range)
                sequence_status['raw_pixel_video_temporal_patch_sizes_und'].append(temporal_patch_size)
                sequence_status['packed_pixel_video_patch_indexes_und'].extend(
                    range(curr, curr + num_video_tokens)
                )
                curr += num_video_tokens
                curr_split_len += num_video_tokens

                has_end_delimiter = bool(item.get('include_end_delimiter', False))
                if split_start:
                    attn_modes.append(str(item.get("attention_mode", "full")))
                sequence_status['packed_position_ids'].extend([curr_rope_id] * num_video_tokens)
                curr_rope_id += 1
                append_image_mrope(
                    h,
                    w,
                    include_start=append_start_delimiter,
                    include_end=has_end_delimiter,
                )

                if has_end_delimiter:
                    split_lens.append(curr_split_len)
                    sample_lens += curr_split_len
                    curr_split_len = 0
                    append_causal_end_of_image_delimiter(
                        append_mrope=False,
                        item=item,
                    )

            elif item['type'] == 'pixel_und':
                # Understanding: clean pixel patches (no timestep, no noise)
                image_tensor = image_tensor_list.pop(0)
                image_value_range = image_value_range_list.pop(0)
                media_kind = str(item.get('media_kind', 'image'))
                if media_kind not in {'image', 'video'}:
                    raise ValueError(f"pixel_und media_kind must be image/video, got {media_kind!r}")
                cfg_dropped = cfg_dropout_decision(
                    item, self.data_config.pixel_und_cond_dropout_prob
                )
                if cfg_dropped:
                    continue

                append_start_delimiter = bool(item.get('append_start_delimiter', True))
                if append_start_delimiter:
                    append_start_of_image_delimiter(append_mrope=False)

                H, W = image_tensor.shape[1:]
                h = H // self.data_config.pixel_token_patch_size
                w = W // self.data_config.pixel_token_patch_size
                num_img_tokens = h * w
                sequence_status['raw_pixel_images_und'].append(image_tensor)
                sequence_status['raw_pixel_image_ranges_und'].append(image_value_range)
                sequence_status['packed_pixel_patch_indexes_und'].extend(range(curr, curr + num_img_tokens))
                curr += num_img_tokens
                curr_split_len += num_img_tokens

                has_end_delimiter = (
                    bool(item['include_end_delimiter'])
                    if 'include_end_delimiter' in item
                    else next_item_is_target_text(item_index)
                )
                if split_start:
                    attn_modes.append(str(item.get("attention_mode", "full")))
                sequence_status['packed_position_ids'].extend([curr_rope_id] * num_img_tokens)
                curr_rope_id += 1
                if has_end_delimiter:
                    split_lens.append(curr_split_len)
                    sample_lens += curr_split_len
                    curr_split_len = 0
                    append_causal_end_of_image_delimiter(
                        append_mrope=False,
                        item=item,
                    )
                append_image_mrope(
                    h,
                    w,
                    include_start=append_start_delimiter,
                    include_end=has_end_delimiter,
                )

            elif item['type'] == 'pixel_gen':
                # Generation: pixel patches with timestep (noise added in model forward)
                image_tensor = image_tensor_list.pop(0)
                image_value_range = image_value_range_list.pop(0)
                cfg_dropped = cfg_dropout_decision(
                    item, self.data_config.pixel_gen_cond_dropout_prob
                )
                if cfg_dropped:
                    continue

                append_start_delimiter = item.get('append_start_delimiter', True)
                if append_start_delimiter:
                    append_start_of_image_delimiter(append_mrope=False)

                H, W = image_tensor.shape[1:]
                h = H // self.data_config.pixel_token_patch_size
                w = W // self.data_config.pixel_token_patch_size
                num_img_tokens = h * w
                sequence_status['raw_pixel_images_gen'].append(image_tensor)
                sequence_status['raw_pixel_image_ranges_gen'].append(image_value_range)
                sequence_status['packed_pixel_patch_indexes_gen'].extend(range(curr, curr + num_img_tokens))
                sequence_status['pixel_grid_hw_gen'].append((h, w))
                if item['loss'] == 1:
                    # Only track grid_hw for loss=1 images used by packed MSE targets.
                    sequence_status['gen_image_grid_hw'].append((h, w))
                    if split_start:
                        timestep = np.random.randn()
                        current_gen_timestep = timestep
                    else:
                        if current_gen_timestep is None:
                            raise ValueError("pixel_gen with split_start=False has no active generation timestep")
                        timestep = current_gen_timestep
                else:
                    timestep = float('-inf')

                sequence_status['packed_timesteps'].extend([timestep] * num_img_tokens)
                curr += num_img_tokens
                curr_split_len += num_img_tokens

                if split_start:
                    if item['loss'] == 1 and 'frame_delta' not in item.keys():
                        attn_modes.append("noise")
                    else:
                        attn_modes.append("clean")
                sequence_status['packed_position_ids'].extend([curr_rope_id] * num_img_tokens)
                if 'frame_delta' in item.keys():
                    curr_rope_id += item['frame_delta']
                elif item['loss'] == 0:
                    curr_rope_id += 1
                append_image_mrope(
                    h,
                    w,
                    include_start=append_start_delimiter,
                    include_end=False,
                )

            elif item['type'] == 'pixel_video_gen':
                video_tensor = video_tensor_list.pop(0)
                video_value_range = video_value_range_list.pop(0)
                cfg_dropped = cfg_dropout_decision(
                    item, self.data_config.pixel_gen_cond_dropout_prob
                )
                if cfg_dropped:
                    continue

                append_start_delimiter = item.get('append_start_delimiter', True)
                if append_start_delimiter:
                    append_start_of_image_delimiter(append_mrope=False)

                num_frames, _, H, W = video_tensor.shape
                temporal_patch_size = item.get(
                    'temporal_patch_size',
                    self.data_config.pixel_video_temporal_patch_size,
                )
                if num_frames % temporal_patch_size:
                    raise ValueError(
                        f"video frames={num_frames} must be divisible by temporal_patch_size={temporal_patch_size}"
                    )
                if H % self.data_config.pixel_token_patch_size or W % self.data_config.pixel_token_patch_size:
                    raise ValueError(
                        f"video shape {(num_frames, H, W)} is not divisible by "
                        f"patch={self.data_config.pixel_token_patch_size}"
                    )
                temporal_groups = num_frames // temporal_patch_size
                num_video_tokens = (
                    temporal_groups
                    * (H // self.data_config.pixel_token_patch_size)
                    * (W // self.data_config.pixel_token_patch_size)
                )
                sequence_status['raw_pixel_videos_gen'].append(video_tensor)
                sequence_status['raw_pixel_video_ranges_gen'].append(video_value_range)
                sequence_status['raw_pixel_video_temporal_patch_sizes_gen'].append(int(temporal_patch_size))
                sequence_status['packed_pixel_video_patch_indexes_gen'].extend(range(curr, curr + num_video_tokens))
                sequence_status['pixel_video_patch_seqlens_gen'].append(num_video_tokens)
                sequence_status['pixel_video_grid_hw_gen'].append(
                    (H // self.data_config.pixel_token_patch_size, W // self.data_config.pixel_token_patch_size)
                )
                if item['loss'] == 1:
                    sequence_status['gen_video_lens'].append(num_video_tokens)
                    if split_start:
                        timestep = np.random.randn()
                        current_gen_timestep = timestep
                    else:
                        if current_gen_timestep is None:
                            raise ValueError("pixel_video_gen with split_start=False has no active generation timestep")
                        timestep = current_gen_timestep
                else:
                    timestep = float('-inf')

                sequence_status['packed_timesteps'].extend([timestep] * num_video_tokens)
                curr += num_video_tokens
                curr_split_len += num_video_tokens

                if split_start:
                    if item['loss'] == 1:
                        attn_modes.append("noise")
                    else:
                        attn_modes.append("clean")
                sequence_status['packed_position_ids'].extend([curr_rope_id] * num_video_tokens)
                curr_rope_id += 1
                append_video_mrope(
                    temporal_groups,
                    H // self.data_config.pixel_token_patch_size,
                    W // self.data_config.pixel_token_patch_size,
                    include_start=append_start_delimiter,
                    include_end=False,
                )

            if item.get('split_end', True) and curr_split_len > 0:
                split_lens.append(curr_split_len)
                sample_lens += curr_split_len
            if item.get('split_end', True):
                current_gen_timestep = None

        sequence_status['curr'] = curr
        sequence_status['sample_lens'].append(sample_lens)
        sequence_status['split_lens'].extend(split_lens)
        sequence_status['attn_modes'].extend(attn_modes)

        return sequence_status


class SimpleCustomBatch:
    """Pinned-memory container for one PixelUMM packed sequence."""

    # All tensor attributes that need pin_memory / .to(device)
    _TENSOR_ATTRS = [
        'packed_text_ids', 'packed_text_indexes', 'packed_position_ids', 'packed_mrope_position_ids',
        'packed_pixel_patch_indexes_und',
        'packed_pixel_video_patch_indexes_und',
        'packed_pixel_patch_indexes_gen', 'packed_pixel_grid_hw_gen',
        'packed_gen_image_lens',
        'packed_pixel_video_patch_indexes_gen', 'pixel_video_patch_seqlens_gen',
        'pixel_video_grid_hw_gen',
        'packed_gen_video_lens',
        'packed_timesteps',
        'packed_label_ids', 'ce_loss_indexes', 'ce_loss_weights',
    ]
    _TENSOR_LIST_ATTRS = [
        'raw_pixel_images_und', 'raw_pixel_videos_und',
        'raw_pixel_images_gen',
        'raw_pixel_videos_gen',
    ]
    _NON_TENSOR_ATTRS = [
        'raw_pixel_image_ranges_und',
        'raw_pixel_video_ranges_und',
        'raw_pixel_video_temporal_patch_sizes_und',
        'raw_pixel_image_ranges_gen',
        'raw_pixel_video_ranges_gen',
        'raw_pixel_video_temporal_patch_sizes_gen',
    ]

    def __init__(self, batch):
        data = batch[0]
        self.sequence_length = data["sequence_length"]
        self.sample_lens = data["sample_lens"]
        self.packed_text_ids = data["packed_text_ids"]
        self.packed_text_indexes = data["packed_text_indexes"]
        self.packed_position_ids = data["packed_position_ids"]
        self.split_lens = data["split_lens"]
        self.attn_modes = data["attn_modes"]

        # Optional fields — set only if present
        for key in [
            'packed_mrope_position_ids',
            'raw_pixel_images_und',
            'raw_pixel_image_ranges_und',
            'packed_pixel_patch_indexes_und',
            'raw_pixel_videos_und',
            'raw_pixel_video_ranges_und', 'raw_pixel_video_temporal_patch_sizes_und',
            'packed_pixel_video_patch_indexes_und',
            'raw_pixel_images_gen',
            'raw_pixel_image_ranges_gen',
            'packed_pixel_patch_indexes_gen', 'packed_pixel_grid_hw_gen',
            'packed_gen_image_lens',
            'raw_pixel_videos_gen',
            'raw_pixel_video_ranges_gen',
            'raw_pixel_video_temporal_patch_sizes_gen',
            'packed_pixel_video_patch_indexes_gen', 'pixel_video_patch_seqlens_gen',
            'pixel_video_grid_hw_gen',
            'packed_gen_video_lens',
            'packed_timesteps',
            'packed_label_ids', 'ce_loss_indexes', 'ce_loss_weights',
        ]:
            if key in data:
                setattr(self, key, data[key])

    def pin_memory(self):
        for attr in self._TENSOR_ATTRS:
            if hasattr(self, attr):
                setattr(self, attr, getattr(self, attr).pin_memory())
        for attr in self._TENSOR_LIST_ATTRS:
            if hasattr(self, attr):
                setattr(self, attr, [x.pin_memory() for x in getattr(self, attr)])
        return self

    def cuda(self, device):
        for attr in self._TENSOR_ATTRS:
            if hasattr(self, attr):
                setattr(self, attr, getattr(self, attr).to(device, non_blocking=True))
        for attr in self._TENSOR_LIST_ATTRS:
            if hasattr(self, attr):
                setattr(self, attr, [x.to(device, non_blocking=True) for x in getattr(self, attr)])
        return self

    def to_dict(self):
        data = dict(
            sequence_length=self.sequence_length,
            sample_lens=self.sample_lens,
            packed_text_ids=self.packed_text_ids,
            packed_text_indexes=self.packed_text_indexes,
            packed_position_ids=self.packed_position_ids,
            split_lens=self.split_lens,
            attn_modes=self.attn_modes,
        )

        for attr in self._TENSOR_ATTRS[3:]:  # skip text/position (already in data)
            if hasattr(self, attr):
                data[attr] = getattr(self, attr)
        for attr in self._TENSOR_LIST_ATTRS + self._NON_TENSOR_ATTRS:
            if hasattr(self, attr):
                data[attr] = getattr(self, attr)

        return data


def collate_wrapper():
    def collate_fn(batch):
        return SimpleCustomBatch(batch)
    return collate_fn
