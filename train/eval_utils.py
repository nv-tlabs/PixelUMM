# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PixelUMM image, video, and multimodal inference helpers."""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch
from PIL import Image

from data.data_utils import untubeify


def save_video_mp4(frames, output_dir, fps=16, filename="t2v.mp4"):
    if not frames:
        return None
    path = str(Path(output_dir) / filename)
    width, height = frames[0].size
    fps = max(float(fps), 1.0)
    writer = imageio_ffmpeg.write_frames(
        path,
        (width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        output_params=["-movflags", "+faststart"],
        ffmpeg_log_level="error",
    )
    writer.send(None)
    try:
        for frame in frames:
            writer.send(np.asarray(frame.convert("RGB")))
    finally:
        writer.close()
    return path


def _video_tubes_to_pil_frames(video_tubes, image_size, num_frames, patch_size):
    if not torch.isfinite(video_tubes).all().item():
        raise RuntimeError("Video generation produced non-finite pixels")
    frames = untubeify(video_tubes, image_size, num_frames, patch_size)
    frames = ((frames.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8).numpy()
    return [Image.fromarray(frames[i]) for i in range(num_frames)]


def _prefill_text_to_visual_chatml_context(
    model, tokenizer, new_token_ids, prompt, device, kv, kv_lens, ropes,
):
    """Prefill the exact T2I/T2V training ChatML context into a KV cache."""
    text_input, kv_lens, ropes = model.prepare_text_to_visual_chatml_prompts(
        curr_kvlens=kv_lens,
        curr_rope=ropes,
        prompts=[prompt],
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )
    text_input = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in text_input.items()
    }
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        kv = model.forward_cache_update_text(kv, **text_input)
    return kv, kv_lens, ropes


def _prefill_generation_visual_start(
    model, new_token_ids, device, kv, kv_lens, ropes,
):
    """Cache the causal ``<vision_start>`` used by PixelUMM generation."""
    start_input, kv_lens, ropes = model.prepare_visual_delimiter_tokens(
        kv_lens, ropes, new_token_ids, delimiter="start",
    )
    start_input = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in start_input.items()
    }
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        kv = model.forward_cache_update_text(kv, **start_input)
    return kv, kv_lens, ropes, True


def _generate_t2i_one(
    model, tokenizer, new_token_ids, prompt, image_size, device,
    num_timesteps, timestep_shift, sampler,
    cfg_text_scale, cfg_img_scale, cfg_interval, cfg_renorm_type,
):
    """Run one T2I sample through the FSDP-wrapped model. Mirrors
    test_inference_pixel.py:generate_t2i. Returns a PIL image."""
    from modeling.pixelumm.qwen3_navit import NaiveCache

    n_layers = model.config.llm_config.num_hidden_layers
    kv = NaiveCache(n_layers)
    kv_lens, ropes = [0], [0]

    cfg_text_kv = deepcopy(kv) if cfg_text_scale > 1.0 else None
    cfg_text_kv_lens = list(kv_lens) if cfg_text_scale > 1.0 else None
    cfg_text_ropes = list(ropes) if cfg_text_scale > 1.0 else None
    cfg_img_kv = deepcopy(kv) if cfg_img_scale > 1.0 else None
    cfg_img_kv_lens = list(kv_lens) if cfg_img_scale > 1.0 else None
    cfg_img_ropes = list(ropes) if cfg_img_scale > 1.0 else None

    kv, kv_lens, ropes = _prefill_text_to_visual_chatml_context(
        model, tokenizer, new_token_ids, prompt, device, kv, kv_lens, ropes,
    )

    kv, kv_lens, ropes, visual_start_prefilled = _prefill_generation_visual_start(
        model, new_token_ids, device, kv, kv_lens, ropes,
    )

    generation_input = model.prepare_pixel_noise(
        curr_kvlens=kv_lens, curr_rope=ropes,
        image_sizes=[image_size],
        visual_start_prefilled=visual_start_prefilled,
    )
    generation_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in generation_input.items()}
    # Preserve the full-precision RNG sample. Model forward remains under BF16
    # autocast, while every sampler starts from an FP32 numerical state.
    generation_input['packed_init_noises'] = generation_input['packed_init_noises'].float()

    cfg_kwargs = {}
    if cfg_text_scale > 1.0 and cfg_text_kv is not None:
        # CFG text-uncond reference must match training's ChatML shell dropout, NOT an
        # empty prefill. Training drops the caption to cfg_dropout_input_ids =
        # <|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n[EOS] (an empty caption
        # through the same ChatML builder). Feed prompts=[""] so the uncond context the
        # model denoises from equals what it saw as the dropout target.
        cfg_text_kv, cfg_text_kv_lens, cfg_text_ropes = _prefill_text_to_visual_chatml_context(
            model, tokenizer, new_token_ids, "", device,
            cfg_text_kv, cfg_text_kv_lens, cfg_text_ropes,
        )
        (
            cfg_text_kv,
            cfg_text_kv_lens,
            cfg_text_ropes,
            cfg_text_start_prefilled,
        ) = _prefill_generation_visual_start(
            model, new_token_ids, device,
            cfg_text_kv, cfg_text_kv_lens, cfg_text_ropes,
        )
        cfg_text_input = model.prepare_pixel_noise(
            curr_kvlens=cfg_text_kv_lens, curr_rope=cfg_text_ropes,
            image_sizes=[image_size],
            visual_start_prefilled=cfg_text_start_prefilled,
        )
        cfg_text_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in cfg_text_input.items()}
        cfg_kwargs.update(
            cfg_text_past_key_values=cfg_text_kv,
            cfg_text_packed_position_ids=cfg_text_input['packed_position_ids'],
            cfg_text_packed_query_indexes=cfg_text_input['packed_indexes'],
            cfg_text_key_values_lens=cfg_text_input['key_values_lens'],
            cfg_text_packed_key_value_indexes=cfg_text_input['packed_key_value_indexes'],
        )
    if cfg_img_scale > 1.0 and cfg_img_kv is not None:
        (
            cfg_img_kv,
            cfg_img_kv_lens,
            cfg_img_ropes,
            cfg_img_start_prefilled,
        ) = _prefill_generation_visual_start(
            model, new_token_ids, device,
            cfg_img_kv, cfg_img_kv_lens, cfg_img_ropes,
        )
        cfg_img_input = model.prepare_pixel_noise(
            curr_kvlens=cfg_img_kv_lens, curr_rope=cfg_img_ropes,
            image_sizes=[image_size],
            visual_start_prefilled=cfg_img_start_prefilled,
        )
        cfg_img_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in cfg_img_input.items()}
        cfg_kwargs.update(
            cfg_img_past_key_values=cfg_img_kv,
            cfg_img_packed_position_ids=cfg_img_input['packed_position_ids'],
            cfg_img_packed_query_indexes=cfg_img_input['packed_indexes'],
            cfg_img_key_values_lens=cfg_img_input['key_values_lens'],
            cfg_img_packed_key_value_indexes=cfg_img_input['packed_key_value_indexes'],
        )

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        unpacked_pixels = model.generate_image(
            past_key_values=kv,
            num_timesteps=num_timesteps,
            timestep_shift=timestep_shift,
            sampler=sampler,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=0.0,
            cfg_renorm_type=cfg_renorm_type,
            **cfg_kwargs,
            **generation_input,
        )

    H, W = image_size
    ps = model.config.patch_size
    h_patches, w_patches = H // ps, W // ps
    pixels = unpacked_pixels[0]
    img = pixels.float().cpu()
    img = img.reshape(h_patches, w_patches, ps, ps, 3)
    img = img.permute(0, 2, 1, 3, 4).reshape(H, W, 3)
    img = ((img.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8).numpy()
    return Image.fromarray(img)


def _generate_t2v_one(
    model, tokenizer, new_token_ids, prompt, video_size, device,
    num_timesteps, timestep_shift, sampler,
    cfg_text_scale, cfg_img_scale, cfg_interval, cfg_renorm_type,
    video_fps=16,
    negative_prompt="",
):
    from modeling.pixelumm.qwen3_navit import NaiveCache

    n_layers = model.config.llm_config.num_hidden_layers
    kv = NaiveCache(n_layers)
    kv_lens, ropes = [0], [0]

    cfg_text_kv = deepcopy(kv) if cfg_text_scale > 1.0 else None
    cfg_text_kv_lens = list(kv_lens) if cfg_text_scale > 1.0 else None
    cfg_text_ropes = list(ropes) if cfg_text_scale > 1.0 else None
    cfg_img_kv = deepcopy(kv) if cfg_img_scale > 1.0 else None
    cfg_img_kv_lens = list(kv_lens) if cfg_img_scale > 1.0 else None
    cfg_img_ropes = list(ropes) if cfg_img_scale > 1.0 else None

    kv, kv_lens, ropes = _prefill_text_to_visual_chatml_context(
        model, tokenizer, new_token_ids, prompt, device, kv, kv_lens, ropes,
    )

    kv, kv_lens, ropes, visual_start_prefilled = _prefill_generation_visual_start(
        model, new_token_ids, device, kv, kv_lens, ropes,
    )

    generation_input = model.prepare_pixel_video_noise(
        curr_kvlens=kv_lens, curr_rope=ropes,
        video_sizes=[video_size],
        fps=video_fps,
        visual_start_prefilled=visual_start_prefilled,
    )
    generation_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in generation_input.items()}
    # Preserve the full-precision RNG sample. Model forward remains under BF16
    # autocast, while every sampler starts from an FP32 numerical state.
    generation_input['packed_init_noises'] = generation_input['packed_init_noises'].float()

    cfg_kwargs = {}
    if cfg_text_scale > 1.0 and cfg_text_kv is not None:
        # Empty preserves the training-dropout shell. An explicit negative
        # prompt reproduces the published video's CFG reference context.
        cfg_text_kv, cfg_text_kv_lens, cfg_text_ropes = _prefill_text_to_visual_chatml_context(
            model, tokenizer, new_token_ids, negative_prompt, device,
            cfg_text_kv, cfg_text_kv_lens, cfg_text_ropes,
        )
        (
            cfg_text_kv,
            cfg_text_kv_lens,
            cfg_text_ropes,
            cfg_text_start_prefilled,
        ) = _prefill_generation_visual_start(
            model, new_token_ids, device,
            cfg_text_kv, cfg_text_kv_lens, cfg_text_ropes,
        )
        cfg_text_input = model.prepare_pixel_video_noise(
            curr_kvlens=cfg_text_kv_lens, curr_rope=cfg_text_ropes,
            video_sizes=[video_size],
            fps=video_fps,
            visual_start_prefilled=cfg_text_start_prefilled,
        )
        cfg_text_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in cfg_text_input.items()}
        cfg_kwargs.update(
            cfg_text_past_key_values=cfg_text_kv,
            cfg_text_packed_position_ids=cfg_text_input['packed_position_ids'],
            cfg_text_packed_query_indexes=cfg_text_input['packed_indexes'],
            cfg_text_key_values_lens=cfg_text_input['key_values_lens'],
            cfg_text_packed_key_value_indexes=cfg_text_input['packed_key_value_indexes'],
        )
    if cfg_img_scale > 1.0 and cfg_img_kv is not None:
        (
            cfg_img_kv,
            cfg_img_kv_lens,
            cfg_img_ropes,
            cfg_img_start_prefilled,
        ) = _prefill_generation_visual_start(
            model, new_token_ids, device,
            cfg_img_kv, cfg_img_kv_lens, cfg_img_ropes,
        )
        cfg_img_input = model.prepare_pixel_video_noise(
            curr_kvlens=cfg_img_kv_lens, curr_rope=cfg_img_ropes,
            video_sizes=[video_size],
            fps=video_fps,
            visual_start_prefilled=cfg_img_start_prefilled,
        )
        cfg_img_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in cfg_img_input.items()}
        cfg_kwargs.update(
            cfg_img_past_key_values=cfg_img_kv,
            cfg_img_packed_position_ids=cfg_img_input['packed_position_ids'],
            cfg_img_packed_query_indexes=cfg_img_input['packed_indexes'],
            cfg_img_key_values_lens=cfg_img_input['key_values_lens'],
            cfg_img_packed_key_value_indexes=cfg_img_input['packed_key_value_indexes'],
        )

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        unpacked_video = model.generate_video(
            past_key_values=kv,
            num_timesteps=num_timesteps,
            timestep_shift=timestep_shift,
            sampler=sampler,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=0.0,
            cfg_renorm_type=cfg_renorm_type,
            **cfg_kwargs,
            **generation_input,
        )

    num_frames, H, W = video_size
    return _video_tubes_to_pil_frames(
        unpacked_video[0],
        (H, W),
        num_frames,
        model.config.patch_size,
    )


def generate_vlm_interleaved(
    model,
    tokenizer,
    new_token_ids,
    images,
    instruction,
    device,
    max_length,
    temperature=0.0,
    fixed_length=False,
    image_placeholder_policy="strict",
    reasoning_mode="standard_bare",
    exact_max_new_tokens=False,
    return_generation_metadata=False,
):
    """Generate an I2T answer with train-aligned interleaved media prefill.

    Training represents every image as four consecutive attention segments:
    causal user text, causal ``<vision_start>``, a full-attention patch block,
    then causal ``<vision_end>``.  This helper applies the same sequence to one
    or more images and preserves ``<image>`` / ``<image N>`` locations in the
    benchmark question.  When a question has no placeholders, all images are
    placed before the question, matching ordinary F14 I2T samples.

    ``fixed_length=True`` is reserved for in-training FSDP eval, where every
    rank must execute the same number of decode iterations.  Standalone
    benchmark replicas can stop at EOS and should leave it false.

    ``reasoning_mode="standard_bare"`` preserves the train-aligned bare
    assistant prefix. Source responses may begin directly, with an empty
    shell, or with a filled trace. ``exact_max_new_tokens=True`` makes the standalone benchmark's
    output budget match Hugging Face/lmms-eval semantics without changing the
    fixed-iteration contract used by in-training FSDP eval. The defaults
    preserve all existing callers.
    """
    from modeling.pixelumm.qwen3_navit import NaiveCache

    images = list(images)
    if not images:
        raise ValueError("VLM generation requires at least one image")

    if image_placeholder_policy not in {"strict", "numbered_references"}:
        raise ValueError(
            f"Unsupported image placeholder policy: {image_placeholder_policy}"
        )
    if reasoning_mode != "standard_bare":
        raise ValueError(f"Unsupported VLM reasoning mode: {reasoning_mode}")

    placeholder = re.compile(r"<image(?:\s+(\d+))?>", flags=re.IGNORECASE)
    matches = list(placeholder.finditer(instruction))

    parts = []
    if matches:
        numbered = [match.group(1) is not None for match in matches]
        if any(numbered) and not all(numbered):
            raise ValueError(
                "VLM prompts cannot mix <image> and numbered <image N> placeholders"
            )

        if image_placeholder_policy == "numbered_references" and not all(numbered):
            raise ValueError(
                "numbered_references requires numbered <image N> placeholders"
            )

        if image_placeholder_policy == "numbered_references":
            labels = [int(match.group(1)) for match in matches]
            unique_labels = sorted(set(labels))
            if len(images) == 1 and len(unique_labels) > 1:
                # Some benchmarks provide one pre-composed panel while their
                # question uses <image N> as textual sub-panel references.
                normalized = placeholder.sub(
                    lambda match: f"Panel {int(match.group(1))}",
                    instruction,
                )
                parts.extend((
                    ("image", 0),
                    ("text", normalized),
                ))
            else:
                expected_labels = list(range(1, len(images) + 1))
                if unique_labels != expected_labels:
                    raise ValueError(
                        "Numbered image references must cover each benchmark visual: "
                        f"labels={unique_labels} expected={expected_labels}"
                    )
                cursor = 0
                seen_labels = set()
                for label, match in zip(labels, matches):
                    parts.append(("text", instruction[cursor : match.start()]))
                    if label not in seen_labels:
                        parts.append(("image", label - 1))
                        seen_labels.add(label)
                    else:
                        parts.append(("text", f"Picture {label}"))
                    cursor = match.end()
                parts.append(("text", instruction[cursor:]))
        else:
            if len(matches) != len(images):
                raise ValueError(
                    "Image placeholder count must match the number of benchmark visuals: "
                    f"placeholders={len(matches)} images={len(images)}"
                )
            image_indices = (
                [int(match.group(1)) - 1 for match in matches]
                if all(numbered)
                else list(range(len(matches)))
            )
            if sorted(image_indices) != list(range(len(images))):
                raise ValueError(
                    "Numbered image placeholders must reference every visual exactly once: "
                    f"indices={image_indices} images={len(images)}"
                )
            cursor = 0
            for image_index, match in zip(image_indices, matches):
                parts.append(("text", instruction[cursor : match.start()]))
                parts.append(("image", image_index))
                cursor = match.end()
            parts.append(("text", instruction[cursor:]))
    else:
        parts.extend(("image", image_index) for image_index in range(len(images)))
        parts.append(("text", instruction))

    n_layers = model.config.llm_config.num_hidden_layers
    kv = NaiveCache(n_layers)
    kv_lens, ropes = [0], [0]

    def to_device(inputs):
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

    def update_text_ids(token_ids):
        nonlocal kv, kv_lens, ropes
        if not token_ids:
            return
        text_input, kv_lens, ropes = model._prepare_token_ids_as_text_cache_input(
            kv_lens,
            ropes,
            [list(token_ids)],
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            kv = model.forward_cache_update_text(kv, **to_device(text_input))

    prefix_input, kv_lens, ropes = model.prepare_image_to_text_chatml_prefixes(
        curr_kvlens=kv_lens,
        curr_rope=ropes,
        tokenizer=tokenizer,
    )
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        kv = model.forward_cache_update_text(kv, **to_device(prefix_input))

    if not parts or parts[-1][0] != "text":
        raise AssertionError("Interleaved VLM prompt must end in a text segment")
    trailing_instruction = parts[-1][1]

    for part_type, value in parts[:-1]:
        if part_type == "text":
            update_text_ids(model._encode_prompt_text(tokenizer, value))
            continue

        image_index = int(value)
        if len(images) > 1:
            update_text_ids(
                model._encode_prompt_text(tokenizer, f"Picture {image_index + 1}: ")
            )

        start_input, kv_lens, ropes = model.prepare_visual_delimiter_tokens(
            kv_lens,
            ropes,
            new_token_ids,
            delimiter="start",
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            kv = model.forward_cache_update_text(kv, **to_device(start_input))

        image_input, kv_lens, ropes = model.prepare_pixel_images(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            images=[images[image_index]],
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            kv = model.forward_cache_update_pixel(kv, **to_device(image_input))

        end_input, kv_lens, ropes = model.prepare_visual_delimiter_tokens(
            kv_lens,
            ropes,
            new_token_ids,
            delimiter="end",
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            kv = model.forward_cache_update_text(kv, **to_device(end_input))

    # Tokenize the trailing question text and ChatML turn boundary together,
    # in one encode call to preserve the boundary before <|im_end|>.
    suffix_ids = model._image_to_text_chatml_suffix_ids(
        tokenizer,
        trailing_instruction,
    )
    assistant_prefill = ""
    if not suffix_ids:
        raise ValueError("Image-to-text ChatML suffix cannot be empty")
    update_text_ids(suffix_ids[:-1])
    start_token_ids = [suffix_ids[-1]]
    start_input = to_device(
        model.prepare_text_start_tokens(kv_lens, ropes, start_token_ids)
    )

    # ``generate_text`` counts ``start_token_ids`` in its returned length.
    # Add one iteration when exact_max_new_tokens is requested; the start
    # token is removed below.
    decode_max_length = int(max_length) + (1 if exact_max_new_tokens else 0)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        token_ids, generation_metadata = model.generate_text(
            past_key_values=kv,
            max_length=decode_max_length,
            do_sample=(temperature > 0.0),
            temperature=max(temperature, 1e-5),
            end_token_id=None if fixed_length else new_token_ids["eos_token_id"],
            return_generation_metadata=True,
            **start_input,
        )

    token_ids = token_ids.squeeze(-1).cpu().tolist()
    eos_id = new_token_ids["eos_token_id"]
    bos_id = new_token_ids["bos_token_id"]
    text_ids = []
    hit_eos = bool(generation_metadata["hit_end_token"])
    for token_id in token_ids:
        if token_id == eos_id:
            break
        text_ids.append(token_id)
    if text_ids and text_ids[0] == start_token_ids[0]:
        text_ids = text_ids[1:]
    if text_ids and text_ids[0] == bos_id:
        text_ids = text_ids[1:]
    text = tokenizer.decode(text_ids, skip_special_tokens=True)
    if not return_generation_metadata:
        return text
    return text, {
        "assistant_prefill": assistant_prefill,
        "generated_tokens": len(text_ids),
        "hit_eos": hit_eos,
        "max_new_tokens": int(max_length),
        "reasoning_mode": reasoning_mode,
    }


def _v2t_frames_to_und_tubes(model, frames, timestamps, *, timestamp_mode="tube_start"):
    """Group processed PIL frames into the released video-UND tube contract."""
    import numpy as np

    temporal_patch_size = int(model.config.pixel_video_temporal_patch_size)
    mrope_type = str(model.config.llm_config.qwen3_mrope_type)
    if mrope_type != "sensenova":
        raise ValueError(f"PixelUMM V2T requires qwen3_mrope_type='sensenova', got {mrope_type!r}")
    if timestamp_mode != "tube_start":
        raise ValueError(f"PixelUMM V2T requires timestamp_mode='tube_start', got {timestamp_mode!r}")
    if temporal_patch_size <= 0:
        raise ValueError(
            f"pixel_video_temporal_patch_size must be positive, got {temporal_patch_size}"
        )
    if not frames:
        raise ValueError("V2T tube eval requires at least one processed frame")

    timeline_origin = float(timestamps[0])
    local_timestamps = [float(value) - timeline_origin for value in timestamps]
    pad = (-len(frames)) % temporal_patch_size
    if pad:
        frames = list(frames) + [frames[-1]] * pad
        local_timestamps = local_timestamps + [local_timestamps[-1]] * pad

    def frame_to_tensor(frame):
        array = np.asarray(frame.convert("RGB"), dtype=np.float32)
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return tensor / 255.0 * 2.0 - 1.0

    tubes = []
    for start in range(0, len(frames), temporal_patch_size):
        tube_frames = frames[start : start + temporal_patch_size]
        tube_start_seconds = float(local_timestamps[start])
        tube_tensor = torch.stack([frame_to_tensor(frame) for frame in tube_frames], dim=0)
        tubes.append(
            {
                "tensor": tube_tensor,
                "timestamp_seconds": tube_start_seconds,
            }
        )
    return tubes


def _prefill_v2t_und_tubes(
    model,
    tokenizer,
    new_token_ids,
    kv,
    kv_lens,
    ropes,
    frames,
    timestamps,
    device,
    *,
    timestamp_mode="tube_start",
):
    """Prefill the model-aligned tube V2T media context, split-by-split.

    Training packs, per tube: a causal ``<t.t seconds>`` text span, a causal
    ``<vision_start>``, one full-attention tube, then a causal ``<vision_end>``.
    """

    def to_device(prepared):
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in prepared.items()
        }

    tubes = _v2t_frames_to_und_tubes(
        model, frames, timestamps, timestamp_mode=timestamp_mode
    )
    for tube in tubes:
        text_input, kv_lens, ropes = model.prepare_text_spans_for_cache(
            kv_lens,
            ropes,
            [f"<{tube['timestamp_seconds']:.1f} seconds>"],
            tokenizer,
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            kv = model.forward_cache_update_text(kv, **to_device(text_input))

        start_input, kv_lens, ropes = model.prepare_visual_delimiter_tokens(
            kv_lens, ropes, new_token_ids, delimiter="start"
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            kv = model.forward_cache_update_text(kv, **to_device(start_input))

        tube_input, kv_lens, ropes = model.prepare_pixel_video_und_tubes_for_text(
            kv_lens,
            ropes,
            [tube["tensor"]],
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            kv = model.forward_cache_update_pixel(kv, **to_device(tube_input))

        end_input, kv_lens, ropes = model.prepare_visual_delimiter_tokens(
            kv_lens, ropes, new_token_ids, delimiter="end"
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            kv = model.forward_cache_update_text(kv, **to_device(end_input))

    return kv, kv_lens, ropes


def _prefill_v2t_und_frames(
    model,
    tokenizer,
    new_token_ids,
    kv,
    kv_lens,
    ropes,
    frames,
    timestamps,
    device,
):
    """Prefill sparse long-video frames through the image UND embedder.

    This is the released Video4 sparse-frame contract: each sampled frame gets
    its own lexical clip-relative
    timestamp and visual island. No temporal tube projection or physical-time
    MRoPE is used.
    """

    def to_device(prepared):
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in prepared.items()
        }

    timeline_origin = float(timestamps[0])
    local_timestamps = [float(value) - timeline_origin for value in timestamps]
    for frame, timestamp_seconds in zip(frames, local_timestamps, strict=True):
        text_input, kv_lens, ropes = model.prepare_text_spans_for_cache(
            kv_lens,
            ropes,
            [f"<{timestamp_seconds:.1f} seconds>"],
            tokenizer,
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            kv = model.forward_cache_update_text(kv, **to_device(text_input))

        start_input, kv_lens, ropes = model.prepare_visual_delimiter_tokens(
            kv_lens, ropes, new_token_ids, delimiter="start"
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            kv = model.forward_cache_update_text(kv, **to_device(start_input))

        image_input, kv_lens, ropes = model.prepare_pixel_images(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            images=[frame],
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            kv = model.forward_cache_update_pixel(kv, **to_device(image_input))

        end_input, kv_lens, ropes = model.prepare_visual_delimiter_tokens(
            kv_lens, ropes, new_token_ids, delimiter="end"
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            kv = model.forward_cache_update_text(kv, **to_device(end_input))

    return kv, kv_lens, ropes


def _generate_v2t_one(
    model,
    tokenizer,
    new_token_ids,
    frames,
    timestamps,
    instruction,
    device,
    max_length,
    temperature,
    *,
    temporal_patch_size=None,
    timestamp_mode="tube_start",
):
    """Decode one V2T video context (frame-as-image or video-UND tubes)."""
    from modeling.pixelumm.qwen3_navit import NaiveCache

    if len(frames) != len(timestamps):
        raise ValueError(
            f"V2T frame/timestamp length mismatch: {len(frames)} != {len(timestamps)}"
        )
    n_layers = model.config.llm_config.num_hidden_layers
    kv = NaiveCache(n_layers)
    kv_lens, ropes = [0], [0]

    prefix_input, kv_lens, ropes = model.prepare_image_to_text_chatml_prefixes(
        curr_kvlens=kv_lens,
        curr_rope=ropes,
        tokenizer=tokenizer,
    )
    prefix_input = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in prefix_input.items()
    }
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        kv = model.forward_cache_update_text(kv, **prefix_input)

    actual_temporal_patch_size = int(
        temporal_patch_size
        if temporal_patch_size is not None
        else model.config.pixel_video_temporal_patch_size
    )
    if actual_temporal_patch_size == 1:
        kv, kv_lens, ropes = _prefill_v2t_und_frames(
            model, tokenizer, new_token_ids, kv, kv_lens, ropes,
            frames, timestamps, device,
        )
    elif actual_temporal_patch_size == int(model.config.pixel_video_temporal_patch_size):
        kv, kv_lens, ropes = _prefill_v2t_und_tubes(
            model, tokenizer, new_token_ids, kv, kv_lens, ropes,
            frames, timestamps, device, timestamp_mode=timestamp_mode,
        )
    else:
        raise ValueError(
            "Unsupported V2T temporal patch override: "
            f"{actual_temporal_patch_size}; expected 1 or "
            f"{model.config.pixel_video_temporal_patch_size}"
        )

    return _decode_v2t_suffix(
        model,
        tokenizer,
        new_token_ids,
        kv,
        kv_lens,
        ropes,
        instruction,
        device,
        max_length,
        temperature,
    )


def _decode_v2t_suffix(
    model,
    tokenizer,
    new_token_ids,
    kv,
    kv_lens,
    ropes,
    instruction,
    device,
    max_length,
    temperature,
):
    """Shared V2T ChatML suffix prefill and answer decoding."""
    suffix_input, kv_lens, ropes, start_token_ids = (
        model.prepare_image_to_text_chatml_suffix_contexts_for_decode(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            instructions=[instruction],
            tokenizer=tokenizer,
        )
    )
    suffix_input = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in suffix_input.items()
    }
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        kv = model.forward_cache_update_text(kv, **suffix_input)

    start_input = model.prepare_text_start_tokens(kv_lens, ropes, start_token_ids)
    start_input = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in start_input.items()
    }
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        token_ids = model.generate_text(
            past_key_values=kv,
            max_length=max_length,
            do_sample=(temperature > 0.0),
            temperature=max(temperature, 1e-5),
            end_token_id=None,
            **start_input,
        )

    token_ids = token_ids.squeeze(-1).cpu().tolist()
    eos_id = new_token_ids["eos_token_id"]
    bos_id = new_token_ids["bos_token_id"]
    text_ids = []
    for token_id in token_ids:
        if token_id == eos_id:
            break
        text_ids.append(token_id)
    if text_ids and text_ids[0] == start_token_ids[0]:
        text_ids = text_ids[1:]
    if text_ids and text_ids[0] == bos_id:
        text_ids = text_ids[1:]
    return tokenizer.decode(text_ids, skip_special_tokens=True)
