# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load complete PixelUMM checkpoints using a model configuration."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Sequence

import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import load_file

from train.config import DataArguments, ModelArguments, TrainingArguments, load_release_config
from train.model_factory import build_pixelumm_model
from train.checkpoint_export import EXPORT_MANIFEST, validate_export_manifest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE_CONFIG = ROOT / "experiments" / "s8_f22_r07" / "release.yaml"


@dataclass(frozen=True)
class ReleasedModel:
    model: torch.nn.Module
    tokenizer: object
    new_token_ids: dict
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    checkpoint_path: Path
    config_path: Path


def dtype_from_name(name: str) -> torch.dtype:
    try:
        return {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "half": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
            "float": torch.float32,
        }[str(name).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported PixelUMM dtype: {name}") from exc


def parse_release_arguments(
    config_path: str | os.PathLike[str] = DEFAULT_RELEASE_CONFIG,
    *,
    extra_args: Sequence[str] = (),
) -> tuple[ModelArguments, DataArguments, TrainingArguments]:
    """Parse the checked-in release YAML without rebuilding a training CLI."""

    config_path = Path(config_path).expanduser().resolve()
    extra_args = tuple(map(str, extra_args))
    if extra_args:
        if len(extra_args) != 2 or extra_args[0] != "--llm_path":
            raise ValueError(
                "Released config accepts only the explicit --llm_path override"
            )
        llm_path = extra_args[1]
    else:
        llm_path = None
    model_args, data_args, training_args = load_release_config(
        config_path,
        llm_path=llm_path,
    )
    _validate_release_contract(model_args, data_args, training_args)
    return model_args, data_args, training_args


def _validate_release_contract(model_args, data_args, training_args) -> None:
    checks = {
        "llm_backend=qwen3": model_args.llm_backend == "qwen3",
        "mrope=sensenova": model_args.qwen3_mrope_type == "sensenova",
        "image_patch=16": int(model_args.pixel_token_patch_size) == 16,
        "video_enabled": bool(model_args.enable_pixel_video),
        "video_und_embedder=boolean": isinstance(model_args.pixel_video_separate_und_gen_embedder, bool),
        "video_temporal_patch=4": int(model_args.pixel_video_temporal_patch_size) == 4,
        "image_embedder=image_raw_patch_linear": model_args.pixel_embedder_type
        == "image_raw_patch_linear",
        "video_embedder=video_raw_tube_linear": model_args.pixel_video_embedder_type
        == "video_raw_tube_linear",
        "inference_attention=flash_attn_varlen": model_args.inference_attention_backend
        == "flash_attn_varlen",
        "layout=pixelumm_mot": data_args.packed_sequence_layout == "pixelumm_mot",
        "visual_gen": bool(training_args.visual_gen),
        "visual_und": bool(training_args.visual_und),
        "gen_space=pixel": training_args.gen_space == "pixel",
        "und_space=pixel": training_args.und_space == "pixel",
        "prediction=x": training_args.prediction_type == "x",
        "loss=v_loss": training_args.loss_type == "v_loss",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "Released checkpoint config is not a supported PixelUMM contract: "
            + ", ".join(failed)
        )


def validate_released_checkpoint(checkpoint_path: str | os.PathLike[str]) -> Path:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Released checkpoint directory is missing: {checkpoint_path}")
    export_manifest = checkpoint_path / EXPORT_MANIFEST
    exported_files = validate_export_manifest(checkpoint_path) if export_manifest.is_file() else None
    if not (checkpoint_path / "__SAVE_COMPLETE").is_file() and exported_files is None:
        raise FileNotFoundError(
            f"Released checkpoint lacks __SAVE_COMPLETE or verified {EXPORT_MANIFEST}: {checkpoint_path}"
        )
    if (checkpoint_path / "model" / ".metadata").is_file():
        required_files = _validate_dcp_model(checkpoint_path)
        if exported_files is not None and exported_files != required_files:
            raise RuntimeError("Export manifest must cover exactly DCP metadata and all referenced shards")
        if exported_files is not None and (checkpoint_path / "model.safetensors").exists():
            raise RuntimeError("DCP export must not contain an unverified model.safetensors")
    else:
        safetensors_path = checkpoint_path / "model.safetensors"
        if not safetensors_path.is_file() or safetensors_path.stat().st_size <= 0:
            raise FileNotFoundError(
                "Released checkpoint has neither complete DCP model/.metadata nor "
                f"a non-empty model.safetensors: {checkpoint_path}"
            )
    return checkpoint_path


def _validate_dcp_model(checkpoint_path: Path) -> set[str]:
    """Fail closed on missing or truncated DCP model shards."""

    from torch.distributed.checkpoint import FileSystemReader

    model_dir = checkpoint_path / "model"
    metadata_path = model_dir / ".metadata"
    metadata = FileSystemReader(str(model_dir)).read_metadata()
    if not metadata.storage_data:
        raise RuntimeError(f"Released checkpoint DCP metadata is empty: {metadata_path}")

    required_sizes: dict[str, int] = {}
    for storage_info in metadata.storage_data.values():
        relative_path = str(storage_info.relative_path)
        required_sizes[relative_path] = max(
            required_sizes.get(relative_path, 0),
            int(storage_info.offset) + int(storage_info.length),
        )
    missing = []
    truncated = []
    for relative_path, required_size in sorted(required_sizes.items()):
        shard = model_dir / relative_path
        if shard.resolve().parent != model_dir.resolve() or not relative_path.endswith('.distcp'):
            raise RuntimeError(f"Invalid DCP shard path: {relative_path}")
        if not shard.is_file():
            missing.append(relative_path)
        elif shard.stat().st_size < required_size:
            truncated.append((relative_path, shard.stat().st_size, required_size))
    if missing or truncated:
        raise RuntimeError(
            f"Incomplete released DCP model at {checkpoint_path}: "
            f"missing={missing[:8]} truncated={truncated[:8]}"
        )
    return {"model/.metadata", *(f"model/{name}" for name in required_sizes)}


def validate_checkpoint_model_schema(
    checkpoint_path: str | os.PathLike[str], model: torch.nn.Module
) -> None:
    """Compare stored keys/shapes, including DCP keys a load plan would ignore.

    A model on the meta device is sufficient: this check reads only checkpoint
    metadata, without allocating model weights or touching a CUDA device.
    """
    checkpoint_path = Path(checkpoint_path)
    safetensors_path = checkpoint_path / "model.safetensors"
    if safetensors_path.is_file():
        with safe_open(str(safetensors_path), framework="pt", device="cpu") as reader:
            shapes = {key: tuple(reader.get_slice(key).get_shape()) for key in reader.keys()}
    else:
        from torch.distributed.checkpoint import FileSystemReader
        from torch.distributed.checkpoint.metadata import TensorStorageMetadata

        metadata = FileSystemReader(str(checkpoint_path / "model")).read_metadata()
        non_tensors = [
            key for key, value in metadata.state_dict_metadata.items()
            if not isinstance(value, TensorStorageMetadata)
        ]
        if non_tensors:
            raise RuntimeError(f"Non-tensor entries in released model: {non_tensors[:8]}")
        shapes = {key: tuple(value.size) for key, value in metadata.state_dict_metadata.items()}

    state = model.state_dict()
    missing = sorted(set(state) - set(shapes))
    unexpected = sorted(set(shapes) - set(state))
    mismatched = [
        (key, tuple(state[key].shape), shapes[key])
        for key in sorted(set(state) & set(shapes))
        if tuple(state[key].shape) != shapes[key]
    ]
    if missing or unexpected or mismatched:
        raise RuntimeError(
            f"Checkpoint/model schema mismatch at {checkpoint_path}: "
            f"missing={missing[:8]} unexpected={unexpected[:8]} "
            f"shape_mismatches(model, checkpoint)={mismatched[:8]}"
        )


def load_checkpoint_weights(
    checkpoint_path: str | os.PathLike[str],
    model: torch.nn.Module,
    *,
    logger: logging.Logger | None = None,
) -> torch.nn.Module:
    """Strictly load only learned model tensors from a released checkpoint."""

    checkpoint_path = validate_released_checkpoint(checkpoint_path)
    validate_checkpoint_model_schema(checkpoint_path, model)
    logger = logger or logging.getLogger("pixelumm.release")
    safetensors_path = checkpoint_path / "model.safetensors"
    if safetensors_path.is_file():
        logger.info("Loading released safetensors weights from %s", safetensors_path)
        state_dict = load_file(str(safetensors_path), device="cpu")
    else:
        from torch.distributed.checkpoint import FileSystemReader, load

        model_dir = checkpoint_path / "model"
        logger.info("Loading released DCP weights from %s", model_dir)
        state_dict = model.state_dict()
        load(state_dict, storage_reader=FileSystemReader(str(model_dir)))
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Strict released checkpoint load returned incompatible keys: "
            f"missing={incompatible.missing_keys[:8]} "
            f"unexpected={incompatible.unexpected_keys[:8]}"
        )
    return model


def initialize_single_process(device: str | torch.device = "cuda") -> torch.device:
    """Initialize the world-size-one process group required by DCP loading."""

    device = torch.device(device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Released PixelUMM inference requires a CUDA GPU")
        index = 0 if device.index is None else int(device.index)
        torch.cuda.set_device(index)
        device = torch.device("cuda", index)
    if not dist.is_initialized():
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29501")
        dist.init_process_group("gloo", timeout=timedelta(seconds=1800))
    if dist.get_world_size() != 1:
        raise RuntimeError(
            "release inference.py is a one-process entrypoint; use the benchmark "
            "launchers for distributed evaluation"
        )
    return device


def load_released_model(
    checkpoint_path: str | os.PathLike[str],
    *,
    config_path: str | os.PathLike[str] = DEFAULT_RELEASE_CONFIG,
    device: str | torch.device = "cuda",
    dtype: str = "bf16",
    llm_path: str | None = None,
    logger: logging.Logger | None = None,
) -> ReleasedModel:
    """Load a released DCP/safetensors checkpoint without training lineage state."""

    checkpoint_path = validate_released_checkpoint(checkpoint_path)
    config_path = Path(config_path).expanduser().resolve()
    extra_args = [] if llm_path is None else ["--llm_path", str(llm_path)]
    model_args, data_args, training_args = parse_release_arguments(
        config_path, extra_args=extra_args
    )
    device = initialize_single_process(device)
    parameter_dtype = dtype_from_name(dtype)
    logger = logger or logging.getLogger("pixelumm.release")
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    # The released checkpoint owns every learned tensor.  The Qwen path is a
    # config/tokenizer dependency only; loading its five weight shards here
    # would waste tens of GB and then be immediately overwritten by DCP.
    previous_default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(parameter_dtype)
        model, tokenizer, new_token_ids = build_pixelumm_model(
            model_args,
            data_args,
            training_args,
            logger,
        )
    finally:
        torch.set_default_dtype(previous_default_dtype)
    model = load_checkpoint_weights(checkpoint_path, model, logger=logger)
    model = model.to(device=device, dtype=parameter_dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return ReleasedModel(
        model=model,
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )
