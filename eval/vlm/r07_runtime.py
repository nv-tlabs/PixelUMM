# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared R07 checkpoint and training-contract helpers for VLM evaluation."""

from __future__ import annotations

import os
from pathlib import Path

from eval.vlm.environment import default_hf_home


def resolve_vlm_hf_home(
    suite: str,
    environment: dict[str, str] | None = None,
) -> str:
    """Resolve one frozen eval cache while allowing explicit overrides."""
    environment = os.environ if environment is None else environment
    contracts = {
        "regular21": "PIXELUMM_REGULAR21_HF_HOME",
        "video4": "PIXELUMM_VIDEO4_HF_HOME",
    }
    try:
        suite_variable = contracts[suite]
    except KeyError as error:
        raise ValueError(f"unknown PixelUMM VLM suite: {suite}") from error
    if suite_variable in environment:
        return environment[suite_variable]
    if "PIXELUMM_VLM_HF_HOME" in environment:
        return environment["PIXELUMM_VLM_HF_HOME"]
    return str(default_hf_home(environment))


def validate_dcp_checkpoint(checkpoint_path: Path) -> None:
    """Verify the completion marker and every referenced model shard range."""
    from torch.distributed.checkpoint import FileSystemReader

    checkpoint_path = Path(checkpoint_path).resolve()
    if not (checkpoint_path / "__SAVE_COMPLETE").is_file():
        raise FileNotFoundError(
            f"Checkpoint lacks __SAVE_COMPLETE: {checkpoint_path}"
        )
    model_dir = checkpoint_path / "model"
    metadata_path = model_dir / ".metadata"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Checkpoint lacks DCP metadata: {metadata_path}")
    metadata = FileSystemReader(str(model_dir)).read_metadata()
    if not metadata.storage_data:
        raise RuntimeError(f"Checkpoint DCP metadata is empty: {metadata_path}")

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
        if not shard.is_file():
            missing.append(relative_path)
        elif shard.stat().st_size < required_size:
            truncated.append((relative_path, shard.stat().st_size, required_size))
    if missing or truncated:
        raise RuntimeError(
            f"Incomplete DCP payload at {checkpoint_path}: "
            f"missing={missing[:8]} truncated={truncated[:8]}"
        )
