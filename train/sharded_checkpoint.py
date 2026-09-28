# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize and load released weights after FSDP sharding, without CPU replicas."""
from __future__ import annotations

from functools import partial
from pathlib import Path

import torch
from safetensors import safe_open
from torch.distributed.checkpoint import FileSystemReader, load
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardedStateDictConfig, StateDictType

from modeling.pixelumm.qwen3_navit import SenseNovaQwen3RotaryEmbedding


def materialize_module(module: torch.nn.Module, *, device: torch.device) -> None:
    """Allocate only this module; regenerate deterministic non-checkpoint buffers."""
    buffers = {}
    for name, value in module.named_buffers(recurse=False):
        if value.is_meta:
            if isinstance(module, SenseNovaQwen3RotaryEmbedding) and name == "inv_freq":
                value, module.attention_scaling = module.rope_init_fn(module.config, device)
            else:
                raise RuntimeError(f"No initializer for meta buffer {type(module).__name__}.{name}")
        buffers[name] = value.to(device=device)
    module.to_empty(device=device, recurse=False)
    for name, value in buffers.items():
        module._buffers[name] = value


def materializer(device: torch.device):
    return partial(materialize_module, device=device)


def _copy_safetensors_shards(path: Path, state: dict) -> None:
    """Read only each rank's local tensor slices, never the full safetensors file."""
    with safe_open(str(path), framework="pt", device="cpu") as reader:
        for name, destination in state.items():
            if hasattr(destination, "local_shards"):
                for shard in destination.local_shards():
                    slices = tuple(slice(start, start + size) for start, size in
                                   zip(shard.metadata.shard_offsets, shard.metadata.shard_sizes))
                    value = reader.get_slice(name)[slices]
                    shard.tensor.copy_(value)
            else:
                destination.copy_(reader.get_tensor(name))


def load_sharded_weights(model: FSDP, checkpoint: Path) -> None:
    """Caller must validate full names/shapes against the unwrapped meta model first."""
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT,
                             ShardedStateDictConfig(offload_to_cpu=False)):
        state = model.state_dict()
        safe_path = checkpoint / "model.safetensors"
        if safe_path.is_file():
            _copy_safetensors_shards(safe_path, state)
        else:
            load(state, storage_reader=FileSystemReader(str(checkpoint / "model")))
        model.load_state_dict(state, strict=True)
