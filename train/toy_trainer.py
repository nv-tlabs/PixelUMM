# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small, auditable FSDP trainer for the released local JSONL interface.

This is intentionally not a continuation trainer.  It restores model weights,
starts a fresh AdamW optimizer, consumes finite local media manifests, and
writes one portable DCP checkpoint for ``inference.py`` without a full CPU gather.
"""

from __future__ import annotations

import logging
import os
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
import torch.distributed as dist
import yaml
from torch.distributed.checkpoint import save as save_dcp
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    BackwardPrefetch,
    ShardedStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.utils.data import DataLoader

from data.dataset_base_pixel import DataConfig, PackedDataset, collate_wrapper
from data.gpu_vision_tubeify import materialize_packed_vision_on_gpu
from modeling.pixelumm.loss_reduction import loss_weights_for_reduction
from modeling.pixelumm.qwen3_navit import Qwen3MoTDecoderLayer
from train.model_factory import build_pixelumm_model
from train.release_checkpoint import parse_release_arguments, validate_checkpoint_model_schema
from train.sharded_checkpoint import load_sharded_weights, materializer


@dataclass(frozen=True)
class ToyTrainingConfig:
    checkpoint: Path
    toy_root: Path
    output: Path
    model_config: Path
    dataset_config: Path
    llm_path: Path | None
    steps: int = 4
    learning_rate: float = 1e-5
    seed: int = 4396
    max_num_tokens: int = 82_000
    expected_num_tokens: int = 60_000

    def to_dict(self) -> dict[str, object]:
        return {
            name: (str(value) if isinstance(value, Path) else value)
            for name, value in self.__dict__.items()
        }


def _logger(rank: int) -> logging.Logger:
    logger = logging.getLogger("pixelumm.toy_train")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    return logger


def _init_distributed() -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("PixelUMM toy training requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank = dist.get_rank()
    return rank, dist.get_world_size(), torch.device("cuda", local_rank)


def _auto_wrap_policy(module, recurse: bool, nonwrapped_numel: int) -> bool:
    del nonwrapped_numel
    return True if recurse else isinstance(module, Qwen3MoTDecoderLayer)


def _checkpoint_layer(module) -> bool:
    return isinstance(module, Qwen3MoTDecoderLayer)


def _seed_training(seed: int) -> None:
    """Seed both model noise and Python/NumPy-based packing/augmentation."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def _build_loader(
    config: ToyTrainingConfig,
    *,
    model_args,
    tokenizer,
    new_token_ids,
    rank: int,
    world_size: int,
) -> DataLoader:
    dataset_meta = yaml.safe_load(config.dataset_config.read_text(encoding="utf-8"))
    if not isinstance(dataset_meta, dict) or not dataset_meta:
        raise RuntimeError(f"Toy dataset config is empty: {config.dataset_config}")
    data_config = DataConfig(
        grouped_datasets=dataset_meta,
        text_cond_dropout_prob=model_args.text_cond_dropout_prob,
        pixel_und_cond_dropout_prob=model_args.pixel_und_cond_dropout_prob,
        pixel_gen_cond_dropout_prob=model_args.pixel_gen_cond_dropout_prob,
        pixel_token_patch_size=model_args.pixel_token_patch_size,
        pixel_video_temporal_patch_size=model_args.pixel_video_temporal_patch_size,
    )
    dataset = PackedDataset(
        data_config,
        tokenizer=tokenizer,
        special_tokens=new_token_ids,
        local_rank=rank,
        world_size=world_size,
        expected_num_tokens=config.expected_num_tokens,
        max_num_tokens_per_sample=config.max_num_tokens,
        max_num_tokens=config.max_num_tokens,
    )
    dataset.set_epoch(config.seed)
    return DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_wrapper(),
        drop_last=True,
    )


def _distributed_objective(
    losses: torch.Tensor | None,
    weights: torch.Tensor | None,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    if losses is None or losses.numel() == 0:
        local_numerator = torch.zeros((), device=device)
        local_denominator = torch.zeros((), device=device)
    else:
        losses = losses.reshape(-1)
        if weights is None:
            weights = torch.ones_like(losses, dtype=torch.float32)
        weights = weights.reshape(-1).to(device=device, dtype=torch.float32)
        if losses.numel() != weights.numel():
            raise RuntimeError(
                f"Loss/weight mismatch: {losses.numel()} vs {weights.numel()}"
            )
        local_numerator = (losses * weights).sum()
        local_denominator = weights.sum()

    reduced = torch.stack(
        (local_numerator.detach().float(), local_denominator.detach().float())
    )
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    if reduced[1].item() == 0.0:
        return local_numerator * 0.0, 0.0
    objective = local_numerator * dist.get_world_size() / reduced[1]
    return objective, float((reduced[0] / reduced[1]).item())


def _save_checkpoint(model: FSDP, output: Path, step: int, rank: int) -> Path:
    checkpoint_root = output / "checkpoints"
    final_path = checkpoint_root / f"{step:07d}"
    temporary_path = checkpoint_root / f".tmp-{step:07d}"
    path_conflict = torch.tensor(
        int(final_path.exists() or temporary_path.exists()),
        device=next(model.parameters()).device,
    )
    dist.all_reduce(path_conflict, op=dist.ReduceOp.MAX)
    if path_conflict.item():
        raise FileExistsError(
            f"Refusing to overwrite an existing toy checkpoint: {final_path}"
        )
    if rank == 0:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        temporary_path.mkdir()
    dist.barrier()
    with FSDP.state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
        ShardedStateDictConfig(offload_to_cpu=False),
    ):
        state_dict = model.state_dict()
        save_dcp(state_dict, checkpoint_id=temporary_path / "model")
    dist.barrier()
    if rank == 0:
        (temporary_path / "__SAVE_COMPLETE").write_text(
            f"step={step}\nformat=pixelumm_release_dcp_v1\n",
            encoding="utf-8",
        )
        temporary_path.rename(final_path)
    dist.barrier()
    return final_path


def run_toy_training(config: ToyTrainingConfig) -> Path:
    rank, world_size, device = _init_distributed()
    logger = _logger(rank)
    _seed_training(config.seed + rank)

    extra_args = (
        ()
        if config.llm_path is None
        else ("--llm_path", str(config.llm_path))
    )
    model_args, data_args, training_args = parse_release_arguments(
        config.model_config,
        extra_args=extra_args,
    )
    # FP32 master parameters are created on meta, then materialized layer by
    # layer on CUDA and sharded. No rank holds a full model in host memory.
    with torch.device("meta"):
        model, tokenizer, new_token_ids = build_pixelumm_model(
            model_args,
            data_args,
            training_args,
            logger,
        )
    validate_checkpoint_model_schema(config.checkpoint, model)
    model = model.float().train()
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=_auto_wrap_policy,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        device_id=device,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        use_orig_params=True,
        param_init_fn=materializer(device),
    )
    logger.info("FSDP initialized; loading only local checkpoint shards")
    load_sharded_weights(fsdp_model, config.checkpoint)
    logger.info("Sharded weights loaded")
    apply_activation_checkpointing(
        fsdp_model,
        checkpoint_wrapper_fn=lambda module: checkpoint_wrapper(
            module,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            preserve_rng_state=True,
        ),
        check_fn=_checkpoint_layer,
    )
    optimizer = torch.optim.AdamW(
        fsdp_model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        foreach=False,
    )
    loader = _build_loader(
        config,
        model_args=model_args,
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
        rank=rank,
        world_size=world_size,
    )
    iterator = iter(loader)
    if rank == 0:
        config.output.mkdir(parents=True, exist_ok=False)
        (config.output / "training-config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    dist.barrier()

    for step in range(1, config.steps + 1):
        batch = next(iterator).cuda(device).to_dict()
        logger.info("step=%d/%d forward; packed_tokens=%d", step, config.steps, batch["sequence_length"])
        task_keys = ("raw_pixel_images_gen", "raw_pixel_videos_gen",
                     "raw_pixel_images_und", "raw_pixel_videos_und")
        task_counts = torch.tensor([len(batch.get(key, [])) for key in task_keys], device=device)
        dist.all_reduce(task_counts, op=dist.ReduceOp.SUM)
        materialize_packed_vision_on_gpu(
            batch,
            patch_size=model_args.pixel_token_patch_size,
        )
        ce_weights = batch.pop("ce_loss_weights", None)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = fsdp_model(**batch)
        ce_losses = outputs["ce"]
        mse_losses = outputs["mse"]
        mse_weights = outputs["mse_loss_reduction_weights"]
        ce_objective, ce_value = _distributed_objective(
            ce_losses,
            loss_weights_for_reduction(
                ce_losses,
                "token",
                non_token_weights=ce_weights,
            ) if ce_losses is not None and ce_losses.numel() else None,
            device=device,
        )
        mse_token_losses = (
            mse_losses.mean(dim=-1)
            if mse_losses is not None and mse_losses.numel()
            else None
        )
        mse_objective, mse_value = _distributed_objective(
            mse_token_losses,
            loss_weights_for_reduction(
                mse_token_losses,
                training_args.mse_loss_reduction,
                non_token_weights=mse_weights,
            ) if mse_token_losses is not None else None,
            device=device,
        )
        loss = ce_objective + mse_objective
        graph_anchor = outputs.get("conditional_graph_anchor")
        if graph_anchor is not None:
            loss = loss + graph_anchor
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite toy-training loss at step {step}: {loss.item()}"
            )
        loss.backward()
        grad_norm = fsdp_model.clip_grad_norm_(1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(
                f"Non-finite toy-training gradient norm at step {step}: {grad_norm.item()}"
            )
        probes = [p for p in fsdp_model.parameters() if p.requires_grad and p.numel()]
        before = torch.stack([p.detach().reshape(-1)[0].clone() for p in probes])
        optimizer.step()
        after = torch.stack([p.detach().reshape(-1)[0] for p in probes])
        update_max = (after - before).abs().max()
        dist.all_reduce(update_max, op=dist.ReduceOp.MAX)
        if not torch.isfinite(update_max) or update_max.item() == 0:
            raise RuntimeError("No finite nonzero optimizer update observed in parameter probes")
        if rank == 0:
            metrics = {"step": step, "ce": ce_value, "mse": mse_value,
                       "grad_norm": float(grad_norm.item()), "update_probe_max": update_max.item(),
                       "global_media_segments": dict(zip(("t2i", "t2v", "image_vlm", "video_vlm"), task_counts.tolist())),
                       "peak_memory_gib_rank0": torch.cuda.max_memory_allocated(device) / 1024**3}
            with (config.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
            logger.info(
                "step=%d/%d ce=%.6f mse=%.6f grad_norm=%.6f update_max=%.9f peak_GiB=%.2f",
                step,
                config.steps,
                ce_value,
                mse_value,
                float(grad_norm.item()),
                update_max.item(),
                metrics["peak_memory_gib_rank0"],
            )

    optimizer.zero_grad(set_to_none=True)
    del optimizer, batch, outputs, loss, ce_losses, mse_losses, mse_token_losses
    torch.cuda.empty_cache()
    checkpoint = _save_checkpoint(fsdp_model, config.output, config.steps, rank)
    if rank == 0:
        logger.info("Saved portable toy-trained checkpoint to %s", checkpoint)
    dist.destroy_process_group()
    return checkpoint
