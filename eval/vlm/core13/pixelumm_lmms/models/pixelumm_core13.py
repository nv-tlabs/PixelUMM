# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""lmms-eval model adapter backed by the train-aligned PixelUMM runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from tqdm import tqdm

from eval.vlm.core13.runtime import PixelUMMImageRuntime


@register_model("pixelumm_core13")
class PixelUMMCore13(lmms):
    def __init__(
        self,
        checkpoint_path: str,
        code_root: str = "",
        device: Optional[str] = None,
        batch_size: int = 1,
        dtype: str = "bf16",
        max_new_tokens_cap: int = 256,
        reasoning_mode: str = "standard_bare",
        reasoning_trace_path: str = "",
        reasoning_trace_resume: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            raise TypeError(f"Unexpected PixelUMM Regular21 args: {kwargs}")
        if int(batch_size) != 1:
            raise ValueError("PixelUMM Regular21 currently requires batch_size=1")
        if reasoning_mode != "standard_bare":
            raise ValueError(
                "PixelUMM-release Regular21 requires reasoning_mode=standard_bare"
            )
        if not code_root:
            raise ValueError("PixelUMM-release Regular21 requires code_root")

        if (
            not torch.distributed.is_initialized()
            or torch.distributed.get_world_size() != 1
            or torch.distributed.get_backend() != "gloo"
        ):
            raise RuntimeError(
                "Regular21 bare-DCP workers require an independent world-size-one "
                "Gloo group; cross-worker distributed groups are unsupported."
            )
        self.accelerator = None
        self._device = torch.device("cuda:0" if device is None else device)
        if self._device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("PixelUMM Regular21 requires one visible CUDA GPU")
        torch.cuda.set_device(self._device.index)
        self._rank = 0
        self._world_size = 1
        self.batch_size_per_gpu = 1
        self.reasoning_trace_path = (
            Path(reasoning_trace_path).resolve() if reasoning_trace_path else None
        )
        if self.reasoning_trace_path is not None:
            self.reasoning_trace_path.parent.mkdir(parents=True, exist_ok=True)
            if self.reasoning_trace_path.exists():
                if not bool(reasoning_trace_resume):
                    raise FileExistsError(
                        f"Refusing to append to an existing reasoning trace: "
                        f"{self.reasoning_trace_path}"
                    )
                for line_number, line in enumerate(
                    self.reasoning_trace_path.read_text().splitlines(), start=1
                ):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Malformed resumable reasoning trace at "
                            f"{self.reasoning_trace_path}:{line_number}"
                        ) from exc
                    if record.get("reasoning_mode") != reasoning_mode:
                        raise RuntimeError(
                            "Reasoning-mode mismatch in resumable trace: "
                            f"expected={reasoning_mode!r} "
                            f"observed={record.get('reasoning_mode')!r}"
                        )
        self.runtime = PixelUMMImageRuntime(
            checkpoint_path=checkpoint_path,
            code_root=code_root,
            device=str(self._device),
            dtype=dtype,
            max_new_tokens_cap=int(max_new_tokens_cap),
            reasoning_mode=reasoning_mode,
        )

    @property
    def model(self):
        return self.runtime.model

    @property
    def tokenizer(self):
        return self.runtime.tokenizer

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self.runtime.max_new_tokens_cap

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    def loglikelihood(
        self,
        requests: List[Instance],
    ) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Regular21 tasks use generate_until")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        outputs = []
        for request_index, request in enumerate(
            tqdm(requests, desc="PixelUMM Regular21")
        ):
            context, generation_kwargs, doc_to_visual, doc_id, task, split = request.args[:6]
            document = self.task_dict[task][split][doc_id]
            visuals = doc_to_visual(document)
            if not isinstance(visuals, list):
                visuals = [visuals]
            generation_kwargs = dict(generation_kwargs or {})
            temperature = float(generation_kwargs.get("temperature", 0.0) or 0.0)
            num_beams = int(generation_kwargs.get("num_beams", 1) or 1)
            do_sample = bool(generation_kwargs.get("do_sample", False))
            if temperature != 0.0 or num_beams != 1 or do_sample:
                raise ValueError(
                    "Core-11 official evaluation is greedy-only: "
                    f"task={task} temperature={temperature} num_beams={num_beams} "
                    f"do_sample={do_sample}"
                )
            until = generation_kwargs.get("until", ())
            if isinstance(until, str):
                until = [until]
            generation = self.runtime.generate(
                instruction=context,
                images=visuals,
                max_new_tokens=generation_kwargs.get("max_new_tokens", 128),
                until=until,
                image_placeholder_policy=(
                    "numbered_references"
                    if task in {
                        "mmmu_val",
                        "mmmu_pro_standard",
                        "mmstar",
                        "mmstar_strict_direct",
                        "mmstar_careful_direct",
                        "mmstar_honey_final",
                        "mmstar_honey_final_stratified",
                        "mmstar_honey_final_stratified_shard0",
                        "mmstar_honey_final_stratified_shard1",
                        "mmstar_honey_final_stratified_shard2",
                    }
                    else "strict"
                ),
            )
            output = generation.scored_output
            if self.reasoning_trace_path is not None:
                record = {
                    "request_index": request_index,
                    "task": str(task),
                    "split": str(split),
                    "doc_id": str(doc_id),
                    "instruction_sha256": hashlib.sha256(
                        str(context).encode("utf-8")
                    ).hexdigest(),
                    "instruction": str(context),
                    "generation_kwargs": generation_kwargs,
                    **generation.to_dict(),
                }
                # The trace lives on shared storage.  Re-create its parent
                # defensively before every append so a transient directory
                # replacement cannot turn an otherwise valid long-running
                # benchmark into a late FileNotFoundError.
                self.reasoning_trace_path.parent.mkdir(parents=True, exist_ok=True)
                with self.reasoning_trace_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            outputs.append(output)
            self.cache_hook.add_partial(
                "generate_until",
                (context, generation_kwargs),
                output,
            )
        return outputs

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("Regular21 contains no multi-round tasks")
