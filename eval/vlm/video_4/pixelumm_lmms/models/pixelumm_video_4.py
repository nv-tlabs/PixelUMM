# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""lmms-eval adapter backed by the R07 train-aligned Video4 runtime."""

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

from eval.vlm.video_4.runtime import PixelUMMVideoRuntime


@register_model("pixelumm_video_4")
class PixelUMMVideo4(lmms):
    def __init__(
        self,
        checkpoint_path: str,
        code_root: str = "",
        device: Optional[str] = None,
        batch_size: int = 1,
        dtype: str = "bf16",
        max_new_tokens_cap: int = 64,
        decode_timeout_seconds: int = 120,
        trace_path: str = "",
        resume_trace: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            raise TypeError(f"Unexpected PixelUMM Video4 args: {kwargs}")
        if int(batch_size) != 1:
            raise ValueError("PixelUMM Video4 requires batch_size=1")
        if not code_root:
            raise ValueError("PixelUMM-release Video4 requires code_root")
        if (
            not torch.distributed.is_initialized()
            or torch.distributed.get_world_size() != 1
            or torch.distributed.get_backend() != "gloo"
        ):
            raise RuntimeError(
                "Bare DCP load requires one independent world-size-one Gloo group"
            )
        self.accelerator = None
        self._device = torch.device("cuda:0" if device is None else device)
        if self._device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("PixelUMM Video4 requires one visible CUDA GPU")
        torch.cuda.set_device(self._device.index)
        self._rank = 0
        self._world_size = 1
        self.batch_size_per_gpu = 1
        self.trace_path = Path(trace_path).resolve() if trace_path else None
        self._resume_trace = str(resume_trace).lower() in {"1", "true", "yes"}
        self._trace_cache = {}
        self._trace_cache_hits = 0
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            if self.trace_path.exists():
                if not self._resume_trace:
                    raise FileExistsError(
                        f"Refusing existing generation trace: {self.trace_path}"
                    )
                for line_number, line in enumerate(
                    self.trace_path.read_text().splitlines(), start=1
                ):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Malformed resume trace {self.trace_path}:{line_number}"
                        ) from exc
                    key = self._request_key(
                        record["task"], record["split"], record["doc_id"],
                        record["instruction_sha256"],
                    )
                    if key in self._trace_cache:
                        raise ValueError(f"Duplicate request in resume trace: {key}")
                    if not isinstance(record.get("scored_output"), str):
                        raise ValueError(
                            f"Missing scored_output in resume trace: {key}"
                        )
                    self._trace_cache[key] = record
        self.runtime = PixelUMMVideoRuntime(
            checkpoint_path=checkpoint_path,
            code_root=code_root,
            device=str(self._device),
            dtype=dtype,
            max_new_tokens_cap=int(max_new_tokens_cap),
            decode_timeout_seconds=int(decode_timeout_seconds),
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

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Video4 tasks use generate_until")

    @staticmethod
    def _request_key(task, split, doc_id, instruction_sha256):
        return (
            str(task), str(split), str(doc_id), str(instruction_sha256),
        )

    def generate_until(self, requests: List[Instance]) -> List[str]:
        outputs = []
        for request_index, request in enumerate(
            tqdm(requests, desc="PixelUMM R07 Video4")
        ):
            context, generation_kwargs, doc_to_visual, doc_id, task, split = request.args[:6]
            generation_kwargs = dict(generation_kwargs or {})
            temperature = float(generation_kwargs.get("temperature", 0.0) or 0.0)
            num_beams = int(generation_kwargs.get("num_beams", 1) or 1)
            do_sample = bool(generation_kwargs.get("do_sample", False))
            if temperature != 0.0 or num_beams != 1 or do_sample:
                raise ValueError(
                    "Video4 is greedy-only: "
                    f"task={task} temperature={temperature} beams={num_beams} sample={do_sample}"
                )
            until = generation_kwargs.get("until", ())
            if isinstance(until, str):
                until = [until]
            instruction_sha256 = hashlib.sha256(
                str(context).encode("utf-8")
            ).hexdigest()
            cache_key = self._request_key(
                task, split, doc_id, instruction_sha256,
            )
            cached = self._trace_cache.get(cache_key)
            if cached is not None:
                if cached.get("instruction") != str(context):
                    raise ValueError(f"Resume trace instruction mismatch: {cache_key}")
                output = cached["scored_output"]
                self._trace_cache_hits += 1
                outputs.append(output)
                self.cache_hook.add_partial(
                    "generate_until", (context, generation_kwargs), output
                )
                continue
            document = self.task_dict[task][split][doc_id]
            visuals = doc_to_visual(document)
            if not isinstance(visuals, list):
                visuals = [visuals]
            if len(visuals) != 1 or not isinstance(visuals[0], (str, Path)):
                raise ValueError(
                    f"Video4 expects exactly one video path: task={task} visuals={visuals!r}"
                )
            generation = self.runtime.generate(
                instruction=str(context),
                video_path=str(visuals[0]),
                max_new_tokens=int(generation_kwargs.get("max_new_tokens", 16)),
                until=until,
            )
            output = generation.scored_output
            if self.trace_path is not None:
                record = {
                    "request_index": request_index,
                    "task": str(task),
                    "split": str(split),
                    "doc_id": str(doc_id),
                    "instruction_sha256": instruction_sha256,
                    "instruction": str(context),
                    "generation_kwargs": generation_kwargs,
                    **generation.to_dict(),
                }
                with self.trace_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            outputs.append(output)
            self.cache_hook.add_partial(
                "generate_until", (context, generation_kwargs), output
            )
        if self._resume_trace:
            print(
                f"[resume] reused {self._trace_cache_hits}/"
                f"{len(outputs)} generation trace records",
                flush=True,
            )
        return outputs

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("Video4 contains no multi-round tasks")
