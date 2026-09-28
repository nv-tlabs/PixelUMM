#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a local PixelUMM toy training package from source media.

The output contains four JSONL manifests and copied media files.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


UPSTREAM_EXAMPLE_URL = (
    "https://lf3-static.bytednsdoc.com/obj/eden-cn/nuhojubrps/"
    "bagel_example.zip"
)
SCHEMA = "pixelumm-toy-data-v1"
TASKS = ("t2i", "t2v", "image_vlm", "video_vlm")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_source_path(manifest: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{manifest}: {field} must be a non-empty relative path")
    raw = value.strip()
    if "://" in raw or Path(raw).is_absolute():
        raise ValueError(f"{manifest}: {field} must be local and relative")
    root = manifest.parent.resolve()
    resolved = (root / raw).resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError(f"{manifest}: {field} escapes its source directory")
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(f"{manifest}: missing {field}: {resolved}")
    return resolved


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    path.write_text(text, encoding="utf-8")


def _save_rgb(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="PNG", optimize=True)


def _caption_from_upstream(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value.strip()
    if isinstance(value, dict):
        for key in sorted(value):
            caption = str(value[key]).strip()
            if caption:
                return caption
    raise ValueError("BAGEL T2I row has no usable caption")


def _prepare_upstream_t2i(root: Path, output: Path, count: int) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("prepare_toy_data.py requires pyarrow for BAGEL parquet") from exc

    parquet_paths = sorted((root / "t2i").glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"BAGEL T2I parquet shards are missing under {root / 't2i'}")
    records: list[dict[str, Any]] = []
    seen_examples: set[str] = set()
    for parquet_path in parquet_paths:
        parquet = pq.ParquetFile(parquet_path)
        for row_group in range(parquet.num_row_groups):
            table = parquet.read_row_group(row_group, columns=["image", "captions"])
            payload = table.to_pylist()
            for row_index, row in enumerate(payload):
                try:
                    image_bytes = bytes(row["image"])
                    with Image.open(io.BytesIO(image_bytes)) as source:
                        image = source.convert("RGB")
                    prompt = _caption_from_upstream(row["captions"])
                except Exception:
                    continue
                identity = hashlib.sha256(
                    image_bytes + b"\0" + prompt.encode("utf-8")
                ).hexdigest()
                if identity in seen_examples:
                    continue
                seen_examples.add(identity)
                index = len(records)
                relative_media = Path("media") / "t2i" / f"upstream_{index:03d}.png"
                _save_rgb(image, output / relative_media)
                records.append(
                    {
                        "id": f"upstream-t2i-{index:03d}",
                        "task": "t2i",
                        "image": relative_media.as_posix(),
                        "prompt": prompt,
                        "source": {
                            "dataset": "BAGEL official example T2I",
                            "shard": parquet_path.name,
                            "row_group": row_group,
                            "row": row_index,
                        },
                    }
                )
                if len(records) == count:
                    return records
    if records:
        return records
    raise RuntimeError("BAGEL T2I yielded no distinct valid examples")


def _conversation_pair(record: dict[str, Any]) -> tuple[str, str] | None:
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        return None
    for index, message in enumerate(conversations[:-1]):
        next_message = conversations[index + 1]
        if not isinstance(message, dict) or not isinstance(next_message, dict):
            continue
        if message.get("from") != "human" or next_message.get("from") != "gpt":
            continue
        instruction = str(message.get("value", "")).replace("<image>", " ").strip()
        answer = str(next_message.get("value", "")).strip()
        if instruction and answer:
            return instruction, answer
    return None


def _prepare_upstream_image_vlm(
    root: Path, output: Path, count: int
) -> list[dict[str, Any]]:
    manifest = root / "vlm" / "llava_ov_si.jsonl"
    image_root = root / "vlm" / "images"
    if not manifest.is_file():
        raise FileNotFoundError(f"BAGEL VLM manifest is missing: {manifest}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        image_value = raw.get("image")
        if isinstance(image_value, list):
            if len(image_value) != 1:
                continue
            image_value = image_value[0]
        pair = _conversation_pair(raw)
        if not isinstance(image_value, str) or pair is None:
            continue
        source_path = (image_root / image_value).resolve()
        if image_root.resolve() not in source_path.parents or not source_path.is_file():
            continue
        try:
            with Image.open(source_path) as source:
                image = source.convert("RGB")
        except Exception:
            continue
        index = len(records)
        relative_media = Path("media") / "image_vlm" / f"upstream_{index:03d}.png"
        _save_rgb(image, output / relative_media)
        instruction, answer = pair
        records.append(
            {
                "id": f"upstream-image-vlm-{index:03d}",
                "task": "image_vlm",
                "image": relative_media.as_posix(),
                "instruction": instruction,
                "answer": answer,
                "source": {
                    "dataset": "BAGEL official example LLaVA-OneVision",
                    "line": line_number,
                    "original_id": raw.get("id"),
                },
            }
        )
        if len(records) == count:
            return records
    raise RuntimeError(
        f"BAGEL image VLM yielded only {len(records)} valid single-image examples; need {count}"
    )


def _prepare_r05_t2v(root: Path, output: Path, count: int) -> list[dict[str, Any]]:
    receipt_root = root / ".t2v_resume"
    if not receipt_root.is_dir():
        receipt_root = root / "metadata"  # Public demo export, same receipt schema.
    receipts = sorted(receipt_root.glob("*sana_long*.json"))
    if len(receipts) < count:
        raise FileNotFoundError(f"R05 SANA receipts under {receipt_root}: {len(receipts)} < {count}")
    records = []
    for index, receipt_path in enumerate(receipts[:count]):
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        contract = payload.get("contract") or {}
        expected_name = f"sana_long_{index:03d}"
        if contract.get("item_name") != expected_name:
            raise RuntimeError(
                f"Unexpected R05 SANA ordering at {receipt_path}: "
                f"{contract.get('item_name')!r} != {expected_name!r}"
            )
        if payload.get("frames") != 96 or float(contract.get("video_fps", 0.0)) != 24.0:
            raise RuntimeError(f"R05 SANA contract changed at {receipt_path}")
        prompt = str(contract.get("prompt", "")).strip()
        source_video = _relative_source_path(root / "receipts.jsonl", payload.get("mp4"), field="mp4")
        if not prompt or not source_video.is_file() or source_video.stat().st_size <= 0:
            raise RuntimeError(f"Incomplete R05 SANA artifact at {receipt_path}")
        relative_media = Path("media") / "t2v" / f"r05_sana_{index:03d}.mp4"
        (output / relative_media).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_video, output / relative_media)
        records.append(
            {
                "id": f"r05-sana-t2v-{index:03d}",
                "task": "t2v",
                "video": relative_media.as_posix(),
                "prompt": prompt,
                "source": {
                    "dataset": "S8-F22-R05 step5000 SANA qualitative eval",
                    "receipt": receipt_path.name,
                    "receipt_sha256": _sha256(receipt_path),
                    "eval_step": contract.get("eval_step"),
                    "sampler": contract.get("sampler"),
                    "steps": contract.get("num_timesteps"),
                    "shift": contract.get("timestep_shift"),
                    "cfg": contract.get("cfg_text_scale"),
                    "fps": contract.get("video_fps"),
                    "frames": payload.get("frames"),
                },
            }
        )
    return records


def _prepare_video_vlm(
    manifest: Path, output: Path, count: int
) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        raw = json.loads(line)
        instruction = str(raw.get("instruction", "")).strip()
        answer = str(raw.get("answer", "")).strip()
        if not instruction or not answer:
            raise ValueError(f"{manifest}:{line_number} lacks instruction/answer")
        source_video = _relative_source_path(manifest, raw.get("video"), field="video")
        index = len(records)
        suffix = source_video.suffix.lower()
        if suffix not in {".mp4", ".webm", ".mkv", ".mov", ".avi", ".gif"}:
            raise ValueError(f"Unsupported video extension in {manifest}:{line_number}")
        relative_media = Path("media") / "video_vlm" / f"videochat_{index:03d}{suffix}"
        (output / relative_media).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_video, output / relative_media)
        records.append(
            {
                "id": f"videochat-flash-vlm-{index:03d}",
                "task": "video_vlm",
                "video": relative_media.as_posix(),
                "instruction": instruction,
                "answer": answer,
                "source": {
                    "dataset": "VideoChat-Flash",
                    "line": line_number,
                    "export": "sparse 1fps local media",
                    "record_id": raw.get("source_record_id"),
                    "revision": raw.get("source_revision"),
                    "annotations": raw.get("source_annotations"),
                },
            }
        )
        if len(records) == count:
            return records
    raise RuntimeError(f"VideoChat source yielded {len(records)} examples; need {count}")


def _inventory(root: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "MANIFEST.json":
            continue
        relative = path.relative_to(root).as_posix()
        result[relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    return result


def build_package(
    *,
    upstream_example_root: Path,
    r05_t2v_root: Path,
    video_vlm_manifest: Path,
    output: Path,
    count: int,
) -> Path:
    upstream_example_root = upstream_example_root.expanduser().resolve()
    r05_t2v_root = r05_t2v_root.expanduser().resolve()
    video_vlm_manifest = video_vlm_manifest.expanduser().resolve()
    output = output.expanduser().resolve()
    if count <= 0:
        raise ValueError("--count must be positive")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing toy package: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp.", dir=output.parent))
    try:
        prepared = {
            "t2i": _prepare_upstream_t2i(upstream_example_root, staging, count),
            "t2v": _prepare_r05_t2v(r05_t2v_root, staging, count),
            "image_vlm": _prepare_upstream_image_vlm(upstream_example_root, staging, count),
            "video_vlm": _prepare_video_vlm(video_vlm_manifest, staging, count),
        }
        for task in TASKS:
            _write_jsonl(staging / f"{task}.jsonl", prepared[task])
        manifest = {
            "schema": SCHEMA,
            "requested_max_records_per_task": count,
            "tasks": {task: len(prepared[task]) for task in TASKS},
            "sources": {
                "upstream_image_examples": {
                    "url": UPSTREAM_EXAMPLE_URL,
                    "selection": "first ten valid official example rows",
                },
                "r05_t2v": {
                    "dataset": "S8-F22-R05 step5000 SANA qualitative eval",
                    "selection": "sana_long_000 through sana_long_009",
                },
                "videochat_flash_export": {
                    "dataset": "VideoChat-Flash",
                    "selection": "ten local sparse-1fps exports",
                },
            },
            "files": _inventory(staging),
        }
        (staging / "MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-example-root", required=True, type=Path)
    parser.add_argument("--r05-t2v-root", required=True, type=Path)
    parser.add_argument("--video-vlm-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = build_package(
        upstream_example_root=args.upstream_example_root,
        r05_t2v_root=args.r05_t2v_root,
        video_vlm_manifest=args.video_vlm_manifest,
        output=args.output,
        count=args.count,
    )
    print(output)


if __name__ == "__main__":
    main()
