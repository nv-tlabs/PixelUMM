#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify a separately distributed toy package before using it for training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

TASKS = ("t2i", "t2v", "image_vlm", "video_vlm")


def verify_package(root: Path) -> dict:
    root = root.resolve()
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "pixelumm-toy-data-v1":
        raise ValueError("Unsupported toy-data schema")
    inventory = manifest.get("files")
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("Missing file inventory")
    for name, expected in inventory.items():
        relative = Path(name)
        path = root / relative
        if relative.is_absolute() or ".." in relative.parts or root not in path.resolve().parents:
            raise ValueError(f"Unsafe inventory path: {name}")
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or unsafe file: {name}")
        if path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Size mismatch: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {name}")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != set(inventory) | {"MANIFEST.json"}:
        raise ValueError("Unlisted or missing files in package")
    counts = {}
    ids = set()
    for task in TASKS:
        name = f"{task}.jsonl"
        if name not in inventory:
            raise ValueError(f"Manifest not covered by hashes: {name}")
        records = [json.loads(line) for line in (root / name).read_text(encoding="utf-8").splitlines() if line.strip()]
        if not records or len(records) != manifest["tasks"][task]:
            raise ValueError(f"Record count mismatch: {task}")
        for record in records:
            ident = record.get("id")
            if not ident or ident in ids or record.get("task") != task:
                raise ValueError(f"Invalid/duplicate record in {task}")
            ids.add(ident)
            media = record.get("image" if task in {"t2i", "image_vlm"} else "video")
            if media not in inventory:
                raise ValueError(f"Media not covered by hashes: {media}")
            fields = ("prompt",) if task in {"t2i", "t2v"} else ("instruction", "answer")
            if any(not isinstance(record.get(key), str) or not record[key].strip() for key in fields):
                raise ValueError(f"Missing text fields: {ident}")
        counts[task] = len(records)
    return {"schema": manifest["schema"], "tasks": counts, "files_verified": len(inventory),
            "bytes_verified": sum(value["bytes"] for value in inventory.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toy-root", type=Path, required=True)
    parser.add_argument("--decode", action="store_true", help="CPU-decode every record using the training dataset")
    parser.add_argument("--llm-path", type=Path, help="Pinned local Qwen config/tokenizer directory for --decode")
    args = parser.parse_args()
    if args.decode and args.llm_path is None:
        parser.error("--decode requires --llm-path")
    report = verify_package(args.toy_root)
    if args.decode:
        import gc
        import torch
        from transformers import AutoTokenizer
        from data.local_jsonl_dataset import PixelUMMLocalJSONLDataset

        torch.set_num_threads(2)
        tokenizer = AutoTokenizer.from_pretrained(args.llm_path, local_files_only=True)
        report["decoded"] = {}
        for task in TASKS:
            dataset = PixelUMMLocalJSONLDataset(tokenizer, [str(args.toy_root / f"{task}.jsonl")], allowed_task=task)
            checks = []
            for record in dataset.records:
                sample = dataset._to_sample(record)
                if sample["num_tokens"] <= 0:
                    raise ValueError(f"Empty sample: {record['id']}")
                shapes = [list(t.shape) for name in ("image_tensor_list", "video_tensor_list") for t in sample.get(name, [])]
                checks.append({"id": record["id"], "num_tokens": sample["num_tokens"], "media_shapes": shapes})
                del sample
                gc.collect()
            report["decoded"][task] = checks
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
