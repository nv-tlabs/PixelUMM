# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integrity receipts for static DCP exports without a training completion marker.

This verifies locally supplied model files, not upstream training completion,
publisher authenticity, optimizer resumability, or model/config compatibility.
Only prepare receipts for a trusted, fully downloaded, immutable export.
"""

import hashlib
import json
from pathlib import Path, PurePosixPath

EXPORT_MANIFEST = "checkpoint_export.json"
EXPORT_SCHEMA = "pixelumm_dcp_export_v1"


def _model_file(root: Path, name: str) -> Path:
    if not isinstance(name, str):
        raise ValueError("Export file path must be a string")
    parts = PurePosixPath(name).parts
    if (
        len(parts) != 2 or parts[0] != "model"
        or name != "/".join(parts) or "\\" in name
        or (parts[1] != ".metadata" and not parts[1].endswith(".distcp"))
    ):
        raise ValueError(f"Invalid export model path: {name!r}")
    path = root / name
    if path.resolve().parent != (root / "model").resolve() or path.is_symlink():
        raise ValueError(f"Export model path must stay inside model/: {name!r}")
    return path


def _digest(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Export changed during verification: {path}")
    return digest.hexdigest()


def validate_export_manifest(root: Path) -> set[str]:
    root = Path(root)
    payload = json.loads((root / EXPORT_MANIFEST).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != EXPORT_SCHEMA:
        raise ValueError("Unsupported checkpoint export manifest schema")
    entries = payload.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Checkpoint export manifest has no files")
    names = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}:
            raise ValueError("Malformed checkpoint export file entry")
        name = entry["path"]
        path = _model_file(root, name)
        if name in names:
            raise ValueError(f"Duplicate export file: {name}")
        names.add(name)
        if type(entry["bytes"]) is not int or entry["bytes"] <= 0:
            raise ValueError(f"Invalid export file size: {name}")
        expected = entry["sha256"]
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError(f"Invalid export SHA256: {name}")
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise RuntimeError(f"Missing or wrong-size export file: {name}")
        if _digest(path) != expected:
            raise RuntimeError(f"Export SHA256 mismatch: {name}")
    if "model/.metadata" not in names:
        raise ValueError("Export manifest must include model/.metadata")
    return names


def prepare_export_manifest(root: Path) -> Path:
    from train.release_checkpoint import _validate_dcp_model

    root = Path(root).expanduser().resolve()
    output = root / EXPORT_MANIFEST
    if output.exists():
        raise FileExistsError(f"Preserving existing export manifest: {output}")
    required = _validate_dcp_model(root)
    if (root / "model.safetensors").exists():
        raise ValueError("Prepare a DCP-only export, without model.safetensors")
    files = []
    for name in sorted(required):
        path = _model_file(root, name)
        files.append({"path": name, "bytes": path.stat().st_size, "sha256": _digest(path)})
    payload = {
        "schema": EXPORT_SCHEMA,
        "verification": "local DCP file integrity; not a training completion assertion",
        "files": files,
    }
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2) + "\n")
    return output
