# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict loader for immutable PixelUMM scientific YAML configurations.

The release launcher owns runtime state such as checkpoint paths, output
directories, topology, and tracking identity. This module deliberately loads
only model/data/training science knobs into the typed release configuration.
Keeping that boundary explicit prevents a portable
scientific config from silently capturing one cluster's filesystem state.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


SCHEMA = "pixelumm_scientific_config_v1"
SECTIONS = ("model", "data", "training")


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    # PyYAML normally keeps the last duplicate key.  A reviewed scientific
    # contract must reject that ambiguity instead of silently changing it.
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicated = key in mapping
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicated:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cli_scalar(value: Any, *, location: str) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, (str, int, float)):
        return str(value)
    raise RuntimeError(
        f"scientific config value must be a scalar at {location}; "
        f"got {type(value).__name__}"
    )


def load_scientific_config(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Load one fail-closed, sectioned scientific configuration."""

    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"scientific config is absent or empty: {path}")
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise RuntimeError(f"scientific config identity mismatch: {path}")
    try:
        payload = yaml.load(
            path.read_text(encoding="utf-8"),
            Loader=_UniqueKeyLoader,
        )
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"scientific config is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("scientific config must be a YAML mapping")
    expected_top_level = {"schema", *SECTIONS}
    if set(payload) != expected_top_level:
        raise RuntimeError(
            "scientific config top-level keys changed: "
            f"expected={sorted(expected_top_level)} actual={sorted(payload)}"
        )
    if payload["schema"] != SCHEMA:
        raise RuntimeError(
            f"unexpected scientific config schema: {payload['schema']!r}"
        )

    sections: dict[str, dict[str, Any]] = {}
    owners: dict[str, str] = {}
    for section in SECTIONS:
        values = payload[section]
        if not isinstance(values, dict) or not values:
            raise RuntimeError(
                f"scientific config section {section!r} must be a non-empty mapping"
            )
        normalized: dict[str, Any] = {}
        for raw_name, value in values.items():
            if not isinstance(raw_name, str) or not raw_name:
                raise RuntimeError(
                    f"scientific config contains an invalid key in {section!r}"
                )
            if raw_name.startswith("--") or not raw_name.replace("_", "a").isalnum():
                raise RuntimeError(
                    f"scientific config key must be a dataclass field name: {raw_name!r}"
                )
            if raw_name in owners:
                raise RuntimeError(
                    f"scientific config key {raw_name!r} appears in both "
                    f"{owners[raw_name]!r} and {section!r}"
                )
            _cli_scalar(value, location=f"{section}.{raw_name}")
            owners[raw_name] = section
            normalized[raw_name] = value
        sections[section] = normalized
    return sections
