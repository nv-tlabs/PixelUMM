#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a self-contained R07 Video4 benchmark HTML report."""

from __future__ import annotations

import html
import json
import math
import os
from pathlib import Path


_BASELINES_PATH = Path(__file__).with_name("comparison_baselines.json")
_VIDEOMME_DETAILS_PATH = Path(__file__).with_name("comparison_videomme_details.json")
_METRICS = {
    "mvbench": ("mvbench_accuracy", 1.0),
    "videomme": ("videomme_perception_score", 1.0),
    "longvideobench_val_v": ("lvb_acc", 100.0),
    "lvbench": ("lvbench_score", 100.0),
}
_VIDEO_COLUMNS = (
    "mvbench",
    "videomme",
    "videomme_wo_sub_short",
    "videomme_wo_sub_medium",
    "videomme_wo_sub_long",
    "longvideobench_val_v",
    "lvbench",
)
_VIDEO_LABELS = {
    "mvbench": "MVBench test",
    "videomme": "Video-MME w/o sub Overall",
    "videomme_wo_sub_short": "Video-MME w/o sub Short",
    "videomme_wo_sub_medium": "Video-MME w/o sub Medium",
    "videomme_wo_sub_long": "Video-MME w/o sub Long",
    "longvideobench_val_v": "LongVideoBench val",
    "lvbench": "LVBench test",
}
_INPUT_FIELDS = (
    ("train_frames", "Train frames"),
    ("train_resolution", "Train resolution"),
    ("eval_frames", "Eval frames"),
    ("eval_resolution", "Eval resolution"),
)
_R07_INPUT = {
    "train_frames": "Sparse 1 FPS image, ≤96 unique frames",
    "train_resolution": "Native aspect ratio; ≤200,704 pixels/frame (≈448² area), p16",
    "eval_frames": "Strict 1 FPS, ≤96 unique frames; uniform fallback",
    "eval_resolution": "Same as training: native AR, ≤200,704 pixels/frame, p16",
}


def _current_model_input(receipt: dict) -> tuple[str, dict[str, str], str]:
    profile = str(receipt.get("representation_profile", ""))
    if profile != "short_image":
        raise ValueError(f"Unexpected PixelUMM Video4 profile: {profile!r}")
    detail = (
        "R07 short=image strict 1 FPS/≤96; duration>96s uses uniform≤96 "
        "unique frames; source FPS outside [1,240] rejected; "
        "native AR/448²-area/p16; tube-start timestamps"
    )
    return "PixelUMM R07 Video4 (short=image)", dict(_R07_INPUT), detail


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def _load_baselines() -> dict:
    payload = json.loads(_BASELINES_PATH.read_text())
    if payload.get("schema_version") != 2:
        raise ValueError(f"Unsupported baseline schema: {payload.get('schema_version')!r}")
    known_sources = set(payload["sources"])
    unknown = {
        row.get("source_id") for row in payload["rows"]
        if row.get("source_id") not in known_sources
    }
    if unknown:
        raise ValueError(f"Unknown baseline source ids: {sorted(unknown)}")
    known_input_sources = set(payload.get("input_sources", {}))
    unknown_input_sources = {
        source_id
        for row in payload["rows"]
        for source_id in row.get("input_source_ids", ())
        if source_id not in known_input_sources
    }
    if unknown_input_sources:
        raise ValueError(f"Unknown input source ids: {sorted(unknown_input_sources)}")
    missing_input_fields = {
        str(row.get("model")): [field for field, _ in _INPUT_FIELDS if not row.get(field)]
        for row in payload["rows"]
        if any(not row.get(field) for field, _ in _INPUT_FIELDS)
    }
    if missing_input_fields:
        raise ValueError(f"Missing baseline input fields: {missing_input_fields}")
    return payload


def _load_videomme_details() -> dict:
    payload = json.loads(_VIDEOMME_DETAILS_PATH.read_text())
    if payload.get("schema_version") != 2:
        raise ValueError(
            f"Unsupported Video-MME detail schema: {payload.get('schema_version')!r}"
        )
    known_sources = set(payload["sources"])
    unknown = {
        row.get("source_id") for row in payload.get("rows", ())
        if row.get("source_id") not in known_sources
    }
    if unknown:
        raise ValueError(f"Unknown Video-MME detail source ids: {sorted(unknown)}")
    return payload


def _score(metrics: dict, task: str) -> tuple[float, str]:
    metric_name, scale = _METRICS[task]
    candidates = [
        (str(name), value) for name, value in metrics.items()
        if str(name).split(",", 1)[0] == metric_name
        and isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if not candidates:
        raise ValueError(f"Missing {metric_name!r} metric for {task}: {sorted(metrics)}")
    candidates.sort(key=lambda item: (not item[0].endswith(",none"), item[0]))
    name, value = candidates[0]
    value = float(value) * scale
    if not math.isfinite(value):
        raise ValueError(f"Non-finite score for {task}: {value}")
    return value, name


def _rankings(
    rows: list[dict], current: dict[str, float], tasks: tuple[str, ...]
) -> dict[str, tuple[float, float | None]]:
    rankings = {}
    for task in tasks:
        values = [
            float(row["scores"][task]) for row in rows
            if task in row.get("scores", {})
            and task not in row.get("rank_excluded_tasks", ())
        ]
        if task in current:
            values.append(current[task])
        distinct = sorted(set(round(value, 8) for value in values), reverse=True)
        if distinct:
            rankings[task] = (distinct[0], distinct[1] if len(distinct) > 1 else None)
    return rankings


def _ranked(value: float | None, task: str, rankings: dict, *, eligible: bool = True) -> str:
    if value is None:
        return "—"
    shown = f"{value:.2f}"
    if not eligible:
        return f'<span title="source protocol does not match the required split">{shown}†</span>'
    best, second = rankings[task]
    rounded = round(value, 8)
    if rounded == best:
        return f"<strong>{shown}</strong>"
    if second is not None and rounded == second:
        return f"<u>{shown}</u>"
    return shown


def _local_source_html(source: dict) -> str:
    local = source.get("local", ())
    paths = [local] if isinstance(local, str) else list(local)
    if not paths:
        return ""
    return " " + " ".join(f"<code>{html.escape(path)}</code>" for path in paths)


def _input_cells(row: dict) -> str:
    return "".join(
        f'<td class="input">{html.escape(str(row[field]))}</td>'
        for field, _ in _INPUT_FIELDS
    )


def _input_evidence_html(baselines: dict, row: dict) -> str:
    links = []
    for source_id in row.get("input_source_ids", ()):
        source = baselines["input_sources"][source_id]
        links.append(
            f'<a href="{html.escape(source["url"], quote=True)}">'
            f'{html.escape(source["label"])}</a>'
        )
    return ", ".join(links) if links else "input setting not separately sourced"


def write_summary_html(output_root: Path, summary: dict, receipt: dict) -> Path:
    """Write summary.html next to the JSON/Markdown aggregate."""
    baselines = _load_baselines()
    videomme_details = _load_videomme_details()
    tasks = baselines["task_order"]
    missing = set(tasks).difference(summary.get("tasks", {}))
    if missing:
        raise ValueError(f"Cannot build complete dashboard; missing tasks: {sorted(missing)}")
    current = {}
    metric_names = {}
    for task in tasks:
        current[task], metric_names[task] = _score(summary["tasks"][task], task)
    duration_breakdown = summary.get("duration_breakdowns", {}).get("videomme", {})
    for bucket in ("short", "medium", "long"):
        entry = duration_breakdown.get(bucket, {})
        value = entry.get("accuracy")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            key = f"videomme_wo_sub_{bucket}"
            current[key] = float(value)
            metric_names[key] = f"duration_breakdowns.videomme.{bucket}.accuracy"

    detail_overlays = {
        (str(row["model"]), str(row["size"])): row
        for row in videomme_details.get("rows", ())
    }
    baseline_rows = []
    for row in baselines["rows"]:
        merged = dict(row)
        merged["scores"] = dict(row.get("scores", {}))
        overlay = detail_overlays.get((str(row["model"]), str(row["size"])))
        if overlay:
            merged["scores"].update(overlay.get("scores", {}))
        baseline_rows.append((merged, overlay))

    rankings = _rankings([row for row, _ in baseline_rows], current, _VIDEO_COLUMNS)
    headers = "".join(
        f"<th>{html.escape(_VIDEO_LABELS[task])}</th>" for task in _VIDEO_COLUMNS
    )

    current_cells = "".join(
        f'<td title="{html.escape(metric_names.get(task, "metric unavailable"))}">{_ranked(current.get(task), task, rankings)}</td>'
        for task in _VIDEO_COLUMNS
    )
    checkpoint = html.escape(str(receipt["checkpoint"]))
    current_label, current_input, current_detail = _current_model_input(receipt)
    rows = [
        f'<tr class="current"><td>{html.escape(current_label)}</td><td>8B</td>'
        + _input_cells(current_input)
        + current_cells
        + f"<td>checkpoint {checkpoint}; official lmms-eval task prompts/scorers</td></tr>"
    ]
    sources = baselines["sources"]
    detail_sources = videomme_details["sources"]
    for row, overlay in baseline_rows:
        cells = "".join(
            f"<td>{_ranked(row.get('scores', {}).get(task), task, rankings, eligible=task not in row.get('rank_excluded_tasks', ()))}</td>"
            for task in _VIDEO_COLUMNS
        )
        source = sources[row["source_id"]]
        source_links = [
            f'<a href="{html.escape(source["url"], quote=True)}">'
            f'{html.escape(source["label"])}</a>'
        ]
        if overlay and overlay["source_id"] != row["source_id"]:
            detail_source = detail_sources[overlay["source_id"]]
            source_links.append(
                f'<a href="{html.escape(detail_source["url"], quote=True)}">'
                "Video-MME detail</a>"
            )
        rows.append(
            "<tr>"
            f'<td>{html.escape(row["model"])}</td>'
            f'<td>{html.escape(row["size"])}</td>'
            + _input_cells(row)
            + cells
            + f'<td><b>scores:</b> {" + ".join(source_links)}<br><b>inputs:</b> {_input_evidence_html(baselines, row)}</td></tr>'
        )

    preprocess = summary["preprocess"]
    reasoning = summary["reasoning"]
    source_items = "".join(
        "<li>"
        f'<a href="{html.escape(source["url"], quote=True)}">{html.escape(source["label"])}</a>: '
        f'{html.escape(source["protocol"])} '
        f'{_local_source_html(source)}'
        "</li>"
        for source in sources.values()
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PixelUMM R07 Video4 benchmark</title>
<style>
body{{font:15px/1.5 system-ui;margin:28px;color:#1f2328;max-width:1900px}}code{{background:#f6f8fa;padding:2px 5px;border-radius:4px}}.cards{{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0}}.card{{border:1px solid #d0d7de;border-radius:8px;padding:10px 14px;background:#f6f8fa}}.wrap{{overflow-x:auto;border:1px solid #d0d7de;border-radius:8px}}table{{border-collapse:collapse;min-width:2100px;font-variant-numeric:tabular-nums}}th,td{{border-right:1px solid #d0d7de;border-bottom:1px solid #d0d7de;padding:9px 12px;text-align:right;white-space:nowrap;vertical-align:top}}th{{background:#f6f8fa;position:sticky;top:0}}th:first-child,td:first-child,th:last-child,td:last-child{{text-align:left}}td.input{{text-align:left;white-space:normal;min-width:180px;max-width:260px}}td:last-child{{white-space:normal;min-width:220px;max-width:340px}}tr.current{{background:#eef7ff;font-weight:650}}td strong{{font-weight:800;color:#b42318}}td u{{color:#b45309;text-decoration-color:#b45309;text-decoration-thickness:2px;text-underline-offset:2px}}.note{{color:#59636e;max-width:1450px}}li{{margin:7px 0}}a{{color:#0969da}}
</style></head><body>
<h1>PixelUMM R07 Video4 benchmark</h1>
<p>Checkpoint <code>{checkpoint}</code>. Tasks, datasets, prompts, splits, and scorers come from the pinned official <code>lmms-eval v0.7.1</code> release; model preprocessing and chat construction follow the R07 release contract.</p>
<div class="cards"><div class="card"><b>{int(summary['num_generations']):,}</b><br>generations</div><div class="card"><b>{preprocess['sampled_frames_min']} / {preprocess['sampled_frames_mean']:.1f} / {preprocess['sampled_frames_max']}</b><br>frames min / mean / max</div><div class="card"><b>{preprocess['packed_video_tokens_min']} / {preprocess['packed_video_tokens_mean']:.1f} / {preprocess['packed_video_tokens_max']}</b><br>raw patch tokens min / mean / max</div><div class="card"><b>{reasoning['completed']} complete, {reasoning['incomplete']} incomplete</b><br>optional reasoning traces</div></div>
<div class="wrap"><table><thead><tr><th>Model</th><th>Size</th>{''.join(f'<th>{label}</th>' for _, label in _INPUT_FIELDS)}{headers}<th>Source / protocol</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="note"><b>Input-field rule:</b> {html.escape(baselines['input_field_note'])}</p>
<p class="note">Video-MME without subtitles is reported as Overall plus the official 900-question Short (&lt;2 min), Medium (4–15 min), and Long (30–60 min) duration buckets. All four values are derived from the same full 2,700-question run.</p>
<p class="note">All score cells are percentage points. Best and second-best distinct values in each column are shown in <strong style="color:#b42318">bold red</strong> and <u style="color:#b45309;text-decoration-color:#b45309">underlined amber</u>. A dagger (†) marks a reported score whose source split does not match, or does not identify, the required split; it is excluded from ranks. Published rows preserve each paper's own preprocessing, frame budget, prompt/decode details, and scorer version; they are reference baselines, not strict reruns under the PixelUMM harness. The PixelUMM row follows its frozen training recipe: {html.escape(current_detail)}, with Qwen ChatML, bare assistant prefix, and greedy decoding.</p>
<h2>Baseline provenance</h2><ul>{source_items}</ul>
</body></html>"""
    path = output_root / "summary.html"
    _atomic_write(path, document)
    return path
