# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic local scorers for judge-dependent Regular21 tasks."""

from __future__ import annotations

import re

import pandas as pd


def mmbench_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def _mmbench_options(doc):
    return {
        letter: str(doc[letter])
        for letter in "ABCDE"
        if letter in doc and pd.notna(doc[letter]) and str(doc[letter]) != "nan"
    }


def mmbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    options = _mmbench_options(doc)
    option_text = "\n".join(f"{key}. {value}" for key, value in options.items())
    hint = "" if pd.isna(doc.get("hint")) else str(doc.get("hint", "")).strip()
    pieces = [piece for piece in (hint, str(doc["question"]), option_text) if piece]
    prompt = "\n".join(pieces)
    post_prompt = (lmms_eval_specific_kwargs or {}).get(
        "post_prompt",
        "\nAnswer with the option's letter from the given choices directly.",
    )
    return prompt + post_prompt


def _extract_choice(prediction, options):
    prediction = str(prediction).strip()
    letters = "".join(options)
    patterns = (
        rf"^\s*\(?([{letters}])\)?(?:[\s\.,:\)]|$)",
        rf"(?:answer|option|choice)\s*(?:is|:)?\s*\(?([{letters}])\)?(?:[\s\.,:\)]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, prediction, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    lowered = prediction.lower()
    text_matches = [key for key, value in options.items() if value.lower() in lowered]
    return text_matches[0] if len(text_matches) == 1 else ""


def mmbench_process_results(doc, results):
    options = _mmbench_options(doc)
    prediction = results[0].strip()
    parsed = _extract_choice(prediction, options)
    return {
        "local_mc_accuracy": {
            "correct": float(parsed == str(doc["answer"]).strip().upper()),
            "prediction": prediction,
            "parsed": parsed,
            "answer": str(doc["answer"]),
            "category": str(doc.get("category", "")),
        }
    }


def mean_correct(results):
    return sum(float(result["correct"]) for result in results) / max(1, len(results))


def seedbench_image_process_docs(dataset):
    """Select the image rows used by the published SEEDBench-image metric."""
    return dataset.filter(
        lambda data_type: str(data_type).lower() == "image",
        input_columns=["data_type"],
        desc="Selecting SEEDBench image rows",
    )


def _seedbench_upstream_utils():
    # Keep generic Regular21 contract checks independent of the full lmms-eval
    # runtime while delegating benchmark semantics to upstream at task runtime.
    from lmms_eval.tasks.seedbench import utils

    return utils


def seedbench_upstream_doc_to_visual(doc):
    return _seedbench_upstream_utils().seed_doc_to_visual(doc)


def seedbench_upstream_doc_to_text(doc):
    return _seedbench_upstream_utils().seed_doc_to_text(doc)


def seedbench_image_process_result(doc, results):
    """Keep the upstream scorer byte-for-byte and expose only its image key."""
    result = _seedbench_upstream_utils().seed_process_result(doc, results)
    return {"seed_image": result["seed_image"]}


def seedbench_upstream_aggregation(results):
    return _seedbench_upstream_utils().seed_aggregation_result(results)


def hallusion_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def hallusion_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return str(doc["question"])


def hallusion_process_results(doc, results):
    prediction = results[0].strip()
    explicit = re.search(
        r"(?:^|\b(?:answer|response)\s*(?:is|:)\s*)\b(yes|no|true|false)\b",
        prediction,
        flags=re.IGNORECASE,
    )
    parsed = explicit.group(1).lower() if explicit else ""
    answer = {"yes": "1", "true": "1", "no": "0", "false": "0"}.get(
        parsed,
        "",
    )
    record = {
        key: value
        for key, value in doc.items()
        if key != "image"
    }
    record.update(
        {
            "model_prediction": prediction,
            "answer": answer,
            "parsed": parsed,
            "correct": answer == str(doc["gt_answer"]),
        }
    )
    return {
        "local_aAcc": record,
        "local_qAcc": record,
        "local_fAcc": record,
    }


def hallusion_aacc(results):
    return sum(bool(result["correct"]) for result in results) / max(1, len(results))


def _hallusion_group_accuracy(results, keys):
    groups = {}
    for result in results:
        key = tuple(result[field] for field in keys)
        groups.setdefault(key, []).append(bool(result["correct"]))
    return sum(all(values) for values in groups.values()) / max(1, len(groups))


def hallusion_qacc(results):
    return _hallusion_group_accuracy(
        results,
        ("category", "subcategory", "set_id", "question_id"),
    )


def hallusion_facc(results):
    return _hallusion_group_accuracy(
        results,
        ("category", "subcategory", "set_id", "figure_id"),
    )
