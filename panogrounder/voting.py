"""Deterministic multi-view voting for PanoGrounder predictions."""

from __future__ import annotations

import ast
import csv
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Mapping


GROUP_COLUMNS = ("scene_id", "text_idx")
PREDICTION_COLUMNS = (*GROUP_COLUMNS, "correct_gt_id", "matched_gt_id")
METADATA_KEYS = ("scene_id", "cam_id", "text_idx")


@dataclass(frozen=True)
class VoteResult:
    scene_id: str
    text_idx: str
    correct_gt_id: int
    predicted_gt_id: int
    correct: bool
    votes: int
    unique_multiple: str = ""
    easy_hard: str = ""
    view_dep_indep: str = ""


def _as_int(value: object, default: int = -1) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _candidate_ids(row: Mapping[str, object]) -> set[int] | None:
    if "distractor_gt_id" not in row:
        return None
    raw = row.get("distractor_gt_id", "[]")
    try:
        distractors = ast.literal_eval(str(raw))
    except (SyntaxError, ValueError):
        distractors = []
    return {_as_int(row.get("correct_gt_id")), *(_as_int(value) for value in distractors)}


def _valid_bbox(value: object) -> bool:
    try:
        bbox = ast.literal_eval(value) if isinstance(value, str) else value
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return False
        x1, y1, x2, y2 = (float(item) for item in bbox)
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0:
            return False
        return max(width / height, height / width) <= 999
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError):
        return False


def vote_rows(rows: Iterable[Mapping[str, object]], dataset: str) -> list[VoteResult]:
    """Vote once per ``(scene_id, text_idx)`` query.

    ScanRefer uses majority voting. Sr3D and Nr3D weight each matched
    instance by the cube of its visible proportion. Ties are resolved by
    the smallest instance id, making the result independent of row order.
    """
    dataset_key = dataset.lower()
    if dataset_key not in {"scanrefer", "sr3d", "nr3d"}:
        raise ValueError(f"Unsupported dataset: {dataset}")
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        missing = [column for column in PREDICTION_COLUMNS if column not in row]
        if missing:
            raise ValueError(f"Missing prediction columns: {', '.join(missing)}")
        grouped[(str(row["scene_id"]), str(row["text_idx"]))].append(row)

    results: list[VoteResult] = []
    for (scene_id, text_idx), group in sorted(grouped.items()):
        correct_id = _as_int(group[0]["correct_gt_id"])
        scores: dict[int, float] = defaultdict(float)
        valid_votes = 0
        allowed: set[int] | None = None
        if dataset_key in {"sr3d", "nr3d"}:
            allowed = set()
            for row in group:
                allowed.update(_candidate_ids(row) or set())
        fallback = min(allowed) if allowed else -1
        for row in group:
            matched_id = _as_int(row.get("matched_gt_id"))
            if not _valid_bbox(row.get("pred_bbox")):
                continue
            if matched_id < 0 or (allowed is not None and matched_id not in allowed):
                continue
            weight = 1.0
            if dataset_key in {"sr3d", "nr3d"}:
                if "visibility_ratio" not in row:
                    raise ValueError(f"{dataset} voting requires a visibility_ratio column")
                weight = max(0.0, _as_float(row.get("visibility_ratio"))) ** 3
            scores[matched_id] += weight
            valid_votes += 1
        if not scores or max(scores.values()) <= 0:
            predicted_id = fallback
        else:
            predicted_id = min(scores, key=lambda item: (-scores[item], item))
        results.append(VoteResult(
            scene_id, text_idx, correct_id, predicted_id,
            predicted_id == correct_id, valid_votes,
            str(group[0].get("unique_multiple", "")),
            str(group[0].get("easy_hard", "")),
            str(group[0].get("view_dep_indep", "")),
        ))
    return results


def vote_file(
    input_csv: Path,
    output_csv: Path,
    dataset: str,
    metadata_csv: Path | None = None,
) -> list[VoteResult]:
    if input_csv.resolve() == output_csv.resolve():
        raise ValueError("Voting output must not overwrite the prediction CSV")
    with input_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if metadata_csv is not None:
        with metadata_csv.open(newline="", encoding="utf-8") as handle:
            metadata = list(csv.DictReader(handle))
        rows = merge_metadata(rows, metadata)
    results = vote_rows(rows, dataset)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[field.name for field in fields(VoteResult)]
        )
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    return results


def _metric_lines(results: list[VoteResult], attribute: str) -> list[str]:
    values = sorted({getattr(result, attribute) for result in results if getattr(result, attribute)})
    if not values:
        return []
    lines = [f"[{attribute}]"]
    for value in values:
        subset = [result for result in results if getattr(result, attribute) == value]
        correct = sum(result.correct for result in subset)
        accuracy = 100.0 * correct / len(subset)
        lines.append(f"{value}: {correct}/{len(subset)} ({accuracy:.2f}%)")
    return lines


def format_report(results: list[VoteResult], dataset: str) -> str:
    correct = sum(result.correct for result in results)
    accuracy = 100.0 * correct / len(results) if results else 0.0
    lines = [
        f"dataset: {dataset}",
        f"queries: {len(results)}",
        f"correct: {correct}",
        f"accuracy: {accuracy:.2f}%",
    ]
    if dataset.lower() == "scanrefer":
        lines.extend(_metric_lines(results, "unique_multiple"))
    else:
        lines.extend(_metric_lines(results, "easy_hard"))
        lines.extend(_metric_lines(results, "view_dep_indep"))
    return "\n".join(lines) + "\n"


def _normalize_cam(value: object) -> str:
    text = str(value).strip()
    digits = "".join(character for character in text if character.isdigit())
    return f"cam_{int(digits)}" if digits else text


def _metadata_key(row: Mapping[str, object]) -> tuple[str, str, str]:
    missing = [column for column in METADATA_KEYS if column not in row]
    if missing:
        raise ValueError(f"Missing metadata keys: {', '.join(missing)}")
    try:
        text_idx = str(int(float(str(row["text_idx"]))))
    except ValueError:
        text_idx = str(row["text_idx"]).strip()
    return str(row["scene_id"]).strip(), _normalize_cam(row["cam_id"]), text_idx


def merge_metadata(
    predictions: list[dict[str, object]],
    metadata: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Join visibility and evaluation attributes without modifying inputs."""
    metadata_by_key: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in metadata:
        key = _metadata_key(row)
        if key in metadata_by_key:
            raise ValueError(f"Duplicate visibility metadata key: {key}")
        metadata_by_key[key] = row

    copied_columns = (
        "visibility_ratio", "distractor_gt_id", "unique_multiple",
        "easy_hard", "view_dep_indep",
    )
    merged: list[dict[str, object]] = []
    for prediction in predictions:
        key = _metadata_key(prediction)
        if key not in metadata_by_key:
            raise ValueError(f"Missing visibility metadata for prediction key: {key}")
        metadata_row = metadata_by_key[key]
        merged_row = dict(prediction)
        for column in copied_columns:
            if column in metadata_row:
                merged_row[column] = metadata_row[column]
        merged.append(merged_row)
    return merged
