"""Compute per-view visibility from instance-ID panoramas."""

from __future__ import annotations

import ast
import csv
from pathlib import Path

import numpy as np
from PIL import Image


REQUIRED_COLUMNS = ("scene_id", "cam_id", "text_idx", "correct_gt_id")


def parse_instance_ids(value: object) -> set[int]:
    if value is None or str(value).strip() in {"", "nan"}:
        return set()
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"Invalid instance IDs: {value}") from error
    if isinstance(parsed, (int, float)):
        return {int(parsed)}
    if isinstance(parsed, (list, tuple, set)):
        return {int(item) for item in parsed}
    raise ValueError(f"Invalid instance IDs: {value}")


def compute_visibility_rows(
    rows: list[dict[str, str]],
    instance_dir: Path,
    render_mode: str,
) -> list[dict[str, object]]:
    """Add ``visibility_ratio`` to annotation rows."""
    cache: dict[Path, set[int]] = {}
    output: list[dict[str, object]] = []
    for row in rows:
        missing = [column for column in REQUIRED_COLUMNS if column not in row]
        if missing:
            raise ValueError(f"Missing annotation columns: {', '.join(missing)}")
        target_ids = parse_instance_ids(row["correct_gt_id"])
        target_ids.update(parse_instance_ids(row.get("distractor_gt_id")))
        if not target_ids:
            raise ValueError(
                f"No candidate instances for {row['scene_id']} / {row['text_idx']}"
            )
        image_path = instance_dir / f"{row['scene_id']}-{render_mode}-{row['cam_id']}.png"
        if image_path not in cache:
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing instance-ID panorama: {image_path}")
            with Image.open(image_path) as image:
                cache[image_path] = {int(value) for value in np.unique(np.asarray(image))}
        visible = target_ids.intersection(cache[image_path])
        output.append({**row, "visibility_ratio": len(visible) / len(target_ids)})
    return output


def compute_visibility_file(
    annotations_csv: Path,
    instance_dir: Path,
    output_csv: Path,
    render_mode: str,
) -> None:
    if annotations_csv.resolve() == output_csv.resolve():
        raise ValueError("Visibility output must not overwrite the annotation CSV")
    with annotations_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [
            field for field in (reader.fieldnames or []) if field != "visibility_ratio"
        ]
        rows = list(reader)
    output_rows = compute_visibility_rows(rows, instance_dir, render_mode)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fieldnames, "visibility_ratio"])
        writer.writeheader()
        writer.writerows(output_rows)
