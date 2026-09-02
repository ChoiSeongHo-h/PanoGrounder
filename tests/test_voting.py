from __future__ import annotations

import csv
from pathlib import Path

import pytest

from panogrounder.voting import format_report, merge_metadata, vote_file, vote_rows


def _row(matched: int, visibility: float = 1.0) -> dict[str, object]:
    return {
        "scene_id": "scene0000_00",
        "text_idx": "7",
        "correct_gt_id": "2",
        "matched_gt_id": str(matched),
        "distractor_gt_id": "[3]",
        "visibility_ratio": str(visibility),
        "pred_bbox": "(0, 0, 10, 10)",
    }


def test_scanrefer_uses_majority_vote() -> None:
    result = vote_rows([_row(2), _row(3), _row(2)], "ScanRefer")[0]
    assert result.predicted_gt_id == 2
    assert result.correct


def test_referit3d_cubes_visibility() -> None:
    rows = [_row(2, 0.6), _row(2, 0.6), _row(3, 0.8)]
    result = vote_rows(rows, "Sr3D")[0]
    assert result.predicted_gt_id == 3  # 0.8^3 > 2 * 0.6^3


def test_ties_use_smallest_instance_id() -> None:
    result = vote_rows([_row(3, 1), _row(2, 1)], "Nr3D")[0]
    assert result.predicted_gt_id == 2


def test_referit3d_requires_visibility() -> None:
    row = _row(2)
    del row["visibility_ratio"]
    with pytest.raises(ValueError, match="visibility"):
        vote_rows([row], "Sr3D")


def test_voting_never_overwrites_input(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_row(2))
        writer.writeheader()
        writer.writerow(_row(2))
    with pytest.raises(ValueError, match="must not overwrite"):
        vote_file(path, path, "ScanRefer")


def test_report_includes_dataset_breakdowns() -> None:
    rows = [_row(2), _row(3)]
    rows[0]["easy_hard"] = "easy"
    rows[1]["easy_hard"] = "easy"
    report = format_report(vote_rows(rows, "Nr3D"), "Nr3D")
    assert "accuracy: 100.00%" in report
    assert "[easy_hard]" in report


def test_visibility_metadata_is_joined_by_view() -> None:
    prediction = {**_row(2), "cam_id": "cam_0"}
    prediction.pop("visibility_ratio")
    metadata = {
        "scene_id": "scene0000_00",
        "cam_id": "0",
        "text_idx": "7.0",
        "visibility_ratio": "0.75",
        "distractor_gt_id": "[3]",
    }
    merged = merge_metadata([prediction], [metadata])
    assert merged[0]["visibility_ratio"] == "0.75"


def test_invalid_boxes_do_not_vote() -> None:
    invalid = _row(3, 1.0)
    invalid["pred_bbox"] = "(-1, -1, -1, -1)"
    result = vote_rows([_row(2, 0.5), invalid], "Sr3D")[0]
    assert result.predicted_gt_id == 2
