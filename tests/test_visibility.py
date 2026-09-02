from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from panogrounder.visibility import compute_visibility_rows, parse_instance_ids


def test_parse_instance_ids() -> None:
    assert parse_instance_ids("7") == {7}
    assert parse_instance_ids("[2, 3]") == {2, 3}


def test_visibility_is_fraction_of_candidate_instances(tmp_path: Path) -> None:
    instance_ids = np.array([[0, 2], [2, 9]], dtype=np.uint16)
    Image.fromarray(instance_ids).save(
        tmp_path / "scene0000_00-back_culled-cam_0.png"
    )
    rows = [{
        "scene_id": "scene0000_00",
        "cam_id": "cam_0",
        "text_idx": "1",
        "correct_gt_id": "2",
        "distractor_gt_id": "[3]",
    }]
    output = compute_visibility_rows(rows, tmp_path, "back_culled")
    assert output[0]["visibility_ratio"] == 0.5
