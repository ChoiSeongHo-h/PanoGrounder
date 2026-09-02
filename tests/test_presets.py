from pathlib import Path

import pytest

from panogrounder.launcher import build_eval_command, build_train_command
from panogrounder.presets import EVAL_PRESETS, TRAIN_PRESETS


def test_canonical_preset_count() -> None:
    assert set(TRAIN_PRESETS) == {"scanrefer", "sr3d", "nr3d", "all"}
    assert set(EVAL_PRESETS) == {
        "scanrefer", "sr3d", "nr3d", "all-scanrefer", "all-sr3d", "all-nr3d"
    }


def test_hyperparameters_are_fixed_inside_train_command() -> None:
    command = build_train_command(
        "sr3d", [Path("sr3d.csv")], Path("base-model"), Path("outputs")
    )
    assert command[0] == "deepspeed"
    assert command[command.index("--lora_rank") + 1] == "32"
    assert command[command.index("--train-iters") + 1] == "9192"
    assert "--use_emd" in command
    assert "--augment_align_data" in command


def test_all_preset_requires_three_csvs() -> None:
    with pytest.raises(ValueError, match="requires 3"):
        build_train_command("all", [Path("one.csv")], Path("model"), Path("out"))


def test_eval_rotation_is_fixed() -> None:
    command = build_eval_command(
        "all-nr3d", Path("nr3d.csv"), Path("checkpoint"), Path("outputs")
    )
    assert command[command.index("--num_rot_eval") + 1] == "4"
    assert command[command.index("--lora_rank") + 1] == "64"


def test_referit3d_eval_uses_result_render_mode() -> None:
    command = build_eval_command(
        "sr3d", Path("sr3d.csv"), Path("checkpoint"), Path("outputs")
    )
    assert command[command.index("--render_mode") + 1] == (
        "ransac_nearcam_obs_dim_multi_many_more"
    )
    assert command[command.index("--max_length") + 1] == "1600"
