"""Canonical experiment presets used for the paper release.

The model hyperparameters intentionally do not form part of the public CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainPreset:
    name: str
    datasets: tuple[str, ...]
    render_modes: tuple[str, ...]
    train_iters: int
    max_length: int
    lora_rank: int
    lora_alpha: int
    alignment_ratio: float
    warmup: float
    augment_alignment: bool
    use_emd: bool
    gpus: int
    deepspeed_config: str


@dataclass(frozen=True)
class EvalPreset:
    name: str
    dataset: str
    train_preset: str
    render_mode: str
    max_length: int
    lora_rank: int
    lora_alpha: int
    deepspeed_config: str


TRAIN_PRESETS = {
    "scanrefer": TrainPreset(
        "scanrefer", ("ScanRefer",), ("back_culled",), 11490, 1500,
        64, 64, 0.5, 0.02, False, False, 4,
        "deepspeed_bf16_batch8.json",
    ),
    "sr3d": TrainPreset(
        "sr3d", ("Sr3D",), ("back_culled",),
        9192, 1600, 32, 32, 0.5, 0.02, True, True, 4,
        "deepspeed_bf16_batch8.json",
    ),
    "nr3d": TrainPreset(
        "nr3d", ("Nr3D",), ("back_culled",),
        7152, 1600, 32, 32, 0.5, 0.02, True, True, 4,
        "deepspeed_bf16_batch8.json",
    ),
    "all": TrainPreset(
        "all", ("ScanRefer", "Nr3D", "Sr3D"),
        ("back_culled", "back_culled", "back_culled"),
        31153, 1600, 64, 64, 0.4, 0.01, True, True, 8,
        "deepspeed_bf16_batch8_8gpu.json",
    ),
}

EVAL_PRESETS = {
    "scanrefer": EvalPreset("scanrefer", "ScanRefer", "scanrefer", "back_culled", 1500, 64, 64, "deepspeed_bf16_batch8.json"),
    "sr3d": EvalPreset("sr3d", "Sr3D", "sr3d", "ransac_nearcam_obs_dim_multi_many_more", 1600, 32, 32, "deepspeed_bf16_eval16.json"),
    "nr3d": EvalPreset("nr3d", "Nr3D", "nr3d", "ransac_nearcam_obs_dim_multi_many_more", 1600, 32, 32, "deepspeed_bf16_eval16.json"),
    "all-scanrefer": EvalPreset("all-scanrefer", "ScanRefer", "all", "back_culled", 1500, 64, 64, "deepspeed_bf16_batch8.json"),
    "all-sr3d": EvalPreset("all-sr3d", "Sr3D", "all", "ransac_nearcam_obs_dim_multi_many_more", 1600, 64, 64, "deepspeed_bf16_batch8.json"),
    "all-nr3d": EvalPreset("all-nr3d", "Nr3D", "all", "ransac_nearcam_obs_dim_multi_many_more", 1600, 64, 64, "deepspeed_bf16_batch8.json"),
}


def require_paths(paths: list[Path], label: str) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(f"Missing {label}:\n  - {joined}")
