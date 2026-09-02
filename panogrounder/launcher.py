"""Build commands for the fixed PanoGrounder experiment presets."""

from __future__ import annotations

import shlex
from pathlib import Path

from .presets import EVAL_PRESETS, TRAIN_PRESETS

ROOT = Path(__file__).resolve().parents[1]
COGVLM = ROOT / "third_party" / "CogVLM"
TRAIN_ENTRY = COGVLM / "finetune_demo" / "finetune_cogvlm_demo.py"
EVAL_ENTRY = COGVLM / "finetune_demo" / "evaluate_cogvlm_demo_batch.py"

_METHOD_ARGS = (
    "--use_multi_view_feature", "--multi_view_feature_dim", "384",
    "--use_fusion", "--use_deep_feature",
    "--use_semantic_head", "--num_semantic_classes", "38",
    "--aux_loss_weight", "0.5",
    "--use_xyz_encoder", "--use_deep_xyz", "--use_range_dino",
    "--use_vit_interpolation", "--use_relative_range_head",
    "--use_intermediate_relative_range_head",
    "--relative_range_loss_weight", "0.5", "--use_relative_xyz_head",
    "--use_intermediate_relative_xyz_head", "--control_only",
)


def _model_args(model_path: Path, max_length: int, rank: int, alpha: int) -> list[str]:
    return [
        "--from_pretrained", str(model_path), "--max_length", str(max_length),
        "--lora_rank", str(rank), "--lora_alpha", str(alpha), "--use_lora",
        "--local_tokenizer", "lmsys/vicuna-7b-v1.5", "--version", "base",
    ]


def build_train_command(
    preset_name: str,
    data_csvs: list[Path],
    model_path: Path,
    output_dir: Path,
    *,
    hostfile: Path | None = None,
) -> list[str]:
    preset = TRAIN_PRESETS[preset_name]
    if len(data_csvs) != len(preset.datasets):
        raise ValueError(f"{preset_name} requires {len(preset.datasets)} --data CSV path(s)")
    command = ["deepspeed"]
    if hostfile is not None:
        command.extend(["--hostfile", str(hostfile)])
    command.append(str(TRAIN_ENTRY))
    command.extend(_model_args(model_path, preset.max_length, preset.lora_rank, preset.lora_alpha))
    command.extend(["--dataset_name", *preset.datasets, "--render_mode", *preset.render_modes])
    command.extend([
        "--loss_types", "ce_l1", "--lr_scale", "1.0", *_METHOD_ARGS,
        "--use_semantic_head_decay", "--use_range_head_decay",
        "--use_last_range_head_decay_only",
    ])
    command.extend(["--ratio_align_data", str(preset.alignment_ratio)])
    if preset.augment_alignment:
        command.append("--augment_align_data")
    if preset.use_emd:
        command.append("--use_emd")
    command.extend([
        "--experiment-name", preset.name, "--model-parallel-size", "1",
        "--mode", "finetune", "--train-iters", str(preset.train_iters),
        "--resume-dataloader", "--train-data", *(str(path) for path in data_csvs),
        "--valid-data", str(data_csvs[0]), "--distributed-backend", "nccl",
        "--lr-decay-style", "cosine", "--warmup", str(preset.warmup),
        "--checkpoint-activations", "--vit_checkpoint_activations",
        "--save-interval", "99999999", "--eval-interval", "100",
        "--save", str(output_dir), "--eval-iters", "50", "--eval-batch-size", "1",
        "--split", "1.", "--deepspeed_config",
        str(ROOT / "configs" / preset.deepspeed_config), "--skip-init", "--seed", "2023",
    ])
    return command


def build_eval_command(
    preset_name: str,
    data_csv: Path,
    checkpoint: Path,
    output_dir: Path,
) -> list[str]:
    preset = EVAL_PRESETS[preset_name]
    command = ["deepspeed", str(EVAL_ENTRY)]
    command.extend(_model_args(checkpoint, preset.max_length, preset.lora_rank, preset.lora_alpha))
    command.extend([
        "--dataset_name", preset.dataset, "--render_mode", preset.render_mode,
        "--eval", *_METHOD_ARGS,
        "--experiment-name", preset.name, "--model-parallel-size", "1",
        "--mode", "finetune", "--train-iters", "0", "--test-data", str(data_csv),
        "--distributed-backend", "nccl", "--strict-eval", "--eval-batch-size", "16",
        "--num_rot_eval", "4", "--output_root", str(output_dir),
        "--deepspeed_config", str(ROOT / "configs" / preset.deepspeed_config),
        "--skip-init", "--seed", "2023",
    ])
    return command


def shell_join(command: list[str]) -> str:
    return shlex.join(command)
