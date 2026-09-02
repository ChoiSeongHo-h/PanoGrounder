# PanoGrounder

Official code for **PanoGrounder: Bridging 2D and 3D with Panoramic Scene
Representations for VLM-based 3D Visual Grounding** (ECCV 2026).

This release contains the CogVLM-based training and inference code and the
multi-view voting used for ScanRefer, Sr3D, and Nr3D.

## Release scope

Available:

- four fixed training presets: ScanRefer, Sr3D, Nr3D, and their union;
- six fixed evaluation presets;
- majority voting for ScanRefer;
- visibility-weighted voting for Sr3D and Nr3D;
- the PanoGrounder extensions on CogVLM.

Coming soon:

- scene construction and panoramic rendering;
- ScanRefer multi-view aggregation;
- PanoGrounder checkpoints.

## Installation

PanoGrounder targets Python 3.10 and a CUDA-enabled PyTorch environment. Install
PyTorch for the CUDA version on your system, then install this repository:

```bash
python -m pip install -e '.[train]'
```

The base model is `cogvlm-grounding-generalist-v1.1`. Model parameters are not
distributed in this repository and remain subject to the CogVLM model license.

## Training

The public interface intentionally exposes only a preset and run-specific
paths. Model and optimization hyperparameters are fixed by the preset.

```bash
panogrounder train scanrefer --data /path/to/scanrefer_train.csv \
  --model-path cogvlm-grounding-generalist-v1.1 --output-dir checkpoints/scanrefer --execute

panogrounder train sr3d --data /path/to/sr3d_train.csv \
  --model-path cogvlm-grounding-generalist-v1.1 --output-dir checkpoints/sr3d --execute

panogrounder train nr3d --data /path/to/nr3d_train.csv \
  --model-path cogvlm-grounding-generalist-v1.1 --output-dir checkpoints/nr3d --execute

panogrounder train all \
  --data /path/to/scanrefer_train.csv \
  --data /path/to/nr3d_train.csv \
  --data /path/to/sr3d_train.csv \
  --model-path cogvlm-grounding-generalist-v1.1 --output-dir checkpoints/all --execute
```

Without `--execute`, the command prints the exact DeepSpeed invocation. The
Slurm wrappers in `scripts/slurm` call the same interface.

## Evaluation

The evaluation presets are `scanrefer`, `sr3d`, `nr3d`, `all-scanrefer`,
`all-sr3d`, and `all-nr3d`.

```bash
panogrounder eval all-nr3d --data /path/to/nr3d_test.csv \
  --checkpoint /path/to/checkpoint --output-dir outputs --execute
```

All evaluation presets use four panorama rotations.

## Voting

```bash
panogrounder vote ScanRefer --input predictions.csv --output votes.csv

panogrounder visibility --annotations annotations.csv \
  --instance-dir /path/to/instance_ids --render-mode back_culled \
  --output visibility.csv

panogrounder vote Sr3D --input predictions.csv \
  --metadata visibility.csv --output votes.csv

panogrounder vote Nr3D --input predictions.csv \
  --metadata visibility.csv --output votes.csv
```

The prediction CSV must contain `scene_id`, `text_idx`, `correct_gt_id`,
`matched_gt_id`, and `pred_bbox` (`[x1, y1, x2, y2]`). Sr3D and Nr3D
additionally require `visibility_ratio` and may provide `distractor_gt_id`.
ScanRefer uses majority voting, while Sr3D and Nr3D use visibility-weighted
voting.

## Third-party code

CogVLM is vendored under `third_party/CogVLM` from upstream commit
`14d1141b44ca0ad0439539d37d2023814f9fd100`. See `THIRD_PARTY.md`, `NOTICE`,
and the licenses in that directory.

## Citation

```bibtex
@inproceedings{jungchoi2026panogrounder,
  title={PanoGrounder: Bridging 2D and 3D with Panoramic Scene Representations for VLM-based 3D Visual Grounding},
  author={Seongmin Jung and Seongho Choi and Gunwoo Jeon and Minsu Cho and Jongwoo Lim},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2026}
}
```

## License

PanoGrounder source code is released under Apache-2.0. Third-party code and
model parameters retain their respective licenses.
