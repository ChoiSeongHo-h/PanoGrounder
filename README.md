# PanoGrounder: Bridging 2D and 3D with Panoramic Scene Representations for VLM-based 3D Visual Grounding

[![arXiv](https://img.shields.io/badge/arXiv-2512.20907-b31b1b.svg)](https://arxiv.org/abs/2512.20907)

This is the official repository for **PanoGrounder**.

## 📢 News
- **[Coming Soon]** The official code will be released shortly. We are currently cleaning up the codebase for public release. Stay tuned!
- **[2025-12]** Our paper is now available on [arXiv](https://arxiv.org/abs/2512.20907).

## Abstract
3D Visual Grounding (3DVG) is a critical bridge from vision-language perception to robotics, requiring both language understanding and 3D scene reasoning. Traditional supervised models leverage explicit 3D geometry but exhibit limited generalization, owing to the scarcity of 3D vision-language datasets and the limited reasoning capabilities compared to modern vision-language models (VLMs). We propose PanoGrounder, a generalizable 3DVG framework that couples multi-modal panoramic representation with pretrained 2D VLMs for strong vision-language reasoning. Panoramic renderings, augmented with 3D semantic and geometric features, serve as an intermediate representation between 2D and 3D, and offer two major benefits: (i) they can be directly fed to VLMs with minimal adaptation and (ii) they retain long-range object-to-object relations thanks to their 360-degree field of view. We devise a three-stage pipeline that places a compact set of panoramic viewpoints considering the scene layout and geometry, grounds a text query on each panoramic rendering with a VLM, and fuses per-view predictions into a single 3D bounding box via lifting. Our approach achieves state-of-the-art results on ScanRefer and Nr3D, and demonstrates superior generalization to unseen 3D datasets and text rephrasings.

## Citation
If you find our work useful in your research, please consider citing:

```bibtex
@article{jungchoi2025panogrounder,
    title={PanoGrounder: Bridging 2D and 3D with Panoramic Scene Representations for VLM-based 3D Visual Grounding},
    author={Seongmin Jung and Seongho Choi and Gunwoo Jeon and Minsu Cho and Jongwoo Lim},
    journal={arXiv preprint arXiv:2512.20907},
    year={2025}
}
```
