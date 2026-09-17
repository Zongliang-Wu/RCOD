<div align="center">

# Realism Control One-step Diffusion for Real-World Image Super-Resolution

<br>

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://zongliang-wu.github.io/RCOD-SR/)
[![arXiv](https://img.shields.io/badge/arXiv-2509.10122-b31b1b.svg)](https://arxiv.org/abs/2509.10122)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📌 Introduction

While one-step diffusion (OSD) methods significantly improve efficiency in real-world image super-resolution (Real-ISR), they typically lack flexible control mechanisms to balance fidelity and realism across diverse scenarios.

**RCOD (Realism Controlled One-step Diffusion)** addresses this challenge by introducing:
1. **Latent Domain Grouping (LDG)**: Adaptively assigns timesteps according to latent degradation severity during training, unlocking multi-tradeoff generation capability in a **single model**.
2. **Degradation-Aware Sampling (DAS)**: Aligns distillation regularization with the grouping scheme.
3. **Visual Prompt Injection Module (VPIM)**: Replaces conventional text prompts with degradation-aware visual tokens for enhanced restoration and semantic accuracy.

<div align="center">
  <img src="https://zongliang-wu.github.io/RCOD-SR/static/images/SR/diff_time_step_principle_01.png" alt="RCOD Latent Domain Grouping Overview" width="92%">
  <p align="left"><em><b>Figure 1:</b> (a) Vanilla One-Step Diffusion (OSD) learns an "average" degradation, limiting generation flexibility. (b) Proposed RCOD uses Latent Domain Grouping to adaptively associate timesteps with latent degradation degrees, allowing explicit fidelity-realism trade-off control during inference.</em></p>
</div>

<div align="center">
  <img src="https://zongliang-wu.github.io/RCOD-SR/static/images/SR/fig1.png" alt="Fidelity Realism Tradeoff" width="92%">
  <p align="left"><em><b>Figure 2:</b> RCOD provides explicit control to output images tailored for high Fidelity (Ours-Fid.), Neutral Realism (Ours-Neu.), or high Realism (Ours-Real.).</em></p>
</div>

---

## 🛠️ Installation & Environment

```bash
git clone https://github.com/Zongliang-Wu/RCOD.git
cd RCOD

# Create conda environment (Python 3.12 recommended)
conda create -n rcod python=3.12 -y
conda activate rcod

# Install dependencies
pip install -r requirements.txt
```

*(Optional: If running in mainland China, set `export USE_HF_MIRROR=1` to use the HuggingFace domestic mirror).*

---

## 📦 Checkpoints & Weights

> 🔗 Trained RCOD checkpoints are hosted on Hugging Face: [**`MMQDD/RCOD`**](https://huggingface.co/MMQDD/RCOD)

You can download all pretrained model weights automatically using our download helper script:

```bash
bash scripts/download_weights.sh
```

Or download manually into the `weights/` directory:

### 1. RCOD Trained Models (from [Hugging Face](https://huggingface.co/MMQDD/RCOD))
| Model | Base | Checkpoint File | Description |
| :--- | :--- | :--- | :--- |
| **RCOD_O** | SD 2.1 base | `weights/rcod_o.pkl` | Unified denoiser (Fidelity / Neutral / Realism) |
| **RCOD_S** | SD-Turbo | `weights/rcod_s.pkl` | Unified denoiser (Fidelity / Neutral / Realism) |
| **RCOD_S MEM** | MLP | `weights/rcod_s_mem.pkl` | Metric Estimation Module for Adaptive inference |

### 2. Base & Auxiliary Models
- **SD 2.1-base**: [stabilityai/stable-diffusion-2-1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) (for RCOD_O)
- **SD-Turbo**: [stabilityai/sd-turbo](https://huggingface.co/stabilityai/sd-turbo) (for RCOD_S)
- **de_net.pth**: Degradation estimator from [S3Diff](https://github.com/ArcticHare105/S3Diff) (included in `RCOD_S/assets/mm-realsr/de_net.pth`)
- **DAPE.pth**: Domain-Adaptive Prior Extractor from [OSEDiff / DAPE](https://github.com/cswry/OSEDiff) ([Google Drive](https://drive.google.com/file/d/1KIV6VewwO2eDC9g4Gcvgm-a0LDI7Lmwm/view?usp=drive_link))
- **ram_swin_large_14m.pth**: RAM feature extractor from [Recognize Anything](https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/ram_swin_large_14m.pth)

---

## 🚀 Quick Inference Demo

We provide lightweight sample images in `assets/samples/` for an immediate out-of-the-box demo run:

### 1. RCOD_O (OSEDiff Base)
```bash
# Run with Neutral realism (timestep=500)
bash scripts/run_rcod_o.sh 0 500

# Custom run with high fidelity (timestep=250) or high realism (timestep=750):
python RCOD_O/inference_rcod_o.py \
    -i assets/samples \
    -o results/RCOD_O_Fid \
    --osediff_path weights/rcod_o.pkl \
    --timestep 250
```

### 2. RCOD_S (S3Diff Base)
```bash
# Run with adaptive timestep selection via MEM
bash scripts/run_rcod_s.sh 0 adaptive

# Or manual timestep setting:
bash scripts/run_rcod_s.sh 0 250
```

---

## 📏 Metric Evaluation

Evaluate full IQA metrics (PSNR, SSIM, LPIPS, DISTS, NIQE, MUSIQ, MANIQA, CLIPIQA, FID) in a single command:

```bash
python evaluate_metrics.py \
    --inp_imgs results/RCOD_O_t500 \
    --gt_imgs /path/to/test_HR \
    --calc_fid
```

## 📜 Citation

If you find this work useful for your research, please cite:

```bibtex
@inproceedings{rcod2026,
  title={Realism Control One-step Diffusion for Real-World Image Super-Resolution},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year={2026}
}
```

---

## 🙏 Acknowledgements

This project is built upon and inspired by the following open-source repositories:
- [OSEDiff](https://github.com/cswry/OSEDiff): One-Step Effective Diffusion Network for Real-World Image Super-Resolution
- [S3Diff](https://github.com/ArcticHare105/S3Diff): Real-World Super-Resolution via Structural and Semantic Single-Step Diffusion

We sincerely thank the authors for sharing their excellent codebases and foundational research.

---

## 📄 License
This project is released under the [MIT License](LICENSE).
