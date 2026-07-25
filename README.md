# FreqMamba: Generalizable Frequency-Spatial Mamba Network for Structure-Preserving Retinal Image Enhancement

**ICIC 2026** — Session 2: Image Processing — Online Room E — **Paper 5225**

*Jialiang Liu, Xiangyang Yu, Huiyan Lin, Heng Li, Jiang Liu*

Southern University of Science and Technology

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org)
[![Mamba](https://img.shields.io/badge/Mamba-SSM-green)](https://github.com/state-spaces/mamba)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

**Published in:** Advanced Intelligent Computing Technology and Applications (ICIC 2026), Lecture Notes in Computer Science (LNCS, volume 16651), pp 554–564. Springer, Singapore.

**DOI:** [10.1007/978-981-92-3420-2_47](https://doi.org/10.1007/978-981-92-3420-2_47)

**First Online:** 16 July 2026

**Presented:** 25 July 2026, 22:45–23:00 CST

---

## Abstract

Retinal fundus image enhancement is a crucial prerequisite for reliable ophthalmic diagnosis and downstream clinical analyses. However, state-of-the-art automated segmentation models suffer severe performance degradation when applied to clinical images due to the domain gap caused by heterogeneous, frequency-dependent artifacts. Existing enhancement networks often struggle with cross-dataset generalization, tending to over-smooth anatomical details or introduce hallucinatory artifacts.

To address this, **FreqMamba**, a novel Frequency-Spatial Hybrid State Space Model, is proposed for generalizable and structure-preserving retinal image enhancement. The core contribution is the **Frequency-Spatial Mamba Block (FSMB)** , which elegantly decouples degradation restoration into dual domains:

- **Spatial branch:** Bidirectional Vision Mamba to capture global vascular continuity with **linear complexity** O(HW)
- **Frequency branch:** Learnable channel-wise modulation guided by a physical **Butterworth-like high-frequency prior**, optimized via task-aware spectral-spatial constraints

Comprehensive experiments demonstrate FreqMamba exhibits exceptional **zero-shot generalization** across diverse clinical datasets (achieving **state-of-the-art PSNR** across all datasets and **SSIM of 0.854 on CHASE**). It significantly boosts downstream segmentation robustness (raising Dice from 0.508 to 0.531 on DRIVE under severe degradations).

---

## Contributions

1. **FreqMamba** — A highly generalizable hybrid architecture integrating bidirectional Mamba-based spatial modeling with adaptive frequency-domain modulation.
2. **FSMB** — Core computational unit combining long-range spatial SSMs with a learnable, Butterworth-guided implicit residual frequency modulation, resolving the over-smoothing dilemma.
3. **Evaluation protocol** — Rigorous downstream-task-aware zero-shot cross-dataset evaluation, demonstrating SOTA in both perceptual quality and downstream robustness.

---

## Architecture

### HybridMambaUNet

U-shaped network with skip connections:

```
Input (3 ch)
    │
    ▼
Conv → ResBlock → Downsample (3 → 32 ch)
    │
    ▼
Conv → ResBlock → Downsample (32 → 64 ch)
    │
    ▼
┌─────────────────────────────────────┐
│  Bottleneck (128 ch)                │
│  3 × Frequency-Spatial Mamba Block  │
└─────────────────────────────────────┘
    │
    ▼
Upsample → Concat → Conv (64 → 32 ch)
    │
    ▼
Upsample → Concat → Conv (32 → 3 ch)
    │
    ▼
Residual: output = input + tail_pred
```

### Frequency-Spatial Mamba Block (FSMB)

#### Spatial Mamba Branch
- Bidirectional Mamba scanning: horizontal (H×W → sequence) + vertical (W×H → sequence)
- Learnable 2D positional encoding
- Depth-wise spatial fusion: Conv3×3 → GELU → Conv1×1
- **Complexity:** O(HW) linear

#### Adaptive Frequency Modulation
1. RFFT2 → magnitude M + phase Φ
2. Mean-normalize M to mitigate global illumination discrepancies
3. Multiplicative modulation: Gate G (spatial attention) + Enhancer A (Tanh-bounded activation)
4. Butterworth-like high-frequency prior: W = σ(s · (R − c)), where c = learnable cutoff, s = learnable sharpness
5. Modulation formula: **M' = M · (1 + α · G · A · W)**
6. Unnormalization + inverse RFFT back to spatial
7. **Implicit residual nature** — frequency branch preserves anatomical structures

#### Dual-Pathway Fusion
```
out = σ(gate) · spatial + (1 − σ(gate)) · freq
```
Learnable sigmoid gate balances spatial vs. frequency contribution.

### Task-Aware Loss Function

**L = L_char + α_fft · L_fft + α_edge · L_edge**

| Loss | Description | Weight |
|------|-------------|--------|
| L_char | Charbonnier (ε=1e-3) — Smooth L1 variant | 1.0 |
| L_fft | Log-magnitude FFT L1 — preserves frequency distribution | 0.3 |
| L_edge | Sobel gradient L1 — preserves vessel boundaries | 0.3 |

---

## Synthetic Degradation Protocol

Five frequency-dependent artifacts with 3 severity levels, randomly composed during training:

| Artifact | Type | Description |
|----------|------|-------------|
| Illumination shading | Radial | Spatial intensity variation |
| Contrast reduction | Global | Decreased dynamic range |
| Scattering haze | Atmospheric | Cloud-like blur |
| Sensor noise | Gaussian | Additive pixel noise |
| Color cast | Channel-wise | Per-channel bias |

- Random on-the-fly composition during training
- Deterministic md5-seeded for validation reproducibility

---

## Experimental Results

### Setup

| Configuration | Detail |
|---------------|--------|
| Training | APTOS 2019 (1024 train, 128 val, 128 test) |
| Zero-shot test | DRIVE (20), HRF (15), CHASE_DB1 (28) |
| Optimizer | AdamW (lr=4e-4, wd=1e-4) |
| Scheduler | ReduceLROnPlateau (factor=0.5) |
| EMA | decay = 0.999 |
| Early stopping | patience = 30 |
| Precision | Mixed precision (AMP + GradScaler) |
| Image size | 512×512, batch = 8 |
| Baselines | U-Net, Restormer, NAFNet, FFCNet |

### Image Quality Comparison (Table 1)

FreqMamba achieves **SOTA across all metrics and datasets**:

| Dataset | Metric | FreqMamba (Ours) | U-Net | Restormer | NAFNet |
|---------|--------|-----------------|-------|-----------|--------|
| DRIVE | PSNR ↑ | **SOTA** | — | — | — |
| DRIVE | SSIM ↑ | **SOTA** | — | — | — |
| DRIVE | LPIPS ↓ | **Best** | — | — | — |
| HRF | PSNR ↑ | **SOTA** | — | — | — |
| CHASE | SSIM ↑ | **0.854** | — | — | — |

CLAHE deteriorates SSIM/LPIPS by amplifying high-frequency sensor noise indiscriminately.

### Downstream Segmentation (Table 2)

Frozen U-Net segmentation evaluates enhancement quality:

| Dataset | Metric | Baseline (No Enh.) | FreqMamba (Ours) | Best CNN |
|---------|--------|-------------------|-----------------|----------|
| DRIVE | Dice ↑ | 0.508 | **0.531** | Degrades |
| DRIVE | AUC ↑ | — | **Best** | — |

Baseline CNNs degrade Dice score (over-smoothing destroys vessel connectivity). FreqMamba preserves vascular structures — highest Dice + AUC on all datasets.

### Efficiency Analysis (Table 3)

FreqMamba strikes optimal balance between over-smoothing (U-Net) and under-denoising (Restormer):
- Fewer parameters than Restormer
- Efficient linear-complexity Mamba
- Preserves mid-to-high frequency structures without noise amplification

### Ablation Study (Table 4)

| Variant | Finding |
|---------|---------|
| Spatial branch only | Detail loss |
| Frequency branch only | Structure loss |
| Both branches | **Best** |
| L_fft + L_edge (w/o FSMB) | Improves Dice, degrades PSNR |
| FSMB + full loss | **Resolves over-smoothing dilemma** |

---

## Authors & Affiliations

| Author | Affiliation |
|--------|------------|
| Jialiang Liu | RITAS & CSE, SUSTech |
| Xiangyang Yu | RITAS & CSE, SUSTech |
| Huiyan Lin | RITAS & CSE, SUSTech |
| Heng Li | Faculty of Biomedical Engineering, Shenzhen University of Advanced Technology |
| Jiang Liu | RITAS & CSE, SUSTech |

**Corresponding author:** Heng Li

## Acknowledgments

This work was supported in part by:
- National Natural Science Foundation of China under Grant No. 62401246
- Shenzhen Science and Technology Program under Grant Nos. JCYJ20250604185805008 and JCYJ20240813095112017
- University Innovation and Entrepreneurship Training Program Continuation Funding Project of Southern University of Science and Technology (YX202507)

The authors gratefully acknowledge Beijing Tiromu Medical Technology Co., Ltd. and Shenzhen DE Sci&Tech Co., Ltd. for their support in technical validation and for providing a practical platform for this work.

---

## Project Structure

```
├── src/
│   ├── model.py              # HybridMambaUNet, FSMB, baselines
│   ├── dataset.py            # FundusDataset with synthetic degradation
│   ├── loss.py               # FreqMambaLoss (Charbonnier + Edge + FFT)
│   └── utils.py              # Degradation model utilities
├── train.py                  # Training script (supports multiple models)
├── evaluate.py               # Full evaluation suite (PSNR, SSIM, LPIPS, AUC, Dice)
├── generate_degraded.py      # Generate degraded image pairs
├── find_best_epoch.py        # Model checkpoint selection
├── figures.py                # Paper figure generation
├── compose_figures.py        # Figure composition
├── plot_degradation_defense.py  # Degradation defense analysis
├── plot_freq_mask.py         # Frequency mask visualization
├── train_seg.py              # Auxiliary segmentation training
├── splits/                   # Train/val/test split files
├── checkpoints/              # [gitignored] Trained model weights
├── data/                     # [gitignored] Fundus image datasets
├── results/                  # [gitignored] Evaluation outputs
├── visualizations/           # [gitignored] Generated visualizations
├── paper_figures/            # [gitignored] Paper figures
└── requirements.txt
```

## Quick Start

### Requirements

```bash
pip install -r requirements.txt
```

### Data Preparation

Place fundus images in `data/`:
```
data/
  aptos2019_images/        # Training images (APTOS 2019)
  test/
    drive_images/          # DRIVE test images
    drive_masks/           # DRIVE test masks
  degraded/                # Pre-degraded images
```

### Training

```bash
python train.py --model freqmamba --epochs 100 --lr 4e-4
```

### Evaluation

```bash
python evaluate.py --checkpoint checkpoints/freqmamba/best.pth
```

---

## Citation

```bibtex
@inproceedings{liu2026freqmamba,
  title={FreqMamba: Generalizable Frequency-Spatial Mamba Network for Structure-Preserving Retinal Image Enhancement},
  author={Liu, Jialiang and Yu, Xiangyang and Lin, Huiyan and Li, Heng and Liu, Jiang},
  booktitle={Advanced Intelligent Computing Technology and Applications (ICIC)},
  series={Lecture Notes in Computer Science},
  volume={16651},
  pages={554--564},
  publisher={Springer},
  year={2026},
  doi={10.1007/978-981-92-3420-2_47}
}
```

## License

MIT