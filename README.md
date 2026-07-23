# FreqMamba: Generalizable Frequency-Spatial Mamba Network for Structure-Preserving Retinal Image Enhancement

**ICIC 2026** — Session 2: Image Processing
**Online Room E — Paper 5225**
*Jialiang Liu, Xiangyang Yu, Huiyan Lin, Heng Li, Jiang Liu*

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org)
[![Mamba](https://img.shields.io/badge/Mamba-SSM-green)](https://github.com/state-spaces/mamba)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Abstract

We propose **FreqMamba**, a novel frequency-spatial Mamba network for structure-preserving retinal image enhancement. The network integrates:

- **Frequency-Spatial Mamba Block** that jointly models long-range dependencies in both frequency and spatial domains via bidirectional Mamba scanning (horizontal + vertical)
- **Learnable frequency modulation** that adaptively enhances amplitude components based on normalized frequency magnitude with learnable cutoff and sharpness parameters
- **Structure-preserving loss** combining Charbonnier, edge (Sobel), and FFT losses for perceptually faithful enhancement

Our method achieves state-of-the-art performance on multiple fundus image enhancement benchmarks while generalizing well across datasets.

## Architecture

### Frequency-Spatial Mamba Block

The core building block operates through two parallel pathways:

1. **Spatial Pathway**: Bidirectional Mamba scanning (horizontal + vertical) with learnable positional embeddings and spatial fusion
2. **Frequency Pathway**: FFT-based amplitude modulation with learnable frequency gating, cutoff, and sharpness parameters

Outputs are adaptively fused via a learnable gating parameter.

### HybridMambaUNet

A U-Net architecture with:
- Encoder: Conv + ResBlock downsampling (3 -> 32 -> 64 channels)
- Bottleneck: 3 stacked Frequency-Spatial Mamba Blocks (128 channels)
- Decoder: Upsampling with skip connections
- Residual learning: output = input + tail prediction

## Results

| Metric | FreqMamba (Ours) | U-Net | Restormer | NAFNet |
|--------|-------------------|-------|-----------|--------|
| PSNR | **SOTA** | — | — | — |
| SSIM | **SOTA** | — | — | — |

## Project Structure

```
src/
  model.py              # HybridMambaUNet, FrequencySpatialMambaBlock, baselines
  dataset.py            # FundusDataset with synthetic degradation
  loss.py               # FreqMambaLoss (Charbonnier + Edge + FFT)
  utils.py              # Degradation model utilities
train.py                # Training script (supports multiple models)
evaluate.py             # Full evaluation suite (PSNR, SSIM, LPIPS, AUC, etc.)
generate_degraded.py    # Generate degraded image pairs
find_best_epoch.py      # Model checkpoint selection
figures.py              # Paper figure generation
compose_figures.py      # Figure composition
plot_degradation_defense.py  # Degradation defense analysis
plot_freq_mask.py       # Frequency mask visualization
train_seg.py            # Auxiliary segmentation training
splits/                 # Train/val/test split files
checkpoints/            # [gitignored] Trained model weights
data/                   # [gitignored] Fundus image datasets
results/                # [gitignored] Evaluation outputs
visualizations/         # [gitignored] Generated visualizations
paper_figures/          # [gitignored] Paper figures
```

## Quick Start

### Requirements

```bash
pip install -r requirements.txt
```

### Data Preparation

Place fundus images in `data/` with the following structure:
```
data/
  aptos2019_images/        # Training images
  test/
    drive_images/          # DRIVE test images
    drive_masks/           # DRIVE test masks
  degraded/                # Pre-degraded images
```

### Training

```bash
# Train FreqMamba
python train.py --model freqmamba --epochs 100 --lr 1e-4

# Train baseline (U-Net)
python train.py --model unet --epochs 100 --lr 1e-4
```

### Evaluation

```bash
python evaluate.py --checkpoint checkpoints/freqmamba/best.pth
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{liu2026freqmamba,
  title={FreqMamba: Generalizable Frequency-Spatial Mamba Network for Structure-Preserving Retinal Image Enhancement},
  author={Liu, Jialiang and Yu, Xiangyang and Lin, Huiyan and Li, Heng and Liu, Jiang},
  booktitle={International Conference on Intelligent Computing (ICIC)},
  year={2026}
}
```

## License

MIT
