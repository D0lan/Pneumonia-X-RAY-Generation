# Variational Autoencoder (VAE) for Chest X-ray Data

This repository contains the implementation and experimental pipeline for a **convolutional Variational Autoencoder (VAE)** trained on a chest X-ray dataset. The model is used for image reconstruction and synthetic image generation, supporting reproducible experimentation.

---

## Reproducibility Checklist and Main Scripts

To facilitate reproduction of the experiments, this checklist summarizes the key steps, dependencies, and scripts required to run the training and evaluation pipelines for the convolutional VAE.

---

## Environment Setup

- **Python version**: 3.10+ (recommended)
- **Required libraries** (install via `pip`):
  - `torch` (PyTorch)
  - `torchvision`
  - `Pillow`
  - `numpy`
- **Optional**: GPU with CUDA support for faster training

---

## Dataset Preparation

1. Download the Chest X-ray dataset and organize it as follows:

```text
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/

## Output Organization
output3/
├── reconstructions3/     # Original vs reconstructed images
│   ├── normal/
│   └── pneumonia/
├── generated3/           # Synthetic images
│   ├── normal/
│   └── pneumonia/
└── checkpoints3/         # Model checkpoints
