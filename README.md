# Pneumonia-X-RAY-Generation and Classification

This repository contains a unified framework for generating synthetic chest X-ray images and evaluating their effectiveness for Pneumonia classification. The project combines **generative modeling** (GANs and VAEs) with a **CNN-based classifier** to study data quality, variability, and cross-domain generalization between real and synthetic medical images.

---

## Project Overview

- **GANs (StyleGAN):** Learn high-fidelity image distributions to generate realistic synthetic chest X-rays.
- **VAEs:** Generate and reconstruct chest X-rays with controlled latent representations, used to study the impact of lower-quality synthetic data.
- **CNN Classifier:** A lightweight CNN used as an evaluation model to quantify how well synthetic images preserve discriminative features relevant to Pneumonia detection.
