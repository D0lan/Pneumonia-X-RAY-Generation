# CNN-Based Chest X-ray Classification

This repository contains a lightweight Convolutional Neural Network (CNN) framework used to evaluate the effectiveness of synthetic chest X-ray images generated using Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs) for Pneumonia classification.

The CNN is **not used for image generation**; instead, it serves as an evaluation model to assess data quality and cross-domain generalization between real and synthetic datasets.

---

## Overview

The project investigates whether synthetic chest X-ray images can preserve clinically relevant features for downstream classification tasks. Models are trained and evaluated across multiple data domains to quantify generalization performance.

**Classification task:**  
- Binary classification: **NORMAL vs. PNEUMONIA**

**Image format:**  
- Grayscale, resized to **128 × 128**

---

## Datasets

Four datasets are used:

- **Original**: Real chest X-ray images (reference distribution)
- **GANs**: Synthetic images generated using GANs
- **VAEs**: Synthetic images generated using VAEs (smaller and lower-quality)
- **Mixed**: Combined training set of Original, GANs, and VAEs

All datasets use an **80/20 train–test split**, except the Mixed dataset, which is evaluated on the individual test sets of the other datasets.

---

## Model Architecture

- Lightweight CNN with:
  - 3 convolutional blocks (Conv → ReLU → 2×2 MaxPooling)
  - Fully connected output layer
- No additional regularization (to ensure performance reflects data quality rather than model capacity)

---

## Training Configuration

- **Loss:** Categorical cross-entropy  
- **Optimizer:** Adam  
- **Learning rate:** 1e-3  
- **Batch size:** 16  
- **Epochs:** 5  
- **Seed:** Fixed for reproducibility  

---

## Evaluation Strategy

Four training/testing scenarios are evaluated:

1. Train on Original → Test on all datasets  
2. Train on GANs → Test on all datasets  
3. Train on VAEs → Test on all datasets  
4. Train on Mixed → Test on all datasets  

---

## Metrics

Performance is evaluated using:
- Accuracy
- Precision
- Recall (Sensitivity)
- F1-score
- Confusion Matrices

These metrics assess both overall performance and cross-domain generalization behavior.

---

## Key Findings

- GAN-generated images generalize well to real data and support effective classification.
- VAEs-generated data are limited by image quality and dataset size.
- Mixed training improves robustness and cross-domain performance.

---

## Usage

This repository is intended for:
- Evaluating synthetic medical image quality
- Studying cross-domain generalization
- Research and academic experimentation

---

## Disclaimer

This project is for **research purposes only** and is **not intended for clinical diagnosis or deployment**.
