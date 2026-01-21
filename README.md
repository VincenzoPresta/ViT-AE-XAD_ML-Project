# ViT-AE-XAD: Vision Transformer Autoencoder for Explainable Anomaly Detection

This repository contains the implementation and experimental analysis of **ViT-AE-XAD**, a Machine Learning project that investigates the integration of **Vision Transformers (ViT)** into the **AE-XAD (AutoEncoder-based eXplainable Anomaly Detection)** framework for industrial visual anomaly detection.

The project is developed as part of an academic Machine Learning course and focuses on **faithfully re-engineering the original AE-XAD pipeline** while replacing the CNN-based encoder with a Vision Transformer, in order to study representational alignment, localization behavior, and performance trade-offs.

---

## 📌 Project Motivation

AE-XAD is a reconstruction-based anomaly detection method designed to produce **explainable anomaly maps** by exploiting feature reconstruction errors.  
However, the original framework relies on **CNN encoders**, whose inductive biases differ significantly from those of **Vision Transformers**, which learn **global, patch-based representations**.

This project explores the following research question:

> *Can a Vision Transformer be integrated into the AE-XAD framework without breaking its explainability and localization properties?*

---

## 🧠 Method Overview

The proposed **ViT-AE-XAD** pipeline preserves the original AE-XAD structure while modifying the encoder:

- **Encoder**: Vision Transformer (ViT-B/16)
- **Decoder**: Original AE-XAD convolutional decoder (unchanged)
- **Training Objective**:
  - Reconstruction loss
  - XAD-specific anomaly scoring
- **Evaluation**:
  - Image-level AUROC
  - Pixel-level AUROC
  - PRO score
  - Qualitative anomaly heatmaps

The project explicitly **does not redesign the loss or detection logic**, in order to isolate the effect of the encoder architecture.

---

## 🔍 Key Experiments

The analysis includes:

- **Frozen ViT encoder vs. trainable ViT**
- **LayerNorm-only fine-tuning**
- **Effect of ViT global representations on anomaly localization**
- **Comparison with CNN-based AE-XAD**
- **Impact on thin and small-scale defects**

Experiments are conducted on the **MVTec AD dataset**, following the original AE-XAD evaluation protocol.

---

## 📊 Main Findings

- Vision Transformers tend to produce **more diffuse and less localized anomaly maps**
- End-to-end fine-tuning of ViT **does not consistently improve performance**
- The AE-XAD framework appears **implicitly aligned with convolutional inductive biases**
- A single global threshold based on local reconstruction statistics becomes less effective with ViT features

These results highlight a **representation–decision mismatch** when global ViT features are paired with local, pixel-wise anomaly scoring.
