# MedVision 3D
### AI-Powered 3D Medical Image Segmentation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-%23EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Domain: Medical Imaging](https://img.shields.io/badge/Domain-Medical%20Imaging-0077b6.svg)](#)
[![Framework: nnU-Net v2](https://img.shields.io/badge/Framework-nnU--Net%20v2-48cae4.svg)](https://github.com/MIC-DKFZ/nnUNet)

---

## 🌟 Introduction

**MedVision 3D** is a medical imaging segmentation project maintained and customized by **Amen Allah Ben Abdrabah**, focusing on state-of-the-art AI-powered analysis of 2D and 3D medical images. Built upon the robust foundation of the **nnU-Net v2** framework, MedVision 3D streamlines volumetric biomedical image segmentation through adaptive deep learning pipelines, automated hyperparameter configuration, and standardized preclinical experimentation workflows.

---

## 🔍 Project Overview

Accurate, automated segmentation of anatomical structures, organs, and pathological lesions is a critical cornerstone of modern clinical radiology, radiation oncology, surgical planning, and computational biomedical discovery. However, volumetric scans (such as high-resolution CT and MRI) present unique computational challenges: wide variance in voxel spacing, anisotropic resolutions, varying image modalities, heterogeneous slice thicknesses, and extreme class imbalance.

**MedVision 3D** tackles these hurdles by providing an end-to-end deep learning framework that automatically tailors its entire data processing and modeling lifecycle to any given medical image dataset. By removing manual heuristic tuning, the platform enables researchers to transition from raw volumetric scans to high-fidelity segmentations with minimal human intervention and maximum reproducibility.

---

## 🎯 Main Objectives

- **Automated Medical Image Analysis**: Eliminate manual guesswork in network design, patch extraction, and data normalization through automated dataset fingerprinting.
- **3D Volumetric Segmentation**: Deliver native 3D convolutional models (including full-resolution 3D U-Nets and cascading architectures) capable of processing large-scale spatial contexts.
- **Organ & Tumor Delineation**: Achieve high-precision boundaries on complex multi-organ datasets and small, low-contrast oncological targets (e.g., liver lesions, brain tumors, lung nodules).
- **Clinical Workflow Support**: Accelerate translational biomedical research by offering high-throughput inference tools, multi-GPU scalability, and modular configuration presets.

---

## ✨ Features

- **Automated Tumor Segmentation**: Specialized handling for multi-region and low-contrast oncological structures across varied anatomical regimes.
- **Organ Segmentation**: High-accuracy multi-class anatomical delineation for organs at risk (OARs), abdominal structures, musculoskeletal regions, and neuroanatomy.
- **Native 3D Medical Image Segmentation**: End-to-end support for 2D, 3D full-resolution, 3D low-resolution, and 3D hierarchical cascades.
- **Broad Medical Imaging Format Support**: Native integration with NIfTI (`.nii.gz`), DICOM (via conversion scripts), TIFF, PNG, and 2D/3D biomedical image formats.
- **AI-Based Image Processing**: Modality-specific intensity normalization (CT z-score clipping, MRI percentile clipping), spatial resampling with spline interpolation, and extensive on-the-fly spatial and color data augmentations.
- **Self-Configuring Pipeline**: Dynamic estimation of optimal patch sizes, batch sizes, network depth, pooling strides, and receptive fields matched to available GPU VRAM.
- **Multi-GPU Training Support**: Scalable model training with PyTorch Distributed Data Parallel (DDP) for single-node and multi-node clusters.
- **Comprehensive Preprocessing & Postprocessing**: Automated foreground cropping, target spacing resampling, multi-threaded caching, connected-component analysis, and automated ensembling.

---

## 🛠️ Technologies Used

| Technology | Role & Category | Purpose / Description |
| :--- | :--- | :--- |
| **Python** (>= 3.10) | Core Language | Primary implementation language for CLI tools, scripting, and pipeline automation |
| **PyTorch** (>= 2.1.2) | Deep Learning Engine | GPU-accelerated tensor computations, neural network modules, and backpropagation |
| **SimpleITK** (>= 2.2.1) | Medical Image Processing | Spatial transformations, coordinate system alignment, and medical I/O |
| **nibabel** | Neuroimaging I/O | Reading, writing, and header inspection of NIfTI-1 and NIfTI-2 medical images |
| **NumPy** (>= 1.24) & **SciPy** | Numerical Computation | High-performance array operations, matrix mathematics, and spatial algorithms |
| **Scikit-learn** & **Scikit-image** | Machine Learning & Vision | Connected component analysis, morphological transformations, and metric calculations |
| **batchgenerators** / **batchgeneratorsv2** | Data Augmentation | Multi-threaded real-time 3D spatial transformations, scaling, and rotations |
| **dynamic-network-architectures** | Architecture Engine | Dynamically constructed U-Net variants matching dataset fingerprints |
| **Pandas**, **Matplotlib**, **Seaborn** | Analytics & Logging | Training progress tracking, loss curve plotting, and validation metric accumulation |

---

## 🏗️ System Architecture

MedVision 3D follows a systematic, modular six-stage pipeline that automatically adapts to the statistical and spatial properties of each biomedical dataset:

```
┌──────────────────────────┐      ┌──────────────────────────┐      ┌──────────────────────────┐
│  Dataset Fingerprinting  │ ──►  │   Experiment Planning    │ ──►  │      Preprocessing       │
│  (Spacing, Modality,     │      │  (Architecture, Patch/   │      │  (Cropping, Resampling,  │
│   Median Shapes, Voxel)  │      │   Batch Sizes, Strides)  │      │   Intensity Normalizing) │
└──────────────────────────┘      └──────────────────────────┘      └──────────────────────────┘
                                                                                  │
                                                                                  ▼
┌──────────────────────────┐      ┌──────────────────────────┐      ┌──────────────────────────┐
│        Inference         │ ◄──  │      Postprocessing      │ ◄──  │     Network Training     │
│  (Sliding-Window Patch,  │      │  (Connected Components,  │      │  (2D / 3D FullRes /      │
│   Test-Time Augmentation)│      │   Model Ensembling)      │      │   3D Cascade, 5-Fold CV) │
└──────────────────────────┘      └──────────────────────────┘      └──────────────────────────┘
```

![MedVision 3D Pipeline](documentation/assets/nnU-Net_overview.png)

1. **Dataset Fingerprinting**: Inspects raw training images to extract image dimensions, voxel spacings, modality types, intensity distributions, and target label distributions.
2. **Experiment Planning**: Employs heuristic rule sets to determine whether to train a 2D U-Net, 3D Full-Resolution U-Net, and/or a 3D Low-Resolution Cascade, configuring patch sizes, batch sizes, and network depth based on GPU memory constraints.
3. **Preprocessing**: Normalizes intensities (z-scoring, min-max, or CT clipping window), extracts foreground regions, and resamples images and segmentations to target voxel spacings.
4. **Network Training**: Trains deep convolutional U-Nets using a combined Dice + Cross-Entropy loss with 5-fold cross-validation and rigorous learning rate scheduling.
5. **Postprocessing**: Empirically determines whether removing unconnected foreground components or ensembling multiple model configurations improves the cross-validation Dice score.
6. **Inference**: Applies sliding-window inference with Gaussian patch weighting and optional test-time augmentation (TTA) to predict new cases and project them back to original image space.

---

## 🚀 Installation Guide

### Prerequisites
- **Python**: Version 3.10 or higher.
- **PyTorch**: Install the compatible PyTorch build for your system hardware (CUDA strongly recommended for training) before installing MedVision 3D dependencies:
  ```bash
  # Example for CUDA 12.1 (check https://pytorch.org for current command)
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```

### Quick Install (Package)
Install the `nnunetv2` package directly via `pip`:
```bash
pip install nnunetv2
```

### Development Install (Editable)
For developers wishing to inspect or extend MedVision 3D:
```bash
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```

*For Intel macOS systems, install with the macOS extra:*
```bash
pip install -e ".[intel_macos]"
```

For detailed platform-specific walkthroughs, see the [Installation and Setup Guide](documentation/getting-started/installation-and-setup.md).

---

## 📦 Dependencies and Requirements

The core runtime dependencies specified in `pyproject.toml` include:

```text
torch >= 2.1.2, != 2.9.*
acvl-utils >= 0.2.6, < 0.3
dynamic-network-architectures >= 0.4.4, < 0.5
batchgenerators >= 0.25.6
batchgeneratorsv2 >= 0.3.6
SimpleITK >= 2.2.1
nibabel
numpy >= 1.24
scipy
scikit-learn
scikit-image >= 0.19.3
tqdm
pandas
matplotlib
seaborn
tifffile
imagecodecs
yacs
einops
blosc2 >= 4.1.0
safetensors >= 0.4.0
timm >= 1.0
```

---

## 🌐 Environment Setup

MedVision 3D requires three dedicated storage directory paths defined as environment variables to organize raw data, cached preprocessed data, and trained model weights:

- **`nnUNet_raw`**: Directory containing raw, unprocessed datasets organized in the MedVision 3D format (`DatasetXXX_Name/`).
- **`nnUNet_preprocessed`**: High-speed storage location where cropped, resampled, and normalized arrays are saved for rapid batch generation during training.
- **`nnUNet_results`**: Target directory where checkpoints, training progress logs, validation summaries, and final model weights are saved.

### Recommended Directory Structure
```text
medvision_data/
├── nnUNet_raw/
├── nnUNet_preprocessed/
└── nnUNet_results/
```

### Linux and macOS (Bash / Zsh)
Add the following exports to your shell profile (e.g., `~/.bashrc` or `~/.zshrc`):
```bash
export nnUNet_raw="/path/to/medvision_data/nnUNet_raw"
export nnUNet_preprocessed="/path/to/medvision_data/nnUNet_preprocessed"
export nnUNet_results="/path/to/medvision_data/nnUNet_results"
```

### Windows (PowerShell)
To set temporarily for the current session:
```powershell
$Env:nnUNet_raw = "C:\path\to\medvision_data\nnUNet_raw"
$Env:nnUNet_preprocessed = "C:\path\to\medvision_data\nnUNet_preprocessed"
$Env:nnUNet_results = "C:\path\to\medvision_data\nnUNet_results"
```
To set permanently on Windows, use System Properties > Environment Variables, or PowerShell:
```powershell
[System.Environment]::SetEnvironmentVariable('nnUNet_raw', 'C:\path\to\medvision_data\nnUNet_raw', 'User')
[System.Environment]::SetEnvironmentVariable('nnUNet_preprocessed', 'C:\path\to\medvision_data\nnUNet_preprocessed', 'User')
[System.Environment]::SetEnvironmentVariable('nnUNet_results', 'C:\path\to\medvision_data\nnUNet_results', 'User')
```

### Windows (Command Prompt)
```cmd
set nnUNet_raw=C:\path\to\medvision_data\nnUNet_raw
set nnUNet_preprocessed=C:\path\to\medvision_data\nnUNet_preprocessed
set nnUNet_results=C:\path\to\medvision_data\nnUNet_results
```

---

## 💻 Running the Project - Example Commands

All commands in MedVision 3D are executed via the `nnUNetv2_` command-line interfaces.

### 1. Dataset Preparation
Structure your dataset inside `$nnUNet_raw/Dataset001_Example/`:
```text
Dataset001_Example/
├── dataset.json
├── imagesTr/
│   ├── CASE_001_0000.nii.gz
│   └── CASE_002_0000.nii.gz
├── labelsTr/
│   ├── CASE_001.nii.gz
│   └── CASE_002.nii.gz
└── imagesTs/
    └── CASE_003_0000.nii.gz
```

### 2. Planning and Preprocessing
Analyze the dataset fingerprint, configure U-Net topologies, and generate preprocessed data arrays:
```bash
# Preprocess using dataset ID (e.g. 1 or 001)
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
```

### 3. Model Training
Train a specific configuration (e.g., `2d`, `3d_fullres`, `3d_lowres`, or `3d_cascade_fullres`) on a cross-validation fold (0 to 4, or `all`):
```bash
# Syntax: nnUNetv2_train DATASET_ID CONFIG FOLD
# Example: Train 3D Full Resolution on fold 0
nnUNetv2_train 1 3d_fullres 0

# Train on multiple folds or configurations
nnUNetv2_train 1 3d_fullres 1
nnUNetv2_train 1 2d 0
```

### 4. Determine Best Configuration and Postprocessing
Automatically evaluate trained models, identify optimal configurations or ensembles, and compute connected component rules:
```bash
# Syntax: nnUNetv2_find_best_configuration DATASET_ID
nnUNetv2_find_best_configuration 1
```

### 5. Running Inference
Generate predictions for unseen test images using the trained model weights:
```bash
# Syntax: nnUNetv2_predict -i INPUT -o OUTPUT -d DATASET_ID -c CONFIG -f FOLD
nnUNetv2_predict -i /path/to/test_images/ -o /path/to/predictions/ -d 1 -c 3d_fullres -f 0
```

---

## 📂 Project Structure

```text
MedVision 3D/
├── nnunetv2/                           # Core framework package
│   ├── configuration.py                # Environment variable handling and global settings
│   ├── dataset_conversion/             # Conversion tools for standard formats & MSD datasets
│   ├── evaluation/                     # Metric evaluation, cross-validation summarizers
│   ├── experiment_planning/            # Fingerprinting, rule-based planner, preprocessing
│   ├── inference/                      # Sliding-window predictors, export utilities
│   ├── model_sharing/                  # Model export, packaging, and pretrained downloads
│   ├── paths.py                        # Path resolution for raw, preprocessed, results
│   ├── postprocessing/                 # Connected component filtering and analysis
│   ├── preprocessing/                  # Normalizers, resamplers, cropping pipelines
│   ├── run/                            # Training CLI execution entrypoints
│   ├── training/                       # Network architectures, loss functions, trainers
│   └── utilities/                      # Label utilities, overlay plotting, file helpers
├── documentation/                      # Comprehensive technical documentation
│   ├── assets/                         # Architecture diagrams, logos, visuals
│   ├── getting-started/                # Quickstart, setup, and first-run guides
│   ├── how-to/                         # Step-by-step task instructions and tutorials
│   ├── reference/                      # CLI references, configuration schemas, dataset format
│   ├── explanation/                    # Architecture breakdown, normalization details
│   └── case-studies/                   # Challenge benchmarks and historical writeups
├── pyproject.toml                      # Build specification, dependencies, CLI entry points
├── LICENSE                             # Apache 2.0 license file
└── readme.md                           # MedVision 3D main project documentation
```

---

## 📚 Documentation

Detailed guides and operational documentation are organized in the [`documentation/`](documentation/README.md) directory:

- 🚀 **[Getting Started](documentation/getting-started/README.md)**: First-time setup, environment configuration, and end-to-end tutorial.
- 📖 **[How-to Guides](documentation/how-to/README.md)**: Practical workflows for dataset preparation, custom training schedules, residual encoder presets, and inference execution.
- 📋 **[Reference Manual](documentation/reference/README.md)**: Exhaustive documentation for CLI commands, JSON metadata schemas, preprocessed data structures, and logging formats.
- 💡 **[Explanation & Theory](documentation/explanation/README.md)**: Conceptual deep dives into self-configuring heuristics, intensity normalization, and architectural choices.
- 🏆 **[Case Studies](documentation/case-studies/README.md)**: Segmentation performance notes across international biomedical challenges.

---

## ⚠️ Limitations and Medical Disclaimer

> [!CAUTION]
> **RESEARCH USE ONLY — NOT APPROVED FOR CLINICAL DIAGNOSIS**
> 
> MedVision 3D is strictly an **academic and research software tool**.
> 
> - **Regulatory Status**: This software is **NOT** a certified medical device and has **NOT** undergone clearance or approval by the US Food and Drug Administration (FDA), European Medicines Agency (EMA), or any equivalent health regulatory authority.
> - **Clinical Decision Making**: MedVision 3D must **NOT** be used as a primary diagnostic instrument, for clinical patient triage, or to direct clinical treatment decisions.
> - **Mandatory Verification**: All predicted segmentations and numerical measurements generated by this tool **must be independently reviewed and verified by qualified, board-certified medical professionals** (e.g., radiologists, pathologists, oncologists) prior to any clinical or translational consideration.
> - **Data Sensitivity**: Segmentation accuracy depends heavily on training data distribution, image quality, acquisition parameters, presence of motion artifacts, and patient anatomy variability. Performance may degrade on out-of-distribution scans.

---

## 🏆 Credits and Attribution

**MedVision 3D** is a customized derivative project built upon and powered by the **nnU-Net** biomedical segmentation framework.

### Original Framework Creators
The underlying nnU-Net architecture and algorithms were conceptualized and developed by:
- **Fabian Isensee**
- **Paul F. Jaeger**
- **Simon A. A. Kohl**
- **Jens Petersen**
- **Klaus H. Maier-Hein**

Developed by the Applied Computer Vision Lab (ACVL) at **[Helmholtz Imaging](https://helmholtz-imaging.de)** and the Division of Medical Image Computing at the **[German Cancer Research Center (DKFZ)](https://www.dkfz.de/en/index.html)**.

### Academic Citation
If you utilize this framework in your research, please cite the seminal Nature Methods publication:

```bibtex
@article{isensee2021nnu,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and Jaeger, Paul F. and Kohl, Simon A. A. and Petersen, Jens and Maier-Hein, Klaus H.},
  journal={Nature Methods},
  volume={18},
  number={2},
  pages={203--211},
  year={2021},
  publisher={Nature Publishing Group}
}
```

Additional recent work on residual encoder configurations:
```text
Isensee, F., et al. (2024). nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation. arXiv:2404.09556.
```

- **Original Upstream Repository**: [https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)

---

## 👤 Maintainer

- **Project**: MedVision 3D
- **Maintainer**: **Amen Allah Ben Abdrabah**
- **Project Role**: MedVision 3D is a medical imaging segmentation project maintained and customized by Amen Allah Ben Abdrabah, focusing on AI-powered analysis of 2D and 3D medical images.

---

## 📄 License

This project is licensed under the **Apache License, Version 2.0**. See the [LICENSE](LICENSE) file for complete terms.

All original copyrights, trademarks, and intellectual property rights belonging to the original authors, the German Cancer Research Center (DKFZ), and the Helmholtz Association are fully recognized and preserved.
