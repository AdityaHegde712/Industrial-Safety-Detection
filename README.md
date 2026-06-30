# SimaAI Industry Safety

**Real-time PPE detection for industrial safety compliance, optimized for SiMa.ai edge deployment.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.11-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/ultralytics-8.4-00d4ff.svg)](https://ultralytics.com/)
[![ONNX](https://img.shields.io/badge/onnx-1.21-005CED.svg)](https://onnx.ai/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

SimaAI Industry Safety is a deep learning system that detects Personal Protective Equipment (PPE) compliance in industrial environments. Using YOLOv8 object detection, the system identifies hard hats, safety vests, masks, and other PPE items from video feeds and image directories.

**Target Hardware:** [SiMa.ai Modalix MLSoC](docs/target_hardware.md) -- a purpose-built edge processor delivering 50 TOPS for on-device AI inference.

### Key Features

- **10-Class PPE Detection** -- person, hardhat, no_hardhat, safety_vest, no_safety_vest, mask, no_mask, safety_cone, machinery, vehicle
- **SiMa.ai Optimized** -- Fixed 640x640 input with static ONNX graph for edge deployment
- **Modular Input Handling** -- Local video files, image directories, and RTSP streams (planned)
- **Experiment Tracking** -- Full Weights & Biases integration for training monitoring
- **Class Imbalance Handling** -- Inverse-frequency class weights for minority PPE categories
- **Multi-Object Tracking** -- ByteTrack integration for frame-to-frame person tracking

---

## Performance

| Metric | Value |
|--------|-------|
| **mAP@0.5** | 0.844 |
| mAP@0.5:0.95 | 0.584 |
| Precision | 0.906 |
| Recall | 0.780 |

*YOLOv8s trained for 300 epochs on unified dataset (3,983 images, 10 classes)*

### Per-Class Results

| Class | mAP@0.5 | Status |
|-------|---------|--------|
| person | 0.915 | Excellent |
| hardhat | 0.862 | Good |
| safety_vest | 0.842 | Good |
| no_safety_vest | 0.870 | Good |
| mask | 0.939 | Excellent |
| machinery | 0.958 | Excellent |
| vehicle | 0.812 | Good |
| safety_cone | 0.790 | Needs improvement |
| no_hardhat | 0.749 | Needs improvement |
| no_mask | 0.698 | Needs improvement |

---

## Project Structure

```
SimaAI_Industry_Safety/
├── scripts/                    # Training, validation, export pipelines
│   ├── train_model.py          # Main training script (YOLOv8 + W&B)
│   ├── validate_model.py       # Model validation on dataset
│   ├── validate_best_model.py  # Validate best checkpoint
│   ├── hyperparameter_search.py# Automated HP search (900+ runs)
│   ├── export_onnx.py          # ONNX export for SiMa.ai
│   ├── inference_with_tracking.py # ByteTrack multi-object tracking
│   ├── merge_datasets.py       # Merge Kaggle + Roboflow datasets
│   ├── oversample_minority.py  # Class imbalance mitigation
│   ├── augmentation_template.py# Data augmentation pipeline
│   └── generate_statistics.py  # Dataset statistics generation
│
├── src/                        # Application source code
│   └── input_handlers/         # Input source abstractions
│       ├── base.py             # Abstract InputHandler base class
│       ├── local_file.py       # Local video/image handler
│       └── rtsp_stream.py      # RTSP stream handler (stubbed)
│
├── configs/                    # Hyperparameter search results
│   ├── best_config.yaml        # Best HP configuration found
│   ├── search_results.yaml     # Full search results
│   └── search_results_intermediate.yaml
│
├── data/                       # Datasets (gitignored)
│   └── processed/unified_dataset/
│       ├── data.yaml           # YOLO dataset config
│       ├── data_oversampled.yaml
│       ├── train/              # Training split
│       ├── val/                # Validation split
│       └── test/               # Test split
│
├── models/weights/             # Model weights (gitignored)
│   ├── detection_model.pt      # PyTorch model
│   └── detection_model.onnx    # ONNX model (SiMa.ai ready)
│
├── reports/                    # Generated reports
│   ├── validation_report.md    # Detailed validation metrics
│   ├── validation_metrics.json
│   └── dataset_statistics.md
│
├── docs/                       # Technical documentation
│   ├── problem.md              # Problem statement & pipeline
│   ├── target_hardware.md      # SiMa.ai MLSoC specifications
│   └── class_imbalance_*.md    # Class imbalance analysis
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Hardware Target: SiMa.ai Modalix MLSoC

This project is designed for deployment on the **SiMa.ai Modalix MLSoC**, a purpose-built edge processor:

| Specification | Details |
|---------------|---------|
| **ML Accelerator** | Dedicated SiMa.ai ML Accelerator, 50 TOPS |
| **CPU** | 8x Arm Cortex-A65 @ 1.4GHz |
| **CV Unit** | Quad-core Synopsys ARC EV74 @ 1GHz |
| **Memory** | LPDDR5 (up to 6400 Mbps) |
| **Form Factor** | 25mm x 25mm FCBGA |
| **Model Format** | ONNX (static graph, no dynamic axes) |
| **Input Shape** | Fixed [1, 3, 640, 640] |

> **Critical Constraint:** The SiMa.ai hardware requires static computational graphs. All ONNX exports use `dynamic=False` with fixed input dimensions.

---

## Quick Start

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended for training)
- 8GB+ GPU VRAM (for training)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/SimaAI_Industry_Safety.git
cd SimaAI_Industry_Safety

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

The project uses a unified dataset merged from multiple sources:

1. **Kaggle Construction Site Safety Dataset** (10 classes)
2. **Roboflow Construction Safety Dataset** (5 classes, mapped to unified schema)
3. **COCO 2017** (person class for transfer learning)

Place your dataset in `data/processed/unified_dataset/` with the following structure:
```
data/processed/unified_dataset/
├── data.yaml           # Dataset configuration
├── train/images/       # Training images
├── train/labels/       # Training labels (YOLO format)
├── val/images/         # Validation images
├── val/labels/         # Validation labels
├── test/images/        # Test images
└── test/labels/        # Test labels
```

---

## Usage

### Training

Train YOLOv8s on the unified PPE dataset:

```bash
python scripts/train_model.py
```

**Configuration** (edit in `scripts/train_model.py`):
- Model: YOLOv8s (default) or YOLOv8m/l
- Epochs: 100
- Image Size: 640x640 (SiMa.ai constraint)
- Batch Size: 16
- Early Stopping: Patience 20

### Validation

Evaluate trained model performance:

```bash
python scripts/validate_model.py
```

### Hyperparameter Search

Run automated hyperparameter optimization:

```bash
python scripts/hyperparameter_search.py
```

> **Note:** The search completed 900+ runs. Best configuration is saved in `configs/best_config.yaml`.

### Export to ONNX

Export trained model for SiMa.ai deployment:

```bash
python scripts/export_onnx.py
```

This produces a static ONNX model at `models/weights/detection_model.onnx` with input shape `[1, 3, 640, 640]`.

### Inference with Tracking

Run multi-object tracking on video/images:

```bash
python scripts/inference_with_tracking.py
```

---

## Training Configuration

### Best Hyperparameters (from search)

```yaml
model: yolov8m.pt
optimizer: SGD
batch: 16
epochs: 100
lr0: 0.005
fraction: 0.8
workers: 2
```

### Class Weights (Inverse Frequency)

To address class imbalance, inverse-frequency weights are applied:

| Class | Weight |
|-------|--------|
| person | 0.13 |
| hardhat | 0.57 |
| no_hardhat | 1.32 |
| safety_vest | 0.75 |
| no_safety_vest | 0.66 |
| mask | 2.00 |
| no_mask | 1.03 |
| safety_cone | 0.91 |
| machinery | 0.62 |
| vehicle | 2.01 |

---

## Datasets

### Unified Class Schema

| ID | Class | Source Mapping |
|----|-------|----------------|
| 0 | person | Kaggle:5, Roboflow:person, COCO:person |
| 1 | hardhat | Kaggle:Hardhat(0), Roboflow:helmet(0) |
| 2 | no_hardhat | Kaggle:NO-Hardhat(2), Roboflow:no-helmet(1) |
| 3 | safety_vest | Kaggle:Safety Vest(7), Roboflow:vest(4) |
| 4 | no_safety_vest | Kaggle:NO-Safety Vest(4), Roboflow:no-vest(2) |
| 5 | mask | Kaggle:Mask(1) |
| 6 | no_mask | Kaggle:NO-Mask(3) |
| 7 | safety_cone | Kaggle:Safety Cone(6) |
| 8 | machinery | Kaggle:machinery(8) |
| 9 | vehicle | Kaggle:vehicle(9) |

---

## Model Deliverables

| Artifact | Path | Format | Status |
|----------|------|--------|--------|
| PyTorch Model | `models/weights/detection_model.pt` | .pt | Ready |
| ONNX Model | `models/weights/detection_model.onnx` | .onnx | Ready |
| Best Config | `configs/best_config.yaml` | .yaml | Ready |
| Validation Report | `reports/validation_report.md` | .md | Ready |

---

<!-- ## Roadmap

### Completed

- [x] Training pipeline with YOLOv8 + W&B integration
- [x] Hyperparameter search (900+ configurations)
- [x] ONNX export with static graph for SiMa.ai
- [x] Class imbalance mitigation (oversampling + class weights)
- [x] Multi-object tracking with ByteTrack
- [x] Dataset merging and preprocessing pipeline

### In Progress

- [ ] Core inference pipeline integration
- [ ] RTSP stream processing
- [ ] Safety rules engine
- [ ] Alert/notification system

### Future

- [ ] SiMa.ai Palette SDK integration
- [ ] INT8 quantization for edge optimization
- [ ] Zone-based safety rule enforcement
- [ ] Docker containerization for deployment
- [ ] REST API for inference requests -->

---

## Development

### Code Style

- Python 3.10+ with type hints
- Docstrings in Google style
- Black formatter recommended
- Flake8 for linting

### Testing

```bash
# Run validation
python scripts/validate_model.py

# Check GPU availability
python check_gpu.py

# Monitor resource usage
python resource_monitor.py
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Acknowledgments

- [Ultralytics](https://ultralytics.com/) -- YOLOv8 framework
- [SiMa.ai](https://sima.ai/) -- Edge deployment hardware
- [Weights & Biases](https://wandb.ai/) -- Experiment tracking
- Kaggle Construction Site Safety Dataset
- Roboflow Construction Safety Dataset

*Last Updated: June 2026*
