#!/usr/bin/env python3
"""
SimaAI Industry Safety - Model Training Script

Trains YOLOv8s on unified dataset for PPE detection with W&B logging.
Fixed input size 640x640 for SiMa.ai static graph compliance.

Requirements:
    - ultralytics>=8.0.0
    - torch>=2.0.0
    - wandb (optional, for experiment tracking)
"""

import os
import sys
import shutil
import traceback
from pathlib import Path
from typing import Optional

# Project root path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================
# Model configuration
MODEL_NAME: str = "yolov8s.pt"
EPOCHS: int = 100
IMG_SIZE: int = 640  # Fixed for SiMa.ai static graph compliance
BATCH_SIZE: int = 16  # Adjust based on GPU memory (8.55 GB available)

# Training hyperparameters (as decided by @ml/model-scientist)
PATIENCE: int = 20  # Early stopping patience
LR0: float = 0.001  # Initial learning rate
LRF: float = 0.01  # Final learning rate factor
MOMENTUM: float = 0.937
WEIGHT_DECAY: float = 0
WARMUP_EPOCHS: int = 3
# BOX_LOSS_GAIN: float = 7.5
# CLS_LOSS_GAIN: float = 0.5
# DFL_LOSS_GAIN: float = 1.5
# CLOSE_MOSAIC: int = 10  # Disable mosaic augmentation for last 10 epochs

# Logging configuration
SAVE_PERIOD: int = 5  # Save checkpoint every 5 epochs

# Paths configuration
DATA_YAML: Path = PROJECT_ROOT / "data/processed/unified_dataset/data.yaml"
PROJECT_NAME: str = "simaai_industry_safety"
EXPERIMENT_NAME: str = "yolov8s_unified_dataset"
OUTPUT_DIR: Path = PROJECT_ROOT / "models/weights"
TARGET_MAP50: float = 0.9  # Target mAP@0.5 for success


def _print_gpu_info() -> None:
    """
    Print GPU information for troubleshooting.

    Checks PyTorch version and CUDA availability.
    """
    try:
        import torch
        print(f"\n[INFO] PyTorch version: {torch.__version__}")
        print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[INFO] CUDA version: {torch.version.cuda}")
    except Exception as e:
        print(f"\n[WARNING] Could not check GPU: {e}")


def _init_wandb() -> bool:
    """
    Initialize Weights & Biases for experiment tracking.

    Returns:
        bool: True if W&B initialized successfully, False otherwise.
    """
    try:
        import wandb
        wandb.init(
            project="simaai-industry-safety",
            name=EXPERIMENT_NAME,
            config={
                "model": MODEL_NAME,
                "epochs": EPOCHS,
                "imgsz": IMG_SIZE,
                "batch": BATCH_SIZE,
                "data": str(DATA_YAML),
                "lr0": LR0,
                "lrf": LRF,
                "momentum": MOMENTUM,
                "weight_decay": WEIGHT_DECAY,
                "warmup_epochs": WARMUP_EPOCHS,
                "patience": PATIENCE,
            }
        )
        print("\n[INFO] Weights & Biases initialized")
        return True
    except Exception as e:
        print(f"\n[WARNING] W&B not available: {e}")
        print("[INFO] Training without experiment tracking")
        return False


def _copy_model_to_deliverables(best_model_path: Path) -> bool:
    """
    Copy trained model to deliverables directory.

    Args:
        best_model_path: Path to the best trained model weights.

    Returns:
        bool: True if copy successful, False otherwise.
    """
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / "detection_model.pth"
        shutil.copy2(best_model_path, output_path)
        print(f"Model copied to: {output_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to copy model: {e}")
        return False


def validate_model(model_path: Path) -> dict:
    """
    Validate the trained model on the dataset.

    Args:
        model_path: Path to the trained model weights.

    Returns:
        dict: Validation metrics including mAP scores.
    """
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    val_results = model.val(data=str(DATA_YAML))

    metrics = {
        "map50": val_results.box.map50,
        "map50_95": val_results.box.map,
        "precision": val_results.box.mp,
        "recall": val_results.box.mr,
    }

    return metrics


def train_model() -> bool:
    """
    Train YOLOv8s on unified dataset for PPE detection.

    Returns:
        bool: True if training successful, False otherwise.
    """
    # Print header
    print("=" * 60)
    print("SimaAI Industry Safety - Model Training")
    print("=" * 60)

    # Print GPU information
    _print_gpu_info()

    # Initialize W&B (optional)
    wandb_initialized = _init_wandb()

    # Print configuration
    print(f"\n[CONFIG] Model: {MODEL_NAME}")
    print(f"[CONFIG] Data: {DATA_YAML}")
    print(f"[CONFIG] Epochs: {EPOCHS}")
    print(f"[CONFIG] Image Size: {IMG_SIZE}x{IMG_SIZE} (static for SiMa.ai)")
    print(f"[CONFIG] Batch Size: {BATCH_SIZE}")
    print(f"[CONFIG] Project: {PROJECT_NAME}")
    print(f"[CONFIG] Experiment: {EXPERIMENT_NAME}")

    # Train YOLOv8s
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60 + "\n")

    try:
        from ultralytics import YOLO

        # Load COCO-pretrained YOLOv8s
        model = YOLO(MODEL_NAME)
        print("[INFO] Loaded YOLOv8s with COCO-pretrained weights")

        # Train on unified dataset
        results = model.train(
            data=str(DATA_YAML),
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            name=EXPERIMENT_NAME,
            project=PROJECT_NAME,
            save=True,
            save_period=SAVE_PERIOD,
            patience=PATIENCE,
            lr0=LR0,
            lrf=LRF,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
            warmup_epochs=WARMUP_EPOCHS,
            # box=BOX_LOSS_GAIN,
            # cls=CLS_LOSS_GAIN,
            # dfl=DFL_LOSS_GAIN,
            # close_mosaic=CLOSE_MOSAIC,
            # rect=False,
            val=True,
            plots=True,
        )

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)

        # Print training metrics
        if hasattr(results, 'box'):
            print(f"\nBest mAP@0.5: {results.box.map50:.4f}")
            print(f"Best mAP@0.5:0.95: {results.box.map:.4f}")

        # Get path to best weights
        best_model_path = Path(PROJECT_NAME) / EXPERIMENT_NAME / "weights" / "best.pt"
        print(f"\n[INFO] Best model saved to: {best_model_path}")

        # Copy model to deliverables directory
        if not _copy_model_to_deliverables(best_model_path):
            print("[WARNING] Failed to copy model to deliverables directory")

        # Validate best model
        print("\n[INFO] Validating best model...")
        metrics = validate_model(best_model_path)

        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)
        print(f"mAP@0.5: {metrics['map50']:.4f}")
        print(f"mAP@0.5:0.95: {metrics['map50_95']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")

        # Check if target mAP@0.5 >= 0.9 is achieved
        if metrics['map50'] >= TARGET_MAP50:
            print(f"\n[SUCCESS] TARGET ACHIEVED: mAP@0.5 >= {TARGET_MAP50}!")
        else:
            print(f"\n[WARNING] Target mAP@0.5 >= {TARGET_MAP50} not met.")
            print(f"  Current: {metrics['map50']:.4f}")
            print("  Consider: more epochs, larger model (yolov8m/l), or hyperparameter tuning")

        print(f"\n[INFO] Training and validation complete!")
        print(f"[INFO] Model saved to: {best_model_path}")

        # Finish W&B run if initialized
        if wandb_initialized:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass

        return True

    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        traceback.print_exc()
        return False


def main() -> int:
    """
    Main entry point for the training script.

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    success = train_model()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)