"""
Validation Training Script: Augmented Data Quality Check
=========================================================
Trains YOLOv8s for 30 epochs on oversampled (augmented) dataset
to verify Albumentations-based augmentation produces valid images/labels.

Expected: mAP@0.5 > 0.7 = Augmented data is VALID
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO


def main():
    print("=" * 70)
    print("  Validation Training: Augmented Data Quality Check")
    print("  YOLOv8s | 30 epochs | Over-sampled Dataset")
    print("=" * 70)

    # Paths
    data_yaml = project_root / "data" / "processed" / "unified_dataset" / "data_oversampled.yaml"
    output_dir = project_root / "runs" / "detect"

    # Verify config exists
    if not data_yaml.exists():
        print(f"[ERROR] Data config not found: {data_yaml}")
        sys.exit(1)
    print(f"\n[INFO] Using data config: {data_yaml}")

    # Parse the YAML to confirm paths
    import yaml
    with open(data_yaml, 'r') as f:
        config = yaml.safe_load(f)
    print(f"[INFO] Train path: {config.get('train', 'N/A')}")
    print(f"[INFO] Val path:   {config.get('val', 'N/A')}")
    print(f"[INFO] Classes:    {config.get('names', {})}")

    # Count training images
    train_img_dir = project_root / "data" / "processed" / "unified_dataset" / config["train"]
    if train_img_dir.exists():
        img_count = len(list(train_img_dir.glob("*.*")))
        print(f"[INFO] Training images: {img_count}")
    else:
        print(f"[WARN] Training image directory not found: {train_img_dir}")

    # Load the model
    print("\n[INFO] Loading YOLOv8s pretrained model...")
    model = YOLO(str(project_root / "yolov8s.pt"))
    print("[INFO] Model loaded successfully")

    # Run training
    print("\n[INFO] Starting training (30 epochs)...")
    print("[INFO] Early stopping patience=5")
    print("-" * 70)

    results = model.train(
        data=str(data_yaml),
        epochs=30,
        imgsz=640,
        batch=16,
        name="yolov8s_val_augmented_data",
        patience=5,
        save=True,
        save_period=-1,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        close_mosaic=10,
        device=0,  # Use GPU
    )

    # Extract and print final metrics
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE - Final Metrics")
    print("=" * 70)

    metrics = {}
    if results is not None:
        if hasattr(results, 'box'):
            map50 = results.box.map50
            map = results.box.map
            precision = results.box.p  # per-class precision array
            recall = results.box.r     # per-class recall array

            metrics = {
                "mAP@0.5": float(map50) if map50 is not None else None,
                "mAP@0.5:0.95": float(map) if map is not None else None,
                "precision": float(precision.mean()) if precision is not None else None,
                "recall": float(recall.mean()) if recall is not None else None,
            }

            print(f"\n  mAP@0.5:     {map50:.4f}")
            print(f"  mAP@0.5:0.95: {map:.4f}")
            print(f"  Precision:   {precision:.4f}" if precision else "  Precision:   N/A")
            print(f"  Recall:      {recall:.4f}" if recall else "  Recall:      N/A")

        # Per-class metrics
        if hasattr(results, 'box') and hasattr(results.box, 'ap_class_index') and hasattr(results.box, 'ap50'):
            print("\n  Per-Class mAP@0.5:")
            names = config.get('names', {})
            for i, cls_idx in enumerate(results.box.ap_class_index):
                class_name = names.get(int(cls_idx), f"class_{int(cls_idx)}")
                ap = results.box.ap50[i]
                print(f"    {class_name:20s}: {ap:.4f}")
                metrics[f"mAP@0.5_{class_name}"] = float(ap)

    # Determine data validity
    print("\n" + "-" * 70)
    if metrics.get("mAP@0.5") is not None:
        map50_val = metrics["mAP@0.5"]
        if map50_val > 0.7:
            validity = "VALID ✅"
            detail = f"mAP@0.5={map50_val:.4f} exceeds threshold of 0.7"
        elif map50_val > 0.5:
            validity = "BORDERLINE ⚠️"
            detail = f"mAP@0.5={map50_val:.4f} is between 0.5 and 0.7"
        else:
            validity = "INVALID ❌"
            detail = f"mAP@0.5={map50_val:.4f} is below threshold of 0.5"
        print(f"  Augmented Data Validity: {validity}")
        print(f"  Reason: {detail}")
    else:
        print("  [WARN] Could not compute mAP@0.5 from results object")

    # Save metrics to JSON
    metrics_path = project_root / "reports" / "augmented_data_validation.json"
    os.makedirs(metrics_path.parent, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[INFO] Metrics saved to: {metrics_path}")
    print("=" * 70)

    return metrics


if __name__ == "__main__":
    results_metrics = main()
