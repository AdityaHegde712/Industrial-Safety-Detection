"""
Quick validation script to extract metrics from the best trained model.
"""
import json
import sys
from pathlib import Path
import yaml
from multiprocessing import freeze_support

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO


def main():
    # Paths
    data_yaml = project_root / "data" / "processed" / "unified_dataset" / "data_oversampled.yaml"
    best_pt = project_root / "runs" / "detect" / "yolov8s_val_augmented_data-3" / "weights" / "best.pt"

    # Load config
    with open(data_yaml) as f:
        config = yaml.safe_load(f)
    names = config.get('names', {})

    # Load best model and validate
    model = YOLO(str(best_pt))
    results = model.val(data=str(data_yaml), device=0)

    # Extract metrics
    if results is not None and hasattr(results, 'box'):
    map50 = float(results.box.map50)
    map_ = float(results.box.map)
    precision = float(results.box.p.mean()) if hasattr(results.box.p, 'mean') else float(results.box.p)
    recall = float(results.box.r.mean()) if hasattr(results.box.r, 'mean') else float(results.box.r)

    metrics = {
        "mAP@0.5": map50,
        "mAP@0.5:0.95": map_,
        "precision": precision,
        "recall": recall,
    }

    print("=" * 70)
    print("  BEST MODEL VALIDATION RESULTS")
    print("=" * 70)
    print(f"  mAP@0.5:     {map50:.4f}")
    print(f"  mAP@0.5:0.95: {map_:.4f}")
    print(f"  Precision:   {precision:.4f}")
    print(f"  Recall:      {recall:.4f}")

    # Per-class
    print("\n  Per-Class mAP@0.5:")
    for i, cls_idx in enumerate(results.box.ap_class_index):
        class_name = names.get(int(cls_idx), f"class_{int(cls_idx)}")
        ap = float(results.box.ap50[i])
        print(f"    {class_name:20s}: {ap:.4f}")
        metrics[f"mAP@0.5_{class_name}"] = ap

    # Data validity
    print("\n" + "-" * 70)
    if map50 > 0.7:
        validity = "VALID"
        detail = f"mAP@0.5={map50:.4f} exceeds threshold of 0.7"
    elif map50 > 0.5:
        validity = "BORDERLINE"
        detail = f"mAP@0.5={map50:.4f} is between 0.5 and 0.7"
    else:
        validity = "INVALID"
        detail = f"mAP@0.5={map50:.4f} is below threshold of 0.5"

    metrics["validity"] = validity
    metrics["validity_detail"] = detail
    print(f"  Augmented Data Validity: {validity}")
    print(f"  Reason: {detail}")
    print("=" * 70)

    # Save
    out_path = project_root / "reports" / "augmented_data_validation.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[INFO] Metrics saved to: {out_path}")
else:
    print("[ERROR] No valid results from model validation")
    sys.exit(1)


if __name__ == "__main__":
    freeze_support()
    main()
