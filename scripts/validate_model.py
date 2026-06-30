#!/usr/bin/env python3
"""Validate best model and copy weights to models/weights/"""
from ultralytics import YOLO
import shutil
from pathlib import Path
import json

# Validate best model
print('Loading best model for validation...')
model = YOLO('runs/detect/simaai_industry_safety/yolov8s_unified_dataset/weights/best.pt')
results = model.val(data='data/processed/unified_dataset/data.yaml')

print('\n=== VALIDATION RESULTS ===')
print(f'mAP@0.5: {results.box.map50:.4f}')
print(f'mAP@0.5:0.95: {results.box.map:.4f}')
print(f'Precision: {results.box.mp:.4f}')
print(f'Recall: {results.box.mr:.4f}')

# Save metrics to JSON for report generation
metrics = {
    'mAP50': float(results.box.map50),
    'mAP50_95': float(results.box.map),
    'precision': float(results.box.mp),
    'recall': float(results.box.mr),
    'f1_score': float(2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr + 1e-6))
}
with open('reports/validation_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print(f'\nMetrics saved to: reports/validation_metrics.json')

# Copy to models/weights/
models_dir = Path('models/weights')
models_dir.mkdir(parents=True, exist_ok=True)
dest_path = models_dir / 'detection_model.pth'
shutil.copy2('runs/detect/simaai_industry_safety/yolov8s_unified_dataset/weights/best.pt', dest_path)
print(f'Model copied to: {dest_path}')
print('\nTask 1 COMPLETED')
