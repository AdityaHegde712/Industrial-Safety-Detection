#!/usr/bin/env python3
"""Export best model to ONNX with static input shape for SiMa.ai"""
from ultralytics import YOLO
import onnxruntime as ort
import shutil
from pathlib import Path

print('Loading best model for ONNX export...')
model = YOLO('runs/detect/simaai_industry_safety/yolov8s_unified_dataset/weights/best.pt')

# Export to ONNX with STATIC input shape (CRITICAL for SiMa.ai)
print('\nExporting to ONNX with static input shape...')
model.export(
    format='onnx',
    imgsz=640,
    dynamic=False,  # CRITICAL: Static graph for SiMa.ai
    simplify=True,
    opset=12  # ONNX opset version
)

# Verify ONNX export
onnx_path = 'runs/detect/simaai_industry_safety/yolov8s_unified_dataset/weights/best.onnx'
print(f'\nVerifying ONNX export: {onnx_path}')
session = ort.InferenceSession(onnx_path)
input_shape = session.get_inputs()[0].shape
print(f'ONNX Input Shape: {input_shape}')

# Check if static shape [1, 3, 640, 640]
expected_shape = [1, 3, 640, 640]
is_static = input_shape == expected_shape
print(f'Static Shape Verified: {is_static}')
if not is_static:
    print(f'WARNING: Expected {expected_shape}, got {input_shape}')
    exit(1)

# Test inference with dummy input
import numpy as np
dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
outputs = session.run(None, {session.get_inputs()[0].name: dummy_input})
print(f'Test inference successful. Output shape: {[o.shape for o in outputs]}')

# Copy to models/weights/
models_dir = Path('models/weights')
models_dir.mkdir(parents=True, exist_ok=True)
dest_onnx = models_dir / 'detection_model.onnx'
shutil.copy2(onnx_path, dest_onnx)
print(f'\nONNX copied to: {dest_onnx}')

# Also copy the original ONNX to models/weights for reference
print('\nTask 2 COMPLETED')
print('=' * 50)
print('ONNX Export Summary:')
print(f'  - Source: {onnx_path}')
print(f'  - Destination: {dest_onnx}')
print(f'  - Input Shape: {input_shape} (static)')
print(f'  - SiMa.ai Compliant: {is_static}')
print('=' * 50)
