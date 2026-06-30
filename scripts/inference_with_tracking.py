#!/usr/bin/env python3
"""
Inference with Multi-Object Tracking for SimaAI Industry Safety System
Uses YOLOv8 + ByteTrack for detection and tracking
"""
from ultralytics import YOLO
import sys
import os
from pathlib import Path

def inference_with_tracking(source, output_dir='runs/tracking', conf_threshold=0.25):
    """
    Run detection + tracking on video/images
    
    Args:
        source: Path to image, video, or directory
        output_dir: Output directory for results
        conf_threshold: Confidence threshold for detections
    """
    print(f"\n{'='*60}")
    print(f"SimaAI Industry Safety - Detection + Tracking")
    print(f"{'='*60}")
    
    # Load model - try .pt first, then .pth for backward compatibility
    model_path_pt = 'models/weights/detection_model.pt'
    model_path_pth = 'models/weights/detection_model.pth'
    
    if os.path.exists(model_path_pt):
        model_path = model_path_pt
    elif os.path.exists(model_path_pth):
        model_path = model_path_pth
    else:
        print(f"ERROR: Model not found at {model_path_pt} or {model_path_pth}")
        print(f"Please ensure Task 1 is completed (model copy)")
        return None
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Check if source exists
    if not os.path.exists(source):
        print(f"ERROR: Source not found: {source}")
        return None
    
    print(f"Source: {source}")
    print(f"Tracker: ByteTrack")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"{'='*60}\n")
    
    # Run inference with ByteTrack
    results = model.track(
        source=source,
        tracker='bytetrack.yaml',  # ByteTrack config
        conf=conf_threshold,
        save=True,
        project=output_dir,
        name='tracked_output',
        exist_ok=True
    )
    
    output_path = os.path.join(output_dir, 'tracked_output')
    print(f"\n{'='*60}")
    print(f"Tracking complete!")
    print(f"Output saved to: {output_path}")
    print(f"{'='*60}\n")
    
    # Print summary of results
    if results:
        print("Summary:")
        for i, r in enumerate(results):
            if hasattr(r, 'boxes') and r.boxes is not None:
                print(f"  Frame/Image {i+1}: {len(r.boxes)} detections")
                if hasattr(r.boxes, 'id') and r.boxes.id is not None:
                    unique_ids = len(r.boxes.id.unique()) if len(r.boxes.id) > 0 else 0
                    print(f"    Unique track IDs: {unique_ids}")
    
    return results

def main():
    """Main entry point"""
    # Parse command line arguments
    if len(sys.argv) > 1:
        source = sys.argv[1]
    else:
        # Default to validation set images for testing
        source = 'data/processed/unified_dataset/val/images'
        print(f"No source provided, using default: {source}")
    
    # Optional: confidence threshold
    conf_threshold = 0.25
    if len(sys.argv) > 2:
        try:
            conf_threshold = float(sys.argv[2])
        except ValueError:
            print(f"Invalid confidence threshold: {sys.argv[2]}, using default: 0.25")
    
    # Run inference with tracking
    results = inference_with_tracking(source, conf_threshold=conf_threshold)
    
    if results:
        print("\n✓ Inference with tracking completed successfully!")
    else:
        print("\n✗ Inference failed. Check errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
