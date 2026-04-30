#!/usr/bin/env python3
"""
Dataset Merger for SimaAI Industry Safety System
Merges Kaggle Construction Site Safety and Roboflow Construction Safety datasets
Creates unified 70/15/15 train/val/test split
"""

import os
import shutil
import random
import yaml
from pathlib import Path
import json

# Set random seed for reproducibility
random.seed(42)

# Base directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed" / "unified_dataset"

# Source datasets
KAGGLE_DIR = DATA_RAW / "kaggle_construction" / "css-data"
ROBOFLOW_DIR = DATA_RAW / "roboflow_ppe" / "extracted"

# Class mappings
KAGGLE_TO_UNIFIED = {
    0: 1,  # Hardhat -> hardhat
    1: 5,  # Mask -> mask
    2: 2,  # NO-Hardhat -> no_hardhat
    3: 6,  # NO-Mask -> no_mask
    4: 4,  # NO-Safety Vest -> no_safety_vest
    5: 0,  # Person -> person
    6: 7,  # Safety Cone -> safety_cone
    7: 3,  # Safety Vest -> safety_vest
    8: 8,  # machinery -> machinery
    9: 9,  # vehicle -> vehicle
}

ROBOFLOW_TO_UNIFIED = {
    0: 1,  # helmet -> hardhat
    1: 2,  # no-helmet -> no_hardhat
    2: 4,  # no-vest -> no_safety_vest
    3: 0,  # person -> person
    4: 3,  # vest -> safety_vest
}

def get_yolo_labels(img_dir, label_dir, class_mapping, source_name):
    """Get all image-label pairs with unified class IDs"""
    data_pairs = []
    
    if not img_dir.exists():
        print(f"Warning: {img_dir} does not exist")
        return data_pairs
    
    for img_file in img_dir.glob("*.*"):
        if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
            continue
        
        # Find corresponding label file
        label_file = label_dir / f"{img_file.stem}.txt"
        if not label_file.exists():
            continue
        
        # Read and convert labels
        converted_labels = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                
                old_class_id = int(parts[0])
                # Map to unified class ID
                if source_name == "kaggle":
                    new_class_id = class_mapping.get(old_class_id)
                else:  # roboflow
                    new_class_id = class_mapping.get(old_class_id)
                
                if new_class_id is None:
                    continue
                
                # Keep bbox as is (already normalized YOLO format)
                converted_labels.append({
                    'class_id': new_class_id,
                    'bbox': [float(parts[1]), float(parts[2]), 
                            float(parts[3]), float(parts[4])]
                })
        
        if converted_labels:  # Only add if has valid labels
            data_pairs.append({
                'image_path': str(img_file),
                'label_path': str(label_file),
                'labels': converted_labels,
                'source': source_name
            })
    
    return data_pairs

def stratified_split(data_pairs, train_ratio=0.70, val_ratio=0.15):
    """Split data ensuring class distribution across splits"""
    # Group by class for stratification
    class_groups = {}
    for idx, item in enumerate(data_pairs):
        # Get primary class (first label)
        if item['labels']:
            primary_class = item['labels'][0]['class_id']
            if primary_class not in class_groups:
                class_groups[primary_class] = []
            class_groups[primary_class].append(idx)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    for class_id, indices in class_groups.items():
        random.shuffle(indices)
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])
    
    # Shuffle the splits
    random.shuffle(train_indices)
    random.shuffle(val_indices)
    random.shuffle(test_indices)
    
    return train_indices, val_indices, test_indices

def copy_data(data_pairs, indices, split_name):
    """Copy images and labels to split directory"""
    split_img_dir = DATA_PROCESSED / split_name / "images"
    split_label_dir = DATA_PROCESSED / split_name / "labels"
    split_img_dir.mkdir(parents=True, exist_ok=True)
    split_label_dir.mkdir(parents=True, exist_ok=True)
    
    copied = 0
    for idx in indices:
        item = data_pairs[idx]
        img_path = Path(item['image_path'])
        label_path = Path(item['label_path'])
        
        # Copy image
        dst_img = split_img_dir / img_path.name
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)
        
        # Write converted labels
        dst_label = split_label_dir / f"{img_path.stem}.txt"
        with open(dst_label, 'w') as f:
            for label in item['labels']:
                f.write(f"{label['class_id']} {label['bbox'][0]} {label['bbox'][1]} "
                       f"{label['bbox'][2]} {label['bbox'][3]}\n")
        
        copied += 1
    
    return copied

def create_test_ground_truth(data_pairs, test_indices):
    """Create ground truth JSON for test set"""
    ground_truth = {"images": []}
    
    for idx in test_indices:
        item = data_pairs[idx]
        img_path = Path(item['image_path'])
        
        annotations = []
        for label in item['labels']:
            # Convert YOLO format to [x1, y1, x2, y2] (absolute coords would need image size)
            # For now, store normalized format
            annotations.append({
                "bbox": [label['bbox'][0], label['bbox'][1], 
                        label['bbox'][2], label['bbox'][3]],
                "class_id": label['class_id'],
                "format": "yolo_normalized"
            })
        
        ground_truth["images"].append({
            "file_name": img_path.name,
            "annotations": annotations
        })
    
    # Save ground truth
    gt_path = DATA_PROCESSED / "test_ground_truth.json"
    with open(gt_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    return gt_path

def main():
    print("=" * 60)
    print("Dataset Merger - SimaAI Industry Safety")
    print("=" * 60)
    
    # Step 1: Collect data from Kaggle dataset
    print("\n[1/5] Loading Kaggle dataset...")
    kaggle_train = get_yolo_labels(
        KAGGLE_DIR / "train" / "images",
        KAGGLE_DIR / "train" / "labels",
        KAGGLE_TO_UNIFIED,
        "kaggle"
    )
    kaggle_val = get_yolo_labels(
        KAGGLE_DIR / "valid" / "images",
        KAGGLE_DIR / "valid" / "labels",
        KAGGLE_TO_UNIFIED,
        "kaggle"
    )
    kaggle_test = get_yolo_labels(
        KAGGLE_DIR / "test" / "images",
        KAGGLE_DIR / "test" / "labels",
        KAGGLE_TO_UNIFIED,
        "kaggle"
    )
    print(f"  Kaggle - Train: {len(kaggle_train)}, Val: {len(kaggle_val)}, Test: {len(kaggle_test)}")
    
    # Step 2: Collect data from Roboflow dataset
    print("\n[2/5] Loading Roboflow dataset...")
    roboflow_train = get_yolo_labels(
        ROBOFLOW_DIR / "train" / "images",
        ROBOFLOW_DIR / "train" / "labels",
        ROBOFLOW_TO_UNIFIED,
        "roboflow"
    )
    roboflow_val = get_yolo_labels(
        ROBOFLOW_DIR / "valid" / "images",
        ROBOFLOW_DIR / "valid" / "labels",
        ROBOFLOW_TO_UNIFIED,
        "roboflow"
    )
    roboflow_test = get_yolo_labels(
        ROBOFLOW_DIR / "test" / "images",
        ROBOFLOW_DIR / "test" / "labels",
        ROBOFLOW_TO_UNIFIED,
        "roboflow"
    )
    print(f"  Roboflow - Train: {len(roboflow_train)}, Val: {len(roboflow_val)}, Test: {len(roboflow_test)}")
    
    # Step 3: Merge all data
    print("\n[3/5] Merging datasets...")
    all_data = kaggle_train + kaggle_val + kaggle_test + roboflow_train + roboflow_val + roboflow_test
    print(f"  Total merged: {len(all_data)} images")
    
    # Step 4: Create stratified split (70/15/15)
    print("\n[4/5] Creating 70/15/15 split...")
    train_idx, val_idx, test_idx = stratified_split(all_data, 0.70, 0.15)
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Step 5: Copy data to unified dataset
    print("\n[5/5] Copying data to unified dataset...")
    n_train = copy_data(all_data, train_idx, "train")
    n_val = copy_data(all_data, val_idx, "val")
    n_test = copy_data(all_data, test_idx, "test")
    print(f"  Copied - Train: {n_train}, Val: {n_val}, Test: {n_test}")
    
    # Create ground truth for test set
    print("\nCreating test set ground truth...")
    gt_path = create_test_ground_truth(all_data, test_idx)
    print(f"  Saved to: {gt_path}")
    
    # Print class distribution
    print("\n" + "=" * 60)
    print("Class Distribution in Unified Dataset:")
    print("=" * 60)
    
    # Load unified class names
    with open(DATA_PROCESSED / "data.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    for split_name, indices in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
        class_counts = {i: 0 for i in range(len(config['names']))}
        for idx in indices:
            for label in all_data[idx]['labels']:
                class_counts[label['class_id']] += 1
        
        print(f"\n{split_name} Set:")
        for class_id, count in class_counts.items():
            if count > 0:
                print(f"  {config['names'][class_id]:20s}: {count:5d}")
    
    print("\n" + "=" * 60)
    print("Dataset merge complete!")
    print(f"Unified dataset saved to: {DATA_PROCESSED}")
    print("=" * 60)

if __name__ == "__main__":
    main()
