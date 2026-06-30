"""
Oversampling Script for Minority Classes (Albumentations-based)
Generates augmented variants of images containing underrepresented classes
"""
import albumentations as A
import cv2
import numpy as np
import shutil
from pathlib import Path
from collections import defaultdict

ALBUMENTATIONS_IS_INSTALLED = True

MULTIPLIERS = {
    1: 2,   # hardhat: 1 new augmented sample
    2: 3,   # no_hardhat: 2 new augmented samples
    5: 5,   # mask: 4 new augmented samples
    6: 2,   # no_mask: 1 new augmented sample
    7: 3,   # safety_cone: 2 new augmented samples
    8: 2,   # machinery: 1 new augmented sample
    9: 7,   # vehicle: 6 new augmented samples
}

transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def is_valid_bbox(bbox):
    x, y, w, h = bbox
    if w < 0.01 or h < 0.01 or w > 1.0 or h > 1.0:
        return False
    x1, y1 = x - w/2, y - h/2
    x2, y2 = x + w/2, y + h/2
    return 0.0 <= x1 and x2 <= 1.0 and 0.0 <= y1 and y2 <= 1.0

def augment_yolo_sample(image_path, label_path):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    bboxes = []
    class_labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                bboxes.append([float(p) for p in parts[1:5]])
                class_labels.append(int(parts[0]))
    
    if not bboxes:
        return None, None, None
    
    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    
    valid_bboxes = []
    valid_labels = []
    for bbox, label in zip(transformed['bboxes'], transformed['class_labels']):
        if is_valid_bbox(bbox):
            valid_bboxes.append(bbox)
            valid_labels.append(label)
    
    aug_image = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)
    return aug_image, valid_bboxes, valid_labels

def oversample_dataset():
    source_img_dir = Path('data/processed/unified_dataset/train/images')
    source_lbl_dir = Path('data/processed/unified_dataset/train/labels')
    target_img_dir = Path('data/processed/unified_dataset/train_oversampled/images')
    target_lbl_dir = Path('data/processed/unified_dataset/train_oversampled/labels')
    target_img_dir.mkdir(parents=True, exist_ok=True)
    target_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    image_class_map = {}
    for label_file in source_lbl_dir.glob('*.txt'):
        classes_in_image = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    classes_in_image.append(int(parts[0]))
        image_class_map[label_file.stem] = classes_in_image
    
    total_augmented = 0
    for img_file in source_img_dir.glob('*.*'):
        stem = img_file.stem
        if stem not in image_class_map:
            continue
        
        # Copy original to target
        shutil.copy2(img_file, target_img_dir / img_file.name)
        shutil.copy2(source_lbl_dir / f"{stem}.txt", target_lbl_dir / f"{stem}.txt")
        
        # Find max multiplier for this image
        max_mult = 1
        for cls_id in image_class_map[stem]:
            if cls_id in MULTIPLIERS:
                max_mult = max(max_mult, MULTIPLIERS[cls_id])
        
        if max_mult > 1:
            # Generate augmented variants
            for i in range(max_mult - 1):
                aug_img, aug_bboxes, aug_labels = augment_yolo_sample(img_file, source_lbl_dir / f"{stem}.txt")
                if aug_img is None or len(aug_bboxes) == 0:
                    continue  # Skip if no valid bboxes survived
                
                # Save augmented image
                new_img_name = f"{stem}_aug_{i}{img_file.suffix}"
                cv2.imwrite(str(target_img_dir / new_img_name), aug_img)
                
                # Save augmented labels
                new_lbl_name = f"{stem}_aug_{i}.txt"
                with open(target_lbl_dir / new_lbl_name, 'w') as f:
                    for bbox, label in zip(aug_bboxes, aug_labels):
                        f.write(f"{label} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                
                total_augmented += 1
    
    print(f"Augmentation complete. Generated {total_augmented} new augmented samples.")
    original_count = len(list(source_img_dir.glob('*.*')))
    augmented_count = len(list(target_img_dir.glob('*.*')))
    print(f"Original train images: {original_count}")
    print(f"Augmented dataset images: {augmented_count}")

if __name__ == '__main__':
    oversample_dataset()