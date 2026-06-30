import albumentations as A
import cv2
import numpy as np
from pathlib import Path

# DEFINE TRANSFORMS (NO vertical flip, ±30° rotation limit)
transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, p=0.5),
    A.HorizontalFlip(p=0.5),
    # NO VerticalFlip - unrealistic for PPE
    A.Rotate(limit=30, p=0.5),  # ±30° max
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def is_valid_bbox(bbox):
    """Check if YOLO bbox is within bounds after augmentation."""
    x, y, w, h = bbox
    # Center must be within image, width/height must be positive and reasonable
    if w < 0.01 or h < 0.01 or w > 1.0 or h > 1.0:
        return False
    # Bbox center and half-extent must stay within [0,1]
    x1, y1 = x - w/2, y - h/2
    x2, y2 = x + w/2, y + h/2
    return 0.0 <= x1 and x2 <= 1.0 and 0.0 <= y1 and y2 <= 1.0

def augment_yolo_sample(image_path, label_path):
    """Augment a single YOLO image+label pair."""
    # Read image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Parse YOLO labels
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
    
    # Apply augmentation
    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    
    # Filter invalid bboxes
    valid_bboxes = []
    valid_labels = []
    for bbox, label in zip(transformed['bboxes'], transformed['class_labels']):
        if is_valid_bbox(bbox):
            valid_bboxes.append(bbox)
            valid_labels.append(label)
    
    # Convert back to BGR for OpenCV saving
    aug_image = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)
    
    return aug_image, valid_bboxes, valid_labels

# Example usage
if __name__ == '__main__':
    img = Path('data/processed/unified_dataset/train/images/sample.jpg')
    lbl = Path('data/processed/unified_dataset/train/labels/sample.txt')
    aug_img, bboxes, labels = augment_yolo_sample(img, lbl)
    if aug_img is not None:
        print(f"Augmented image shape: {aug_img.shape}")
        print(f"Valid bboxes remaining: {len(bboxes)}")