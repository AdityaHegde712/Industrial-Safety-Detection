#!/usr/bin/env python3
"""
Generate Dataset Statistics Report for SimaAI Industry Safety System
"""

import yaml
import json
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "processed" / "unified_dataset"
REPORTS_DIR = PROJECT_ROOT / "reports"

def count_labels(label_dir):
    """Count class occurrences in label directory"""
    class_counts = Counter()
    total_bboxes = 0
    bbox_sizes = []
    
    for label_file in label_dir.glob("*.txt"):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
                    total_bboxes += 1
                    # Calculate bbox area (width * height)
                    w = float(parts[3])
                    h = float(parts[4])
                    bbox_sizes.append(w * h)
    
    return class_counts, total_bboxes, bbox_sizes

def get_image_stats(image_dir):
    """Get image dimension statistics"""
    # This would require actually reading images, which can be slow
    # For now, return placeholder
    return {"total_images": len(list(image_dir.glob("*.*")))}

def generate_report():
    """Generate comprehensive dataset statistics report"""
    REPORTS_DIR.mkdir(exist_ok=True)
    
    # Load dataset config
    with open(DATASET_DIR / "data.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    class_names = config['names']
    
    report = []
    report.append("# Dataset Statistics Report")
    report.append("## SimaAI Industry Safety System\n")
    
    report.append("### Dataset Overview")
    report.append(f"- **Total Images**: {sum([len(list((DATASET_DIR / split / 'images').glob('*.*'))) for split in ['train', 'val', 'test']])}")
    report.append(f"- **Train/Val/Test Split**: 70%/15%/15%")
    report.append(f"- **Number of Classes**: {len(class_names)}")
    report.append(f"- **Source Datasets**: Kaggle Construction Site Safety, Roboflow Construction Safety\n")
    
    # Process each split
    for split in ['train', 'val', 'test']:
        img_dir = DATASET_DIR / split / "images"
        label_dir = DATASET_DIR / split / "labels"
        
        n_images = len(list(img_dir.glob("*.*")))
        class_counts, total_bboxes, bbox_sizes = count_labels(label_dir)
        
        report.append(f"## {split.upper()} Set")
        report.append(f"- **Images**: {n_images}")
        report.append(f"- **Total Bounding Boxes**: {total_bboxes}")
        report.append(f"- **Average BBoxes per Image**: {total_bboxes / n_images:.2f}\n")
        
        report.append("### Class Distribution")
        report.append("| Class ID | Class Name | Count | Percentage |")
        report.append("|----------|------------|-------|------------|")
        
        for class_id in sorted(class_counts.keys()):
            count = class_counts[class_id]
            pct = (count / total_bboxes) * 100 if total_bboxes > 0 else 0
            report.append(f"| {class_id} | {class_names[class_id]:20s} | {count:5d} | {pct:6.2f}% |")
        
        report.append("")
        
        # BBox size statistics
        if bbox_sizes:
            bbox_sizes = np.array(bbox_sizes)
            report.append("### Bounding Box Size Statistics (Normalized Area)")
            report.append(f"- **Mean**: {bbox_sizes.mean():.4f}")
            report.append(f"- **Std**: {bbox_sizes.std():.4f}")
            report.append(f"- **Min**: {bbox_sizes.min():.4f}")
            report.append(f"- **Max**: {bbox_sizes.max():.4f}")
            report.append(f"- **Median**: {np.median(bbox_sizes):.4f}\n")
    
    # Save report
    report_path = REPORTS_DIR / "dataset_statistics.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Report saved to: {report_path}")
    
    # Also create a simple visualization
    splits = ['train', 'val', 'test']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, split in enumerate(splits):
        label_dir = DATASET_DIR / split / "labels"
        class_counts, _, _ = count_labels(label_dir)
        
        class_ids = sorted(class_counts.keys())
        counts = [class_counts[c] for c in class_ids]
        labels = [class_names[c] for c in class_ids]
        
        axes[idx].barh(range(len(counts)), counts)
        axes[idx].set_yticks(range(len(counts)))
        axes[idx].set_yticklabels(labels, fontsize=8)
        axes[idx].set_title(f"{split.upper()} Set Class Distribution")
        axes[idx].set_xlabel("Count")
    
    plt.tight_layout()
    viz_path = REPORTS_DIR / "dataset_class_distribution.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {viz_path}")
    
    return report_path

if __name__ == "__main__":
    generate_report()
