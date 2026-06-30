import os
import random
import shutil
from pathlib import Path
import yaml

random.seed(42)

src_labels = Path('data/processed/unified_dataset/train/labels')
src_images = Path('data/processed/unified_dataset/train/images')
out_dir = Path('data/experiments/subset_20pct')

# Clean old output
for d in [out_dir / 'train_search', out_dir / 'val_search']:
    if d.exists():
        shutil.rmtree(d)

# Build class -> image mapping
class_to_images = {i: set() for i in range(10)}
all_stems = set()

label_files = list(src_labels.glob('*.txt'))
for lf in label_files:
    stem = lf.stem
    all_stems.add(stem)
    with open(lf) as f:
        for line in f:
            line = line.strip()
            if line:
                cls = int(line.split()[0])
                class_to_images[cls].add(stem)

# Select 20% of ALL unique images
all_stems = sorted(all_stems)
random.shuffle(all_stems)
n_select = max(1, int(len(all_stems) * 0.2))
selected = set(all_stems[:n_select])

# 90-10 split
selected_list = sorted(selected)
random.shuffle(selected_list)
split = int(len(selected_list) * 0.9)
train_stems = set(selected_list[:split])
val_stems = set(selected_list[split:])

# Create output dirs
(train_out_img := out_dir / 'train_search' / 'images').mkdir(parents=True)
(train_out_lbl := out_dir / 'train_search' / 'labels').mkdir(parents=True)
(val_out_img := out_dir / 'val_search' / 'images').mkdir(parents=True)
(val_out_lbl := out_dir / 'val_search' / 'labels').mkdir(parents=True)

def copy_files(stems, img_dst, lbl_dst):
    for s in stems:
        for ext in ['.jpg', '.jpeg', '.png']:
            src_img = src_images / f"{s}{ext}"
            if src_img.exists():
                shutil.copy2(src_img, img_dst / src_img.name)
                break
        src_lbl = src_labels / f"{s}.txt"
        if src_lbl.exists():
            shutil.copy2(src_lbl, lbl_dst / src_lbl.name)

copy_files(train_stems, train_out_img, train_out_lbl)
copy_files(val_stems, val_out_img, val_out_lbl)

# Create YAML
yaml_content = {
    'path': 'C:/Users/hifia/Projects/SimaAI_Industry_Safety/data/experiments/subset_20pct',
    'train': 'train_search/images',
    'val': 'val_search/images',
    'names': {i: name for i, name in enumerate(['person','hardhat','no_hardhat','safety_vest','no_safety_vest','mask','no_mask','safety_cone','machinery','vehicle'])}
}
with open(out_dir / 'data_experiment.yaml', 'w') as f:
    yaml.dump(yaml_content, f, default_flow_style=False)

# Report
total_orig = len(all_stems)
total_train = len(train_stems)
total_val = len(val_stems)
pct = (total_train + total_val) / total_orig * 100
print(f"Original images: {total_orig}")
print(f"Train images: {total_train}")
print(f"Val images: {total_val}")
print(f"Total subset: {total_train + total_val} ({pct:.1f}% of original)")
print()
print(f"{'Class':20s} {'Orig':>6s} {'Train':>6s} {'Val':>6s} {'Pct':>6s}")
for cls, name in enumerate(['person','hardhat','no_hardhat','safety_vest','no_safety_vest','mask','no_mask','safety_cone','machinery','vehicle']):
    orig = len(class_to_images[cls])
    train_cnt = sum(1 for s in train_stems if s in class_to_images[cls])
    val_cnt = sum(1 for s in val_stems if s in class_to_images[cls])
    subset_pct = (train_cnt + val_cnt) / orig * 100 if orig > 0 else 0
    print(f"{name:20s} {orig:6d} {train_cnt:6d} {val_cnt:6d} {subset_pct:5.1f}%")
