from pathlib import Path
import yaml

# Load class names
with open('data/processed/unified_dataset/data.yaml', 'r') as f:
    config = yaml.safe_load(f)
class_names = config['names']

# Count function
def count_instances(label_dir):
    counts = {i: 0 for i in range(len(class_names))}
    total_bboxes = 0
    
    for label_file in Path(label_dir).glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    counts[class_id] += 1
                    total_bboxes += 1
    
    return counts, total_bboxes

# Count FULL dataset (train + val + test)
print("=" * 60)
print("FULL DATASET (train + val + test)")
print("=" * 60)
full_train_counts, full_train_total = count_instances('data/processed/unified_dataset/train/labels')
full_val_counts, full_val_total = count_instances('data/processed/unified_dataset/val/labels')
full_test_counts, full_test_total = count_instances('data/processed/unified_dataset/test/labels')

full_counts = {i: full_train_counts[i] + full_val_counts[i] + full_test_counts[i] for i in range(10)}
full_total = full_train_total + full_val_total + full_test_total

print(f"\nTotal images: 3983")
print(f"Total bounding boxes: {full_total}\n")
for class_id in sorted(full_counts.keys()):
    count = full_counts[class_id]
    pct = (count / full_total) * 100
    print(f"  Class {class_id:2d} ({class_names[class_id]:20s}): {count:6d} ({pct:5.2f}%)")

# Count TRAIN+VAL only (exclude test set)
print("\n" + "=" * 60)
print("TRAIN + VAL ONLY (exclude test set)")
print("=" * 60)

tv_counts = {i: full_train_counts[i] + full_val_counts[i] for i in range(10)}
tv_total = full_train_total + full_val_total

print(f"\nTotal images: 3376")
print(f"Total bounding boxes: {tv_total}\n")
for class_id in sorted(tv_counts.keys()):
    count = tv_counts[class_id]
    pct = (count / tv_total) * 100
    print(f"  Class {class_id:2d} ({class_names[class_id]:20s}): {count:6d} ({pct:5.2f}%)")

# Calculate RATIOS for opposing pairs
print("\n" + "=" * 60)
print("OPPOSING CLASS RATIOS (for imbalance check)")
print("=" * 60)

pairs = [
    (5, 6, "mask vs no_mask"),
    (1, 2, "hardhat vs no_hardhat"),
    (3, 4, "safety_vest vs no_safety_vest")
]

for class_a, class_b, desc in pairs:
    count_a = tv_counts[class_a]
    count_b = tv_counts[class_b]
    if count_a > count_b:
        ratio = count_a / count_b
        print(f"\n  {desc}:")
        print(f"    {class_names[class_a]}: {count_a}")
        print(f"    {class_names[class_b]}: {count_b}")
        print(f"    Ratio ({class_names[class_a]}:{class_names[class_b]}): 1:{ratio:.2f}")
    else:
        ratio = count_b / count_a
        print(f"\n  {desc}:")
        print(f"    {class_names[class_a]}: {count_a}")
        print(f"    {class_names[class_b]}: {count_b}")
        print(f"    Ratio ({class_names[class_b]}:{class_names[class_a]}): 1:{ratio:.2f}")

# Calculate ratios for generalist classes
print("\n" + "=" * 60)
print("GENERALIST CLASS RATIOS (person vs others)")
print("=" * 60)

person_count = tv_counts[0]
for class_id in [7, 8, 9]:  # safety_cone, machinery, vehicle
    count = tv_counts[class_id]
    ratio = person_count / count
    print(f"\n  person vs {class_names[class_id]}:")
    print(f"    person: {person_count}")
    print(f"    {class_names[class_id]}: {count}")
    print(f"    Ratio (person:{class_names[class_id]}): 1:{ratio:.2f}")

# Save results to file for research analyst
import json
results = {
    "full_dataset": {"counts": full_counts, "total": full_total},
    "train_val_only": {"counts": tv_counts, "total": tv_total},
    "class_names": class_names,
    "imbalance_threshold": 1.2,
    "opposing_pairs": [
        {"classes": [5, 6], "names": ["mask", "no_mask"], "ratio": tv_counts[5] / tv_counts[6] if tv_counts[6] > 0 else float('inf')},
        {"classes": [1, 2], "names": ["hardhat", "no_hardhat"], "ratio": tv_counts[2] / tv_counts[1] if tv_counts[1] > 0 else float('inf')},
        {"classes": [3, 4], "names": ["safety_vest", "no_safety_vest"], "ratio": tv_counts[4] / tv_counts[3] if tv_counts[3] > 0 else float('inf')}
    ],
    "generalist_classes": [
        {"class": 0, "name": "person", "count": tv_counts[0]},
        {"class": 7, "name": "safety_cone", "count": tv_counts[7]},
        {"class": 8, "name": "machinery", "count": tv_counts[8]},
        {"class": 9, "name": "vehicle", "count": tv_counts[9]}
    ]
}

with open('docs/class_imbalance_data.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print("Results saved to: docs/class_imbalance_data.json")
print("=" * 60)