import itertools
import yaml
import torch
import gc
from ultralytics import YOLO
from pathlib import Path

SEARCH_SPACE = {
    'model': ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt'],
    'lr0': [0.001, 0.005, 0.01],
    'optimizer': ['SGD', 'AdamW'],
}

# Raw class imbalance weights
CLS_PW = [0.5, 2.15, 4.98, 2.82, 2.50, 7.54, 3.87, 3.44, 2.35, 7.57]
mean_w = sum(CLS_PW) / len(CLS_PW)
CLS_PW_NORM = [w / mean_w for w in CLS_PW]

DATA_YAML = 'data/experiments/subset_20pct/data_experiment.yaml'
Path('configs').mkdir(exist_ok=True)

results_log = []

for model_name, lr0, optimizer in itertools.product(
    SEARCH_SPACE['model'],
    SEARCH_SPACE['lr0'],
    SEARCH_SPACE['optimizer']
):
    exp_name = f"{Path(model_name).stem}_lr{lr0}_{optimizer.lower()}"
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"Starting: model={model_name}, lr0={lr0}, optimizer={optimizer}")
    print(f"{'='*60}")
    
    try:
        model = YOLO(model_name)
        # Set per-class weights via internal attribute (cls_pw param rejects lists)
        model.model.class_weights = torch.tensor(CLS_PW_NORM, dtype=torch.float32).to(model.device)
        
        extra_kwargs = {}
        if optimizer == 'SGD':
            extra_kwargs['momentum'] = 0.937
        
        results = model.train(
            data=DATA_YAML,
            epochs=30,
            imgsz=640,
            batch=16,
            name=exp_name,
            patience=5,
            save=True,
            save_period=-1,
            lr0=lr0,
            lrf=0.01,
            optimizer=optimizer,
            weight_decay=0.0005,
            warmup_epochs=3,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            close_mosaic=10,
            workers=2,          # Reduced from 8 to save system RAM
            fraction=0.8,       # Cap GPU memory at 80%
            plots=True,
            **extra_kwargs
        )
        
        metrics = {
            'model': model_name,
            'lr0': lr0,
            'optimizer': optimizer,
            'mAP50': results.box.map50 if hasattr(results, 'box') else 0,
            'mAP50_95': results.box.map if hasattr(results, 'box') else 0,
        }
        results_log.append(metrics)
        
        print(f"  ✅ mAP@0.5: {metrics['mAP50']:.4f}")
        print(f"     mAP@0.5:0.95: {metrics['mAP50_95']:.4f}")
        
    except Exception as e:
        print(f"  ❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        results_log.append({
            'model': model_name, 'lr0': lr0, 'optimizer': optimizer,
            'mAP50': 0, 'mAP50_95': 0, 'error': str(e)
        })
    
    finally:
        # CRITICAL: GPU memory cleanup between experiments
        print("  Cleaning up GPU memory...")
        if 'model' in locals():
            del model
        if 'results' in locals():
            del results
        gc.collect()
        torch.cuda.empty_cache()
        # Save intermediate results crash-safe
        with open('configs/search_results_intermediate.yaml', 'w') as f:
            yaml.dump(results_log, f, default_flow_style=False)

# Final cleanup
print("\nAll experiments complete. Final cleanup...")
gc.collect()
torch.cuda.empty_cache()

# Print summary table
print(f"\n{'='*80}")
print(f"HYPERPARAMETER SEARCH RESULTS")
print(f"{'='*80}")
print(f"{'Model':15s} {'lr0':8s} {'Optimizer':10s} {'mAP@0.5':10s} {'mAP@0.5:0.95':10s}")
print(f"{'-'*15} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

sorted_results = sorted(results_log, key=lambda x: x['mAP50'], reverse=True)
for r in sorted_results:
    print(f"{r['model']:15s} {r['lr0']:<8.3f} {r['optimizer']:10s} {r['mAP50']:<10.4f} {r['mAP50_95']:<10.4f}")

# Save best config
best = sorted_results[0]
best_config = {
    'model': best['model'],
    'lr0': best['lr0'],
    'optimizer': best['optimizer'],
    'batch': 16,
    'cls_pw': CLS_PW_NORM,
    'epochs': 100,
    'workers': 2,
    'fraction': 0.8,
    'mAP50_achieved': best['mAP50'],
    'mAP50_95_achieved': best['mAP50_95'],
}

with open('configs/best_config.yaml', 'w') as f:
    yaml.dump(best_config, f, default_flow_style=False)

# Save final results
with open('configs/search_results.yaml', 'w') as f:
    yaml.dump(sorted_results, f, default_flow_style=False)

print(f"\nBest config saved to: configs/best_config.yaml")
print(f"Full results saved to: configs/search_results.yaml")
print(f"Winner: {best['model']} | lr0={best['lr0']} | {best['optimizer']} | mAP@0.5={best['mAP50']:.4f}")
