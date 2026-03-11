from huggingface_hub import HfApi, create_repo

REPO_ID = "raj5517/imu-activity-classifier"
api = HfApi()

# ─────────────────────────────────────────
# 1. Create repo
# ─────────────────────────────────────────
create_repo(REPO_ID, repo_type="model", exist_ok=True)
print(f"✅ Repo ready: https://huggingface.co/{REPO_ID}")

# ─────────────────────────────────────────
# 2. Upload model files
# ─────────────────────────────────────────
uploads = [
    ("models/imu_pruned_int8.tflite",  "imu_pruned_int8.tflite"),
    ("models/imu_pruned_fp16.tflite",  "imu_pruned_fp16.tflite"),
    ("outputs/confusion_matrix.png",   "confusion_matrix.png"),
    ("outputs/training_curves.png",    "training_curves.png"),
    ("outputs/compression_summary.png","compression_summary.png"),
    ("model.py",                       "model.py"),
    ("requirements.txt",               "requirements.txt"),
]

for local_path, repo_path in uploads:
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=REPO_ID,
        repo_type="model",
    )
    print(f"  ✅ Uploaded {repo_path}")

# ─────────────────────────────────────────
# 3. Upload model card
# ─────────────────────────────────────────
model_card = """---
language: en
license: mit
tags:
  - tensorflow
  - tflite
  - time-series
  - activity-recognition
  - imu
  - edge-deployment
  - pruning
  - quantization
datasets:
  - uci-har
metrics:
  - accuracy
---

# IMU Activity Classifier — Pruning + INT8 Quantization

Compact 1D-CNN for human activity recognition from 6-axis IMU signals.
Trained on UCI HAR dataset, compressed via magnitude pruning (78% sparsity)
+ INT8 quantization for edge/microcontroller deployment.

## Results

| Model | Size | Accuracy | Latency |
|-------|------|----------|---------|
| Baseline FP32 | 358 KB | 93.99% | 46.1 ms |
| Pruned FP16 TFLite | 191 KB | 92.40% | 0.054 ms |
| **Pruned INT8 TFLite** | **113 KB** | **92.43%** | **0.026 ms** |

- Size reduction: 68.4%
- Latency speedup: 1775x
- Accuracy drop: 1.56%

## Classes
WALKING · WALKING_UPSTAIRS · WALKING_DOWNSTAIRS · SITTING · STANDING · LAYING

## Compression Pipeline
1. Baseline 1D-CNN trained on UCI HAR (93.99% accuracy)
2. Magnitude pruning with PolynomialDecay → 78% sparsity
3. INT8 post-training quantization → 113KB TFLite model

## Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

## Compression Summary
![Compression Summary](compression_summary.png)

## Training Curves
![Training Curves](training_curves.png)

## Usage
```python
import tensorflow as tf
import numpy as np

interpreter = tf.lite.Interpreter("imu_pruned_int8.tflite")
interpreter.allocate_tensors()
input_det  = interpreter.get_input_details()
output_det = interpreter.get_output_details()

# sample shape: (1, 128, 9) — float32
sample = np.random.randn(1, 128, 9).astype(np.float32)
interpreter.set_tensor(input_det[0]['index'], sample)
interpreter.invoke()
output = interpreter.get_tensor(output_det[0]['index'])

ACTIVITIES = ['WALKING','WALKING_UPSTAIRS','WALKING_DOWNSTAIRS',
              'SITTING','STANDING','LAYING']
print(ACTIVITIES[np.argmax(output)])
```

## Links
- GitHub: https://github.com/RAj5517/imu_activity_classifier
- Dataset: UCI HAR (University of California Irvine)
"""

api.upload_file(
    path_or_fileobj=model_card.encode(),
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="model",
)
print("  ✅ Uploaded README.md (model card)")
print(f"\n🎉 Done! https://huggingface.co/{REPO_ID}")