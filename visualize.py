import os
os.environ['TF_USE_LEGACY_KERAS'] = '0'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from model import ACTIVITIES

os.makedirs('outputs', exist_ok=True)

# ─────────────────────────────────────────
# 1. Load data + model
# ─────────────────────────────────────────
X_test = np.load('data/X_test.npy').astype(np.float32)
y_test = np.load('data/y_test.npy').astype(int)
history = np.load('models/train_history.npy', allow_pickle=True).item()

model  = tf.keras.models.load_model('models/imu_baseline.keras')
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

SHORT = ['WALK', 'W_UP', 'W_DOWN', 'SIT', 'STAND', 'LAY']

# ═══════════════════════════════════════════════════════
# PLOT 1 — Confusion Matrix
# ═══════════════════════════════════════════════════════
print("Generating confusion matrix...")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=SHORT)
disp.plot(ax=ax, colorbar=True, cmap='Blues')

ax.set_title(
    'IMU Activity Classifier — Confusion Matrix\n'
    f'Test Accuracy: 93.99%  |  UCI HAR Dataset',
    fontsize=13, fontweight='bold', pad=15
)
ax.set_xlabel('Predicted Label', fontsize=11)
ax.set_ylabel('True Label', fontsize=11)

# Annotate diagonal with per-class accuracy
for i in range(len(SHORT)):
    total   = cm[i].sum()
    correct = cm[i][i]
    pct     = 100 * correct / total
    ax.text(i, i, f'\n\n{pct:.1f}%',
            ha='center', va='center',
            fontsize=8, color='white' if pct > 50 else 'black')

plt.tight_layout()
plt.savefig('outputs/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ outputs/confusion_matrix.png")

# ═══════════════════════════════════════════════════════
# PLOT 2 — Training Curves
# ═══════════════════════════════════════════════════════
print("Generating training curves...")

epochs = range(1, len(history['accuracy']) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    'IMU Activity Classifier — Training History',
    fontsize=14, fontweight='bold', y=1.01
)

# Accuracy
ax1.plot(epochs, history['accuracy'],     color='#2196F3', linewidth=2, label='Train')
ax1.plot(epochs, history['val_accuracy'], color='#FF5722', linewidth=2,
         linestyle='--', label='Validation')
ax1.axhline(y=0.9399, color='green', linewidth=1.5,
            linestyle=':', label=f'Test: 93.99%')
ax1.set_title('Accuracy', fontsize=12, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.set_ylim([0.7, 1.0])
ax1.grid(True, alpha=0.3)

# Loss
ax2.plot(epochs, history['loss'],     color='#2196F3', linewidth=2, label='Train')
ax2.plot(epochs, history['val_loss'], color='#FF5722', linewidth=2,
         linestyle='--', label='Validation')
ax2.set_title('Loss', fontsize=12, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ outputs/training_curves.png")

# ═══════════════════════════════════════════════════════
# PLOT 3 — Model Compression Summary Bar Chart
# ═══════════════════════════════════════════════════════
print("Generating compression summary chart...")

models   = ['Baseline\n(FP32 Keras)', 'Pruned\n(FP16 TFLite)', 'Pruned\n(INT8 TFLite)']
sizes    = [358.3, 191.4, 113.3]
accs     = [93.99, 92.40, 92.43]
latency  = [46.121, 0.054, 0.026]

colors_size = ['#455A64', '#1976D2', '#388E3C']
colors_acc  = ['#455A64', '#1976D2', '#388E3C']

fig = plt.figure(figsize=(14, 5))
fig.suptitle(
    'IMU Classifier — Compression Pipeline Results',
    fontsize=14, fontweight='bold'
)
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)

# Size bar
ax1 = fig.add_subplot(gs[0])
bars = ax1.bar(models, sizes, color=colors_size, edgecolor='white', width=0.5)
ax1.set_title('Model Size (KB)', fontweight='bold')
ax1.set_ylabel('KB')
for bar, val in zip(bars, sizes):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'{val:.0f}KB', ha='center', fontsize=10, fontweight='bold')
ax1.set_ylim(0, 450)
ax1.grid(axis='y', alpha=0.3)

# Accuracy bar
ax2 = fig.add_subplot(gs[1])
bars = ax2.bar(models, accs, color=colors_acc, edgecolor='white', width=0.5)
ax2.set_title('Test Accuracy (%)', fontweight='bold')
ax2.set_ylabel('Accuracy %')
for bar, val in zip(bars, accs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.2f}%', ha='center', fontsize=10, fontweight='bold')
ax2.set_ylim(88, 96)
ax2.grid(axis='y', alpha=0.3)

# Latency bar (log scale)
ax3 = fig.add_subplot(gs[2])
bars = ax3.bar(models, latency, color=colors_acc, edgecolor='white', width=0.5)
ax3.set_title('Avg Inference Latency (ms)\n[log scale]', fontweight='bold')
ax3.set_ylabel('ms (log scale)')
ax3.set_yscale('log')
for bar, val in zip(bars, latency):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.3,
             f'{val:.3f}ms', ha='center', fontsize=10, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

plt.savefig('outputs/compression_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ outputs/compression_summary.png")

# ─────────────────────────────────────────
# Done
# ─────────────────────────────────────────
print("\n── All plots saved to outputs/ ──────────")
print("  confusion_matrix.png")
print("  training_curves.png")
print("  compression_summary.png")