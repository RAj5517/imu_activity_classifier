import os
os.environ['TF_USE_LEGACY_KERAS'] = '0'

import numpy as np
import tensorflow as tf
import tf_keras
import tensorflow_model_optimization as tfmot
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import time

from model import ACTIVITIES

# ─────────────────────────────────────────
# 0. Setup
# ─────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

os.makedirs('models', exist_ok=True)

# ─────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────
print("Loading data...")
X_train = np.load('data/X_train.npy').astype(np.float32)
X_test  = np.load('data/X_test.npy').astype(np.float32)
y_train = np.load('data/y_train.npy').astype(int)
y_test  = np.load('data/y_test.npy').astype(int)

Y_train = to_categorical(y_train, 6)
Y_test  = to_categorical(y_test,  6)

X_tr, X_val, Y_tr, Y_val = train_test_split(
    X_train, Y_train,
    test_size=0.15,
    random_state=42,
    stratify=y_train
)

# ─────────────────────────────────────────
# 2. Rebuild in tf_keras + copy weights
#    tfmot 0.8.0 requires tf_keras Functional
#    not Keras 3 Functional
# ─────────────────────────────────────────
print("Loading baseline model...")

def build_imu_tfkeras(input_shape=(128, 9), n_classes=6):
    inputs = tf_keras.Input(shape=input_shape, name='imu_input')

    x = tf_keras.layers.Conv1D(32, kernel_size=5, padding='same',
                                activation='relu', name='conv1')(inputs)
    x = tf_keras.layers.BatchNormalization(name='bn1')(x)
    x = tf_keras.layers.MaxPooling1D(pool_size=2, name='pool1')(x)
    x = tf_keras.layers.Dropout(0.1, name='drop1')(x)

    x = tf_keras.layers.Conv1D(64, kernel_size=3, padding='same',
                                activation='relu', name='conv2')(x)
    x = tf_keras.layers.BatchNormalization(name='bn2')(x)
    x = tf_keras.layers.MaxPooling1D(pool_size=2, name='pool2')(x)
    x = tf_keras.layers.Dropout(0.1, name='drop2')(x)

    x = tf_keras.layers.Conv1D(128, kernel_size=3, padding='same',
                                activation='relu', name='conv3')(x)
    x = tf_keras.layers.BatchNormalization(name='bn3')(x)
    x = tf_keras.layers.MaxPooling1D(pool_size=2, name='pool3')(x)
    x = tf_keras.layers.Dropout(0.1, name='drop4')(x)

    x = tf_keras.layers.Conv1D(128, kernel_size=3, padding='same',
                                activation='relu', name='conv4')(x)
    x = tf_keras.layers.BatchNormalization(name='bn4')(x)

    x = tf_keras.layers.GlobalAveragePooling1D(name='gap')(x)
    x = tf_keras.layers.Dense(64, activation='relu', name='dense1')(x)
    x = tf_keras.layers.Dropout(0.3, name='drop5')(x)
    outputs = tf_keras.layers.Dense(n_classes, activation='softmax',
                                    name='activity')(x)

    return tf_keras.Model(inputs, outputs, name='IMU_Classifier')

# Load trained Keras 3 model, copy weights into tf_keras model
keras3_model   = tf.keras.models.load_model('models/imu_baseline.keras')
baseline_model = build_imu_tfkeras()
baseline_model.set_weights(keras3_model.get_weights())
baseline_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

_, base_acc = baseline_model.evaluate(X_test, Y_test, verbose=0)
base_size   = baseline_model.count_params() * 4 / 1024
print(f"Baseline accuracy : {base_acc*100:.2f}%")
print(f"Baseline size     : {base_size:.1f} KB  (FP32)")

# ═══════════════════════════════════════════════════════
# STAGE 1 — PRUNING (80% sparsity)
# ═══════════════════════════════════════════════════════
print("\n── Stage 1: Pruning ─────────────────────────────")

end_step = len(X_tr) // 32 * 30

pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.80,
        begin_step=0,
        end_step=end_step
    )
}

prunable_model = tfmot.sparsity.keras.prune_low_magnitude(
    baseline_model, **pruning_params
)

prunable_model.compile(
    optimizer=tf_keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

pruning_callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    # No EarlyStopping — pruning schedule MUST run all 30 epochs
    # to reach 80% sparsity via PolynomialDecay
]

start = time.time()
prunable_model.fit(
    X_tr, Y_tr,
    validation_data=(X_val, Y_val),
    epochs=30,
    batch_size=32,
    callbacks=pruning_callbacks,
    verbose=1
)
print(f"Pruning time: {(time.time()-start)/60:.1f} min")

# Strip pruning wrappers
# Strip pruning wrappers
pruned_model = tfmot.sparsity.keras.strip_pruning(prunable_model)

# Must recompile after stripping — strip removes optimizer state
pruned_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
pruned_model.save('models/imu_pruned.keras')

_, pruned_acc = pruned_model.evaluate(X_test, Y_test, verbose=0)
print(f"\nPruned accuracy (80% sparsity): {pruned_acc*100:.2f}%")
print(f"Accuracy drop : {(base_acc - pruned_acc)*100:.2f}%")

# Actual sparsity check
total_weights = 0
zero_weights  = 0
for layer in pruned_model.layers:
    for weight in layer.weights:
        w = weight.numpy()
        total_weights += w.size
        zero_weights  += (w == 0).sum()

actual_sparsity = 100 * zero_weights / total_weights
print(f"Actual sparsity : {actual_sparsity:.1f}%")

# ═══════════════════════════════════════════════════════
# STAGE 2a — FP16 TFLITE
# ═══════════════════════════════════════════════════════
print("\n── Stage 2a: FP16 TFLite Conversion ────────────")

converter_fp16 = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
converter_fp16.target_spec.supported_types = [tf.float16]
tflite_fp16 = converter_fp16.convert()

with open('models/imu_pruned_fp16.tflite', 'wb') as f:
    f.write(tflite_fp16)
print(f"FP16 model size: {len(tflite_fp16)/1024:.1f} KB")

# ═══════════════════════════════════════════════════════
# STAGE 2b — INT8 TFLITE
# ═══════════════════════════════════════════════════════
print("\n── Stage 2b: INT8 TFLite Conversion ────────────")

rep_data = X_train[:200].astype(np.float32)

def representative_dataset():
    for i in range(len(rep_data)):
        yield [rep_data[i:i+1]]

converter_int8 = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_dataset
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type  = tf.float32
converter_int8.inference_output_type = tf.float32

tflite_int8 = converter_int8.convert()

with open('models/imu_pruned_int8.tflite', 'wb') as f:
    f.write(tflite_int8)
print(f"INT8 model size: {len(tflite_int8)/1024:.1f} KB")

# ═══════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════
fp16_size = len(tflite_fp16) / 1024
int8_size = len(tflite_int8) / 1024

print("\n" + "═"*52)
print("  COMPRESSION PIPELINE — FINAL RESULTS")
print("═"*52)
print(f"  Baseline  (FP32 Keras)   {base_size:>7.1f} KB   {base_acc*100:.2f}%")
print(f"  Pruned    (FP32 Keras)   {base_size:>7.1f} KB   {pruned_acc*100:.2f}%  ({actual_sparsity:.0f}% zeros)")
print(f"  Pruned    (FP16 TFLite)  {fp16_size:>7.1f} KB")
print(f"  Pruned    (INT8 TFLite)  {int8_size:>7.1f} KB")
print(f"\n  Total compression  :  {(1 - int8_size/base_size)*100:.1f}% size reduction")
print(f"  Accuracy retained  :  {pruned_acc*100:.2f}%  (vs {base_acc*100:.2f}% baseline)")
print("═"*52)
print("\n✅ Models saved to models/")