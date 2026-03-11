import os
os.environ['TF_USE_LEGACY_KERAS'] = '0'

import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.utils import to_categorical
from model import ACTIVITIES

# ─────────────────────────────────────────
# 1. Load test data
# ─────────────────────────────────────────
X_test = np.load('data/X_test.npy').astype(np.float32)
y_test = np.load('data/y_test.npy').astype(int)
Y_test = to_categorical(y_test, 6)

# ─────────────────────────────────────────
# 2. Benchmark TFLite model
# ─────────────────────────────────────────
def benchmark_tflite(model_path, X_test, y_test, n_runs=500):
    print(f"\n── {model_path} ──────────────────────────")

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_det  = interpreter.get_input_details()
    output_det = interpreter.get_output_details()

    # Warmup — 20 runs to stabilize CPU/cache
    for i in range(20):
        sample = X_test[i:i+1]
        interpreter.set_tensor(input_det[0]['index'], sample)
        interpreter.invoke()

    # Benchmark latency
    times = []
    for i in range(n_runs):
        sample = X_test[i % len(X_test):i % len(X_test) + 1]
        start  = time.perf_counter()
        interpreter.set_tensor(input_det[0]['index'], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_det[0]['index'])
        end    = time.perf_counter()
        times.append((end - start) * 1000)

    # Accuracy on full test set
    correct = 0
    for i in range(len(X_test)):
        sample = X_test[i:i+1]
        interpreter.set_tensor(input_det[0]['index'], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_det[0]['index'])
        if np.argmax(output) == y_test[i]:
            correct += 1

    accuracy  = 100 * correct / len(X_test)
    avg_ms    = np.mean(times)
    p50_ms    = np.percentile(times, 50)
    p95_ms    = np.percentile(times, 95)
    p99_ms    = np.percentile(times, 99)
    size_kb   = os.path.getsize(model_path) / 1024

    print(f"  Size          : {size_kb:.1f} KB")
    print(f"  Accuracy      : {accuracy:.2f}%")
    print(f"  Avg latency   : {avg_ms:.3f} ms")
    print(f"  P50 latency   : {p50_ms:.3f} ms")
    print(f"  P95 latency   : {p95_ms:.3f} ms")
    print(f"  P99 latency   : {p99_ms:.3f} ms")
    print(f"  Sub-10ms      : {'✅ YES' if avg_ms < 10 else '⚠️  NO'}")
    print(f"  Sub-50ms      : {'✅ YES' if avg_ms < 50 else '⚠️  NO'}")

    return {
        'path': model_path,
        'size_kb': size_kb,
        'accuracy': accuracy,
        'avg_ms': avg_ms,
        'p95_ms': p95_ms,
    }

# ─────────────────────────────────────────
# 3. Benchmark Keras baseline (full model)
# ─────────────────────────────────────────
def benchmark_keras(model_path, X_test, y_test, n_runs=200):
    print(f"\n── {model_path} ──────────────────────────")

    model = tf.keras.models.load_model(model_path)

    # Warmup
    model.predict(X_test[:10], verbose=0)

    # Single-sample latency
    times = []
    for i in range(n_runs):
        sample = X_test[i % len(X_test):i % len(X_test) + 1]
        start  = time.perf_counter()
        model.predict(sample, verbose=0)
        end    = time.perf_counter()
        times.append((end - start) * 1000)

    y_pred   = np.argmax(model.predict(X_test, verbose=0), axis=1)
    accuracy = 100 * (y_pred == y_test).mean()
    avg_ms   = np.mean(times)
    p95_ms   = np.percentile(times, 95)
    if os.path.isdir(model_path):
        size_kb = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, filenames in os.walk(model_path)
            for f in filenames
        ) / 1024
    else:
        size_kb = os.path.getsize(model_path) / 1024

    print(f"  Size          : {size_kb:.1f} KB")
    print(f"  Accuracy      : {accuracy:.2f}%")
    print(f"  Avg latency   : {avg_ms:.3f} ms")
    print(f"  P95 latency   : {p95_ms:.3f} ms")

    return {
        'path': model_path,
        'size_kb': size_kb,
        'accuracy': accuracy,
        'avg_ms': avg_ms,
        'p95_ms': p95_ms,
    }

# ─────────────────────────────────────────
# 4. Run all benchmarks
# ─────────────────────────────────────────
print("=" * 56)
print("  IMU ACTIVITY CLASSIFIER — BENCHMARK RESULTS")
print("=" * 56)

results = []
results.append(benchmark_keras('models/imu_baseline.keras',    X_test, y_test))
results.append(benchmark_tflite('models/imu_pruned_fp16.tflite', X_test, y_test))
results.append(benchmark_tflite('models/imu_pruned_int8.tflite', X_test, y_test))

# ─────────────────────────────────────────
# 5. Final comparison table
# ─────────────────────────────────────────
print("\n" + "=" * 56)
print("  FINAL COMPARISON TABLE")
print("=" * 56)
print(f"  {'Model':<28} {'Size':>7}  {'Acc':>7}  {'Avg ms':>8}")
print(f"  {'-'*28} {'-'*7}  {'-'*7}  {'-'*8}")

labels = ['Baseline (Keras FP32)', 'Pruned FP16 TFLite', 'Pruned INT8 TFLite']
for label, r in zip(labels, results):
    print(f"  {label:<28} {r['size_kb']:>6.1f}K  {r['accuracy']:>6.2f}%  {r['avg_ms']:>7.3f}ms")

baseline_size = results[0]['size_kb']
int8_size     = results[2]['size_kb']
speedup       = results[0]['avg_ms'] / results[2]['avg_ms']

# Use TFLite FP16 as size baseline if keras size failed
ref_size = 358.3  # known FP32 param size from model.py
print(f"\n  Size reduction  : {(1 - int8_size/ref_size)*100:.1f}%  (358KB FP32 → {int8_size:.1f}KB INT8)")
print(f"  Latency speedup : {speedup:.0f}x  (Keras single-sample → INT8 TFLite)")
print("=" * 56)