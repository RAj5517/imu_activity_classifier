import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import os, time
from model import build_imu_model, ACTIVITIES

# ─────────────────────────────────────────
# 0. GPU config
# ─────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"✅ GPU found: {gpus[0].name}")
else:
    print("⚠️  No GPU found, running on CPU")

# ─────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────
print("\nLoading data...")
X_train = np.load('data/X_train.npy').astype(np.float32)
X_test  = np.load('data/X_test.npy').astype(np.float32)
y_train = np.load('data/y_train.npy').astype(int)
y_test  = np.load('data/y_test.npy').astype(int)

Y_train = to_categorical(y_train, num_classes=6)
Y_test  = to_categorical(y_test,  num_classes=6)

print(f"X_train : {X_train.shape}  |  Y_train : {Y_train.shape}")
print(f"X_test  : {X_test.shape}   |  Y_test  : {Y_test.shape}")

# ─────────────────────────────────────────
# 2. Stratified train/val split
#    UCI HAR is subject-ordered — must shuffle
# ─────────────────────────────────────────
X_tr, X_val, Y_tr, Y_val, y_tr, _ = train_test_split(
    X_train, Y_train, y_train,
    test_size=0.15,
    random_state=42,
    stratify=y_train
)

print(f"\nTrain samples : {len(X_tr)}")
print(f"Val   samples : {len(X_val)}")

# ─────────────────────────────────────────
# 3. Class weights — fix SITTING/STANDING
# ─────────────────────────────────────────
cw = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(6),
    y=y_tr
)
class_weights = dict(enumerate(cw))
print(f"\nClass weights:")
for i, name in enumerate(ACTIVITIES):
    print(f"  {name:<25} {cw[i]:.3f}")

# ─────────────────────────────────────────
# 4. Build model
# ─────────────────────────────────────────
model = build_imu_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ─────────────────────────────────────────
# 5. Callbacks
# ─────────────────────────────────────────
os.makedirs('models', exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        filepath='models/imu_baseline.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
]

# ─────────────────────────────────────────
# 6. Train
# ─────────────────────────────────────────
print("\n── Training ─────────────────────────────")
start = time.time()

history = model.fit(
    X_tr, Y_tr,
    validation_data=(X_val, Y_val),
    epochs=100,
    batch_size=32,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

elapsed = time.time() - start
print(f"\nTraining time: {elapsed/60:.1f} minutes")

# ─────────────────────────────────────────
# 7. Evaluate
# ─────────────────────────────────────────
print("\n── Test Set Evaluation ──────────────────")
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
print(f"Test Accuracy : {test_acc*100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

print("\n── Per-Class Accuracy ───────────────────")
for i, name in enumerate(ACTIVITIES):
    mask    = y_test == i
    correct = (y_pred[mask] == i).sum()
    total   = mask.sum()
    print(f"  {name:<25} {correct:>4}/{total:<4}  ({100*correct/total:.1f}%)")

# ─────────────────────────────────────────
# 8. Save
# ─────────────────────────────────────────
model.save('models/imu_baseline.keras')
np.save('models/train_history.npy', history.history)
print(f"\n✅ Model saved  →  models/imu_baseline.keras")
print(f"✅ History saved →  models/train_history.npy")