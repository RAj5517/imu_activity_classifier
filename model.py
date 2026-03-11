import tensorflow as tf
from tensorflow.keras import layers, Model

ACTIVITIES = [
    'WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS',
    'SITTING', 'STANDING', 'LAYING'
]
N_CLASSES    = len(ACTIVITIES)
INPUT_SHAPE  = (128, 9)   # 128 timesteps, 9 channels


def build_imu_model(input_shape=INPUT_SHAPE, n_classes=N_CLASSES):
    """
    Compact 1D-CNN for IMU activity classification.
    Designed from scratch for pruning + INT8 quantization.

    Architecture rationale:
      - Conv1D blocks  → extract local motion patterns per axis
      - BatchNorm      → stable training, works well post-quantization
      - MaxPooling     → downsample before deeper layers
      - GlobalAvgPool  → replaces Flatten, much fewer params, less overfit
      - Dense head     → final classifier

    Total params: ~85K   (intentionally small for edge deployment)
    """

    inputs = tf.keras.Input(shape=input_shape, name='imu_input')

    # ── Block 1 ──────────────────────────────
    x = layers.Conv1D(32, kernel_size=5, padding='same',
                      activation='relu', name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.MaxPooling1D(pool_size=2, name='pool1')(x)   # 128 → 64
    x = layers.Dropout(0.1, name='drop1')(x)

    # ── Block 2 ──────────────────────────────
    x = layers.Conv1D(64, kernel_size=3, padding='same',
                      activation='relu', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.MaxPooling1D(pool_size=2, name='pool2')(x)   # 64 → 32
    x = layers.Dropout(0.1, name='drop2')(x)

    # ── Block 3 ──────────────────────────────
    x = layers.Conv1D(128, kernel_size=3, padding='same',
                      activation='relu', name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.MaxPooling1D(pool_size=2, name='pool3')(x)   # 32 → 16
    x = layers.Dropout(0.1, name='drop4')(x)

    # ── Block 4 (deeper feature extraction) ──
    x = layers.Conv1D(128, kernel_size=3, padding='same',
                      activation='relu', name='conv4')(x)
    x = layers.BatchNormalization(name='bn4')(x)

    # ── Pooling + Head ────────────────────────
    x = layers.GlobalAveragePooling1D(name='gap')(x)        # [N, 128]

    x = layers.Dense(64, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.3, name='drop5')(x)

    outputs = layers.Dense(
        n_classes, activation='softmax', name='activity'
    )(x)

    model = Model(inputs, outputs, name='IMU_Classifier')
    return model


# ─────────────────────────────────────────────
# Quick sanity check when run directly
# ─────────────────────────────────────────────
if __name__ == '__main__':
    model = build_imu_model()
    model.summary()

    total   = model.count_params()
    size_kb = total * 4 / 1024   # FP32 = 4 bytes per param

    print(f"\n── Model Info ──────────────────────────")
    print(f"Total parameters  : {total:,}")
    print(f"FP32 size estimate: {size_kb:.1f} KB")
    print(f"Input shape       : {INPUT_SHAPE}")
    print(f"Output classes    : {N_CLASSES}  →  {ACTIVITIES}")
    print(f"\n✅ Model built successfully.")