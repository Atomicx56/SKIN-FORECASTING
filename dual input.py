# Dual-Input CNN: T1 + T2 -> Predict T3 (TensorFlow) with Improved Loss and Regularization

import tensorflow as tf
from tensorflow import keras
layers = keras.layers
models = keras.models
import numpy as np
import os
import matplotlib.pyplot as plt

# Load data
DATA_PATH = r"C:\Users\vishn\OneDrive\keshav and sri\Desktop\Project\Dataset\processed"
T1_train = np.load(os.path.join(DATA_PATH, "T1_train.npy"))
T2_train = np.load(os.path.join(DATA_PATH, "T2_train.npy"))
T3_train = np.load(os.path.join(DATA_PATH, "T3_train.npy"))
T1_test = np.load(os.path.join(DATA_PATH, "T1_test.npy"))
T2_test = np.load(os.path.join(DATA_PATH, "T2_test.npy"))
T3_test = np.load(os.path.join(DATA_PATH, "T3_test.npy"))

# Ensure channels last (H, W, C)
if T1_train.shape[-1] != 3:
    T1_train = np.transpose(T1_train, (0, 2, 3, 1))
    T2_train = np.transpose(T2_train, (0, 2, 3, 1))
    T3_train = np.transpose(T3_train, (0, 2, 3, 1))
    T1_test = np.transpose(T1_test, (0, 2, 3, 1))
    T2_test = np.transpose(T2_test, (0, 2, 3, 1))
    T3_test = np.transpose(T3_test, (0, 2, 3, 1))

# Custom SSIM metric
def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

# Combined loss: MSE + SSIM
def combined_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    ssim = tf.reduce_mean(1 - tf.image.ssim(y_true, y_pred, max_val=1.0))
    return 0.4 * mse + 0.6 * ssim

# Dual-branch encoder model
def dual_input_model(input_shape=(256, 256, 3)):
    # Input A (T1)
    input_A = layers.Input(shape=input_shape)
    x1 = layers.Conv2D(32, 3, activation='relu', padding='same')(input_A)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.MaxPooling2D()(x1)
    x1 = layers.Conv2D(64, 3, activation='relu', padding='same')(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.MaxPooling2D()(x1)

    # Input B (T2)
    input_B = layers.Input(shape=input_shape)
    x2 = layers.Conv2D(32, 3, activation='relu', padding='same')(input_B)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.MaxPooling2D()(x2)
    x2 = layers.Conv2D(64, 3, activation='relu', padding='same')(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.MaxPooling2D()(x2)

    # Merge branches
    merged = layers.Concatenate()([x1, x2])
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(merged)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    output = layers.Conv2D(3, 1, activation='sigmoid')(x)

    model = models.Model(inputs=[input_A, input_B], outputs=output)
    return model

# Build model
model = dual_input_model()
model.compile(optimizer='adam', loss=combined_loss, metrics=[ssim_metric, 'mae'])
model.summary()

# Learning rate reduction on plateau
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

# Train
history = model.fit(
    [T1_train, T2_train], T3_train,
    validation_data=([T1_test, T2_test], T3_test),
    epochs=100,
    batch_size=8,
    callbacks=[lr_schedule]
)

# Save model
model.save(os.path.join(DATA_PATH, "dual_input_t1_t2_to_t3.keras"))

# Predict on test set
preds = model.predict([T1_test, T2_test])

# Directory to save predictions
save_dir = os.path.join(DATA_PATH, "predicted_T3_images")
os.makedirs(save_dir, exist_ok=True)

for i in range(min(10, len(preds))):  # Save first 10 predictions
    pred_img = (preds[i] * 255).astype(np.uint8)
    # Ensure RGB order for matplotlib
    plt.imsave(os.path.join(save_dir, f"predicted_T3_{i:03}.png"), pred_img)
    # Optional: Visualize side by side
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(T1_test[i])
    axs[0].set_title("T1 Input")
    axs[1].imshow(T2_test[i])
    axs[1].set_title("T2 Input")
    axs[2].imshow(pred_img)
    axs[2].set_title("Predicted T3 (RGB)")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"comparison_{i:03}.png"))
    plt.close()
