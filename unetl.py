# Enhanced U-Net for T1 -> T3 Skin Disease Progression Forecasting with Visualization

import tensorflow as tf
from tensorflow import keras
layers = keras.layers
models = keras.models
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import ImageDataGenerator

# Load preprocessed data
DATA_PATH = r"C:\Users\vishn\OneDrive\keshav and sri\Desktop\Project\Dataset\processed"
T1_train = np.load(os.path.join(DATA_PATH, "T1_train.npy"))
T3_train = np.load(os.path.join(DATA_PATH, "T3_train.npy"))
T1_test = np.load(os.path.join(DATA_PATH, "T1_test.npy"))
T3_test = np.load(os.path.join(DATA_PATH, "T3_test.npy"))

# Ensure channels last (H, W, C)
if T1_train.shape[-1] != 3:
    T1_train = np.transpose(T1_train, (0, 2, 3, 1))
    T3_train = np.transpose(T3_train, (0, 2, 3, 1))
    T1_test = np.transpose(T1_test, (0, 2, 3, 1))
    T3_test = np.transpose(T3_test, (0, 2, 3, 1))

# Normalize data to [0, 1]
T1_train = T1_train / 255.0
T3_train = T3_train / 255.0
T1_test = T1_test / 255.0
T3_test = T3_test / 255.0

# Advanced data augmentation
# from tensorflow.keras.layers import Dropout

data_gen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    contrast_stretching=True if hasattr(ImageDataGenerator, 'contrast_stretching') else False,
    fill_mode='nearest'
)

# Deeper U-Net++ with Dropout

def unetpp_model(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)
    # Encoder
    x00 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x00 = layers.BatchNormalization()(x00)
    x00 = layers.Dropout(0.1)(x00)
    x00 = layers.Conv2D(64, 3, activation='relu', padding='same')(x00)
    x00 = layers.BatchNormalization()(x00)
    x00 = layers.Dropout(0.1)(x00)
    p0 = layers.MaxPooling2D((2, 2))(x00)

    x10 = layers.Conv2D(128, 3, activation='relu', padding='same')(p0)
    x10 = layers.BatchNormalization()(x10)
    x10 = layers.Dropout(0.1)(x10)
    x10 = layers.Conv2D(128, 3, activation='relu', padding='same')(x10)
    x10 = layers.BatchNormalization()(x10)
    x10 = layers.Dropout(0.1)(x10)
    p1 = layers.MaxPooling2D((2, 2))(x10)

    x20 = layers.Conv2D(256, 3, activation='relu', padding='same')(p1)
    x20 = layers.BatchNormalization()(x20)
    x20 = layers.Dropout(0.2)(x20)
    x20 = layers.Conv2D(256, 3, activation='relu', padding='same')(x20)
    x20 = layers.BatchNormalization()(x20)
    x20 = layers.Dropout(0.2)(x20)
    p2 = layers.MaxPooling2D((2, 2))(x20)

    x30 = layers.Conv2D(512, 3, activation='relu', padding='same')(p2)
    x30 = layers.BatchNormalization()(x30)
    x30 = layers.Dropout(0.3)(x30)
    x30 = layers.Conv2D(512, 3, activation='relu', padding='same')(x30)
    x30 = layers.BatchNormalization()(x30)
    x30 = layers.Dropout(0.3)(x30)

    # Decoder with nested skip connections
    x01 = layers.Conv2D(64, 3, activation='relu', padding='same')(
        layers.Concatenate()([x00, layers.UpSampling2D((2, 2))(x10)]))
    x01 = layers.BatchNormalization()(x01)
    x01 = layers.Dropout(0.1)(x01)
    x01 = layers.Conv2D(64, 3, activation='relu', padding='same')(x01)
    x01 = layers.BatchNormalization()(x01)
    x01 = layers.Dropout(0.1)(x01)

    x11 = layers.Conv2D(128, 3, activation='relu', padding='same')(
        layers.Concatenate()([x10, layers.UpSampling2D((2, 2))(x20)]))
    x11 = layers.BatchNormalization()(x11)
    x11 = layers.Dropout(0.1)(x11)
    x11 = layers.Conv2D(128, 3, activation='relu', padding='same')(x11)
    x11 = layers.BatchNormalization()(x11)
    x11 = layers.Dropout(0.1)(x11)

    x21 = layers.Conv2D(256, 3, activation='relu', padding='same')(
        layers.Concatenate()([x20, layers.UpSampling2D((2, 2))(x30)]))
    x21 = layers.BatchNormalization()(x21)
    x21 = layers.Dropout(0.2)(x21)
    x21 = layers.Conv2D(256, 3, activation='relu', padding='same')(x21)
    x21 = layers.BatchNormalization()(x21)
    x21 = layers.Dropout(0.2)(x21)

    x02 = layers.Conv2D(64, 3, activation='relu', padding='same')(
        layers.Concatenate()([x00, x01, layers.UpSampling2D((2, 2))(x11)]))
    x02 = layers.BatchNormalization()(x02)
    x02 = layers.Dropout(0.1)(x02)
    x02 = layers.Conv2D(64, 3, activation='relu', padding='same')(x02)
    x02 = layers.BatchNormalization()(x02)
    x02 = layers.Dropout(0.1)(x02)

    x12 = layers.Conv2D(128, 3, activation='relu', padding='same')(
        layers.Concatenate()([x10, x11, layers.UpSampling2D((2, 2))(x21)]))
    x12 = layers.BatchNormalization()(x12)
    x12 = layers.Dropout(0.1)(x12)
    x12 = layers.Conv2D(128, 3, activation='relu', padding='same')(x12)
    x12 = layers.BatchNormalization()(x12)
    x12 = layers.Dropout(0.1)(x12)

    x03 = layers.Conv2D(64, 3, activation='relu', padding='same')(
        layers.Concatenate()([x00, x01, x02, layers.UpSampling2D((2, 2))(x12)]))
    x03 = layers.BatchNormalization()(x03)
    x03 = layers.Dropout(0.1)(x03)
    x03 = layers.Conv2D(64, 3, activation='relu', padding='same')(x03)
    x03 = layers.BatchNormalization()(x03)
    x03 = layers.Dropout(0.1)(x03)

    outputs = layers.Conv2D(3, 1, activation='sigmoid')(x03)
    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

# SSIM metric
def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

# Build VGG19 feature extractor ONCE
vgg = VGG19(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
vgg.trainable = False
feature_extractor = keras.Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)

# Perceptual loss using VGG19
def perceptual_loss(y_true, y_pred):
    y_true_features = feature_extractor(y_true)
    y_pred_features = feature_extractor(y_pred)
    return tf.reduce_mean(tf.square(y_true_features - y_pred_features))

def l1_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

# Combined loss: MSE + SSIM + Perceptual + L1 (with higher perceptual weight)
def combined_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    ssim = tf.reduce_mean(1 - tf.image.ssim(y_true, y_pred, max_val=1.0))
    perceptual = perceptual_loss(y_true, y_pred)
    l1 = l1_loss(y_true, y_pred)
    return 0.1 * mse + 0.1 * ssim + 0.65 * perceptual + 0.15 * l1

# Use data augmentation generator for training
train_gen = data_gen.flow(T1_train, T3_train, batch_size=8, seed=42)

model = unetpp_model()
model.compile(optimizer='adam', loss=combined_loss, metrics=[ssim_metric, 'mae'])
model.summary()

# Training with early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_ssim_metric', patience=10, mode='max', restore_best_weights=True
)

# Add ReduceLROnPlateau callback
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

history = model.fit(
    train_gen,
    steps_per_epoch=len(T1_train) // 8,
    validation_data=(T1_test, T3_test),
    epochs=100,
    callbacks=[early_stopping, reduce_lr]
)

# Save model
model.save(os.path.join(DATA_PATH, "unet_t1_to_t3_improved.keras"))

# Predict and save sharp RGB images
preds = model.predict(T1_test)
preds = np.clip(preds, 0, 1)
save_dir = os.path.join(DATA_PATH, "predicted_T3_images")
os.makedirs(save_dir, exist_ok=True)
for i in range(min(10, len(preds))):
    img = (preds[i] * 255).astype(np.uint8)
    plt.imsave(os.path.join(save_dir, f"predicted_T3_{i:03}.png"), img)

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['ssim_metric'], label='Train SSIM')
plt.plot(history.history['val_ssim_metric'], label='Val SSIM')
plt.title('SSIM Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('SSIM')
plt.legend()

plt.tight_layout()
plt.show()
