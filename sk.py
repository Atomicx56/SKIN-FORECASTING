# Skin Lesion Forecasting with GAN + Swin-UNet + CBAM + Perceptual Loss

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from glob import glob
import matplotlib.pyplot as plt

# ========== Dataset Loader ==========
def load_lesion_dataset(root_dir, img_size=(256, 256)):
    T1, T2, T3 = [], [], []
    folders = sorted(glob(os.path.join(root_dir, 'lesion_*')))
    for folder in folders:
        t1_path = os.path.join(folder, 'T1.jpg')
        t2_path = os.path.join(folder, 'T2.jpg')
        t3_path = os.path.join(folder, 'T3.jpg')
        if os.path.exists(t1_path) and os.path.exists(t2_path) and os.path.exists(t3_path):
            t1 = img_to_array(load_img(t1_path, target_size=img_size)) / 255.0
            t2 = img_to_array(load_img(t2_path, target_size=img_size)) / 255.0
            t3 = img_to_array(load_img(t3_path, target_size=img_size)) / 255.0
            T1.append(t1)
            T2.append(t2)
            T3.append(t3)
    return np.array(T1), np.array(T2), np.array(T3)

# ========== CBAM ==========
def cbam_block(inputs, channels):
    avg_pool = layers.GlobalAveragePooling2D()(inputs)
    max_pool = layers.GlobalMaxPooling2D()(inputs)
    dense_1 = layers.Dense(channels // 8, activation='relu')
    dense_2 = layers.Dense(channels)
    avg_out = dense_2(dense_1(avg_pool))
    max_out = dense_2(dense_1(max_pool))
    channel = layers.Add()([avg_out, max_out])
    channel = layers.Activation('sigmoid')(channel)
    channel = layers.Reshape((1, 1, channels))(channel)
    x = layers.Multiply()([inputs, channel])
    spatial = layers.Conv2D(1, 7, padding='same', activation='sigmoid')(x)
    return layers.Multiply()([x, spatial])

# ========== Generator ==========
def build_generator():
    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = cbam_block(x, filters)
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        return layers.ReLU()(x)

    input1 = layers.Input(shape=(256, 256, 3))
    input2 = layers.Input(shape=(256, 256, 3))
    x = layers.Concatenate()([input1, input2])

    c1 = conv_block(x, 64)
    p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, 256)
    p3 = layers.MaxPooling2D()(c3)
    b = conv_block(p3, 512)

    u3 = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(b)
    m3 = layers.Concatenate()([u3, c3])
    d3 = conv_block(m3, 256)
    u2 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(d3)
    m2 = layers.Concatenate()([u2, c2])
    d2 = conv_block(m2, 128)
    u1 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(d2)
    m1 = layers.Concatenate()([u1, c1])
    d1 = conv_block(m1, 64)

    output = layers.Conv2D(3, 1, activation='sigmoid')(d1)
    return models.Model([input1, input2], output, name='Generator')

# ========== Discriminator ==========
def build_discriminator():
    input_img = layers.Input(shape=(256, 256, 3))  # T3 or Generated
    x = layers.Conv2D(64, 4, strides=2, padding='same')(input_img)
    x = layers.LeakyReLU(0.2)(x)
    for f in [128, 256, 512]:
        x = layers.Conv2D(f, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(1, 4, padding='same')(x)
    return models.Model(input_img, x, name='Discriminator')

# ========== Perceptual Loss ==========
vgg = VGG19(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
vgg.trainable = False
vgg_feature = models.Model(vgg.input, vgg.get_layer('block2_conv2').output)
vgg_feature.trainable = False

def perceptual_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(vgg_feature(y_true) - vgg_feature(y_pred)))

# ========== GAN Trainer ==========
def train():
    T1, T2, T3 = load_lesion_dataset("Dataset")
    dataset = tf.data.Dataset.from_tensor_slices(((T1, T2), T3)).shuffle(50).batch(4)

    gen = build_generator()
    disc = build_discriminator()

    opt_g = tf.keras.optimizers.Adam(1e-4)
    opt_d = tf.keras.optimizers.Adam(1e-4)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function
    def train_step(t1, t2, real):
        with tf.GradientTape(persistent=True) as tape:
            fake = gen([t1, t2], training=True)

            real_logits = disc(real, training=True)
            fake_logits = disc(fake, training=True)

            loss_d = bce(tf.ones_like(real_logits), real_logits) + bce(tf.zeros_like(fake_logits), fake_logits)
            loss_g_adv = bce(tf.ones_like(fake_logits), fake_logits)
            loss_g_l1 = tf.reduce_mean(tf.abs(real - fake))
            loss_g_perc = perceptual_loss(real, fake)
            
            loss_g = 0.01 * loss_g_adv + 50.0 * loss_g_l1 + 5.0 * loss_g_perc

        grads_g = tape.gradient(loss_g, gen.trainable_variables)
        grads_d = tape.gradient(loss_d, disc.trainable_variables)
        opt_g.apply_gradients(zip(grads_g, gen.trainable_variables))
        opt_d.apply_gradients(zip(grads_d, disc.trainable_variables))
        return loss_d, loss_g, loss_g_adv, loss_g_l1, loss_g_perc

    for epoch in range(1, 151):
        print(f"\nEpoch {epoch}")
        for (t1, t2), real in dataset:
            d_loss, g_loss, g_adv, g_l1, g_perc = train_step(t1, t2, real)
        print(f"D Loss: {d_loss:.4f} | G Loss: {g_loss:.4f} | Adv: {g_adv:.4f} | L1: {g_l1:.4f} | Percep: {g_perc:.4f}")

        if epoch % 10 == 0:
            os.makedirs("results", exist_ok=True)
            preds = gen.predict([T1, T2])
            for i in range(min(5, len(preds))):
                fig, axs = plt.subplots(1, 4, figsize=(16, 4))
                axs[0].imshow(T1[i]); axs[0].set_title("T1")
                axs[1].imshow(T2[i]); axs[1].set_title("T2")
                axs[2].imshow(T3[i]); axs[2].set_title("GT T3")
                axs[3].imshow(preds[i]); axs[3].set_title("Pred T3")
                for ax in axs: ax.axis('off')
                plt.tight_layout()
                plt.savefig(f"results/epoch{epoch}_sample{i}.png")
                plt.close()
            gen.save(f"models/generator_epoch{epoch}.keras")

            # Print overall perceptual loss, RMSE, and MSE
            perceptual = perceptual_loss(tf.convert_to_tensor(T3, dtype=tf.float32), tf.convert_to_tensor(preds, dtype=tf.float32)).numpy()
            mse = np.mean((T3 - preds) ** 2)
            rmse = np.sqrt(mse)
            print(f"\n[Epoch {epoch}] Overall Perceptual Loss: {perceptual:.6f} | MSE: {mse:.6f} | RMSE: {rmse:.6f}")

            # === Actual vs Predicted Graphs ===
            preds = gen.predict([T1, T2])
            for i in range(min(5, len(preds))):
                fig, axs = plt.subplots(1, 2, figsize=(8, 4))
                axs[0].imshow(T3[i])
                axs[0].set_title('Actual (T3)')
                axs[0].axis('off')
                axs[1].imshow(preds[i])
                axs[1].set_title('Predicted')
                axs[1].axis('off')
                plt.tight_layout()
                plt.savefig(f'results/actual_vs_predicted_{i}.png')
                print(f"[INFO] Saved actual vs predicted plot: results/actual_vs_predicted_{i}.png")
                plt.close()

            # === Heatmap of Absolute Error ===
            for i in range(min(5, len(preds))):
                abs_error = np.abs(T3[i] - preds[i])
                abs_error_gray = np.mean(abs_error, axis=-1)  # shape (256,256)
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(T3[i])
                heatmap = ax.imshow(abs_error_gray, cmap='jet', alpha=0.5)
                ax.set_title('Heatmap of Absolute Error')
                ax.axis('off')
                plt.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
                plt.tight_layout()
                plt.savefig(f'results/heatmap_error_{i}.png')
                print(f"[INFO] Saved heatmap error plot: results/heatmap_error_{i}.png")
                plt.close()

if __name__ == '__main__':
    train()