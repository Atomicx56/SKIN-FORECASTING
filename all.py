# All-in-One Script: Load Lesion Folders, Create .npy, Predict T3, and Save Results

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# ================================
# CONFIGURATION
# ================================
ROOT_DIR = r"C:\Users\vishn\OneDrive\keshav and sri\Desktop\Project\Dataset"
SAVE_DIR = os.path.join(ROOT_DIR, "processed")
MODEL_PATH = os.path.join(SAVE_DIR, "dual_input_t1_t2_to_t3.keras")
IMG_SIZE = (256, 256)

# ================================
# UTILITIES
# ================================
def load_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)
    return np.array(img) / 255.0

# ================================
# 1. Load All Folders and Build Arrays
# ================================
T1_list, T2_list, T3_list = [], [], []

for folder in sorted(os.listdir(ROOT_DIR)):
    folder_path = os.path.join(ROOT_DIR, folder)
    if not os.path.isdir(folder_path):
        continue
    try:
        t1 = load_image(os.path.join(folder_path, "T1.jpg"))
        t2 = load_image(os.path.join(folder_path, "T2.jpg"))
        t3 = load_image(os.path.join(folder_path, "T3.jpg"))
        T1_list.append(t1)
        T2_list.append(t2)
        T3_list.append(t3)
    except Exception as e:
        print(f"❌ Failed to load images in {folder}: {e}")

T1_array = np.array(T1_list)
T2_array = np.array(T2_list)
T3_array = np.array(T3_list)

# Save processed .npy files
os.makedirs(SAVE_DIR, exist_ok=True)
np.save(os.path.join(SAVE_DIR, "T1_all.npy"), T1_array)
np.save(os.path.join(SAVE_DIR, "T2_all.npy"), T2_array)
np.save(os.path.join(SAVE_DIR, "T3_all.npy"), T3_array)
print("✅ Saved .npy files")

# ================================
# 2. Load Model and Predict
# ================================
def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'ssim_metric': ssim_metric})
preds = model.predict([T1_array, T2_array])

# ================================
# 3. Save Predictions + Comparisons
# ================================
PRED_DIR = os.path.join(SAVE_DIR, "predicted_T3_images")
os.makedirs(PRED_DIR, exist_ok=True)

for i in range(len(T1_array)):
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(T1_array[i])
    axs[0].set_title("T1 Input")
    axs[1].imshow(T2_array[i])
    axs[1].set_title("T2 Input")
    axs[2].imshow(T3_array[i])
    axs[2].set_title("Ground Truth T3")
    axs[3].imshow(preds[i])
    axs[3].set_title("Predicted T3")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(PRED_DIR, f"comparison_{i:03}.png"))
    plt.close()

    # Save predicted T3 only
    pred_image = (preds[i] * 255).astype(np.uint8)
    plt.imsave(os.path.join(PRED_DIR, f"predicted_T3_{i:03}.png"), pred_image)

print(f"✅ Saved all predictions and visual comparisons to: {PRED_DIR}")
