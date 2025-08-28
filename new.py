import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# --- Define custom metric and loss again ---
def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def combined_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    ssim = tf.reduce_mean(1 - tf.image.ssim(y_true, y_pred, max_val=1.0))
    return 0.4 * mse + 0.6 * ssim

# --- Paths ---
DATA_PATH = r"C:\Users\vishn\OneDrive\keshav and sri\Desktop\Project\Dataset\processed"
MODEL_PATH = os.path.join(DATA_PATH, "dual_input_t1_t2_to_t3.keras")
SAVE_DIR = os.path.join(DATA_PATH, "predicted_T3_images")

# --- Load model with both custom objects ---
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={
    'ssim_metric': ssim_metric,
    'combined_loss': combined_loss
})

# --- Load test data ---
T1_test = np.load(os.path.join(DATA_PATH, "T1_test.npy"))
T2_test = np.load(os.path.join(DATA_PATH, "T2_test.npy"))
T3_test = np.load(os.path.join(DATA_PATH, "T3_test.npy"))

# --- Predict ---
preds = model.predict([T1_test, T2_test])

# --- Create directory ---
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Visualize and Save Predictions ---
for i in range(T1_test.shape[0]):
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(T1_test[i])
    axs[0].set_title("T1 Input")
    axs[1].imshow(T2_test[i])
    axs[1].set_title("T2 Input")
    axs[2].imshow(T3_test[i])
    axs[2].set_title("Ground Truth T3")
    axs[3].imshow(preds[i])
    axs[3].set_title("Predicted T3")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"comparison_{i:03}.png"))
    plt.close()

    # Save just predicted image
    pred_image = (preds[i] * 255).astype(np.uint8)
    plt.imsave(os.path.join(SAVE_DIR, f"predicted_T3_{i:03}.png"), pred_image)
