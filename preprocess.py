import os
import cv2
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split

# Change this to your local dataset path
DATASET_ROOT = r"C:\Users\vishn\OneDrive\keshav and sri\Desktop\Project\Dataset"
IMG_SIZE = (256, 256)

T1_data, T2_data, T3_data = [], [], []

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Missing: {path}")
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    return img

# Scan through all lesion folders
lesion_folders = sorted(glob(os.path.join(DATASET_ROOT, "lesion_*")))

print(f"Found {len(lesion_folders)} lesion folders...")

for folder in lesion_folders:
    try:
        t1 = load_image(os.path.join(folder, "T1.jpg"))
        t2 = load_image(os.path.join(folder, "T2.jpg"))
        t3 = load_image(os.path.join(folder, "T3.jpg"))

        T1_data.append(t1)
        T2_data.append(t2)
        T3_data.append(t3)

    except Exception as e:
        print(f"⚠️ Skipping {folder}: {e}")

# Convert to NumPy arrays
T1 = np.array(T1_data)
T2 = np.array(T2_data)
T3 = np.array(T3_data)

print(f"\n✅ Loaded {len(T1)} samples.")
print(f"Shape: T1={T1.shape}, T2={T2.shape}, T3={T3.shape}")

# Split train/test
T1_train, T1_test, T2_train, T2_test, T3_train, T3_test = train_test_split(
    T1, T2, T3, test_size=0.2, random_state=42
)

# Save as .npy files
save_dir = os.path.join(DATASET_ROOT, "processed")
os.makedirs(save_dir, exist_ok=True)

np.save(os.path.join(save_dir, "T1_train.npy"), T1_train)
np.save(os.path.join(save_dir, "T2_train.npy"), T2_train)
np.save(os.path.join(save_dir, "T3_train.npy"), T3_train)
np.save(os.path.join(save_dir, "T1_test.npy"), T1_test)
np.save(os.path.join(save_dir, "T2_test.npy"), T2_test)
np.save(os.path.join(save_dir, "T3_test.npy"), T3_test)

print(f"\nAll preprocessed data saved to: {save_dir}")


