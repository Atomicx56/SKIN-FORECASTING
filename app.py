import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# ====== Load the model ======
MODEL_PATH = r"C:\Users\vishn\OneDrive\keshav and sri\Desktop\Project\models\generator_epoch100.keras"

@st.cache_resource
def load_forecasting_model():
    return load_model(MODEL_PATH, compile=False)

model = load_forecasting_model()

# ====== App UI ======
st.set_page_config(page_title="Skin Lesion Forecasting", layout="centered")
st.title("🧠 Skin Disease Progression Forecasting")
st.write("Upload two lesion images (T1 and T2) to generate a forecast of the future lesion state (T3).")

# ====== Image preprocessing ======
def preprocess(image_file):
    image = load_img(image_file, target_size=(256, 256))
    image = img_to_array(image) / 255.0
    return image

# ====== Upload inputs ======
t1_file = st.file_uploader("📷 Upload T1 Image", type=["jpg", "jpeg", "png"])
t2_file = st.file_uploader("📷 Upload T2 Image", type=["jpg", "jpeg", "png"])

if t1_file and t2_file:
    with st.spinner("Processing and forecasting..."):
        t1_img = preprocess(t1_file)
        t2_img = preprocess(t2_file)

        # Predict T3
        pred = model.predict([np.expand_dims(t1_img, axis=0), np.expand_dims(t2_img, axis=0)])[0]

        # ====== Display results ======
        st.subheader("🖼️ Forecasted Lesion Progression (T3)")

        col1, col2, col3 = st.columns(3)
        col1.image(t1_img, caption="T1 Image", use_column_width=True)
        col2.image(t2_img, caption="T2 Image", use_column_width=True)
        col3.image(pred, caption="Predicted T3", use_column_width=True)

        # Optional: Save prediction
        st.success("✅ Forecast complete!")

        save_btn = st.download_button(
            label="📥 Download Forecasted T3",
            data=(pred * 255).astype(np.uint8).tobytes(),
            file_name="forecasted_T3.raw",
            mime="application/octet-stream"
        )
else:
    st.warning("Please upload both T1 and T2 images to continue.")

