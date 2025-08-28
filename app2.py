import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

IMG_SIZE = (256, 256)

# Load model once
model = load_model("models/generator_epoch150.keras", compile=False)

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

def postprocess_image(array):
    array = np.clip(array * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(array)

class LesionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Skin Lesion Forecaster (Tkinter)")
        self.root.geometry("900x500")
        
        self.t1_path = None
        self.t2_path = None

        tk.Label(root, text="Skin Lesion Forecasting using GAN", font=("Arial", 18)).pack(pady=10)

        self.frame = tk.Frame(root)
        self.frame.pack()

        self.btn_t1 = tk.Button(self.frame, text="Upload T1", command=self.upload_t1)
        self.btn_t1.grid(row=0, column=0, padx=10)

        self.btn_t2 = tk.Button(self.frame, text="Upload T2", command=self.upload_t2)
        self.btn_t2.grid(row=0, column=1, padx=10)

        self.btn_predict = tk.Button(self.frame, text="Predict T3", command=self.predict)
        self.btn_predict.grid(row=0, column=2, padx=10)

        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(pady=20)

        self.canvas_t1 = tk.Label(self.canvas_frame, text="T1 Image")
        self.canvas_t1.grid(row=0, column=0, padx=20)

        self.canvas_t2 = tk.Label(self.canvas_frame, text="T2 Image")
        self.canvas_t2.grid(row=0, column=1, padx=20)

        self.canvas_t3 = tk.Label(self.canvas_frame, text="Predicted T3")
        self.canvas_t3.grid(row=0, column=2, padx=20)

    def upload_t1(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if path:
            self.t1_path = path
            img = Image.open(path).resize((150, 150))
            img_tk = ImageTk.PhotoImage(img)
            self.canvas_t1.configure(image=img_tk)
            self.canvas_t1.image = img_tk
            self.canvas_t1['text'] = ""

    def upload_t2(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if path:
            self.t2_path = path
            img = Image.open(path).resize((150, 150))
            img_tk = ImageTk.PhotoImage(img)
            self.canvas_t2.configure(image=img_tk)
            self.canvas_t2.image = img_tk
            self.canvas_t2['text'] = ""

    def predict(self):
        if not self.t1_path or not self.t2_path:
            messagebox.showerror("Error", "Please upload both T1 and T2 images.")
            return

        t1 = preprocess_image(self.t1_path)
        t2 = preprocess_image(self.t2_path)
        pred = model.predict([t1, t2])[0]
        img = postprocess_image(pred)
        img = img.resize((150, 150))
        img_tk = ImageTk.PhotoImage(img)
        self.canvas_t3.configure(image=img_tk)
        self.canvas_t3.image = img_tk
        self.canvas_t3['text'] = ""

if __name__ == "__main__":
    root = tk.Tk()
    app = LesionApp(root)
    root.mainloop()
