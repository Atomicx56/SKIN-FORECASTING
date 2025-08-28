# Forecasting Skin Lesion Progression (T1, T2 → T3) with Swin‑UNet + CBAM & PatchGAN

> Predict the **future state of a skin lesion (T3)** from two prior snapshots **T1 & T2**, using a GAN-based dual‑input generator for sharper, more realistic forecasts.

---

## ✨ Highlights

* **Task**: Forecast T3 image from T1 and T2 lesion images (progression modeling).
* **Approach**: **GAN** with **Swin‑UNet + CBAM** generator and **PatchGAN** discriminator.
* **Losses**: Adversarial + **Perceptual (VGG19 features)** + **L1**.
* **Metrics**: **MSE, MAE, SSIM, PSNR**, plus optional perceptual similarity.
* **Baselines**: Non‑GAN U‑Net/Swin‑UNet regression model (L1 + Perceptual) for ablations.
* **Extras**: Training curves, Grad‑CAM/attention maps (optional), Streamlit demo app.

---

## 📷 Problem Statement

Given two time‑stamped lesion images **T1** and **T2** of the same lesion, generate a realistic forecast **T3** that preserves lesion morphology and texture evolution (e.g., spread, boundary sharpness, redness).

---

## 🧠 Method Overview

### Generator (G): Swin‑UNet + CBAM (Dual‑Input)

* **Inputs**: Concatenated channels `[T1, T2]` (e.g., 2×(H×W×1/3)).
* **Backbone**: Swin Transformer U‑Net encoder‑decoder for multi‑scale context.
* **Attention**: **CBAM** (Channel & Spatial) in skip connections for feature refinement.
* **Output**: Predicted **T3** image (same size as inputs).

### Discriminator (D): PatchGAN

* Operates on local patches (≈70×70) for fine‑grained realism judgment.
* Inputs are pairs `(T1,T2,T3)` → real vs `(T1,T2,G(T1,T2))` → fake.

### Losses

* **Adversarial**: PatchGAN (LSGAN/BCE) on D’s real/fake classification.
* **Perceptual Loss**: `L_perc = Σ_i || ϕ_i(T3) − ϕ_i(G(T1,T2)) ||_1`, where ϕ\_i are VGG19 feature maps.
* **Pixel Loss**: `L_L1 = || T3 − G(T1,T2) ||_1`.
* **Total**: `L_G = λ_adv L_adv + λ_perc L_perc + λ_L1 L_L1` (defaults: 1.0 / 1.0 / 100.0).

---

## 🗂️ Dataset Structure

Organize your dataset as follows (example):

```
DATA_ROOT/
  lesions/
    lesion_001/
      T1.png
      T2.png
      T3.png
    lesion_002/
      T1.png
      T2.png
      T3.png
  splits/
    train.txt   # list of lesion IDs for training
    val.txt     # list of lesion IDs for validation
    test.txt    # list of lesion IDs for testing
```

**Notes**

* Images are resized/cropped to **256×256** (configurable) and normalized to **\[-1, 1]**.
* If your data are RGB, keep channels as‑is; for grayscale, use 1 channel.
* Ensure T1/T2/T3 for a lesion are **aligned and patient‑consistent**.

---

## 🛠️ Installation

### 1) Create environment

```bash
conda create -n lesion-forecast python=3.10 -y
conda activate lesion-forecast
```

### 2) Install requirements

```bash
pip install -r requirements.txt
```

`requirements.txt` (reference):

```
tensorflow>=2.12
tensorflow-addons
opencv-python
scikit-image
pillow
numpy
pandas
matplotlib
albumentations
scikit-learn
tqdm
```

> ✅ GPU recommended. Install a TF build matching your CUDA/CuDNN versions if needed.



## 🚀 Quick Start

### 1) Prepare data

Update `configs/base.yaml` paths, or pass via CLI.

### 2) Train (GAN)

```bash
python training/train_gan.py \
  --data_root /path/to/DATA_ROOT \
  --img_size 256 \
  --batch_size 4 \
  --epochs 200 \
  --lambda_perc 1.0 \
  --lambda_l1 100.0 \
  --g_lr 2e-4 --d_lr 2e-4 \
  --out_dir experiments/runs/gan_swin_cbam
```

### 3) Train (Baseline)

```bash
python training/train_baseline.py \
  --data_root /path/to/DATA_ROOT \
  --img_size 256 \
  --batch_size 8 \
  --epochs 150 \
  --lambda_perc 1.0 --lambda_l1 100.0 \
  --out_dir experiments/runs/baseline_swin
```

### 4) Inference

```bash
python inference/predict.py \
  --data_root /path/to/DATA_ROOT \
  --checkpoint experiments/runs/gan_swin_cbam/best.ckpt \
  --save_dir outputs/predictions
```

### 5) Streamlit Demo

```bash
streamlit run app/streamlit_app.py \
  -- --checkpoint experiments/runs/gan_swin_cbam/best.ckpt \
  --img_size 256
```

---

## 📊 Evaluation

Run evaluation on the test split:

```bash
python inference/visualize.py \
  --data_root /path/to/DATA_ROOT \
  --pred_dir outputs/predictions \
  --metrics mse mae ssim psnr \
  --save_report outputs/metrics.csv
```

**Metrics Tracked**

* **MSE**: Mean Squared Error
* **MAE**: Mean Absolute Error
* **SSIM**: Structural Similarity Index
* **PSNR**: Peak Signal‑to‑Noise Ratio
* **Perceptual loss** (report only): average VGG feature L1

Example (placeholder) results table:

|                                 Model |     MSE ↓ |     MAE ↓ |    SSIM ↑ |   PSNR ↑ |
| ------------------------------------: | --------: | --------: | --------: | -------: |
|                  Baseline (Swin‑UNet) |     0.012 |     0.067 |     0.812 |     25.8 |
| **GAN (Swin‑UNet + CBAM + PatchGAN)** | **0.010** | **0.061** | **0.835** | **26.6** |

> Replace with your actual numbers after training; CSV export is enabled.

---

## 🧪 Ablations & Visuals

* **Ablations**: remove CBAM, change loss weights, swap Swin‑UNet → U‑Net.
* **Qualitative**: grids of `[T1, T2, GT(T3), Pred(T3)]` with SSIM/PSNR overlays.
* **Attention/Grad‑CAM**: visualize attention maps to interpret lesion focus.

---

## ⚙️ Configs (YAML)

Key flags in `configs/base.yaml`:

```yaml
img_size: 256
batch_size: 4
epochs: 200
lambda_adv: 1.0
lambda_perc: 1.0
lambda_l1: 100.0
g_lr: 0.0002
d_lr: 0.0002
norm: instance  # or batch
optimizer: adam
mixed_precision: true
```

---

## 🧮 Math Summary

Given inputs **x₁ = T1**, **x₂ = T2**, and target **y = T3**:

* **G** predicts **ŷ = G(x₁, x₂)**.
* **Adversarial loss (LSGAN)**: $L_{adv} = \mathbb{E}[(D(x, y) - 1)^2] + \mathbb{E}[D(x, ŷ)^2]$.
* **Perceptual loss**: $L_{perc} = \sum_i \| \phi_i(y) - \phi_i(ŷ) \|_1$.
* **Pixel loss**: $L_{L1} = \| y - ŷ \|_1$.
* **Total**: $L_G = \lambda_{adv} L_{adv} + \lambda_{perc} L_{perc} + \lambda_{L1} L_{L1}$.

---

## 🔍 Reproducibility

* **Seeds**: set `--seed 42` for determinism.
* **Checkpoints**: saved every N epochs; `best.ckpt` by val SSIM.
* **Logging**: TensorBoard under `experiments/runs/*/logs`.

---

## 📈 Roadmap

* [ ] Add multi‑task heads (segmentation, redness score)
* [ ] Temporal consistency loss (optical flow / warping)
* [ ] Self‑distillation for few‑shot lesions
* [ ] Model quantization for edge devices
* [ ] Dockerfile & Colab notebook

---

## 🧪 How to Cite

If you use this repo, please cite:

```bibtex
@software{lesion_forecasting_2025,
  author  = {Vishnu Vardhan Pingali},
  title   = {Forecasting Skin Lesion Progression with Swin-UNet + CBAM and PatchGAN},
  year    = {2025},
  url     = {https://github.com/yourname/skin-forecasting}
}
```

---

## 📜 License

This project is licensed under the **MIT License** – see `LICENSE` for details.

---

## 🙏 Acknowledgements

* Swin‑UNet, CBAM, PatchGAN, and VGG19 implementations inspired by their original papers and open‑source repos.
* Thanks to the open‑source community for datasets, tooling, and feedback.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or PR with a clear description, steps to reproduce, and screenshots of qualitative results.
