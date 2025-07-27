# FishSeg: U-Net Based Fish Image Segmentation in the Wild

This project proposes a U-Net-based deep learning model to segment fish images under challenging underwater conditions including noise, lighting variation, and complex backgrounds.

## 🧠 Features

- 🐟 Dataset augmentation with realistic background and noise
- 💡 Preprocessing with Adaptive Histogram Equalization and Gamma Correction
- 🔍 Robust segmentation using U-Net
- 📊 Accuracy > 83% in noisy and bright underwater test sets

## 🔧 Tech Stack

- Python, PyTorch
- OpenCV, NumPy, Matplotlib
- U-Net architecture

## 🚀 How to Run

```bash
pip install -r requirements.txt
python train.py
