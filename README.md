# Real-Time Hand Gesture Recognition — CNN & OpenCV ✋🤖

A real-time deep learning computer vision system combining **Convolutional Neural Networks (CNN)** with **live OpenCV camera streaming** for gesture classification.

---

## 📌 System Architecture

```
[Webcam Video Stream] → [Frame Capture (OpenCV)] → [Resize & Normalization]
                               ↓
                 [Convolutional Neural Network]
               - Conv2D (32, 3x3) + MaxPooling2D
               - Conv2D (64, 3x3) + MaxPooling2D
               - Conv2D (128, 3x3) + MaxPooling2D
               - Dense (128, ReLU) + Softmax Output
                               ↓
         [Real-Time Visual Overlay on Video Stream]
```

---

## 🛠 Tech Stack
* **TensorFlow / Keras 2.x** (Deep learning network design & training)
* **OpenCV (`opencv-python`)** (Video stream capture, text overlay, bounding windows)
* **scikit-learn** (`LabelEncoder`, `train_test_split`)
* **NumPy & Matplotlib**

---

## 🚀 Getting Started

### 1. Installation
```bash
git clone https://github.com/Devanshu155/ARKA_ML_04.git
cd ARKA_ML_04
pip install -r requirements.txt
```

### 2. Execution
```bash
python task4.py
```
*Press `q` to terminate the live webcam stream.*

---

## 👤 Author
**Devanshu Mishra** ([@Devanshu155](https://github.com/Devanshu155))
