## 🤖 Object Detection & Distance Estimation for Robotics Navigation
* 👤 Submitted by: Chaitanya Dhawade


> YOLOv8 • Monocular Distance Estimation • ONNX/INT8 Optimization • Advanced Computer Vision

---

## 🚀 Project Overview

This project implements a **real-time object detection and distance estimation system** for robotics navigation.

### 🔑 Features

* Object detection using YOLOv8 (transfer learning)
* Monocular distance estimation using pinhole camera model
* Real-time inference (image, video, webcam)
* Model optimization (ONNX, FP16, INT8, pruning)
* Advanced vision modules (BEV, Optical Flow, Epipolar Geometry)

---

## 🧠 Pipeline

```
Input → YOLOv8 Detection → Bounding Boxes → Distance Estimation → Output (Label + Distance)
```

---

## 📐 Distance Formula

```
D = (H_real × F) / H_pixel
```

---

## 📁 Project Structure

```
robotics_nav/
├── configs/
├── dataset/
├── models/
├── inference/
├── optimization/
├── benchmarks/
├── utils/
├── outputs/
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

```bash
git clone https://github.com/your-username/robotics-nav-detection.git

cd robotics-nav-detection

pip install -r requirements.txt
```

---

## 📦 Dataset Setup

### BDD100K

```bash
python dataset/bdd100k_dataset.py \
  --json data/bdd100k/labels/det_20/bdd100k_labels_images_train.json \
  --images data/bdd100k/images/100k/train \
  --out-labels data/bdd100k/labels/yolo/train
```

---

## 🏋️ Training

```bash
python models/train.py --config configs/config.yaml
```

---

## 🔍 Inference

### Image

```bash
python inference/detect.py \
  --source image.jpg \
  --weights runs/train/robotics_nav_detection/weights/best.pt
```

### Webcam

```bash
python inference/detect.py \
  --source 0 \
  --weights runs/train/robotics_nav_detection/weights/best.pt
```

---

## ⚙️ Optimization

```bash
python optimization/optimize.py \
  --weights runs/train/robotics_nav_detection/weights/best.pt \
  --mode all
```

---

## 📊 Benchmarking

```bash
python benchmarks/fps_benchmark.py \
  --weights runs/train/robotics_nav_detection/weights/best.pt
```

---

## 🔬 Advanced Vision

### BEV

```bash
python utils/advanced_vision.py --demo bev --source 0
```

### Optical Flow

```bash
python utils/advanced_vision.py --demo optical_flow --source 0
```

### Epipolar Geometry

```bash
python utils/advanced_vision.py \
  --demo epipolar \
  --left left.jpg \
  --right right.jpg
```

---

## 📈 Results

* mAP@0.5: ~0.7
* Distance Error: ~0.3m
* Model Size Reduction: up to 75%
* CPU Speed Gain: ~2×

---

## 🎯 Use Cases

* Autonomous robots
* Self-driving systems
* Surveillance
* Navigation assistants

# 📬 Contact
* 👤 Name: Chaitanya Gajanan Dhawade
* 📧 Email: chaitanyadhawade23@gmail.com
* 🔗 LinkedIn: https://www.linkedin.com/in/chaitanya-dhawade-652380313/
* 💻 GitHub: https://github.com/Chaitanya-dhawade/
