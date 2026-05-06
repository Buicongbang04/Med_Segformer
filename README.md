# 🧠 Medical Image Segmentation with SegFormer

This repository implements a **SegFormer-based model** for **medical image segmentation**, focusing on tasks such as tumor segmentation from medical imaging (e.g., CT/MRI).

The project is built on top of **PyTorch + MMSegmentation**, and extends SegFormer for domain-specific medical datasets.

---

## 📌 Overview

* 🎯 Task: Semantic segmentation (medical images)
* 🧠 Model: SegFormer (Transformer-based segmentation)
* 📊 Loss: CrossEntropy + Dice Loss
* 🏥 Domain: Medical imaging (e.g., tumor segmentation)

SegFormer is a Transformer-based architecture that combines:

* A **hierarchical Transformer encoder**
* A lightweight **MLP decoder**

This design enables efficient multi-scale feature extraction and strong segmentation performance.

---

## 📂 Project Structure

```
Med_Segformer/
│
├── configs/               # Training configs (SegFormer variants)
├── mmseg/                # Core segmentation framework (modified)
├── tools/                # Training & testing scripts
├── datasets/             # Dataset handling (custom)
├── checkpoints/          # Saved models
├── work_dirs/            # Training logs & outputs
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone repository

```bash
git clone https://github.com/Buicongbang04/Med_Segformer.git
cd Med_Segformer
```

---

### 2. Create environment (recommended)

```bash
conda create -n segformer python=3.8 -y
conda activate segformer
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If using GPU:

```bash
pip install mmcv-full==1.2.7 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.1/index.html
```

---

## 📊 Dataset Preparation

Prepare dataset in segmentation format:

```
data/
├── images/
│   ├── train/
│   ├── val/
├── masks/
│   ├── train/
│   ├── val/
```

* `images`: input images
* `masks`: ground truth segmentation
* Masks should be **grayscale (0 = background, 1 = tumor)**

---

## 🚀 Training

### Single GPU

```bash
python tools/train.py configs/your_config.py
```

### Multi-GPU

```bash
./tools/dist_train.sh configs/your_config.py 2
```

---

## 📈 Evaluation

```bash
python tools/test.py configs/your_config.py checkpoints/latest.pth
```

Metrics:

* mIoU
* Dice Score (important for medical segmentation)

---

## 🧪 Inference (Demo)

```bash
python demo/image_demo.py \
    demo/test.png \
    configs/your_config.py \
    checkpoints/latest.pth \
    --device cuda:0
```

---

## ⚙️ Custom Loss

Project uses:

* CrossEntropy Loss
* Dice Loss

```python
Loss = CE + Dice
```

Dice is critical because:

* Tumor regions are small
* Background dominates

---

## 🧠 Model Details

SegFormer consists of:

1. **Encoder (MiT - Mix Vision Transformer)**
2. **Multi-scale feature extraction**
3. **MLP Decoder (lightweight)**

Pipeline:

```
Image → Transformer Encoder → Multi-scale Features
      → MLP Decoder → Segmentation Map
```

---

## 📌 Notes

* acc_seg is not important for medical segmentation
* Focus on:

  * Dice Score
  * Tumor Dice (per class)
* Class imbalance is a major issue → tune loss carefully

---

## 🧪 Tips for Better Results

* ✔ Increase Dice weight
* ✔ Use small batch size (2–4)
* ✔ Normalize input images
* ✔ Use pretrained backbone
* ✔ Apply augmentation (flip, rotate)

---

## 📜 Acknowledgements

* SegFormer (NeurIPS 2021)
* MMSegmentation framework

---

## 📧 Contact

Author: Bui Cong Bang
GitHub: https://github.com/Buicongbang04
