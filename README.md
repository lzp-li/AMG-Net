# AMG-Net: Environment-Adaptive Asymmetric Mutual Guidance for Multispectral Object Detection

**Status:** Currently Under Review in *Pattern Analysis and Applications*.

This repository provides the core implementation of **AMG-Net**. To address challenges such as extreme illumination changes and the degradation of fine-grained spatial structures, AMG-Net designs an asymmetric cross-modal mutual guidance mechanism. It integrates a texture injection module (VCTI) and a structure compensation module (TPDP) to achieve the complementary enhancement of visible high-frequency textures and thermal infrared salient features.

## 1. Prerequisites
Our experiments are implemented and validated based on the following environment:
* **OS:** Ubuntu 24.04.3 LTS
* **GPU:** NVIDIA GeForce RTX 5070 Ti Laptop GPU (12GB VRAM)
* **Framework:** PyTorch v2.10.0, CUDA 13.0

Clone this repository and install the required dependencies:

```bash
git clone [https://github.com/lzp-li/AMG-Net.git](https://github.com/lzp-li/AMG-Net.git)
cd AMG-Net
pip install -r requirements.txt
```

## 2. Data Preparation
Our experiments are conducted on the publicly available **FLIR** (aligned version) and **KAIST** multispectral datasets. 
1. Download the datasets from their official repositories.
2. Organize the images and labels in the standard YOLO format.
3. Modify the dataset path in your `.yaml` configuration file (e.g., `data/FLIR.yaml`).

## 3. Quick Start Guide

### Training
To train the AMG-Net model from scratch on the FLIR/KAIST dataset, run the `train.py` script. The default input resolution is 640×640:

```bash
python train.py --weights '' --cfg models/yolov5l.yaml --data [your_dataset_config.yaml] --epochs 60 --batch-size 8 --img-size 640 --device 0
```
*(Note: Please replace `[your_dataset_config.yaml]` with your local dataset yaml file path).*

### Testing / Inference
To evaluate the model's performance on the test set, use the `test.py` script:

```bash
python test.py --weights runs/train/exp/weights/best.pt --data [your_dataset_config.yaml] --img-size 640 --device 0
```

## 4. Main Results
AMG-Net achieves state-of-the-art performance on multispectral object detection benchmarks:
* **FLIR:** **83.4%** mAP@0.5 and **37.5%** mAP@0.75.
* **KAIST:** The log-average miss rate (MR-all) is reduced to **6.71%**.

## Disclaimer & Code Availability
*Note: This repository currently provides the core structural implementation (`models/` and `common.py`) and step-by-step run scripts for peer-review validation. Full pre-trained weights and highly detailed ablation configurations will be fully released upon the formal acceptance of the manuscript.*
