# Pedestrian Attribute Recognition

## 📚 Contents
- [Pedestrian Attribute Recognition](#pedestrian-attribute-recognition)
  - [📚 Contents](#-contents)
  - [📋 Introduction](#-introduction)
  - [📂 Datasets](#-datasets)
  - [🛠️ Environment](#️-environment)
  - [🚀 Get Started](#-get-started)
  - [💗 Acknowledgement](#-acknowledgement)
  - [🤝 Contribute \& Contact](#-contribute--contact)

## 📋 Introduction

The HAP pre-trained model is fine-tuned for the pedestrian attribute recognition task with respect to:

- Three datasets: PA-100K, RAP, and PETA

## 📂 Datasets

Put the dataset directories outside the HAP project:
```bash
home
├── HAP
├── PA-100K  # PA-100K dataset directory
│   ├── annotation.mat
│   ├── dataset_all.pkl
│   └── data
│       ├── xxx.jpg
│       └── ...
├── RAP  # RAP dataset directory
│   ├── RAP_annotation
│   │   └── RAP_annotation.mat
│   ├── dataset_all.pkl
│   └── RAP_dataset
│       ├── xxx.png
│       └── ...     
└── PETA  # PETA dataset directory
    ├── PETA.mat
    ├── dataset_all.pkl
    └── images
        ├── xxx.png
        └── ...
```

## 🛠️ Environment
Conda is recommended for configuring the environment:
```bash
conda env create -f env_attribute.yaml && conda activate env_attribute

# Install mmcv
cd ../2d_pose_estimation/mmcv && git checkout v1.3.9 && MMCV_WITH_OPS=1 && cd .. && python -m pip install -e mmcv
```

## 🚀 Get Started

It may need 8 GPUs with memory larger than 6GB, such as NVIDIA V100, for training.

```bash
# -------------------- Fine-Tuning HAP for Pedestrian Attribute Recognition --------------------

# Download the checkpoint and move it here
CKPT=ckpt_default_pretrain_pose_mae_vit_base_patch16_LUPersonPose_399.pth

cd Rethinking_of_PAR/

DATA=pa100k  # {pa100k, rapv1, peta}

python -m torch.distributed.launch \
    --nproc_per_node=${NPROC_PER_NODE} \
    --master_port=${PORT} \
    train.py \
    --cfg configs/pedes_baseline/${DATA}.yaml
```

## 💗 Acknowledgement

Our implementation is based on the codebase of [Rethinking_of_PAR
](https://github.com/valencebond/Rethinking_of_PAR).

## 🤝 Contribute & Contact

Feel free to star and contribute to our repository. 

If you have any questions or advice, contact us through GitHub issues or email (yuanjk0921@outlook.com).