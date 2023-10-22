# Person ReID

## 📚 Contents
- [Person ReID](#person-reid)
  - [📚 Contents](#-contents)
  - [📋 Introduction](#-introduction)
  - [📂 Datasets](#-datasets)
  - [🛠️ Environment](#️-environment)
  - [🚀 Get Started](#-get-started)
  - [💗 Acknowledgement](#-acknowledgement)
  - [🤝 Contribute \& Contact](#-contribute--contact)

## 📋 Introduction

The HAP pre-trained model is fine-tuned for the conventional person ReID task with respect to:

- Two datasets: MSMT17 and Market-1501
- Two resolution sizes: (256, 128) and (384, 128)
- Two model structures: ViT and ViT-lem

## 📂 Datasets

Put the dataset directories outside the HAP project:
```bash
home
├── HAP
├── msmt  # MSMT17 dataset directory
│   ├── bounding_box_train
│   │   ├── xxx.jpg
│   │   └── ...
│   ├── bounding_box_test
│   │   ├── xxx.jpg
│   │   └── ...
│   └── query
│       ├── xxx.jpg
│       └── ...
└── market  # Market-1501 dataset directory
    ├── bounding_box_train
    │   ├── xxx.jpg
    │   └── ...
    ├── bounding_box_test
    │   ├── xxx.jpg
    │   └── ...
    └── query
        ├── xxx.jpg
        └── ...
```

## 🛠️ Environment
Conda is recommended for configuring the environment:
```bash
conda env create -f env-person_reid.yaml && conda activate env_person_reid
```

## 🚀 Get Started

It may need 1 GPU with memory larger than 12GB, such as NVIDIA V100, for training.

```bash
# -------------------- Fine-Tuning HAP for Person ReID --------------------
cd HAP/downstream/person_reid/

# Download the checkpoint and move it here
CKPT=ckpt_default_pretrain_pose_mae_vit_base_patch16_LUPersonPose_399.pth

GPU=0
SEED=0
TAG=default
SIZE=256  # {256, 384}
DATA=msmt  # {msmt, market}
MODEL=vit_base_patch16  # {vit_base_patch16, lem_base_patch16}
OUTPUT=output-person_reid/${TAG}/${DATA}/${MODEL}/${SIZE}/${SEED}/

python main_reid.py \
    --config_file configs/reid/${DATA}.yaml \
    --model ${MODEL} \
    --batch_size 64 \
    --resume ${CKPT} \
    --epochs 100 \
    --warmup_epochs 5 \
    --lr 8e-3 \
    --size ${SIZE} \
    --device ${GPU} \
    --seed ${SEED} \
    --output_dir ${OUTPUT}
```

## 💗 Acknowledgement

Our implementation is based on the codebase of [MALE](https://github.com/YanzuoLu/MALE).

## 🤝 Contribute & Contact

Feel free to star and contribute to our repository. 

If you have any questions or advice, contact us through GitHub issues or email (yuanjk0921@outlook.com).