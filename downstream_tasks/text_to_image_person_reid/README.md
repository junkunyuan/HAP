# Text-to-Image Person ReID


## 📚 Contents
- [Text-to-Image Person ReID](#text-to-image-person-reid)
  - [📚 Contents](#-contents)
  - [📋 Introduction](#-introduction)
  - [📂 Datasets](#-datasets)
  - [🛠️ Environment](#️-environment)
  - [🚀 Get Started](#-get-started)
  - [💗 Acknowledgement](#-acknowledgement)
  - [🤝 Contribute \& Contact](#-contribute--contact)


## 📋 Introduction

The HAP pre-trained model is fine-tuned for the text-to-image person ReID task on CUHK-PEDES, ICFG-PEDES, RSTPReid datasets.

## 📂 Datasets

Put the dataset directories outside the HAP project:

```bash
home
├── HAP
├── CUHK-PEDES  # CUHK-PEDES dataset directory
│   └── imgs
├── ICFG-PEDES  # ICFG-PEDES dataset directory
│   ├── train
│   └── test
└── RSTPReid  # RSTPReid dataset directory
    └── xxx.jpg
```

## 🛠️ Environment

Conda is recommended for configuring the environment:
```bash
conda env create -f env-text_to_image_person_reid.yaml && conda activate env_t2i_person_reid
```

## 🚀 Get Started

We provide BERT checkpoint [here](https://drive.google.com/file/d/1hk0cqGOw3OikQv35y5RKkMt9XhosoW-I/view).

It may need 1 GPU with memory larger than 14GB, such as NVIDIA V100, for training.

```bash
# -------------------- Fine-Tuning HAP for Text-to-Image Person ReID --------------------
cd HAP/downstream/text_to_image_person_reid/

# Download the HAP checkpoint and move it here
# text_to_image_person_reid/ckpt_default_pretrain_pose_mae_vit_base_patch16_LUPersonPose_399.pth

# Download the BERT folder and move it here
# text_to_image_person_reid/bert_base_uncased/

DATASET=CUHK-PEDES  # {CUHK-PEDES, ICFG-PEDES, RSTPReid}
DATA_ROOT=../../../${DATASET}

GPU=0  # Choose an available GPU

python train.py \
  --GPU_id ${GPU} \
  --dataset ${DATASET} \
  --dataroot ${DATA_ROOT}
```

## 💗 Acknowledgement

Our implementation is based on the codebase of [LGUR](https://github.com/ZhiyinShao-H/LGUR).

## 🤝 Contribute & Contact

Feel free to star and contribute to our repository. 

If you have any questions or advice, contact us through GitHub issues or email (yuanjk0921@outlook.com).