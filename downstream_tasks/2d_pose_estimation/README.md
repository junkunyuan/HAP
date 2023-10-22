# 2D Pose Estimation

## 📚 Contents
- [2D Pose Estimation](#2d-pose-estimation)
  - [📚 Contents](#-contents)
  - [📋 Introduction](#-introduction)
  - [📂 Datasets](#-datasets)
  - [🛠️ Environment](#️-environment)
  - [🚀 Get Started](#-get-started)
  - [💗 Acknowledgement](#-acknowledgement)
  - [🤝 Contribute \& Contact](#-contribute--contact)

## 📋 Introduction

The HAP pre-trained model is fine-tuned for the 2d human pose estimation task with respect to:

- Three datasets: MPII, COCO, and AIC (AI Challenger)
- Two resolution sizes: (256, 192) and (384, 288)
- Two training settings: single-dataset and multi-dataset training

## 📂 Datasets

Put the dataset directories outside the HAP project:
```bash
home
├── HAP
├── mpii  # MPII dataset directory
│   ├── annotations
│   │   ├── mpii_train.json
│   │   └── mpii_val.json
│   └── images
│       ├── xxx.jpg
│       └── ...
├── coco  # COCO dataset directory
│   ├── annotations
│   │   ├── person_keypoints_train2017.json
│   │   └── person_keypoints_val2017.json
│   ├── train2017
│   │   ├── xxx.jpg
│   │   └── ...
│   └── val2017
│       ├── xxx.jpg
│       └── ...
└── aic  # AIC dataset directory
    ├── annotations
    │   ├── aic_train.json
    │   └── aic_val.json
    ├── ai_challenger_keypoint_train_20170902
    │   └──keypoint_train_images_20170902
    │       ├── xxx.jpg
    │       └── ...
    └── ai_challenger_keypoint_validation_20170911
        └──keypoint_validation_images_20170911
            ├── xxx.jpg
            └── ...
```

## 🛠️ Environment
Conda is recommended for configuring the environment:
```bash
conda env create -f env-2d_pose.yaml && conda activate env_2d_pose

# Install mmcv and mmpose of ViTPose
cd mmcv && git checkout v1.3.9 && MMCV_WITH_OPS=1 && cd .. && python -m pip install -e mmcv
python -m pip install -v -e ViTPose
```

## 🚀 Get Started

It may need 8 GPUs with memory larger than 16GB, such as NVIDIA V100, for single-dataset training with resolution of (256, 128).

It may need 8 GPUs with memory larger than 40GB, such as NVIDIA A100, for single-dataset training with resolution of (384, 288) and multi-dataset training.

```bash
# -------------------- Fine-Tuning HAP for 2D Pose Estimation --------------------
cd HAP/downstream/2d_pose_estimation/
ViTPose/

# Download the checkpoint
CKPT=ckpt_default_pretrain_pose_mae_vit_base_patch16_LUPersonPose_399.pth  # checkpoint path

rm -rf mmcv_custom/checkpoint.py
cp mmcv_custom/checkpoint-hap.py mmcv_custom/checkpoint.py

# ---------- For Single-Dataset Training ----------
DATA=mpii  # {mpii, coco, aic}
RESOLUTION=256x192  # {256x192, 384x288}
OUTPUT_DIR=output-2d_pose_estimation/${DATA}/${RESOLUTION}/
CONFIG=configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/${DATA}/ViTPose_base_${DATA}_${RESOLUTION}.py

# ---------- For Multi-Dataset Training ----------
# RESOLUTION=256x192  # {256x192, 384x288}
# CONFIG=configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco+aic+mpii_${RESOLUTION}.py
# OUTPUT_DIR=output-person_reid/coco+aic+mpii/${RESOLUTION}/

python -m torch.distributed.launch \
   --nnodes ${NNODES} \
   --node_rank ${RANK} \
   --nproc_per_node ${NPROC_PER_NODE} \
   --master_addr ${ADDRESS} \
   --master_port ${PORT} \
   tools/train.py \
   ${CONFIG} \
   --work-dir ${OUTPUT_DIR} \
   --launcher pytorch \
   --cfg-options model.pretrained=${CKPT}
   # --resume-from
```

After multi-dataset training, split the model and evaluate it on MPII and AIC (COCO has been tested during training) : 

```bash
# -------------------- Split Model --------------------
# We simply split the latest one. Maybe you can choose the best one.
python tools/model_split.py \
   --source ${OUTPUT_DIR}latest.pth

# -------------------- Test on MPII and AIC --------------------
DATA=mpii  # {mpii, aic}
TEST_MODEL=${DATA}.pth
RESOLUTION=256x192  # {256x192, 384x288}
CONFIG=configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/${DATA}/ViTPose_base_${DATA}_${RESOLUTION}.py

python -m torch.distributed.launch \
   --nproc_per_node=${NNODES} \
   --node_rank ${RANK} \
   --nproc_per_node ${NPROC_PER_NODE} \
   --master_addr ${ADDRESS} \
   --master_port ${PORT} \
   tools/test.py \
   ${CONFIG} \
   ${OUTPUT_DIR}${TEST_MODEL} \
   --launcher pytorch
``` 

## 💗 Acknowledgement

Our implementation is based on the codebase of [ViTPose](https://github.com/ViTAE-Transformer/ViTPose), [mmcv](https://github.com/open-mmlab/mmcv), [mmpose](https://github.com/open-mmlab/mmpose).

## 🤝 Contribute & Contact

Feel free to star and contribute to our repository. 

If you have any questions or advice, contact us through GitHub issues or email (yuanjk0921@outlook.com).