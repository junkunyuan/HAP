# 3D Pose and Shape Estimation

## 📚 Contents
- [3D Pose and Shape Estimation](#3d-pose-and-shape-estimation)
  - [📚 Contents](#-contents)
  - [📋 Introduction](#-introduction)
  - [📂 Datasets](#-datasets)
  - [🛠️ Environment](#️-environment)
  - [🚀 Get Started](#-get-started)
  - [💗 Acknowledgement](#-acknowledgement)
  - [🤝 Contribute \& Contact](#-contribute--contact)

## 📋 Introduction

The HAP pre-trained model is fine-tuned on Human3.6M, MuCo, COCO, MPII datasets and evaluated on 3DPW dataset for the 3d human pose and shape estimation task.

## 📂 Datasets

Put the dataset directories outside the HAP project:

```bash
home
├── HAP
├── Human36M  # Human3.6M dataset directory
│   ├── annotations
│   └── images
├── MuCo  # MuCo dataset directory
│   └── data
│       ├── augmented_set
│       ├── unaugmented_set
│       ├── MuCo-3DHP.json
│       └── smpl_param.json
├── coco  # COCO dataset directory
│   ├── annotations
│   └── images
│       ├── train2017
│       └── val2017
├── mpii  # MPII dataset directory
│   └── data
│       ├── annotations
│       └── images
└── PW3D  # 3DPW dataset directory
    └── data
        ├── 3DPW_latest_train.json
        ├── 3DPW_latest_validation.json
        ├── 3DPW_latest_test.json
        ├── 3DPW_validation_crowd_hhrnet_result.json
        └── imageFiles
```

## 🛠️ Environment

Conda is recommended for configuring the environment:
```bash
conda env create -f env-3d_pose.yaml && conda activate env_3d_pose

cd HAP/downstream_tasks/3d_pose_estimation/3DCrowdNet_RELEASE/ && sh requirements.sh
```

Download the required files following [3DCrowdNet](https://github.com/hongsukchoi/3DCrowdNet_RELEASE).

Prepare them as well as datasets by

```bash
cd HAP/downstream_tasks/3d_pose_estimation/3DCrowdNet_RELEASE/data/

DATA_PATH=../../../../../../

# Download J_regressor_extra.npy and move it here

# ---------- Prepare Human3.6M data ----------
cd Human36M
ln -s ${DATA_PATH}Human36M/images/ && ln -s ${DATA_PATH}Human36M/annotations/
# Download J_regressor_h36m_correct.npy and move it here
cd ..

# ---------- Prepare MuCo ----------
cd  MuCo && ln -s ${DATA_PATH}MuCo/data/ && cd ..

# ---------- Prepare COCO ----------
cd MSCOCO
# Download J_regressor_coco_hip_smpl.npy and MSCOCO_train_SMPL_NeuralAnnot.json and move them here
ln -s ${DATA_PATH}coco/images/ && ln -s ${DATA_PATH}coco/annotations/
cd ..

# ---------- Prepare MPII ----------
cd MPII
mkdir data && cd data
ln -s ${DATA_PATH}../mpii/images && ln -s ${DATA_PATH}../mpii/annotations
cd annotations
# Download MPII_train_SMPL_NeuralAnnot.json and move it here
cd ../..

# ---------- Prepare PW3D ----------
cd PW3D && ln -s ${DATA_PATH}/PW3D/data && cd ..

# ---------- Prepare SMPL models ----------
cd ../common/utils/smplpytorch/smplpytorch/native/models/
# Download basicModel_neutral_lbs_10_207_0_v1.0.0.pkl, basicModel_m_lbs_10_207_0_v1.0.0.pkl, basicModel_f_lbs_10_207_0_v1.0.0.pkl, and move them here
cd -

# ---------- Prepare vposer ----------
VP_PAHT=../common/utils/human_model_files/smpl/VPOSER_CKPT/
mkdir -p ${VP_PAHT} && cd ${VP_PAHT}
# Download vposer.zip and unzip it here 
cd -

# ---------- Prepare pre-trained 2D pose model
# You may need to run the 2d pose task and get the checkpoint of latest.pth
mv latest.pth ../
```

## 🚀 Get Started

It may need 8 GPUs with memory larger than 32GB, such as NVIDIA A100, for training.

```bash
# -------------------- Fine-Tuning HAP for 3D Pose and Shape Estimation --------------------
cd HAP/downstream_tasks/3d_pose_estimation/3DCrowdNet_RELEASE/main/

# ---------- Training ----------
python train.py \
    --gpu 0-7 \
    --amp \
    --cfg ../assets/yaml/3dpw_vit_b_4gpu.yml

# ---------- Evaluation ----------
for((EPOCH=0; EPOCH<11; EPOCH++)); do python test.py \
   --gpu 0-7 \
   --test_epoch ${EPOCH} \
   --exp_dir ../output-3d_pose/ \
   --cfg ../assets/yaml/3dpw_vit_b_4gpu.yml \
echo 'finish test for epoch '${EPOCH} \
done
```

## 💗 Acknowledgement

Our implementation is based on the codebase of [3DCrowdNet](https://github.com/hongsukchoi/3DCrowdNet_RELEASE).

## 🤝 Contribute & Contact

Feel free to star and contribute to our repository. 

If you have any questions or advice, contact us through GitHub issues or email (yuanjk0921@outlook.com).
