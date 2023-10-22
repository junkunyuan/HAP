# ----------------------------------------------------------------------
# HAP: Structure-Aware Masked Image Modeling for Human-Centric Perception
# Written by Junkun Yuan (yuanjk0921@outlook.com)
# ----------------------------------------------------------------------
# Load LUPerson dataset (with keypoints)
# ----------------------------------------------------------------------

import os
import cv2
import pickle
import random
import numpy as np

import torch
from torch.utils.data import Dataset


def load_image_pose(img_path, pose_path, transform):
    """Load and transform images and poses."""

    # Load image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load pose
    pose_load = np.load(pose_path, allow_pickle=True)
    keypoints = pose_load[0][0]['keypoints']  # list with size (n, 3)

    # Remove out-of-bound keypoints before transforms by setting them to -10000 
    H, W, _ = img.shape
    for kp_idx, kp in enumerate(keypoints):
        if not (0. <= kp[0] < W and 0. <= kp[1] < H):
            keypoints[kp_idx] = [-10000., -10000., -10000.]

    # Apply transforms to images and keypoints
    img = transform(image=img, keypoints=keypoints)
    img_trans, keypoints_trans = img['image'], img['keypoints']

    # Remove out-of-bound keypoints after transforms by setting them to -10000 
    _, H, W = img_trans.shape
    keypoints_trans_in = []  # in-bound keypoints
    for kp_idx, kp in enumerate(keypoints_trans):
        if not (0. <= kp[0] < W and 0. <= kp[1] < H):
            keypoints_trans[kp_idx] = [-10000., -10000., -10000.]
        else:
            keypoints_trans_in.append(kp)

    num_keypoints = len(keypoints_trans_in)
    
    # Fill keypoints
    num_kps_full = 17
    while len(keypoints_trans_in) < num_kps_full:
        keypoints_trans_in.append([-10000., -10000., -10000.])
    
    keypoints_trans_in = torch.tensor(keypoints_trans_in)
    keypoints_trans = torch.tensor(keypoints_trans)

    return img_trans, keypoints_trans_in, num_keypoints, keypoints_trans


class LUPersonPose(Dataset):
    """Build LUPerson pose dataset."""
    def __init__(self, cfg, transform, is_train=True):
        self.transform = transform

        self.ALL_LUPerson_NUM = 4180243

        # Ratio < 1: use a certain ratio of data
        if cfg.DATA.SAMPLE_RATIO < 1:
            num_sample = int(self.ALL_LUPerson_NUM * cfg.DATA.SAMPLE_RATIO)    
        # Ratio > 1: use a certain number of samples
        elif cfg.DATA.SAMPLE_RATIO > 1:
            num_sample = cfg.DATA.SAMPLE_RATIO
        # Ratio = 1: use all samples
        else:
            num_sample = self.ALL_LUPerson_NUM

        self.img_items = []
        # If sample data randomly
        if cfg.DATA.SAMPLE_SOURCE == 'random':
            for root, _, files in os.walk(cfg.DATA.ROOT_DIR):
                for file in files:
                    img_path = os.path.join(root, file)
                    pose_file = file[:-4] + '.npy'
                    pose_path = os.path.join(cfg.DATA.ROOT_DIR_POSE, pose_file)
                    self.img_items.append((img_path, pose_path))

            assert len(self.img_items) == self.ALL_LUPerson_NUM

            self.img_items = random.sample(self.img_items, num_sample)
            print(f'=> Use {len(self.img_items)} random LUPerson samples with keypoints')

        # If sample data using a list
        elif cfg.DATA.SAMPLE_SOURCE.endswith('.pkl'):
            f = open(cfg.DATA.SAMPLE_SOURCE, 'rb')
            source_list = pickle.load(f)

            assert len(source_list) == self.ALL_LUPerson_NUM

            for file in source_list[:num_sample]:
                img_path = os.path.join(cfg.DATA.ROOT_DIR, file)
                pose_file = file[:-4] + '.npy'
                pose_path = os.path.join(cfg.DATA.ROOT_DIR_POSE, pose_file)
                self.img_items.append((img_path, pose_path))
            print(f'=> Use {len(self.img_items)} LUPerson samples with keypoints from source {cfg.DATA.SAMPLE_SOURCE}')
        else:
            raise NotImplementedError
    
    def __len__(self):
        return len(self.img_items)
    
    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path, pose_path = img_item

        # img: transformed images
        # kps_in: keypoints within images
        # num_kps: number of keypoints within images
        # kps_all: all keypoints in order
        img, kps_in, num_kps, kps_all = load_image_pose(img_path, pose_path, self.transform)

        return (img, kps_in, num_kps, kps_all)
