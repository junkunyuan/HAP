# ----------------------------------------------------------------------
# HAP: Structure-Aware Masked Image Modeling for Human-Centric Perception
# Written by Junkun Yuan (yuanjk0921@outlook.com)
# ----------------------------------------------------------------------
# Build data loader with transforms
# ----------------------------------------------------------------------
# References:
# MALE: https://github.com/YanzuoLu/MALE
# MAE: https://github.com/facebookresearch/mae
# ----------------------------------------------------------------------

import cv2

import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import datasets

# ----------------------------------------------------------------------
# Build data transforms

BICUBIC = InterpolationMode.BICUBIC

def transforms_mae(size):
    """Pre-training transforms for ImageNet dataset, copied from MAE."""
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.2, 1.0), interpolation=BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    return transform


def transforms_male(size):
    """Pre-training transforms for LUPerson dataset, copied from MALE."""
    aspect_ratio = size[1] / size[0]
    ratio = (aspect_ratio * 3. / 4., aspect_ratio * 4. / 3.)
    
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.2, 1.0), ratio=ratio, interpolation=BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    
    return transform


def transforms_pose(size):
    """Pre-training transforms for LUPerson-pose dataset."""
    aspect_ratio = size[1] / size[0]
    ratio = (aspect_ratio * 3. / 4., aspect_ratio * 4. / 3.)
    
    transform = A.Compose([
            A.RandomResizedCrop(height=size[0], width=size[1], scale=(0.8, 1.0), 
                ratio=ratio, interpolation=cv2.INTER_CUBIC),
            A.HorizontalFlip(),
            A.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    return transform
# ----------------------------------------------------------------------


def initialize_data_loader(cfg):
    """Build data loader and save them in a dict."""
    data_loader = dict()

    transforms_train = globals()[cfg.DATA.TRANSFORMS](cfg.DATA.INPUT_SIZE)
    dataset_train = datasets.__dict__[cfg.DATA.NAME](cfg, transforms_train, is_train=True)
    if cfg.TASK == 'pretrain':
        sampler_train = torch.utils.data.DistributedSampler(
            dataset=dataset_train, 
            num_replicas=cfg.DIST.WORLD_SIZE, 
            rank=cfg.DIST.RANK, 
            shuffle=True
        )
        data_loader['train'] = torch.utils.data.DataLoader(
            dataset=dataset_train,
            batch_size=cfg.DATA.BATCH_SIZE,
            sampler=sampler_train,
            num_workers=cfg.DATA.NUM_WORKERS,
            pin_memory=cfg.DATA.PIN_MEMORY,
            drop_last=True
        )
    else:
        raise NotImplementedError
    
    return data_loader
