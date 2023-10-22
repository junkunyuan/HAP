# ----------------------------------------------------------------------
# HAP: Structure-Aware Masked Image Modeling for Human-Centric Perception
# Written by Junkun Yuan (yuanjk0921@outlook.com)
# ----------------------------------------------------------------------
# Configuration file for pre-training
# ----------------------------------------------------------------------
# References:
# Swin Transformer: https://github.com/microsoft/Swin-Transformer
# ----------------------------------------------------------------------

import os
import yaml
from yacs.config import CfgNode as CN

import timm
import torch
import torchvision

_C = CN()

# Base config files
_C.BASE = ['']

# OW: configs that could be overwritten by command line arguments
# ----------------------------------------------------------------------
# Data settings
# ----------------------------------------------------------------------
_C.DATA = CN()
# Dataset name, OW
_C.DATA.NAME = 'LUPerson'
# Path to dataset, OW
_C.DATA.ROOT_DIR = '../LUPerson-data'
# Path to pose data, OW
_C.DATA.ROOT_DIR_POSE = '../LUPerson-pose'
# Sample source, 'random' or pkl file (e.g., cfs_list.pkl from TransReID-SSL), OW
_C.DATA.SAMPLE_SOURCE = 'random'
# Ratio of samples to train, use decimal like 0.5 or integer like 1281167 
_C.DATA.SAMPLE_RATIO = 0.5
# Image input size
_C.DATA.INPUT_SIZE = (256, 128)
# Data transforms, OW
_C.DATA.TRANSFORMS = 'transforms_pose'
# Workers to load data
_C.DATA.NUM_WORKERS = 16
# Training data batch size per gpu, OW
_C.DATA.BATCH_SIZE = None
# Effective training data batch size
_C.DATA.TOTAL_BATCH_SIZE = 4096
# Accumulate gradient iterasions, OW
_C.DATA.ACCUM_ITER = 1
# Pin CPU memory
_C.DATA.PIN_MEMORY = True

# ----------------------------------------------------------------------
# Model settings
# ----------------------------------------------------------------------
_C.MODEL = CN()
# Model name
_C.MODEL.NAME = 'pose_mae_vit_base_patch16'
# Checkpoint to resume, OW
_C.MODEL.RESUME = ''
# Position embedding size of checkpoint model, OW
_C.MODEL.CHECKPOINT_POS_EMBED_SIZE = (16, 8)
# Masking ratio (percentage of removed patches), OW
_C.MODEL.MASK_RATIO = 0.5
# Weight to align, OW
_C.MODEL.ALIGN = 0.05
# Normalization of target pixel values, OW
_C.MODEL.NORM_PIX_LOSS = True

# ----------------------------------------------------------------------
# Training settings
# ----------------------------------------------------------------------
_C.TRAIN = CN()
# Training epochs, OW
_C.TRAIN.EPOCHS = 400
# Start epoch
_C.TRAIN.START_EPOCH = 0
# Warmup epochs, OW
_C.TRAIN.WARMUP_EPOCHS = 40
# Base learning rate, OW
_C.TRAIN.BLR = 1.5e-4
# Learning rate, default: lr = blr * total_batch_size / 256, OW
_C.TRAIN.LR = None
# Weight decay
_C.TRAIN.WEIGHT_DECAY = 0.05
# Optimizer Betas
_C.TRAIN.BETAS = (0.9, 0.95)
# Device to train, OW
_C.TRAIN.DEVICE = 'cuda'
# If overwrite previous checkpoint
_C.TRAIN.CKPT_OVERWRITE = True
# Print frequency, iters
_C.TRAIN.PRINT_FREQ = 10
# Save frequency, epochs
_C.TRAIN.SAVE_FREQ = 5

# ----------------------------------------------------------------------
# Distributed training settings
# ----------------------------------------------------------------------
_C.DIST = CN()
# Distributed mode
_C.DIST.DIST_MODE = False
# World size, OW
_C.DIST.WORLD_SIZE = 1
# Rank
_C.DIST.RANK = 0
# Local rank, OW
_C.DIST.LOCAL_RANK = 0
# Enable ITP distributed training, OW
_C.DIST.DIST_ON_ITP = False
# URL used to set up distributed training, OW
_C.DIST.DIST_URL = 'env://'
# Backend of distributed training
_C.DIST.BACKEND = 'nccl'
# Log on 'master' process, or main process of each 'node', or 'all' processes
_C.DIST.LOG = 'node'

# ----------------------------------------------------------------------
# Environment settings
# ----------------------------------------------------------------------
_C.VERSION = CN()
# PyTorch version
_C.VERSION.PYTORCH = str(torch.__version__)
# TorchVision version
_C.VERSION.TORCHVISION = str(torchvision.__version__)
# timm version
_C.VERSION.TIMM = str(timm.__version__)

# ----------------------------------------------------------------------
# Misc settings
# ----------------------------------------------------------------------
# Task to run
_C.TASK = 'pretrain'
# Random seed, OW
_C.SEED = 0
# Tag of experiment for log, OW
_C.TAG = 'default'
# Output directory, OW
_C.OUTPUT_DIR = 'output-HAP/temp'
# Flag of print on console
_C.PRINT = False


def _update_config_from_file(cfg, cfg_file):
    """Update config from cfg file."""
    if cfg_file:
        cfg.defrost()
        with open(cfg_file, 'r') as f:
            yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
        # Update from each config file in 'Base' key of cfg file
        for cfg_each in yaml_cfg.setdefault('BASE', ['']):
            if cfg_each:
                _update_config_from_file(cfg, os.path.join(os.path.dirname(cfg_file), cfg_each))
        print(f'=> Merge config from {cfg_file}')
        cfg.merge_from_file(cfg_file)
        cfg.freeze()


def update_config(cfg, args):
    """Update config from cfg file and command line arguments."""

    # Update from cfg file
    _update_config_from_file(cfg, args.config_file)

    # Update from command line arguments
    cfg.defrost()

    # Merge data parameters
    cfg.DATA.NAME = args.dataset
    cfg.DATA.ROOT_DIR = args.data_path
    cfg.DATA.ROOT_DIR_POSE = args.pose_path
    cfg.DATA.SAMPLE_SOURCE = args.sample_split_source
    if cfg.DATA.NAME == 'LUPerson':
        cfg.DATA.TRANSFORMS = 'transforms_male'
    elif cfg.DATA.NAME == 'LUPersonPose':
        cfg.DATA.TRANSFORMS = 'transforms_pose'
    else:
        cfg.DATA.TRANSFORMS = 'transforms_mae'
    cfg.DATA.BATCH_SIZE = args.batch_size
    cfg.DATA.ACCUM_ITER = args.accum_iter

    # Merge model parameters
    cfg.MODEL.NAME = args.model
    cfg.MODEL.RESUME = args.resume
    if args.ckpt_pos_embed:
        cfg.MODEL.CHECKPOINT_POS_EMBED_SIZE = tuple(args.ckpt_pos_embed)
    cfg.MODEL.MASK_RATIO = args.mask_ratio
    cfg.MODEL.ALIGN = args.align

    # Merge training parameters
    cfg.TRAIN.EPOCHS = args.epochs
    cfg.TRAIN.WARMUP_EPOCHS = args.warmup_epochs
    cfg.TRAIN.BLR = args.blr
    cfg.TRAIN.LR = args.lr
    cfg.TRAIN.DEVICE = args.device
    cfg.TRAIN.CKPT_OVERWRITE = args.ckpt_overwrite

    # Merge distributed training parameters
    cfg.DIST.WORLD_SIZE = args.world_size
    cfg.DIST.LOCAL_RANK = args.local_rank
    cfg.DIST.DIST_ON_ITP = args.dist_on_itp
    cfg.DIST.DIST_URL = args.dist_url

    # Merge misc parameters
    cfg.SEED = args.seed
    cfg.TAG = args.tag
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    else:
        cfg.OUTPUT_DIR = os.path.join('output-HAP', cfg.TAG, cfg.TASK, cfg.MODEL.NAME, cfg.DATA.NAME)

    cfg.freeze()


def merge_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    cfg = _C.clone()
    update_config(cfg, args)

    return cfg
