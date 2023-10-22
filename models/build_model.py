# ----------------------------------------------------------------------
# HAP: Structure-Aware Masked Image Modeling for Human-Centric Perception
# Written by Junkun Yuan (yuanjk0921@outlook.com)
# ----------------------------------------------------------------------
# Build the model
# ----------------------------------------------------------------------
# References:
# MALE: https://github.com/YanzuoLu/MALE
# ----------------------------------------------------------------------

import math

import torch

from timm.optim import optim_factory

import models

from utils import NativeScalerWithGradNormCount


def initialize_model(cfg, logger, device):
    """Build model."""
    logger.info(f'=> Build model: {cfg.MODEL.NAME}')
    model = models.__dict__[cfg.MODEL.NAME](cfg.DATA.INPUT_SIZE, cfg.MODEL.NORM_PIX_LOSS)

    # Move model to GPU/CPU device
    model.to(device)
    model_without_ddp = model
    unused = False
    if cfg.DIST.DIST_MODE == True:
        model = torch.nn.parallel.DistributedDataParallel(model, [cfg.DIST.LOCAL_RANK], find_unused_parameters=unused)
        model_without_ddp = model.module
    
    if cfg.TASK == 'pretrain':
        param_groups = optim_factory.add_weight_decay(model_without_ddp, cfg.TRAIN.WEIGHT_DECAY)
        loss_func = None

    # Calculate effective/total batch-size and lr
    cfg.defrost()
    # Total batch size is 4096 by default
    if cfg.DATA.BATCH_SIZE is None:  # reset batch size
        cfg.DATA.BATCH_SIZE = int(cfg.DATA.TOTAL_BATCH_SIZE / cfg.DATA.ACCUM_ITER / cfg.DIST.WORLD_SIZE)
        logger.info(f'=> Reset batch-size to {cfg.DATA.BATCH_SIZE} and accum_iter to {cfg.DATA.ACCUM_ITER}')
    else:  # reset accum iter
        cfg.DATA.ACCUM_ITER = math.ceil(cfg.DATA.TOTAL_BATCH_SIZE / cfg.DATA.BATCH_SIZE / cfg.DIST.WORLD_SIZE)
        logger.info(f'=> Reset accum_iter to {cfg.DATA.ACCUM_ITER}')

    total_batch_size = int(cfg.DATA.BATCH_SIZE * cfg.DATA.ACCUM_ITER * cfg.DIST.WORLD_SIZE)
    if cfg.DATA.TOTAL_BATCH_SIZE != total_batch_size:
        cfg.DATA.TOTAL_BATCH_SIZE =  total_batch_size
        logger.info(f'=> Reset total batch-size to {cfg.DATA.TOTAL_BATCH_SIZE}')
    
    if cfg.TRAIN.LR is None:  # if only base_lr is specified
        cfg.TRAIN.LR = cfg.TRAIN.BLR * cfg.DATA.TOTAL_BATCH_SIZE / 256
        logger.info(f'=> Set lr to {cfg.TRAIN.LR}')
    cfg.freeze()
    
    optimizer = torch.optim.AdamW(param_groups, cfg.TRAIN.LR, cfg.TRAIN.BETAS)
    scaler = NativeScalerWithGradNormCount()

    return model, model_without_ddp, optimizer, scaler, loss_func
