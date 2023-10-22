# ----------------------------------------------------------------------
# HAP: Structure-Aware Masked Image Modeling for Human-Centric Perception
# Written by Junkun Yuan (yuanjk0921@outlook.com)
# ----------------------------------------------------------------------
# Main file for pre-training
# ----------------------------------------------------------------------

import os
import time
import shutil
import argparse
import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from utils import create_logger

from config_pretrain import merge_config

from engine_pretrain import train_one_epoch

from utils import fix_seed, init_distributed_mode

from datasets.build_data_loader import initialize_data_loader

from models.build_model import initialize_model
from models.load_save_model import load_checkpoint, save_checkpoint


def set_config():
    parser = argparse.ArgumentParser('Run HAP for pre-training', add_help=False)

    # One may build file for configuration
    parser.add_argument('--config_file', default=None, type=str, help='path to config file')

    # Data parameters
    parser.add_argument('--dataset', default='LUPersonPose', type=str, help='pre-training dataset')
    parser.add_argument('--data_path', default='../LUPerson-data', type=str, help='path to dataset')
    parser.add_argument('--pose_path', default='../LUPerson-pose', type=str, help='path to pose info')
    parser.add_argument('--sample_split_source', default='../cfs_list.pkl', type=str, help='path to split source, or use random')
    parser.add_argument('--batch_size', default=None, type=int, help='batch size per GPU')
    parser.add_argument('--accum_iter', default=1, type=int, help='accumulate gradient iterations to increase total batch-size')

    # Model parameters
    parser.add_argument('--model', default='pose_mae_vit_base_patch16', type=str, help='model to use')
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    parser.add_argument('--ckpt_pos_embed', default='', nargs='+', type=int, help='position embedding size of checkpoint model')
    parser.add_argument('--mask_ratio', default=0.5, type=float, help='masking ratio')
    parser.add_argument('--align', default=0.05, type=float, help='weight for alignment loss')

    # Training parameters
    parser.add_argument('--epochs', default=400, type=int, help='training epochs')
    parser.add_argument('--warmup_epochs', default=40, type=int, help='epochs to warmup LR')
    parser.add_argument('--blr', default=1.5e-4, type=float, help='base lr')
    parser.add_argument('--lr', default=None, type=float, help='default: lr = base_lr * total_batch_size / 256')
    parser.add_argument('--device', default='cuda', type=str, help='device to use')
    parser.add_argument('--ckpt_overwrite', action='store_true', help='overwrite previous checkpoints during training')
    parser.set_defaults(ckpt_overwrite=False)

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=0, type=int, help='index of current process on current node')
    parser.add_argument('--dist_on_itp', action='store_true', help='distributed training on itp')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Misc parameters
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--tag', default='default', type=str, help='tag of experiment for log')
    parser.add_argument('--output_dir', default='', type=str, help='output path, default: output-HAP/<tag>/<task>/<model>/<dataset>')

    args = parser.parse_args()

    # Merge basic config, cfg file, and arguments 
    cfg = merge_config(args)

    return cfg


def main(cfg, logger):
    # Fix random seed
    fix_seed(cfg)

    # Build model, update batchsize and lr, initialize optimizer
    device = torch.device(cfg.TRAIN.DEVICE)
    model, model_without_ddp, optimizer, scaler, _ = initialize_model(cfg, logger, device)

    # Load model from checkpoint
    state = load_checkpoint(cfg, model_without_ddp, optimizer, scaler, logger)

    # Load data and build loader
    data_loader = initialize_data_loader(cfg)

    # Build tensorboard
    summary_writer = SummaryWriter(cfg.OUTPUT_DIR) if cfg.DIST.LOCAL_RANK == 0 else None

    # Save config
    logger.info(f'Running with config:\n{str(cfg)}')

    # Start run
    start_time = time.time()
    for epoch in range(cfg.TRAIN.START_EPOCH, cfg.TRAIN.EPOCHS):
        logger.info(f'=> TAG: {cfg.TAG}  TASK: {cfg.TASK}  MODEL: {cfg.MODEL.NAME}  DATA: {cfg.DATA.NAME}')

        state.epoch = epoch
        if cfg.DIST.DIST_MODE:
            data_loader['train'].sampler.set_epoch(epoch)
        
        # One may change masking strategy here
        mask_pose = {'mask_type': ['block', 'parts'], 'num_parts': 6}

        train_one_epoch(cfg, data_loader['train'], model, optimizer, scaler, epoch, device, summary_writer, logger, **mask_pose)

        if epoch % cfg.TRAIN.SAVE_FREQ == 0 or epoch + 1 == cfg.TRAIN.EPOCHS:
            save_checkpoint(cfg, state, epoch)

    total_time = time.time() - start_time
    logger.info(f'Total training time {datetime.timedelta(seconds=int(total_time))}')
    if summary_writer:
        summary_writer.close()


if __name__ == '__main__':
    # Set config parameters
    cfg = set_config()

    # Create directory
    if cfg.DIST.LOCAL_RANK == 0:
        # delete anyway, one may choose not to do
        if os.path.exists(cfg.OUTPUT_DIR):
            shutil.rmtree(cfg.OUTPUT_DIR)
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Initialize distributed training
    init_distributed_mode(cfg)
    
    # Create logger
    logger = create_logger(cfg.OUTPUT_DIR, cfg.DIST.RANK, cfg.MODEL.NAME, cfg.PRINT)

    main(cfg, logger)
