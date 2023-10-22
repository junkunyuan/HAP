import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# OW: config that will be overwritten by command line arguments

# -----------------------------------------------------------------
# Dataset settings
# -----------------------------------------------------------------
# Image input size
_C.DATA = CN()
# Dataset name, OW
_C.DATA.NAME = 'Market1501'
# Path to dataset, OW
_C.DATA.ROOT_DIR = '../market/'
# Image input size, OW
_C.DATA.INPUT_SIZE = (256, 128)
# Data transforms
_C.DATA.TRANSFORM = 'transreid_aug'
# Workers to load data
_C.DATA.NUM_WORKERS = 16
# Number of instances in a batch
_C.DATA.NUM_INSTANCES = 8
# Training data batch size
_C.DATA.BATCH_SIZE = 64


# -----------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------
_C.MODEL = CN()
# Model name
_C.MODEL.NAME = 'vit_base_patch16'
# Checkpoint to resume, OW
_C.MODEL.RESUME = ''
# Position embedding size of checkpoint model
_C.MODEL.CHECKPOINT_POS_EMBED_SIZE = (16, 8)
# Drop path rate
_C.MODEL.DROP_PATH = 0.1
# GPU device
_C.MODEL.DEVICE_ID = 0
# Weight of identity loss
_C.MODEL.ID_LOSS_WEIGHT = 0.5
# Weight of triplet loss
_C.MODEL.TRI_LOSS_WEIGHT = 0.5

# -----------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------
_C.TRAIN = CN()
# Training epochs, OW
_C.TRAIN.EPOCHS = 100
# Warmup epochs, OW
_C.TRAIN.WARMUP_EPOCHS = 5
# Learning rate, OW
_C.TRAIN.LR = 8e-3
# Weight decay
_C.TRAIN.WEIGHT_DECAY = 0.05
# Layer decay
_C.TRAIN.LAYER_DECAY = 0.4
# Optimizer Betas
_C.TRAIN.BETAS = (0.9, 0.999)

# -----------------------------------------------------------------
# Evaluation settings
# -----------------------------------------------------------------
_C.VALIDATE = CN()
# Batch size for evaluation
_C.VALIDATE.BATCH_SIZE = 256
# Evaluation frequence
_C.VALIDATE.EVAL_FREQ = 10
# Evaluation data transform
_C.VALIDATE.TRANSFORM = 'no_aug'
# Feature normalization
_C.VALIDATE.FEAT_NORM = True
# Evaluation before training
_C.VALIDATE.BEFORE_TRAIN = False

# -----------------------------------------------------------------
# Misc settings
# -----------------------------------------------------------------
# Output directory, OW
_C.OUTPUT_DIR = 'output-person_reid'
# If overwrite previous checkpoint
_C.CHECKPOINT_OVERWRITE = True
# Print frequency
_C.PRINT_FREQ = 20
# Random seed, OW
_C.SEED = 0


def _update_config_from_file(cfg, cfg_file):
    """Update config from cfg file."""
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
    cfg.DATA.BATCH_SIZE = args.batch_size
    cfg.DATA.INPUT_SIZE = (args.size, 128)

    # Merge model parameters
    cfg.MODEL.NAME = args.model
    if args.resume:
        cfg.MODEL.RESUME = args.resume
    cfg.MODEL.DEVICE_ID = args.device

    # Merge training parameters
    cfg.TRAIN.EPOCHS = args.epochs
    cfg.TRAIN.WARMUP_EPOCHS = args.warmup_epochs
    cfg.TRAIN.LR = args.lr

    # Merge misc parameters
    cfg.SEED = args.seed
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    cfg.freeze()


def merge_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    cfg = _C.clone()
    update_config(cfg, args)

    return cfg
