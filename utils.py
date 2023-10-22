# ----------------------------------------------------------------------
# HAP: Structure-Aware Masked Image Modeling for Human-Centric Perception
# Written by Junkun Yuan (yuanjk0921@outlook.com)
# ----------------------------------------------------------------------
# Utils
# ----------------------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# ----------------------------------------------------------------------

import os
import sys
import math
import random
import logging
import datetime
import builtins
import functools
import numpy as np

import torch
from torch._six import inf
import torch.distributed as dist


def fix_seed(cfg):
    """Fix the random seed for reproducibility."""
    seed = max(cfg.SEED + cfg.DIST.RANK, 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def adjust_learning_rate(optimizer, epoch, epochs, warmup_epochs, lr):
    """Decay the learning rate with half-cycle cosine after warmup."""
    if epoch < warmup_epochs:
        lr = lr * epoch / warmup_epochs 
    else:
        # lr = lr_min + 0.5 * (lr_max - lr_min)(1 + cos(T_cur / T_all * pi))
        lr = lr * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


@functools.lru_cache()
def create_logger(output_dir, rank, name, if_print):
    """Create logger to print and save log."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Create formatter
    fmt = '[%(asctime)s](%(filename)s %(lineno)d) %(message)s'
    try:
        from termcolor import colored
        color_fmt = colored('[%(asctime)s]', 'green') + \
                    colored('(%(filename)s %(lineno)d) ', 'yellow') + \
                    '%(message)s'
    except:
        color_fmt = fmt
    datefmt = f'%Y-%m-%d %H:%M:%S'

    # Create console handlers for pre-set process
    if if_print:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter(fmt=color_fmt, datefmt=datefmt))
        logger.addHandler(console_handler)
    
    # Create file handlers  
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{rank}.txt'), mode='a')
    file_handler.setFormatter(logging.Formatter(fmt, datefmt))
    logger.addHandler(file_handler)

    return logger


# ----------------------------------------------------------------------
# Gradient normalization
class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm
# ----------------------------------------------------------------------


# ---------------------------------------------------------------------- 
# Distributed training settings
def setup_for_distributed(if_print):
    """Enable print on pre-set process."""
    builtin_print = builtins.print
    
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if if_print or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def check_log_process(log, rank, local_rank):
    """Print and save log on pre-set process."""
    if_master = (log == 'master') and (rank == 0)  # on master process
    if_node = (log == 'node') and (local_rank == 0)  # on main process of each node
    if_all = (log == 'all')  # on all processes

    return if_master or if_node or if_all


def init_distributed_mode(cfg):
    """Initialize distributed training."""
    cfg.defrost()
    if cfg.DIST.DIST_ON_ITP:
        cfg.DIST.RANK = int(os.environ['OMPI_COMM_WORLD_RANK'])
        cfg.DIST.WORLD_SIZE = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        cfg.DIST.LOCAL_RANK = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        cfg.DIST.DIST_URL = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(cfg.DIST.LOCAL_RANK)
        os.environ['RANK'] = str(cfg.DIST.RANK)
        os.environ['WORLD_SIZE'] = str(cfg.DIST.WORLD_SIZE)
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        cfg.DIST.RANK = int(os.environ["RANK"])
        cfg.DIST.WORLD_SIZE = int(os.environ['WORLD_SIZE'])
        cfg.DIST.LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        cfg.DIST.RANK = int(os.environ['SLURM_PROCID'])
        cfg.DIST.LOCAL_RANK = cfg.DIST.RANK % torch.cuda.device_count()
    else:
        print('=> Not using distributed mode')
        cfg.DIST.DIST_MODE = False
        cfg.PRINT = True
        setup_for_distributed(True)
        return
    
    cfg.DIST.DIST_MODE = True

    torch.cuda.set_device(cfg.DIST.LOCAL_RANK)
    print(f'=> Distributed init: url {cfg.DIST.DIST_URL}  rank {cfg.DIST.RANK})  gpu {cfg.DIST.LOCAL_RANK}')
    dist.init_process_group(backend=cfg.DIST.BACKEND, init_method=cfg.DIST.DIST_URL, 
                            world_size=cfg.DIST.WORLD_SIZE, rank=cfg.DIST.RANK)
    dist.barrier()

    cfg.PRINT = check_log_process(cfg.DIST.LOG, cfg.DIST.RANK, cfg.DIST.LOCAL_RANK)
    setup_for_distributed(cfg.PRINT)
    
    cfg.freeze()
# ----------------------------------------------------------------------
