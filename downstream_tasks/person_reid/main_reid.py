import argparse
import copy
import datetime
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from timm.utils import AverageMeter
from torch.utils.tensorboard import SummaryWriter

import datasets
import models
import utils.transforms
from config_reid import _C as cfg
from utils.eval import eval_func
from utils.logger import setup_logger
from utils.lr_decay import param_groups_lrd
from utils.lr_scheduler import adjust_learning_rate
from utils.pos_embed import interpolate_pos_embed
from utils.sampler import RandomIdentitySampler
from utils.scaler import NativeScalerWithGradNormCount
from utils.triplet_loss import TripletLoss

from config_reid import merge_config


def set_config():
    parser = argparse.ArgumentParser('Finetune for person ReID task', add_help=False)

    parser.add_argument('--config_file', default='configs/reid/market.yaml', type=str, help='path to config file')

    # Data parameters
    parser.add_argument('--batch_size', default=64, type=int, help='batch size per GPU')
    parser.add_argument('--size', default=256, type=int, help='input size')

    # Model parameters
    parser.add_argument('--model', default='', type=str, help='model to finetune')
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    parser.add_argument('--device', default=0, type=int, help='gpu device')

    # Training parameters
    parser.add_argument('--epochs', default=100, type=int, help='training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='epochs to warmup LR')
    parser.add_argument('--lr', type=float, default=8e-3, help='lr')

    # Misc parameters
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--output_dir', default='', type=str, help='root of output folder')
    
    args = parser.parse_args()

    # Merge config, cfg file, and command line arguments 
    cfg = merge_config(args)

    return cfg


def main(cfg):
    seed = cfg.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device_id = cfg.MODEL.DEVICE_ID
    torch.cuda.set_device(device_id)
    cudnn.benchmark = True
    logger.info(f'set cuda device = {device_id}')
    
    train_loader, val_loader, num_classes, num_queries = initialize_data_loader(cfg)
    model, criterion, optimizer, scaler = initialize_model(cfg, num_classes, device_id)
    summary_writer = SummaryWriter(cfg.OUTPUT_DIR)

    state = load_checkpoint(cfg, model, optimizer, scaler)

    if cfg.VALIDATE.BEFORE_TRAIN:
        rank1, mAP = validate(val_loader, model, device_id, num_queries, cfg.VALIDATE.FEAT_NORM, cfg.PRINT_FREQ)
        logger.info(f'validation results: Rank-1: {rank1}  mAP: {mAP}')

    start_epoch = state.epoch + 1
    logger.info(f'start_epoch: {start_epoch}')

    start_time = time.time()
    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        state.epoch = epoch

        train_one_epoch(train_loader, model, criterion, optimizer, scaler, epoch, device_id, cfg.TRAIN.EPOCHS, 
                        cfg.TRAIN.WARMUP_EPOCHS, cfg.TRAIN.LR, cfg.PRINT_FREQ, summary_writer)

        # save_checkpoint(state, cfg.OUTPUT_DIR, epoch, cfg.CHECKPOINT_OVERWRITE)

        if ((epoch + 1) % cfg.VALIDATE.EVAL_FREQ == 0) or (epoch + 1 == cfg.TRAIN.EPOCHS):
            rank1, mAP = validate(val_loader, model, device_id, num_queries, cfg.VALIDATE.FEAT_NORM, cfg.PRINT_FREQ)
            logger.info(f'validation results: mAP: {mAP} rank-1: {rank1}  ')
    
    total_time = time.time() - start_time
    logger.info(f'training time {datetime.timedelta(seconds=int(total_time))}')
    if summary_writer:
        summary_writer.close()
    

def train_one_epoch(train_loader, model, criterion, optimizer, scaler, epoch, device_id, 
                    epochs, warmup_epochs, lr, print_freq, summary_writer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    id_losses = AverageMeter()
    tri_losses = AverageMeter()

    model.train()
    num_steps = len(train_loader)

    start = time.time()
    end = time.time()
    for i, (imgs, pid, _) in enumerate(train_loader):
        lr_decayed = adjust_learning_rate(optimizer, i / num_steps + epoch, epochs, warmup_epochs, lr)
        
        imgs = imgs.cuda(device_id, non_blocking=True)
        target = pid.cuda(device_id, non_blocking=True)

        with torch.cuda.amp.autocast():
            feats, logits = model(imgs)
        loss, id_loss, tri_loss = criterion(feats, logits, target)

        loss_value = loss.item()
        id_loss_value = id_loss.item()
        tri_loss_value = tri_loss.item()
        if not math.isfinite(loss_value):
            logger.info(f'loss is {loss_value}, stopping training')
            sys.exit(1)

        scaler(loss, optimizer, parameters=model.parameters())
        optimizer.zero_grad()

        losses.update(loss_value)
        id_losses.update(id_loss_value)
        tri_losses.update(tri_loss_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            etas = batch_time.avg * (num_steps - i)
            logger.info(
                f'Train [{epoch}/{epochs}]({i}/{num_steps})  '
                f'Time {batch_time.val:.4f}({batch_time.avg:.4f})  '
                f'Loss {losses.val:.4f}({losses.avg:.4f})  '
                f'Id_loss {id_losses.val:.4f}({id_losses.avg:.4f})  '
                f'Tri_loss {tri_losses.val:.4f}({tri_losses.avg:.4f})  '
                f'Lr {lr_decayed:.4e}  '
                f'Eta {datetime.timedelta(seconds=int(etas))}'
            )

        if summary_writer:
            summary_writer.add_scalar('Loss', loss_value, epoch * num_steps + i)
            summary_writer.add_scalar('Id_loss', id_loss_value, epoch * num_steps + i)
            summary_writer.add_scalar('Tri_loss', tri_loss_value, epoch * num_steps + i)
            summary_writer.add_scalar('Lr', lr_decayed, epoch * num_steps + i)

    epoch_time = time.time() - start
    logger.info(f'EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}')


def validate(val_loader, model, device_id, num_queries, feat_norm, print_freq):
    batch_time = AverageMeter()

    model.eval()
    num_steps = len(val_loader)

    feats = []
    pids = []
    camids = []

    start = time.time()
    end = time.time()
    with torch.no_grad():
        for i, (img, pid, camid) in enumerate(val_loader):
            img = img.cuda(device_id, non_blocking=True)

            with torch.cuda.amp.autocast():
                feat, _ = model(img)
            
            if isinstance(feat, tuple):
                feat = torch.cat(feat, dim=-1)

            feats.append(feat.cpu())
            pids.extend(np.asarray(pid))
            camids.extend(np.asarray(camid))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                etas = batch_time.avg * (num_steps - i)
                logger.info(
                    f'Valdate ({i}/{num_steps})  '
                    f'Time {batch_time.val:.4f}({batch_time.avg:.4f})  '
                    f'Eta {datetime.timedelta(seconds=int(etas))}'
                )
    
    feats = torch.cat(feats, dim=0)
    if feat_norm:
        feats = F.normalize(feats, dim=1, p=2)
    
    query_feats = feats[:num_queries]
    query_pids = np.asarray(pids[:num_queries])
    query_camids = np.asarray(camids[:num_queries])

    gallery_feats = feats[num_queries:]
    gallery_pids = np.asarray(pids[num_queries:])
    gallery_camids = np.asarray(camids[num_queries:])
    
    q = query_feats.shape[0]
    g = gallery_feats.shape[0]
    dist_mat = torch.pow(query_feats, exponent=2).sum(dim=1, keepdim=True).expand(q, g) + \
               torch.pow(gallery_feats, exponent=2).sum(dim=1, keepdim=True).expand(g, q).t()
    dist_mat.addmm_(query_feats, gallery_feats.t(), beta=1, alpha=-2).cpu().numpy()

    cmc, mAP = eval_func(dist_mat, query_pids, gallery_pids, query_camids, gallery_camids, max_rank=1)

    total_time = time.time() - start
    logger.info(f'validation takes {datetime.timedelta(seconds=int(total_time))}')
    return cmc[0], mAP


def initialize_model(cfg, num_classes, device_id):
    logger.info(f'creating model: {cfg.MODEL.NAME}')
    model = models.__dict__[cfg.MODEL.NAME](cfg, num_classes)
    model.cuda(device_id)
    
    triplet = TripletLoss()
    def loss_func(feats, logits, target):
        if not isinstance(feats, tuple) and not isinstance(logits, tuple):
            id_loss = F.cross_entropy(logits, target)
            tri_loss = triplet(feats, target)[0]
        else:
            id_loss = [F.cross_entropy(logit, target) for logit in logits]
            id_loss = sum(id_loss) / len(id_loss)
            tri_loss = [triplet(feat, target)[0] for feat in feats]
            tri_loss = sum(tri_loss) / len(tri_loss)
        return cfg.MODEL.ID_LOSS_WEIGHT * id_loss + cfg.MODEL.TRI_LOSS_WEIGHT * tri_loss, id_loss, tri_loss

    param_groups = param_groups_lrd(model, cfg.TRAIN.WEIGHT_DECAY, model.no_weight_decay(), cfg.TRAIN.LAYER_DECAY)
    optimizer = torch.optim.AdamW(param_groups, cfg.TRAIN.LR, cfg.TRAIN.BETAS)
    scaler = NativeScalerWithGradNormCount()
    return model, loss_func, optimizer, scaler


def initialize_data_loader(cfg):
    train_transform = utils.transforms.__dict__[cfg.DATA.TRANSFORM](cfg)
    train_dataset = datasets.__dict__[cfg.DATA.NAME](cfg, train_transform, is_train=True)
    num_classes = train_dataset.num_classes
    train_sampler = RandomIdentitySampler(train_dataset, cfg.DATA.BATCH_SIZE, cfg.DATA.NUM_INSTANCES)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.DATA.BATCH_SIZE, 
        num_workers=cfg.DATA.NUM_WORKERS, 
        pin_memory=True, 
        sampler=train_sampler
    )

    val_transform = utils.transforms.__dict__[cfg.VALIDATE.TRANSFORM](cfg)
    val_dataset = datasets.__dict__[cfg.DATA.NAME](cfg, val_transform, is_train=False)
    num_queries = val_dataset.num_queries
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.VALIDATE.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATA.NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, val_loader, num_classes, num_queries


class State:
    def __init__(self, arch, model, optimizer, scaler):
        self.epoch = -1
        self.arch = arch
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
    
    def capture_snapshot(self):
        return {
            'epoch': self.epoch,
            'arch': self.arch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict()
        }
    
    def apply_snapshot(self, obj):
        msg = self.model.load_state_dict(obj['model'], strict=False)
        if 'arch' in obj.keys() and self.arch == obj['arch']:
            self.epoch = obj['epoch']
            self.optimizer.load_state_dict(obj['optimizer'])
            self.scaler.load_state_dict(obj['scaler'])
        return msg
    
    def save(self, f):
        torch.save(self.capture_snapshot(), f)
    
    def load(self, f, pos_embed_size):
        snapshot = torch.load(f, map_location='cpu')
        state_dict = self.model.state_dict()
        for k in list(snapshot['model'].keys()):
            # atl model warps mae
            if k.startswith('mae'):
                snapshot['model'][k[4:]] = copy.deepcopy(snapshot['model'][k])
                del snapshot['model'][k]
                k = k[4:]
            if k not in state_dict or (snapshot['model'][k].shape != state_dict[k].shape and k != 'pos_embed'):
                del snapshot['model'][k]
        interpolate_pos_embed(self.model, snapshot['model'], pos_embed_size, 'pos_embed')
        msg = self.apply_snapshot(snapshot)
        logger.info(msg)


def load_checkpoint(cfg, model_without_ddp, optimizer, scaler):
    state = State(cfg.MODEL.NAME, model_without_ddp, optimizer, scaler)

    if os.path.isfile(cfg.MODEL.RESUME):
        logger.info(f'loading checkpoint file: {cfg.MODEL.RESUME}')
        state.load(cfg.MODEL.RESUME, cfg.MODEL.CHECKPOINT_POS_EMBED_SIZE)
        logger.info(f'loaded checkpoint file: {cfg.MODEL.RESUME}')
    else:
        logger.info(f'Not find checkpoint to load')

    return state


def save_checkpoint(state, output_dir, epoch, checkpoint_overwrite):
    if checkpoint_overwrite:
        filename = os.path.join(output_dir, 'checkpoint.pth')
    else:
        filename = os.path.join(output_dir, 'checkpoint_%04d.pth' % epoch)
    checkpoint_dir = os.path.dirname(filename)
    os.makedirs(checkpoint_dir, exist_ok=True)

    tmp_filename = filename + '.tmp'
    torch.save(state.capture_snapshot(), tmp_filename)
    os.rename(tmp_filename, filename)
    logger.info(f'saved checkpoint for epoch {state.epoch} as {filename}')


if __name__ == '__main__':
    # Set config parameters
    cfg = set_config()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    logger = setup_logger(cfg.OUTPUT_DIR, local_rank=0, name=cfg.MODEL.NAME)
    logger.info(f'running with config:\n{str(cfg)}')

    main(cfg)
