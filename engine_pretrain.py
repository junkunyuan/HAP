# ----------------------------------------------------------------------
# HAP: Structure-Aware Masked Image Modeling for Human-Centric Perception
# Written by Junkun Yuan (yuanjk0921@outlook.com)
# ----------------------------------------------------------------------
# Training file for pre-training
# ----------------------------------------------------------------------
# References:
# MALE: https://github.com/YanzuoLu/MALE
# ----------------------------------------------------------------------

import os
import sys
import time
import math
import json
import datetime

import torch
import torch.distributed as dist

from timm.utils import AverageMeter

from utils import adjust_learning_rate


def train_one_epoch(cfg, train_loader, model, optimizer, scaler, epoch, device, summary_writer, logger, **mask_pose):
    batch_time = AverageMeter()
    losses = AverageMeter()

    if cfg.MODEL.NAME.startswith('pose_mae'):
        losses_ali = AverageMeter()
        losses_rec = AverageMeter()

    accum_iter = cfg.DATA.ACCUM_ITER

    model.train()
    num_steps = len(train_loader)

    start = time.time()
    end = time.time()
    for data_iter_step, (samples, keypoints, num_kps, keypoints_all) in enumerate(train_loader):
        if data_iter_step % accum_iter == 0:
            lr_decayed = adjust_learning_rate(optimizer, data_iter_step / num_steps + epoch, cfg.TRAIN.EPOCHS, cfg.TRAIN.WARMUP_EPOCHS, cfg.TRAIN.LR)

        samples_train = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            if cfg.MODEL.NAME.startswith('pose_mae'):
                align = True if cfg.MODEL.ALIGN > 0 else False
                
                loss_ali, loss_rec, _, _ = model(samples_train, cfg.MODEL.MASK_RATIO, align=align, keypoints=keypoints, num_kps=num_kps, keypoints_all=keypoints_all, **mask_pose)
                
                loss = loss_ali * cfg.MODEL.ALIGN + loss_rec
            else:
                if cfg.DATA.NAME == 'LUPerson':
                    loss, _, _ = model(samples_train, cfg.MODEL.MASK_RATIO)
                elif cfg.DATA.NAME == 'LUPersonPose':
                    loss, _, _ = model(samples_train, cfg.MODEL.MASK_RATIO, keypoints=keypoints, num_kps=num_kps, keypoints_all=keypoints_all, **mask_pose)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            logger.warning(f'Loss is {loss_value}, stopping training')
            sys.exit(1)

        loss /= accum_iter
        scaler(loss, optimizer, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        losses.update(loss_value)
        if cfg.MODEL.NAME.startswith('pose_mae'):
            losses_ali.update(loss_ali.item())
            losses_rec.update(loss_rec.item())

        batch_time.update(time.time() - end)
        end = time.time()

        GB = 1024. ** 3
        if data_iter_step % cfg.TRAIN.PRINT_FREQ == 0:
            etas = batch_time.avg * (num_steps - data_iter_step)
            if cfg.MODEL.NAME.startswith('pose_mae'):
                logger.info(
                    f'Epoch: [{epoch}/{cfg.TRAIN.EPOCHS}] ({data_iter_step}/{num_steps})  '
                    f'loss: {losses.val:.4f} ({losses.avg:.4f})  '
                    f'loss_ali: {losses_ali.val:.4f} ({losses_ali.avg:.4f})  '
                    f'loss_rec: {losses_rec.val:.4f} ({losses_rec.avg:.4f})  '
                    f'lr: {lr_decayed:.4e}  '
                    # f'time: {batch_time.val:.4f} ({batch_time.avg:.4f})  '
                    # f'eta: {datetime.timedelta(seconds=int(etas))}  '
                )
            else:
                logger.info(
                    f'Epoch: [{epoch}/{cfg.TRAIN.EPOCHS}] ({data_iter_step}/{num_steps})  '
                    f'loss: {losses.val:.4f} ({losses.avg:.4f})  '
                    f'lr: {lr_decayed:.4e}  '
                    # f'time: {batch_time.val:.4f} ({batch_time.avg:.4f})  '
                    # f'eta: {datetime.timedelta(seconds=int(etas))}  '
                )

        if cfg.DIST.WORLD_SIZE > 1:
            loss_value_reduce = torch.tensor(loss_value).cuda()
            dist.all_reduce(loss_value_reduce)
            loss_value_reduce_mean = loss_value_reduce / cfg.DIST.WORLD_SIZE
            loss_value = loss_value_reduce_mean.item()

        if summary_writer:
            epoch_1000x = int((data_iter_step / num_steps + epoch) * 1000)
            summary_writer.add_scalar('loss', loss_value, epoch_1000x)
            summary_writer.add_scalar('lr', lr_decayed, epoch_1000x)

    epoch_time = time.time() - start
    epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
    max_men = torch.cuda.max_memory_allocated() / GB
    logger.info(f'EPOCH TIME: {epoch_time_str}    GPU MEMORY: {max_men:.2f} GB')

    if cfg.PRINT:
        if summary_writer is not None:
            summary_writer.flush()
        
        with open(os.path.join(cfg.OUTPUT_DIR, 'result.txt'), mode='a', encoding='utf-8') as f:
            if cfg.MODEL.NAME.startswith('pose_mae'):
                f.write(json.dumps({'epoch': epoch, 'loss': losses.avg, 'loss_ali': losses_ali.avg, 'loss_rec': losses_rec.avg, 'lr': lr_decayed}) + '\n')
            else:
                f.write(json.dumps({'epoch': epoch, 'loss': losses.avg, 'lr': lr_decayed}) + '\n')
            f.flush()
