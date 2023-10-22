# ----------------------------------------------------------------------
# HAP: Structure-Aware Masked Image Modeling for Human-Centric Perception
# Written by Junkun Yuan (yuanjk0921@outlook.com)
# ----------------------------------------------------------------------
# Load and save model
# ----------------------------------------------------------------------
# References:
# MALE: https://github.com/YanzuoLu/MALE
# MAE: https://github.com/facebookresearch/mae
# ----------------------------------------------------------------------

import os

import torch
import torch.nn.functional as F


class State:
    def __init__(self, arch, model_without_ddp, optimizer, scaler):
        self.epoch = -1
        self.arch = arch
        self.model_without_ddp = model_without_ddp
        self.optimizer = optimizer
        self.scaler = scaler

    def capture_snapshot(self):
        """Return current training state."""
        return {
            'epoch': self.epoch,
            'arch': self.arch,
            'model': self.model_without_ddp.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict()
        }

    def apply_snapshot(self, ckpt):
        """Update training state from checkpoint."""
        msg = self.model_without_ddp.load_state_dict(ckpt['model'], strict=False)
        if 'arch' in ckpt.keys() and self.arch == ckpt['arch']:
            self.epoch = ckpt['epoch']
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scaler.load_state_dict(ckpt['scaler'])
        return msg

    def save(self, ckpt_path):
        """Save current training state."""
        torch.save(self.capture_snapshot(), ckpt_path)

    def load(self, ckpt_path, pos_embed_size, logger):
        snapshot = torch.load(ckpt_path, map_location='cpu')
        interpolate_pos_embed(self.model_without_ddp, snapshot['model'], pos_embed_size, 'pos_embed', logger)
        msg = self.apply_snapshot(snapshot)
        logger.info(msg)


def load_checkpoint(cfg, model_without_ddp, optimizer, scaler, logger):
    state = State(cfg.MODEL.NAME, model_without_ddp, optimizer, scaler)

    if os.path.exists(cfg.MODEL.RESUME):
        state.load(cfg.MODEL.RESUME, cfg.MODEL.CHECKPOINT_POS_EMBED_SIZE, logger)
        logger.info(f'=> Load checkpoint from {cfg.MODEL.RESUME}')

        cfg.defrost()
        cfg.TRAIN.START_EPOCH = state.epoch + 1
        cfg.freeze()

    return state


def save_checkpoint(cfg, state, epoch):
    """Save model checkpoint."""
    ckpt_name = f'ckpt_{cfg.TAG}_{cfg.TASK}_{cfg.MODEL.NAME}_{cfg.DATA.NAME}_{epoch}.pth'
    ckpt_path = os.path.join(cfg.OUTPUT_DIR, ckpt_name)

    if cfg.DIST.LOCAL_RANK == 0:
        torch.save(state.capture_snapshot(), ckpt_path)
        # Whether delete previous checkpoints
        if cfg.TRAIN.CKPT_OVERWRITE:
            for epo_idx in range(epoch):
                pre_ckpt_name = f'ckpt_{cfg.TAG}_{cfg.TASK}_{cfg.MODEL.NAME}_{cfg.DATA.NAME}_{epo_idx}.pth'
                pre_ckpt_path = os.path.join(cfg.OUTPUT_DIR, pre_ckpt_name)
                if os.path.exists(pre_ckpt_path):
                    os.remove(pre_ckpt_path)


def interpolate_pos_embed(model, checkpoint_model, orig_size, param_name, logger=None):
    if param_name in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model[param_name]
        embedding_size = pos_embed_checkpoint.shape[-1]
        # height, width for the new position embedding
        new_size_h, new_size_w = model.patch_embed.grid_size
        # class_token and dist_token are kept unchanged
        if orig_size[0] * orig_size[1] != new_size_h * new_size_w:
            info = f"=> Position embedding interpolate from ({orig_size[0]}, {orig_size[1]}) to ({new_size_h}, {new_size_w})"
            if logger:
                logger.info(info)
            else:
                print(info)
            extra_tokens = pos_embed_checkpoint[:, :1]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, 1:]
            pos_tokens = pos_tokens.reshape(-1, orig_size[0], orig_size[1], embedding_size).permute(0, 3, 1, 2)
            pos_tokens = F.interpolate(pos_tokens, size=(new_size_h, new_size_w), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model[param_name] = new_pos_embed