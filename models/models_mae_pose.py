# ----------------------------------------------------------------------
# HAP: Structure-Aware Masked Image Modeling for Human-Centric Perception
# Written by Junkun Yuan (yuanjk0921@outlook.com)
# ----------------------------------------------------------------------
# Build HAP model
# ----------------------------------------------------------------------
# References:
# MALE: https://github.com/YanzuoLu/MALE
# MAE: https://github.com/facebookresearch/mae
# ----------------------------------------------------------------------

import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed

from .mae_attention import BlockAtt
from .mask_block import MaskingGenerator
from .mask_strategy import mask_body_parts

__all__ = [
    'pose_mae_vit_base_patch16',
    'pose_mae_vit_large_patch16'
]


class PoseMaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size
        self.grid_size = self.patch_embed.grid_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            BlockAtt(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            BlockAtt(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = self.build_2d_sincos_position_embedding(self.pos_embed.shape[-1])
        self.pos_embed.data.copy_(pos_embed)

        decoder_pos_embed = self.build_2d_sincos_position_embedding(self.decoder_pos_embed.shape[-1])
        self.decoder_pos_embed.data.copy_(decoder_pos_embed)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def build_2d_sincos_position_embedding(self, embed_dim, temperature=10000.):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='xy')
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos_embed = torch.cat([pe_token, pos_emb], dim=1)
        return pos_embed

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] % p == 0, imgs.shape[3] % p == 0

        h, w = self.patch_embed.grid_size
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h, w = self.patch_embed.grid_size
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs

    def random_masking(self, x, mask_ratio, **mask_pose):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        x = x[:, 1:, :]
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # Mask pose with higher probability
        if mask_pose:
            noise = noise.cpu().numpy()

            if 'block' in mask_pose['mask_type']:  # mask random block
                num_mask = int(L - len_keep)
                mask_gen = MaskingGenerator(input_size=self.grid_size, num_masking_patches=num_mask, min_num_patches=12, max_num_patches=None, min_aspect=0.3, max_aspect=None)
                noise = np.array([mask_gen() for _ in range(N)])
            if 'parts' in mask_pose['mask_type']:  # mask random body parts
                noise = mask_body_parts(noise, N, self.patch_size, self.grid_size, mask_pose['keypoints_all'], mask_pose['num_parts'])
            
            noise = torch.from_numpy(noise)
            noise = noise.to(x.device)

        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first subset
        ids_vis = ids_shuffle[:, :len_keep]
        x_vis = torch.gather(x, dim=1, index=ids_vis.unsqueeze(-1).repeat(1, 1, D))

        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_vis, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, **mask_pose):
        # Embed patches, (N, 3, H, W) -> (N, L, D)
        x = self.patch_embed(x)

        # Add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        x = torch.cat([cls_token.repeat(x.size(0), 1, 1), x], dim=1)

        # Masking, (N, L, D) -> (N, M, D), M = L * (1 - mask_ratio)
        x_vis, mask, ids_restore = self.random_masking(x, mask_ratio, **mask_pose)

        # Append cls token
        x_vis = torch.cat((cls_token.expand(x_vis.shape[0], -1, -1), x_vis), dim=1)
        # Apply Transformer blocks
        for blk in self.blocks:
            x_vis, _ = blk(x_vis)
        x_vis = self.norm(x_vis)

        return x_vis, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # Embed tokens
        x = self.decoder_embed(x)

        # Append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # Add pos embed
        x = x + self.decoder_pos_embed

        # Apply Transformer blocks
        for blk in self.decoder_blocks:
            x, _ = blk(x)
        x = self.decoder_norm(x)

        # Predictor projection
        x = self.decoder_pred(x)

        # Remove cls token
        x = x[:, 1:, :]

        return x

    def forward_reconst(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def align_loss(self, q, k, T=0.2):
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)

        # Gather all negative samples
        k = concat_all_gather(k)

        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * dist.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * T)
    
    def forward_align(self, cls_vis, cls_vis2):
        return self.align_loss(cls_vis, cls_vis2) + self.align_loss(cls_vis2, cls_vis)

    def forward(self, imgs, mask_ratio=0.5, align=False, **mask_pose):
        embed_vis, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, align=align, **mask_pose)
        embed_vis2, mask2, ids_restore2 = self.forward_encoder(imgs, mask_ratio, align=align, **mask_pose)

        if align:
            loss_align = self.forward_align(embed_vis[:, 0, :], embed_vis2[:, 0, :])
        else:
            loss_align = torch.FloatTensor([0]).to(imgs.device)

        pred = self.forward_decoder(embed_vis, ids_restore)  # [N, L, p*p*3]
        loss_rec = self.forward_reconst(imgs, pred, mask)

        pred2 = self.forward_decoder(embed_vis2, ids_restore2)  # [N, L, p*p*3]
        loss_rec = 0.5 * loss_rec + 0.5 * self.forward_reconst(imgs, pred2, mask2)

        return loss_align, loss_rec, pred, mask


def pose_mae_vit_base_patch16(img_size, norm_pix_loss):
    model = PoseMaskedAutoencoderViT(
        img_size=img_size, patch_size=16, embed_dim=768, depth=12, 
        num_heads=12, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=norm_pix_loss)
    return model

def pose_mae_vit_large_patch16(img_size, norm_pix_loss):
    model = PoseMaskedAutoencoderViT(
        img_size=img_size, patch_size=16, embed_dim=1024, depth=24, 
        num_heads=16, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=norm_pix_loss)
    return model


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
