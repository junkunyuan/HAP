import torch
import torch.nn.functional as F


def interpolate_pos_embed(model, checkpoint_model, orig_size, param_name):
    if param_name in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model[param_name]
        embedding_size = pos_embed_checkpoint.shape[-1]
        # height, width for the new position embedding
        new_size_h, new_size_w = model.patch_embed.grid_size
        # class_token and dist_token are kept unchanged
        if orig_size[0] * orig_size[1] != new_size_h * new_size_w:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size[0], orig_size[1], new_size_h, new_size_w))
            extra_tokens = pos_embed_checkpoint[:, :1]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, 1:]
            pos_tokens = pos_tokens.reshape(-1, orig_size[0], orig_size[1], embedding_size).permute(0, 3, 1, 2)
            pos_tokens = F.interpolate(pos_tokens, size=(new_size_h, new_size_w), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model[param_name] = new_pos_embed