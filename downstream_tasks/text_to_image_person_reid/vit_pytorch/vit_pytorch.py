import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import init
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.constant(m.weight.data, 1)
        init.constant(m.bias.data, 0)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            # nn.GELU(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Attention_DECODER(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim*2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, txt, image, mask=None):
        b1, n1, _, h = *image.shape, self.heads
        b2, n2, _, h = *txt.shape, self.heads

        q = self.to_q(txt)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        kv = self.to_kv(image).chunk(2, dim = -1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), kv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(mask, mask_value)
            del mask
        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

class PreNorm_3(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, y, **kwargs):
        return self.fn(self.norm(x), self.norm(y),**kwargs)

class Residual_3(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, y,**kwargs):
        return self.fn(x, y, **kwargs) + x

class Residual_3_y(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, y,**kwargs):
        return self.fn(x, y, **kwargs) + y


class PartQuery(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()

        self.part_net = Residual_3(PreNorm_3(dim, Attention_DECODER(dim, heads=heads,
                                                                    dim_head=dim_head,
                                                                    dropout=dropout)))

    def forward(self, x, y, mask=None):
        x = self.part_net(x, y, mask=mask)
        return x


class Transformer_DECODER(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual_3(PreNorm_3(dim, Attention_DECODER(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, y, mask=None):
        for attn_decode, ff in self.layers:  # attn,
            # x = attn(x, mask = mask)
            x = attn_decode(x, y , mask=mask)
            x = ff(x)
        return x

class Transformer_DECODER_gai(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim,Residual(Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                PreNorm_3(dim,Residual_3(Attention_DECODER(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                PreNorm(dim,Residual(FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, y,mask = None):
        for attn, attn_decode, ff in self.layers:
            x = attn(x, mask = mask)
            x = attn_decode(x, y , mask=mask)
            x = ff(x)
        return x

class Transformer_DECODER_noself(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual_3(PreNorm_3(dim, Attention_DECODER(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, y,mask = None):
        for attn_decode, ff in self.layers:
            # x = attn(x, mask = mask)
            x = attn_decode(x, y , mask=mask)
            x = ff(x)
        return x

class Transformer_DECODER_y(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                PreNorm_3(dim, Attention_DECODER(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, y,mask = None):
        for attn, attn_decode, ff in self.layers:
            x = attn(x, mask = mask)
            x = attn_decode(x, y , mask=mask)
            x = ff(x)
        return x

class TransformerEncode(nn.Module):
    def __init__(self, *,  dim, depth, heads, mlp_dim, pool = 'cls',  dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.to_embedding = nn.Sequential(
            nn.Linear(512, dim)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, 100 + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

    def forward(self, input,caption):
        ignore_mask = (caption == 0)
        x = self.to_embedding(input)
        b, n, c = x.shape
        cls_tokens = repeat(self.cls_token, '() n c -> b n c', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x, None)
        f = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # x = self.to_latent(f)
        return f

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls',
                 channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, mask = None):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x, mask)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)

class pixel_ViT(nn.Module):
    def __init__(self, *, image_size_h, image_size_w ,patch_size , dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        # assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size_h//patch_size) * (image_size_w//patch_size)
        patch_dim = channels * patch_size**2
        # assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2)  -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool

    def forward(self, img,pos_embedding, mask = None):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += pos_embedding
        x = self.dropout(x)

        x = self.transformer(x, mask)

        return x

class pixel_trans(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        patch_dim = channels

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c h w  -> b (h w) c'),
            nn.Linear(patch_dim, dim),
        )

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool

    def forward(self, img, mask = None):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x, mask)
        return x

class ENCODER(nn.Module):
    def __init__(self, *, image_size_h, image_size_w ,patch_size , dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        # assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size_h//patch_size) * (image_size_w//patch_size)
        patch_dim = channels * patch_size**2
        # assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2)  -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool

    def forward(self, img, mask = None):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x, mask)

        x = x[:, 1:]

        return x

class DECODER(nn.Module):
    def __init__(self, *, opt,dim, depth, heads, mlp_dim, pool = 'cls', patch_dim = 2048, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.to_DIM_txt= nn.Linear(patch_dim, dim)
        self.to_DIM_img = nn.Linear(patch_dim, dim)

        self.dropout = nn.Dropout(emb_dropout)
        if opt.res_y:
            print('res_y')
            self.transformer = Transformer_DECODER_y(dim, depth, heads, dim_head, mlp_dim, dropout)
        elif opt.noself:
            print('noself')
            self.transformer = Transformer_DECODER_noself(dim, depth, heads, dim_head, mlp_dim, dropout)
        elif opt.post_norm:
            print('post_norm')
            self.transformer = Transformer_DECODER_gai(dim, depth, heads, dim_head, mlp_dim, dropout)
        else:
            self.transformer = Transformer_DECODER(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool

    def forward(self, txt, img, mask = None): # img [64,48,2048] txt [64,Length,2048]
        img = self.to_DIM_img(img)
        txt = self.to_DIM_txt(txt)
        b_img, n_img, _ = img.shape
        b_txt, n_txt, _ = txt.shape
        x = self.transformer(txt, img, mask)
        return x

from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import init
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class Attention_mydecoder(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim*2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self,txt, image ,kv_mask = None,q_mask = None):
        b1, n1, _, h = *image.shape, self.heads
        b2, n2, _, h = *txt.shape, self.heads

        q = self.to_q(txt)
        q = rearrange(q,'b n (h d) -> b h n d', h=h)
        kv = self.to_kv(image).chunk(2, dim = -1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), kv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max
        if kv_mask is not None:
            assert kv_mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(q_mask, 'b i -> b () i ()') * rearrange(kv_mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask
        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer_mydecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual_3(PreNorm_3(dim, Attention_mydecoder(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, y,kv_mask = None,q_mask = None):
        for attn_decode, ff in self.layers:
            x = attn_decode(x, y , kv_mask=kv_mask,q_mask = q_mask)
            x = ff(x)
        return x

class conv(nn.Module):

    def __init__(self, input_dim, output_dim, relu=False, BN=False):
        super(conv, self).__init__()

        block = []
        block += [nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False)]

        if BN:
            block += [nn.BatchNorm2d(output_dim)]
        if relu:
            block += [nn.ReLU(inplace=True)]

        self.block = nn.Sequential(*block)
        self.block.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.block(x)
        x = x.squeeze(3).squeeze(2)
        return x


class Transformer_DETR(nn.Module):
    def __init__(self, part_num, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.part_num = part_num
        self.conv_DETR = nn.ModuleList()
        for _ in range(depth):
            conv_1X1_2 = nn.ModuleList()
            for _ in range(part_num):
                conv_1X1_2.append(conv(384, 384))
            self.conv_DETR.append(
                conv_1X1_2
            )
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual_3(PreNorm_3(dim, Attention_mydecoder(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, y,kv_mask = None,q_mask = None):
        conv_num = 0
        for attn_decode, ff in self.layers:
            part_feat = []
            x = attn_decode(x, y , kv_mask=kv_mask,q_mask = q_mask)
            x = ff(x)
            for i in range(self.part_num):
                x_i = self.conv_DETR[conv_num][i](x[:,i,:].unsqueeze(2).unsqueeze(2))
                part_feat.append(x_i.unsqueeze(1))
            x = torch.cat(part_feat,dim=1)
        return x

class mydecoder(nn.Module):
    def __init__(self, *, opt,dim, depth, heads, mlp_dim, pool = 'cls', patch_dim = 2048, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer_mydecoder(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool

    def forward(self, txt,img, kv_mask = None,q_mask = None): # img [64,48,2048] txt [64,Length,2048]
        b_img, n_img, _ = img.shape
        b_txt, n_txt, _ = txt.shape
        x = self.transformer(txt, img, kv_mask,q_mask)
        return x

class mydecoder_DETR(nn.Module):
    def __init__(self, *, opt,dim, depth, heads, mlp_dim, pool = 'cls', patch_dim = 2048, dim_head = 64, dropout = 0., emb_dropout = 0.,part_num=6):
        super().__init__()

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer_DETR(part_num,dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool

    def forward(self, txt,img, kv_mask = None,q_mask = None): # img [64,48,2048] txt [64,Length,2048]
        b_img, n_img, _ = img.shape
        b_txt, n_txt, _ = txt.shape
        x = self.transformer(txt, img, kv_mask,q_mask)
        return x