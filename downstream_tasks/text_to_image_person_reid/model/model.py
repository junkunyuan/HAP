# -*- coding: utf-8 -*-
"""
Created on Sat., Aug. 1(st), 2019 at 9:05

@author: zifyloo
"""

from torch import nn
from model.text_feature_extract import TextExtract
from torchvision import models
import torch
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from model_TransREID.backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID


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


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)


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


class TextImgPersonReidNet(nn.Module):

    def __init__(self, opt):
        super(TextImgPersonReidNet, self).__init__()

        self.opt = opt
        resnet50 = models.resnet50(pretrained=True)

        self.ImageExtract = nn.Sequential(*(list(resnet50.children())[:-2]))
        self.TextExtract = TextExtract(opt)

        self.avg_global = nn.AdaptiveMaxPool2d((1, 1))
        # self.avg_global = nn.AdaptiveAvgPool2d((1, 1))

        self.conv_1X1 = conv(2048, opt.feature_length)

    def forward(self, image, caption_id, text_length):

        image_feature = self.img_embedding(image)
        text_feature = self.txt_embedding(caption_id, text_length)
        # print(text_feature.shape)

        return image_feature, text_feature

    def img_embedding(self, image):
        image_feature = self.ImageExtract(image)

        image_feature = self.avg_global(image_feature)
        image_feature = self.conv_1X1(image_feature)

        return image_feature

    def txt_embedding(self, caption_id, text_length):
        text_feature = self.TextExtract(caption_id, text_length)

        text_feature = self.conv_1X1(text_feature)

        return text_feature


class ft_net_TransREID(nn.Module):

    def __init__(self, class_num=751, droprate=0.5, stride=2):
        super(ft_net_TransREID, self).__init__()
        model_ft = vit_base_patch16_224_TransReID(img_size=[256, 128], sie_xishu=3.0,
                                                        camera=0, view=0, stride_size=[16, 16], drop_path_rate=0.1,
                                                        drop_rate= 0.0,
                                                        attn_drop_rate=0.0)
        self.in_planes = 768
        model_path = '/home/zhiying/my_test/text-image-reid/jx_vit_base_p16_224-80ecf9dd.pth'
        model_ft.load_param(model_path)
        print('Loading pretrained ImageNet model from {}'.format(model_path))
        self.model = model_ft

    def forward(self, x):
        x = self.model(x)
        return x

class ft_net_TransREID_local(nn.Module):

    def __init__(self, class_num=751, droprate=0.5, stride=2):
        super(ft_net_TransREID_local, self).__init__()
        model_ft = vit_base_patch16_224_TransReID(img_size=[384, 128], sie_xishu=3.0,local_feature=True,
                                                        camera=0, view=0, stride_size=[16, 16], drop_path_rate=0.1,
                                                        drop_rate= 0.0,
                                                        attn_drop_rate=0.0)
        self.in_planes = 768
        model_path = 'ckpt_default_pretrain_pose_mae_vit_base_patch16_LUPersonPose_399.pth'
        model_ft.load_param(model_path)
        print('Loading pretrained ImageNet model......from {}'.format(model_path))
        self.model = model_ft

    def forward(self, x):
        x = self.model(x)
        x = x[:,1:]
        return x

class ft_net_TransREID_local_smallDeiT(nn.Module):

    def __init__(self, class_num=751, droprate=0.5, stride=2):
        super(ft_net_TransREID_local_smallDeiT, self).__init__()
        model_ft = deit_small_patch16_224_TransReID(img_size=[384, 128], sie_xishu=3.0,local_feature=False,
                                                        camera=0, view=0, stride_size=[16, 16], drop_path_rate=0.1,
                                                        drop_rate= 0.0,
                                                        attn_drop_rate=0.0)
        self.in_planes = 768
        model_path = '/home/zhiyin/deit_small_distilled_patch16_224-649709d9.pth'
        model_ft.load_param(model_path)
        print('Loading pretrained ImageNet model......from {}'.format(model_path))
        self.model = model_ft

    def forward(self, x):
        x = self.model(x)
        return x

class ft_net_TransREID_local_smallVit(nn.Module):

    def __init__(self, class_num=751, droprate=0.5, stride=2):
        super(ft_net_TransREID_local_smallVit, self).__init__()
        model_ft = vit_small_patch16_224_TransReID(img_size=[384, 128], sie_xishu=3.0,local_feature=True,
                                                   camera=0, view=0, stride_size=[16, 16], drop_path_rate=0.1,
                                                   drop_rate=0.0,
                                                   attn_drop_rate=0.0)
        self.in_planes = 768
        model_path = '/home/zhiying/my_test/text-image-reid/vit_small_p16_224-15ec54c9.pth'
        model_ft.load_param(model_path)
        print('Loading pretrained ImageNet model......from {}'.format(model_path))
        self.model = model_ft

    def forward(self, x):
        x = self.model(x)
        x = x[:, 1:]
        return x