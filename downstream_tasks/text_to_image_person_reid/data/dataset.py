# -*- coding: utf-8 -*-
"""
make the CUHK-PEDE dataset

Created on Fri., Aug. 1(st), 2019 at 10:42

@author: zifyloo
"""

import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import os
from utils.read_write_data import read_dict
import cv2
import torchvision.transforms.functional as F
import random


def fliplr(img, dim):
    """
    flip horizontal
    :param img:
    :return:
    """
    inv_idx = torch.arange(img.size(dim) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(dim, inv_idx)
    return img_flip


class CUHKPEDEDataset(data.Dataset):
    def __init__(self, opt, tran):

        self.opt = opt
        self.flip_flag = (self.opt.mode == 'train')
        data_save = read_dict(opt.pkl_root + opt.mode + '_save.pkl')
        print(data_save.keys())
        if opt.dataset == 'CUHK-PEDES':
            self.img_path = [os.path.join(opt.dataroot,  img_path) for img_path in data_save['img_path']]
        elif opt.dataset == 'ICFG-PEDES':
            self.img_path = [os.path.join(opt.dataroot, img_path.split('/')[-3],img_path.split('/')[-2],img_path.split('/')[-1]) for img_path in data_save['img_path']]
        elif opt.dataset == 'RSTPReid':
            self.img_path = [os.path.join(opt.dataroot,  img_path) for img_path in data_save['img_path']]

        self.label = data_save['id']
        if self.opt.wordtype == 'bert':
            self.caption_code = data_save['bert_caption_id']
        elif self.opt.wordtype == 'lstm':
            self.caption_code = data_save['lstm_caption_id']
        self.transform = tran

        self.num_data = len(self.img_path)

    def __getitem__(self, index):
        """
        :param index:
        :return: image and its label
        """

        image = Image.open(self.img_path[index])
        image = self.transform(image)

        label = torch.from_numpy(np.array([self.label[index]])).long()

        caption_code, caption_length, caption_mask= self.caption_mask(self.caption_code[index])

        return image, label, caption_code, caption_length, caption_mask

    def caption_mask(self, caption):
        caption_length = len(caption)
        caption = torch.from_numpy(np.array(caption)).view(-1).float()
        if caption_length < self.opt.caption_length_max:
            zero_padding = torch.zeros(self.opt.caption_length_max - caption_length)
            caption = torch.cat([caption, zero_padding], 0)
        else:
            caption = caption[:self.opt.caption_length_max]
            caption_length = self.opt.caption_length_max
        caption_mask = np.where(caption != 0, 1, 0)
        return caption, caption_length , caption_mask

    def __len__(self):
        return self.num_data


class CUHKPEDE_img_dateset(data.Dataset):
    def __init__(self, opt, tran):

        self.opt = opt

        data_save = read_dict(opt.pkl_root + opt.mode + '_save.pkl')

        if opt.dataset == 'CUHK-PEDES':
            self.img_path = [os.path.join(opt.dataroot,  img_path) for img_path in data_save['img_path']]
        elif opt.dataset == 'ICFG-PEDES':
            self.img_path = [os.path.join(opt.dataroot, img_path.split('/')[-3],img_path.split('/')[-2],img_path.split('/')[-1]) for img_path in data_save['img_path']]
        elif opt.dataset == 'RSTPReid':
            self.img_path = [os.path.join(opt.dataroot,  img_path) for img_path in data_save['img_path']]

        self.label = data_save['id']

        self.transform = tran

        self.num_data = len(self.img_path)

    def __getitem__(self, index):
        """
        :param index:
        :return: image and its label
        """

        image = Image.open(self.img_path[index])
        image = self.transform(image)

        label = torch.from_numpy(np.array([self.label[index]])).long()

        return image, label

    def __len__(self):
        return self.num_data


class CUHKPEDE_txt_dateset(data.Dataset):
    def __init__(self, opt):

        self.opt = opt

        data_save = read_dict(opt.pkl_root + opt.mode + '_save.pkl')

        self.label = data_save['caption_label']
        if self.opt.wordtype == 'bert':
            self.caption_code = data_save['bert_caption_id']
        elif self.opt.wordtype == 'lstm':
            self.caption_code = data_save['lstm_caption_id']

        self.caption_matching_img_index = data_save['caption_matching_img_index']

        self.num_data = len(self.caption_code)

    def __getitem__(self, index):
        """
        :param index:
        :return: image and its label
        """

        label = torch.from_numpy(np.array([self.label[index]])).long()

        caption_code, caption_length , caption_mask= self.caption_mask(self.caption_code[index])
        caption_matching_img_index = self.caption_matching_img_index[index]
        return label, caption_code, caption_length, caption_mask,caption_matching_img_index

    def caption_mask(self, caption):
        caption_length = len(caption)
        caption = torch.from_numpy(np.array(caption)).view(-1).float()
        if caption_length < self.opt.caption_length_max:
            zero_padding = torch.zeros(self.opt.caption_length_max - caption_length)
            caption = torch.cat([caption, zero_padding], 0)
        else:
            caption = caption[:self.opt.caption_length_max]
            caption_length = self.opt.caption_length_max
        caption_mask = np.where(caption != 0, 1, 0)
        return caption, caption_length, caption_mask

    def __len__(self):
        return self.num_data

