# -*- coding: utf-8 -*-
"""
Created on Sat., JUL. 20(th), 2019 at 16:51

@author: zifyloo
"""

from torchvision import transforms
from PIL import Image
import torch
from data.dataset import CUHKPEDEDataset, CUHKPEDE_img_dateset, CUHKPEDE_txt_dateset


def get_dataloader(opt):
    """
    tranforms the image, downloads the image with the id by data.DataLoader
    """

    if opt.mode == 'train':
        transform_list = [
            transforms.Resize((384, 128), interpolation=3),
            transforms.Pad(10),
            transforms.RandomCrop((384, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        tran = transforms.Compose(transform_list)

        dataset = CUHKPEDEDataset(opt, tran)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                                 shuffle=True, drop_last=True, num_workers=3)
        print('{}-{} has {} pohtos'.format(opt.dataset, opt.mode, len(dataset)))

        return dataloader

    else:
        tran = transforms.Compose([
        transforms.Resize((384, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
        )

        img_dataset = CUHKPEDE_img_dateset(opt, tran)

        img_dataloader = torch.utils.data.DataLoader(img_dataset, batch_size=opt.batch_size,
                                                 shuffle=False, drop_last=False, num_workers=3)

        txt_dataset = CUHKPEDE_txt_dateset(opt)

        txt_dataloader = torch.utils.data.DataLoader(txt_dataset, batch_size=opt.batch_size,
                                                 shuffle=False, drop_last=False, num_workers=3)

        print('{}-{} has {} pohtos, {} text'.format(opt.dataset, opt.mode, len(img_dataset), len(txt_dataset)))

        return img_dataloader, txt_dataloader



