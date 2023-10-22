# -*- coding: utf-8 -*-
"""
Created on Thurs., Aug. 1(st), 2019

Update on on Sun., Aug. 4(th), 2019

@author: zifyloo
"""

import argparse
import torch
import logging
import os
from utils.read_write_data import makedir

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class options():
    def __init__(self):
        self._par = argparse.ArgumentParser(description='options for Deep Cross Modal')

        self._par.add_argument('--mode', type=str, default='train', help='choose mode [train or test]')
        self._par.add_argument('--trained', type=bool, default=False, help='whether the network has pretrained model')

        self._par.add_argument('--bidirectional', type=bool, default=True, help='whether the lstm is bidirectional')
        self._par.add_argument('--using_pose', type=bool, default=False, help='whether using pose')
        self._par.add_argument('--last_lstm', type=bool, default=False, help='whether just using the last lstm')
        self._par.add_argument('--using_noun', type=bool, default=False, help='whether just using the noun')

        self._par.add_argument('--epoch', type=int, default=300, help='train epoch')
        self._par.add_argument('--start_epoch', type=int, default=0, help='the start epoch')
        self._par.add_argument('--epoch_decay', type=list, default=[], help='decay epoch')
        self._par.add_argument('--wd', type=float, default=0.00004, help='weight decay')
        self._par.add_argument('--batch_size', type=int, default=16, help='batch size')
        self._par.add_argument('--adam_alpha', type=float, default=0.9, help='momentum term of adam')
        self._par.add_argument('--adam_beta', type=float, default=0.999, help='momentum term of adam')
        self._par.add_argument('--lr', type=float, default=0.002, help='initial learning rate for adam')
        self._par.add_argument('--margin', type=float, default=0.2, help='ranking loss margin')

        self._par.add_argument('--vocab_size', type=int, default=5000, help='the size of vocab')
        self._par.add_argument('--feature_length', type=int, default=512, help='the length of feature')
        self._par.add_argument('--class_num', type=int, default=11003,
                               help='num of class for StarGAN training on second dataset')
        self._par.add_argument('--part', type=int, default=6, help='the num of image part')
        self._par.add_argument('--caption_length_max', type=int, default=100, help='the max length of caption')
        self._par.add_argument('--random_erasing', type=float, default=0.0, help='the probability of random_erasing')

        self._par.add_argument('--Save_param_every', type=int, default=5, help='the frequency of save the param ')
        self._par.add_argument('--save_path', type=str, default='./checkpoints/test',
                               help='save the result during training')
        self._par.add_argument('--GPU_id', type=str, default='0', help='choose GPU ID')
        self._par.add_argument('--device', type=str, default='', help='cuda devie')
        self._par.add_argument('--dataset', type=str, default='CUHK-PEDES', help='choose the dataset ')
        self._par.add_argument('--dataroot', type=str, default='/data1/zhiying/text-image/CUHK-PEDES',
                               help='data root of the Data')
        self._par.add_argument('--pkl_root', type=str,
                               default='/home/zefeng/Exp/code/text-image/code by myself/data/processed_data/',
                               help='data root of the pkl')

        self._par.add_argument('--test_image_num', type=int, default=200, help='the num of images in test mode')

        self.opt = self._par.parse_args()

        self.opt.device = torch.device('cuda:{}'.format(self.opt.GPU_id[0]))


def config(opt):

    log_config(opt)
    model_root = os.path.join(opt.save_path, 'model')
    if os.path.exists(model_root) is False:
        makedir(model_root)


def log_config(opt):
    logroot = os.path.join(opt.save_path, 'log')
    if os.path.exists(logroot) is False:
        makedir(logroot)
    filename = os.path.join(logroot, opt.mode + '.log')
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(handler)
    if opt.mode != 'test':
        logger.info(opt)



