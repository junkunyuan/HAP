# -*- coding: utf-8 -*-
"""
Created on Sat., Aug. 17(rd), 2019 at 15:33

@author: zifyloo
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class classifier(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(classifier, self).__init__()

        self.block = nn.Linear(input_dim, output_dim)
        self.block.apply(weights_init_classifier)

    def forward(self, x):
        x = self.block(x)
        return x


class Id_Loss(nn.Module):

    def __init__(self, opt):
        super(Id_Loss, self).__init__()

        self.opt = opt

        self.W = classifier(opt.feature_length, opt.class_num)
        # self.W_txt = classifier(opt.feature_length, opt.class_num)

    def calculate_IdLoss(self, image_embedding, text_embedding, label):

        label = label.view(label.size(0))

        criterion = nn.CrossEntropyLoss(reduction='mean')

        score_i2t = self.W(image_embedding)
        score_t2i = self.W(text_embedding)
        Lipt_local = criterion(score_i2t, label)
        Ltpi_local = criterion(score_t2i, label)
        pred_i2t = torch.mean((torch.argmax(score_i2t, dim=1) == label).float())
        pred_t2i = torch.mean((torch.argmax(score_t2i, dim=1) == label).float())
        loss = (Lipt_local + Ltpi_local)

        return loss, pred_i2t, pred_t2i

    def forward(self, image_embedding, text_embedding, label):

        loss, pred_i2t, pred_t2i = self.calculate_IdLoss(image_embedding, text_embedding, label)

        return loss, pred_i2t, pred_t2i


class Id_Loss_2(nn.Module):

    def __init__(self, opt):
        super(Id_Loss_2, self).__init__()

        self.opt = opt

        self.W = classifier(opt.feature_length, opt.class_num)
        # self.W_txt = classifier(opt.feature_length, opt.class_num)

    def calculate_IdLoss(self, image_embedding, text_embedding, label):

        label = label.view(label.size(0))

        criterion = nn.CrossEntropyLoss(reduction='mean')

        score_i2t = self.W(image_embedding)
        score_t2i = self.W(text_embedding)
        Lipt_local = criterion(score_i2t, label)
        Ltpi_local = criterion(score_t2i, label)
        pred_i2t = torch.mean((torch.argmax(score_i2t, dim=1) == label).float())
        pred_t2i = torch.mean((torch.argmax(score_t2i, dim=1) == label).float())
        # loss = (Lipt_local + Ltpi_local)

        return Lipt_local, Ltpi_local,pred_i2t, pred_t2i

    def forward(self, image_embedding, text_embedding, label):

        Lipt_local, Ltpi_local, pred_i2t, pred_t2i = self.calculate_IdLoss(image_embedding, text_embedding, label)

        return Lipt_local, Ltpi_local, pred_i2t, pred_t2i


class Id_Loss_3(nn.Module):

    def __init__(self, opt):
        super(Id_Loss_3, self).__init__()

        self.opt = opt

        self.W = classifier(opt.feature_length, opt.class_num)
        # self.W_txt = classifier(opt.feature_length, opt.class_num)

    def calculate_IdLoss(self, image_embedding, image_embedding_2 , text_embedding, label):

        label = label.view(label.size(0))

        criterion = nn.CrossEntropyLoss(reduction='mean')

        score_i2t = self.W(image_embedding)
        score_i2t_MPN = self.W(image_embedding_2)
        score_t2i = self.W(text_embedding)
        Lipt_local = criterion(score_i2t, label)
        Ltpi_local = criterion(score_t2i, label)
        Lipt_local_MPN = criterion(score_i2t_MPN, label)
        pred_i2t = torch.mean((torch.argmax(score_i2t, dim=1) == label).float())
        pred_t2i = torch.mean((torch.argmax(score_t2i, dim=1) == label).float())
        pred_i2t_MPN = torch.mean((torch.argmax(score_i2t_MPN, dim=1) == label).float())
        # loss = (Lipt_local + Ltpi_local)

        return Lipt_local, Ltpi_local,Lipt_local_MPN,pred_i2t, pred_t2i , pred_i2t_MPN

    def forward(self, image_embedding, image_embedding_2 , text_embedding, label):

        Lipt_local, Ltpi_local,Lipt_local_MPN,pred_i2t, pred_t2i , pred_i2t_MPN = self.calculate_IdLoss(image_embedding, image_embedding_2 , text_embedding, label)

        return Lipt_local, Ltpi_local,Lipt_local_MPN,pred_i2t, pred_t2i , pred_i2t_MPN