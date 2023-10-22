# -*- coding: utf-8 -*-
"""
Created on Sat., Aug. 3(rd), 2019 at 16:17

@author: zifyloo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init


class Loss(nn.Module):

    def __init__(self, opt):
        super(Loss, self).__init__()

        self.esp = 1e-8
        self.opt = opt

        self.W_Init()

    def W_Init(self):
        self.W = Parameter(torch.randn(512, self.opt.class_num))
        init.normal_(self.W.data, std=0.001)

    def calculate_CMPMLoss(self, image_embedding, text_embedding, label):

        Image_embedding_norm = image_embedding / image_embedding.norm(dim=1, keepdim=True)
        Text_embedding_norm = text_embedding / text_embedding.norm(dim=1, keepdim=True)

        image_text = torch.mm(image_embedding, Text_embedding_norm.t())
        text_image = torch.mm(text_embedding, Image_embedding_norm.t())
        P_image2text = F.softmax(image_text, dim=1)
        P_text2image = F.softmax(text_image, dim=1)

        labels_reshape = torch.reshape(label, (label.size(0), 1))
        labels_dist = labels_reshape - labels_reshape.t()
        y = (labels_dist == 0)
        Q = y.float() / y.float().sum(dim=1, keepdim=True)

        '''torch.log(P_image2text / (Q + self.esp)) will cause training collapse'''
        Li2t = torch.mean(torch.sum(P_image2text *
                                    (F.log_softmax(image_text, dim=1) - torch.log(Q + self.esp)), dim=1))
        Lt2i = torch.mean(torch.sum(P_text2image *
                                    (F.log_softmax(text_image, dim=1) - torch.log(Q + self.esp)), dim=1))

        CMPM_loss = Li2t + Lt2i

        sim_cos = torch.matmul(Image_embedding_norm, Text_embedding_norm.t())

        positive_sim = torch.mean(sim_cos[y])
        negative_sim = torch.mean(sim_cos[y == 0])

        return CMPM_loss, positive_sim, negative_sim

    def calculate_CMPCLoss(self, image_embedding, text_embedding, label):
        criterion = nn.CrossEntropyLoss(reduction='mean')
        self.W_norm = self.W / self.W.norm(dim=0, keepdim=True)

        Image_embedding_norm = image_embedding / image_embedding.norm(dim=1, keepdim=True)
        Text_embedding_norm = text_embedding / text_embedding.norm(dim=1, keepdim=True)

        z_cap_i2t = torch.sum(image_embedding * Text_embedding_norm, dim=1, keepdim=True) * Text_embedding_norm
        z_cap_t2i = torch.sum(text_embedding * Image_embedding_norm, dim=1, keepdim=True) * Image_embedding_norm

        score_i2t = torch.mm(z_cap_i2t, self.W_norm)
        score_t2i = torch.mm(z_cap_t2i, self.W_norm)

        label = label.view(label.size(0))
        Lipt = criterion(score_i2t, label)
        Ltpi = criterion(score_t2i, label)

        CMPCLoss = Lipt + Ltpi

        pred_i2t = torch.mean((torch.argmax(score_i2t, dim=1) == label).float())
        pred_t2i = torch.mean((torch.argmax(score_t2i, dim=1) == label).float())

        return CMPCLoss, pred_i2t, pred_t2i

    def forward(self, image_embedding, text_embedding, label):

        CMPC_loss = 0.0
        CMPM_loss = 0.0
        pred_i2t = 0.0
        pred_t2i = 0.0
        negative_sim = 0.0
        positive_sim = 0.0

        if self.opt.CMPM:
            CMPM_loss, positive_sim, negative_sim = self.calculate_CMPMLoss(image_embedding, text_embedding, label)
        if self.opt.CMPC:
            CMPC_loss, pred_i2t, pred_t2i = self.calculate_CMPCLoss(image_embedding, text_embedding, label)

        # loss = CMPM_loss + CMPC_loss

        return CMPM_loss, CMPC_loss, pred_i2t, pred_t2i, positive_sim, negative_sim
