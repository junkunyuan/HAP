# -*- coding: utf-8 -*-
"""
Created on Sat., Aug. 17(rd), 2019 at 15:41

@author: zifyloo
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def calculate_similarity_global(image_embedding, text_embedding):
    image_embedding_norm = image_embedding / (image_embedding.norm(dim=1, keepdim=True) + 1e-8)
    text_embedding_norm = text_embedding / (text_embedding.norm(dim=1, keepdim=True) + 1e-8)

    similarity = torch.mm(image_embedding_norm, text_embedding_norm.t())

    return similarity


class RankingLoss(nn.Module):

    def __init__(self, opt):
        super(RankingLoss, self).__init__()

        self.margin = opt.margin
        self.device = opt.device

    def semi_hard_negative(self, loss):
        negative_index = np.where(np.logical_and(loss < self.margin, loss > 0))[0]
        return np.random.choice(negative_index) if len(negative_index) > 0 else None

    def get_triplets(self, similarity, labels):
        similarity = similarity.cpu().data.numpy()

        labels = labels.cpu().data.numpy()
        triplets = []

        for idx, label in enumerate(labels):  # same class calculate together

            negative = np.where(labels != label)[0]

            ap_sim = similarity[idx, idx]
            # print(ap_combination_list.shape, ap_distances_list.shape)

            loss = similarity[idx, negative] - ap_sim + self.margin

            negetive_index = self.semi_hard_negative(loss)

            if negetive_index is not None:
                triplets.append([idx, idx, negative[negetive_index]])

        if len(triplets) == 0:
            triplets.append([idx, idx, negative[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)

    def forward(self, similarity, label):

        image_triplets = self.get_triplets(similarity, label)
        text_triplets = self.get_triplets(similarity.t(), label)

        # print(image_triplets.size(), text_triplets.size())
        image_anchor_loss = F.relu(self.margin
                            - similarity[image_triplets[:, 0], image_triplets[:, 1]]
                            + similarity[image_triplets[:, 0], image_triplets[:, 2]])
        similarity = similarity.t()
        texy_anchor_loss = F.relu(self.margin
                            - similarity[text_triplets[:, 0], text_triplets[:, 1]]
                            + similarity[text_triplets[:, 0], text_triplets[:, 2]])

        loss = torch.sum(image_anchor_loss) + torch.sum(texy_anchor_loss)
        # loss = CMPM_loss + CMPC_loss

        return loss
