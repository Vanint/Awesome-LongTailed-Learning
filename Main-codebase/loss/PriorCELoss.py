"""Copyright (c) Hyperconnect, Inc. and its affiliates.
All rights reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import *


class PriorCELoss(nn.Module):
    # Also named as LADE-CE Loss
    def __init__(self, num_classes, img_max=None, prior=None, prior_txt=None):
        super().__init__()
        self.img_num_per_cls = calculate_prior(num_classes, img_max, prior, prior_txt, return_num=True).float().cuda()
        self.prior = self.img_num_per_cls / self.img_num_per_cls.sum()
        self.criterion = nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def forward(self, x, y):
        logits = x + torch.log(self.prior + 1e-9)
        loss = self.criterion(logits, y)
        return loss


def create_loss(num_classes, img_max=None, prior=None, prior_txt=None):
    print('Loading PriorCELoss Loss.')
    return PriorCELoss(
        num_classes=num_classes,
        img_max=img_max,
        prior=prior,
        prior_txt=prior_txt,
    )
