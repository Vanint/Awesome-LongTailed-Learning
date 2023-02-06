import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *


class CBLoss(nn.Module):
    def __init__(self, prior_txt, loss_type="softmax", beta=0.9999, gamma=1):
        super(CBLoss, self).__init__()  
        self.no_of_classes = 1000
        self.samples_per_cls = calculate_prior(num_classes=1000, img_max=None, prior=None, prior_txt = prior_txt, return_num=True) 
        
        self.loss_type = loss_type
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits, labels): 
        effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
        weights = (1.0 - self.beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * self.no_of_classes
    
        labels_one_hot = F.one_hot(labels, self.no_of_classes).float()
    
        weights = torch.tensor(weights).float().cuda()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1,self.no_of_classes)
    
        if self.loss_type == "focal":
            cb_loss = self.focal_loss(labels_one_hot, logits, weights, self.gamma)
        elif self.loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
        elif self.loss_type == "softmax":
            pred = logits.softmax(dim = 1)
            cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
        return cb_loss

 

    
    def focal_loss(self, labels, logits, alpha, gamma): 
        BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")
    
        if gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
                torch.exp(-1.0 * logits)))
    
        loss = modulator * BCLoss
    
        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)
    
        focal_loss /= torch.sum(labels)
        return focal_loss

 

def create_loss(prior_txt):
    return CBLoss(prior_txt, loss_type="softmax", beta=0.9, gamma=1.0)
 