"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
 
 
 

class SEQL(nn.Module):
    def __init__(self, gamma=0.9, lambda_n=0.00043):
        super(SEQL, self).__init__()
        self.gamma = gamma
        self.lambda_n = lambda_n 
        dist = [0 for _ in range(1000)]
        with open('./data/ImageNet_LT/ImageNet_LT_train.txt') as f:
            for line in f:
                dist[int(line.split()[1])] += 1
        num = sum(dist)
        prob = [i/num for i in dist]
        prob = torch.FloatTensor(prob)     
        self.prob = prob
        class_weight = torch.zeros(1000).cuda()
        for i in range(1000):
            class_weight[i] = 1 if self.prob[i] > lambda_n else 0 
        self.class_weight=class_weight

    def replace_masked_values(self, tensor, mask, replace_with):
        assert tensor.dim() == mask.dim(), '{} vs {}'.format(tensor.shape, mask.shape)
        one_minus_mask = 1 - mask
        values_to_add = replace_with * one_minus_mask
        return tensor * mask + values_to_add


    def forward(self, input, target):
        N, C = input.shape
        not_ignored = self.class_weight.view(1, C).repeat(N, 1)
        over_prob = (torch.rand(input.shape).cuda() > self.gamma).float()
        is_gt = target.new_zeros((N, C)).float()
        is_gt[torch.arange(N), target] = 1

        weights = ((not_ignored + over_prob + is_gt) > 0).float()
        input = self.replace_masked_values(input, weights, -1e7)
        loss = F.cross_entropy(input, target)
        return loss
    
    
    """
    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        
        
 
        bs = target.shape[0]
        b = torch.ones([bs,1000]).cuda()
        T = torch.zeros([bs,1000]).cuda() 
        for i in range(bs):
            if np.random.random()> self.gamma:
                b[i]=0 
            if self.prob[target[i]]< self.lambda_n:
                T[i] = 1 
        instance_weight = b*T 
        target_onehot = F.one_hot(target, num_classes=1000)
        weight = 1-instance_weight* (1-target_onehot)
        logpt = input.detach().cuda()
        for i in range(bs):
            logit_sum = torch.sum(weight[i]*input[i].exp()+input[i].exp())
            logpt[i] = input[i].exp()/logit_sum
        
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)

        loss = -1 * logpt.log()
        return loss.mean()
   """  
def create_loss(prior_txt):
    print('Loading SEQL Loss.')
    return SEQL()