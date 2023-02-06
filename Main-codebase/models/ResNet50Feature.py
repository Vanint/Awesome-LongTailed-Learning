"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from libs.models.ResNetFeature import *
from libs.utils.utils import *
from os import path
        
def create_model(use_fc=False, dropout=None, stage1_weights=False, dataset=None, log_dir=None, test=False, *args):
    
    print('Loading Scratch ResNet 50 Feature Model.')
    resnet50 = ResNet(Bottleneck, [3, 4, 6, 3], use_fc=use_fc, dropout=None)

    if not test:
        if stage1_weights:
            assert(dataset)
            print('Loading %s Stage 1 ResNet 10 Weights.' % dataset)
            if log_dir is not None:
                # subdir = log_dir.strip('/').split('/')[-1]
                # subdir = subdir.replace('stage2', 'stage1')
                # weight_dir = path.join('/'.join(log_dir.split('/')[:-1]), subdir)
                weight_dir = path.join('/'.join(log_dir.split('/')[:-1]), 'stage1')
            else:
                weight_dir = './logs/%s/stage1' % dataset
            print('==> Loading weights from %s' % weight_dir)
            resnet50 = init_weights(model=resnet50,
                                    weights_path=path.join(weight_dir, 'final_model_checkpoint.pth'))
        else:
            print('No Pretrained Weights For Feature Model.')

    return resnet50
