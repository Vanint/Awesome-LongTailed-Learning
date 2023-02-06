"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from models.ResNextFeature import *
from utils import *
from os import path
        
def create_model(use_fc=False, dropout=None, stage1_weights=False, dataset=None, log_dir=None, test=False, last_relu=True, *args):
    
    print('Loading Scratch ResNext 50 Feature Model.')
    print('======> Last ReLU: ', last_relu)
    resnext = ResNext(Bottleneck, [3, 4, 6, 3], use_fc=use_fc, dropout=None,
                       groups=32, width_per_group=4, last_relu=last_relu)

    if not test:
        if stage1_weights:
            assert(dataset)
            print('Loading %s Stage 1 ResNext 50 Weights.' % dataset)
            if log_dir is not None:
                weight_dir = path.join('/'.join(log_dir.split('/')[:-1]), 'stage1')
            else:
                weight_dir = './logs/%s/stage1' % dataset
            print('==> Loading weights from %s' % weight_dir)
            resnext = init_weights(model=resnext,
                                    weights_path=path.join(weight_dir, 'final_model_checkpoint.pth'))
        else:
            print('No Pretrained Weights For Feature Model.')

    return resnext