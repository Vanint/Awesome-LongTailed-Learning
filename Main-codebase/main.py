"""Copyright (c) Hyperconnect, Inc. and its affiliates.
Custmoized by Yifan Zhang
All rights reserved.
"""

import os
import argparse
import pprint
from data import dataloader
from run_networks import model
import warnings
import yaml
import numpy as np
from utils import source_import, update
from pathlib import Path
import torch.backends.cudnn as cudnn


data_root_dict = {'ImageNet': '../data/ImageNet'}

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default=None, type=str)
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--save_feature', default=False, action='store_true')
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--model_dir', type=str, default=None)
parser.add_argument('--alpha', type=float, default=None)
parser.add_argument('--lr', type=float, default=None) 
parser.add_argument("--remine_lambda", default=None, type=float)
parser.add_argument("--work_dir", default="./exp_results", type=str, help="output dir")
parser.add_argument("--exp_name", default="test", type=str, help="exp name")
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument("--no-use-dv", action="store_true")
parser.add_argument("--test_imb_ratio", type=float, default=None,
                    help="Give explicit imbalance ratio for test dataset.")
parser.add_argument("--exist_only", type=int, default=0)
parser.add_argument("--test-reverse", type=int, default=0)
parser.add_argument("--train-reverse", action="store_true")
parser.add_argument('--root', default=None, type=str)

args = parser.parse_args()
args.test_reverse = bool(args.test_reverse)

print(f'args: {args}')

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
output_dir = f'{args.work_dir}/{args.exp_name}'
Path(output_dir).mkdir(parents=True, exist_ok=True)
# ============================================================================
# Random Seed
import torch
import random
if args.seed is not None:
    print('=======> Using Fixed Random Seed <========')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True

# ============================================================================
# LOAD CONFIGURATIONS
with open(args.cfg) as f:
    config = yaml.safe_load(f)
config = update(config, args, output_dir)

test_mode = args.test
save_mode = args.save_feature  # only in eval
training_opt = config['training_opt']
dataset = training_opt['dataset']

if not os.path.isdir(training_opt['log_dir']):
    os.makedirs(training_opt['log_dir'])

if args.root is not None:
    data_root = args.root
else:
    data_root = data_root_dict[dataset.rstrip('_LT')]

print('Loading dataset from: %s' % data_root)
pprint.pprint(config)


# ============================================================================
# TRAINING
if not test_mode:
    # during training, different sampler may be applied
    sampler_defs = training_opt['sampler']
    if sampler_defs:
        if sampler_defs['type'] == 'ClassAwareSampler':
            sampler_dic = {
                'sampler': source_import(sampler_defs['def_file']).get_sampler(),
                'params': {'num_samples_cls': sampler_defs['num_samples_cls']}
            }
        elif sampler_defs['type'] in ['MixedPrioritizedSampler',
                                      'ClassPrioritySampler']:
            sampler_dic = {
                'sampler': source_import(sampler_defs['def_file']).get_sampler(),
                'params': {k: v for k, v in sampler_defs.items() \
                           if k not in ['type', 'def_file']}
            }
    else:
        sampler_dic = None

    # generated sub-datasets all have test split
    splits = ['train', 'val']
    if dataset not in ['iNaturalist18', 'ImageNet']:
        splits.append('test')
    data = {x: dataloader.load_data(data_root=data_root,
                                    dataset=dataset, phase=x,
                                    batch_size=training_opt['batch_size'],
                                    sampler_dic=sampler_dic,
                                    num_workers=training_opt['num_workers'],
                                    top_k_class=training_opt['top_k'] if 'top_k' in training_opt else None,
                                    reverse=args.train_reverse)
            for x in splits}

    training_model = model(config, data, test=False)
    training_model.train()

# ============================================================================
# TESTING
else:
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data",
                            UserWarning)
    print('Under testing phase, we load training data simply to calculate training data number for each class.')

    if 'iNaturalist' in dataset.rstrip('_LT'):
        splits = ['train', 'val']
        test_split = 'val'
    else:
        splits = ['train', 'val', 'test']
        test_split = 'test'


    if 'ImageNet' == training_opt['dataset']:
        splits = ['train', 'val']
        test_split = 'val'
        
    data = {x: dataloader.load_data(data_root=data_root,
                                    dataset=dataset, phase=x,
                                    batch_size=training_opt['batch_size'],
                                    sampler_dic=None,
                                    num_workers=training_opt['num_workers'],
                                    top_k_class=training_opt['top_k'] if 'top_k' in training_opt else None,
                                    shuffle=False,
                                    test_imb_ratio=args.test_imb_ratio,
                                    reverse=args.train_reverse if x == "train" else args.test_reverse)
            for x in splits}

    training_model = model(config, data, test=True,
                           test_imb_ratio=args.test_imb_ratio,
                           test_reverse=args.test_reverse)
    # load checkpoints
    training_model.load_model(args.model_dir)

    training_model.eval(phase=test_split, save_feat=save_mode)

print('='*25, ' ALL COMPLETED ', '='*25)
