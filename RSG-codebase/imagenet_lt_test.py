import argparse
import os
import random
import time
import warnings
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
from collections import OrderedDict
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from imagenet_lt_data import *
from utils import *


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Places Testing')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnext50_32x4d',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnext50_32x4d)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--best_checkpoint', type=str, default='checkpoint_rsg/ImageNet_LT_resnext50_32x4d/ckpt.best.pth.tar')
parser.add_argument('--image_dir', type=str, default='../data/ImageNet')

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    model = models.__dict__[args.arch](num_classes=1000, phase_train=False)
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model

def shot_acc(train_class_count, test_class_count, class_correct, many_shot_thr=100, low_shot_thr=20):
    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
       
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i].detach().cpu().numpy() / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i].detach().cpu().numpy() / test_class_count[i]))
        else:
            median_shot.append((class_correct[i].detach().cpu().numpy() / test_class_count[i]))          

    return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)

def mic_acc_cal(preds, labels):
    acc_mic_top1 = (preds == labels).sum().item()
    return acc_mic_top1

if __name__ == "__main__":

    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    model = load_checkpoint(args.best_checkpoint)
    model = model.cuda(args.gpu)

    transform_test = transforms.Compose([
      transforms.Scale(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_train = transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]) 
    
    train_dataset = ImageNet_LT(args.image_dir, transform_train, 'train')
    test_dataset = ImageNet_LT(args.image_dir, transform_test, 'test')

    cls_num_list_train = train_dataset.get_cls_num_list()
    cls_num_list_test = test_dataset.get_cls_num_list()

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    

    overall_acc = 0.0
    many_shot_overall = 0.0
    median_shot_overall = 0.0
    low_shot_overall = 0.0
    total_num = 0
    correct_class = [0]*1000

    for i, (input, label) in enumerate(test_loader):
         output = model(input.cuda(), phase_train=False)
         predict_ = torch.topk(output, 1, dim=1, largest=True, sorted=True, out=None)[1]
         predict = predict_.cpu().detach().squeeze()
         acc = mic_acc_cal(predict, label.cpu())

         for l in range(0, 1000):
            correct_class[l] += (predict[label == l] == label[label==l]).sum()

         overall_acc += acc
         total_num += len(label.cpu())


    overall_acc = overall_acc * 1.0 / total_num
    many_shot_overall, median_shot_overall, low_shot_overall = shot_acc(cls_num_list_train, cls_num_list_test, correct_class)

    print("The overall accuracy: %.2f. The many shot accuracy: %.2f. The median shot accuracy: %.2f. The low shot accuracy: %.2f." % (overall_acc * 100, many_shot_overall* 100, median_shot_overall * 100, low_shot_overall * 100 ))
