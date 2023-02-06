"""Copyright (c) Facebook, Inc. and its affiliates.
Custmoized by Yifan Zhang.
All rights reserved. 
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score
import torch.nn.functional as F
import importlib
import pdb
import math
from bisect import bisect_right
from collections import Counter

def update(config, args, output_dir):
    # Change parameters
    config['model_dir'] = get_value(config['model_dir'], args.model_dir)
    config['training_opt']['batch_size'] = \
        get_value(config['training_opt']['batch_size'], args.batch_size)
    config['networks']['classifier']['optim_params']['lr'] = \
        get_value(config['networks']['classifier']['optim_params']['lr'], args.lr)

    # if using pretrained model, don't change lr
    if "caffe" in config["networks"]["feat_model"]["params"] and \
            config["networks"]["feat_model"]["params"]["caffe"]:
        pass
    else:
        config['networks']['feat_model']['optim_params']['lr'] = \
            get_value(config['networks']['feat_model']['optim_params']['lr'], args.lr)

    if 'cifar_imb_ratio' in config['training_opt']:
        config['training_opt']['cifar_imb_ratio'] = \
            get_value(config['training_opt']['cifar_imb_ratio'], args.cifar_imb_ratio)
    config['training_opt']['log_dir'] = output_dir

    if "RouteWeightLoss" in config["criterions"].keys():
        if "prior" in config["criterions"]["PerformanceLoss"]["loss_params"]:
            config['criterions']['PerformanceLoss']['loss_params']['prior'] = \
                get_value(config['criterions']['PerformanceLoss']['loss_params']['prior'], args.cifar_imb_ratio)
        if "prior" in config["criterions"]["RouteWeightLoss"]["loss_params"]:
            config['criterions']['RouteWeightLoss']['loss_params']['prior'] = \
                get_value(config['criterions']['RouteWeightLoss']['loss_params']['prior'], args.cifar_imb_ratio)
        if "exist_only" in config["criterions"]["RouteWeightLoss"]["loss_params"]:
            config['criterions']['RouteWeightLoss']['loss_params']['exist_only'] = \
                get_value(config['criterions']['RouteWeightLoss']['loss_params']['exist_only'], bool(args.exist_only))

        config['criterions']['RouteWeightLoss']['weight'] = \
            get_value(config['criterions']['RouteWeightLoss']['weight'], args.alpha)
        config['criterions']['RouteWeightLoss']['loss_params']['remine_lambda'] = \
            get_value(config['criterions']['RouteWeightLoss']['loss_params']['remine_lambda'],
                      args.remine_lambda)

    if args.no_use_dv:
        config["criterions"].pop("RouteWeightLoss", None)

    return config


def source_import(file_path):
    """This function imports python module directly from source code using importlib"""
    spec = importlib.util.spec_from_file_location('', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def batch_show(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(20,20))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)

def print_write(print_str, log_file):
    print(*print_str)
    if log_file is None:
        return
    with open(log_file, 'a') as f:
        print(*print_str, file=f)

def init_weights(model, weights_path, caffe=False, classifier=False):
    """Initialize weights"""
    print('Pretrained %s weights path: %s' % ('classifier' if classifier else 'feature model',
                                              weights_path))
    weights = torch.load(weights_path)
    if not classifier:
        if caffe:
            weights = {k: weights[k] if k in weights else model.state_dict()[k]
                       for k in model.state_dict()}
        else:
            weights = weights['state_dict_best']['feat_model']
            weights = {k: weights['module.' + k] if 'module.' + k in weights else model.state_dict()[k]
                       for k in model.state_dict()}
    else:
        weights = weights['state_dict_best']['classifier']
        weights = {k: weights['module.fc.' + k] if 'module.fc.' + k in weights else model.state_dict()[k]
                   for k in model.state_dict()}
    model.load_state_dict(weights)
    return model

def shot_acc(preds, labels, train_data, many_shot_thr=100, low_shot_thr=20, acc_per_cls=False):

    if isinstance(train_data, np.ndarray):
        training_labels = np.array(train_data).astype(int)
    else:
        training_labels = np.array(train_data.dataset.labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    b = np.load("./data/shot_list.npy")
    many_shot_bool = b[0]
    medium_shot_bool = b[1] 
    few_shot_bool = b[2]

    all_list = []
    for i in range(len(train_class_count)):
        all_list.append((class_correct[i] / test_class_count[i]))
    all_list = np.array(all_list)
    many_shot = all_list[many_shot_bool]    
    median_shot = all_list[medium_shot_bool]
    low_shot = all_list[few_shot_bool]
    #print(len(many_shot))
    
    """
    many_shot = []
    median_shot = []
    low_shot = []
    
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))    
    """

    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)

    if acc_per_cls:
        class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)]
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), class_accs
    else:
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)

def weighted_shot_acc (preds, labels, ws, train_data, many_shot_thr=100, low_shot_thr=20):

    training_labels = np.array(train_data.dataset.labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(ws[labels==l].sum())
        class_correct.append(((preds[labels==l] == labels[labels==l]) * ws[labels==l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))
    return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)

def F_measure(preds, labels, theta=None):
    # Regular f1 score
    return f1_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro')

def mic_acc_cal(preds, labels):
    if isinstance(labels, tuple):
        assert len(labels) == 3
        targets_a, targets_b, lam = labels
        acc_mic_top1 = (lam * preds.eq(targets_a.data).cpu().sum().float() \
                       + (1 - lam) * preds.eq(targets_b.data).cpu().sum().float()) / len(preds)
    else:
        acc_mic_top1 = (preds == labels).sum().item() / len(labels)
    return acc_mic_top1


def weighted_mic_acc_cal(preds, labels, ws):
    acc_mic_top1 = ws[preds == labels].sum() / ws.sum()
    return acc_mic_top1

def class_count (data):
    labels = np.array(data.dataset.labels)
    class_data_num = []
    for l in np.unique(labels):
        class_data_num.append(len(labels[labels == l]))
    return class_data_num


# New Added
def torch2numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, (list, tuple)):
        return tuple([torch2numpy(xi) for xi in x])
    else:
        return x

def logits2score(logits, labels):
    scores = F.softmax(logits, dim=1)
    score = scores.gather(1, labels.view(-1, 1))
    score = score.squeeze().cpu().numpy()
    return score


def logits2entropy(logits):
    scores = F.softmax(logits, dim=1)
    scores = scores.cpu().numpy() + 1e-30
    ent = -scores * np.log(scores)
    ent = np.sum(ent, 1)
    return ent


def logits2CE(logits, labels):
    scores = F.softmax(logits, dim=1)
    score = scores.gather(1, labels.view(-1, 1))
    score = score.squeeze().cpu().numpy() + 1e-30
    ce = -np.log(score)
    return ce


def get_priority(ptype, logits, labels):
    if ptype == 'score':
        ws = 1 - logits2score(logits, labels)
    elif ptype == 'entropy':
        ws = logits2entropy(logits)
    elif ptype == 'CE':
        ws = logits2CE(logits, labels)

    return ws

def get_value(oldv, newv):
    if newv is not None:
        return newv
    else:
        return oldv


# Tang Kaihua New Add
def print_grad_norm(named_parameters, logger_func, log_file, verbose=False):
    if not verbose:
        return None

    total_norm = 0.0
    param_to_norm = {}
    param_to_shape = {}
    for n, p in named_parameters.items():
        if p.grad is not None:
            param_norm = p.grad.norm(2)
            total_norm += param_norm ** 2
            param_to_norm[n] = param_norm
            param_to_shape[n] = p.size()

    total_norm = total_norm ** (1. / 2)

    logger_func(['----------Total norm {:.5f}-----------------'.format(total_norm)], log_file)
    for name, norm in sorted(param_to_norm.items(), key=lambda x: -x[1]):
        logger_func(["{:<50s}: {:.5f}, ({})".format(name, norm, param_to_shape[name])], log_file)
    logger_func(['-------------------------------'], log_file)

    return total_norm

def smooth_l1_loss(input, target, beta=1. / 9, reduction='mean'):
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        print('XXXXXX Error Reduction Type for smooth_l1_loss, use default mean')
        return loss.mean()


def l2_loss(input, target, reduction='mean'):
    return F.mse_loss(input, target, reduction=reduction)


def regression_loss(input, target, l2=False, pre_mean=True, l1=False, moving_average=False, moving_ratio=0.01):
    assert (l2 + l1 + moving_average) == 1
    if l2:
        if (input.shape[0] == target.shape[0]):
            assert not pre_mean
            loss = l2_loss(input, target.clone().detach())
        else:
            assert pre_mean
            loss = l2_loss(input, target.clone().detach().mean(0, keepdim=True))
    elif l1:
        loss = smooth_l1_loss(input, target.clone().detach())
    elif moving_average:
        # input should be register_buffer rather than nn.Parameter
        with torch.no_grad():
            input = (1 - moving_ratio) * input + moving_ratio * target.clone().detach().mean(0, keepdim=True)
        loss = None
    return loss

def gumbel_softmax(logits, tau=1, hard=False, gumbel=True, dim=-1):
    if gumbel:
        gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / tau                        # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)
    else:
        y_soft = logits.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


def gumbel_sigmoid(logits, tau=1, hard=False, gumbel=True):
    if gumbel:
        gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / tau                        # ~Gumbel(logits,tau)
        y_soft = torch.sigmoid(gumbels)
    else:
        y_soft = torch.sigmoid(logits)

    if hard:
        # Straight through.
        y_hard = (y_soft > 0.5).float()
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_epochs=5,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_epochs:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_epochs
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


def calculate_prior(num_classes, img_max=None, prior=None, prior_txt=None, reverse=False, return_num=False):
    if prior_txt:
        labels = []
        with open(prior_txt) as f:
            for line in f:
                labels.append(int(line.split()[1]))
        occur_dict = dict(Counter(labels))
        img_num_per_cls = [occur_dict[i] for i in range(num_classes)]
    else:
        img_num_per_cls = []
        for cls_idx in range(num_classes):
            if reverse:
                num = img_max * (prior ** ((num_classes - 1 - cls_idx) / (num_classes - 1.0)))
            else:
                num = img_max * (prior ** (cls_idx / (num_classes - 1.0)))
            img_num_per_cls.append(int(num))
    img_num_per_cls = torch.Tensor(img_num_per_cls)

    if return_num:
        return img_num_per_cls
    else:
        return img_num_per_cls / img_num_per_cls.sum()
