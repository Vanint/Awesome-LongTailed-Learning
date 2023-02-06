"""https://github.com/facebookresearch/moco"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class NormedLinear_Classifier(nn.Module):

    def __init__(self, num_classes=1000, feat_dim=2048):
        super(NormedLinear_Classifier, self).__init__()
        self.weight = Parameter(torch.Tensor(feat_dim, num_classes))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x, *args):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


def flatten(t):
    return t.reshape(t.shape[0], -1)

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.2, mlp=False, feat_dim=2048, normalize=False, num_classes=1000):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)
        self.linear = nn.Linear(feat_dim, num_classes)
        self.linear_k = nn.Linear(feat_dim, num_classes)


        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.BatchNorm1d(dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.BatchNorm1d(dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.linear.parameters(), self.linear_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient


        # create the queue
        self.register_buffer("queue", torch.randn(K, dim))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_l", torch.randint(0, num_classes, (K,)))

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # cross_entropy
        self.layer = -2 
        self.feat_after_avg_k = None
        self.feat_after_avg_q = None
        self._register_hook()

        self.normalize = normalize

    
    def _find_layer(self, module):
        if type(self.layer) == str:
            modules = dict([*module.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*module.children()]

            return children[self.layer]
        return None

    def _hook_k(self, _, __, output):
        self.feat_after_avg_k = flatten(output)
        if self.normalize: 
           self.feat_after_avg_k = nn.functional.normalize(self.feat_after_avg_k, dim=1)


    def _hook_q(self, _, __, output):
        self.feat_after_avg_q = flatten(output)
        if self.normalize:
           self.feat_after_avg_q = nn.functional.normalize(self.feat_after_avg_q, dim=1)


    def _register_hook(self):
        layer_k = self._find_layer(self.encoder_k)
        assert layer_k is not None, f'hidden layer ({self.layer}) not found'
        handle = layer_k.register_forward_hook(self._hook_k)

        layer_q = self._find_layer(self.encoder_q)
        assert layer_q is not None, f'hidden layer ({self.layer}) not found'
        handle = layer_q.register_forward_hook(self._hook_q)


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.linear.parameters(), self.linear_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        labels = concat_all_gather(labels)


        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity


        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size,:] = keys
        self.queue_l[ptr:ptr + batch_size] = labels

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x, y):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        y_gather = concat_all_gather(y)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], y_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, y, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        y_gather = concat_all_gather(y)

        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]


        return x_gather[idx_this], y_gather[idx_this]

    def _train(self, im_q, im_k, labels):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, labels, idx_unshuffle = self._batch_shuffle_ddp(im_k, labels)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k, labels = self._batch_unshuffle_ddp(k, labels, idx_unshuffle)

        # compute logits
        features = torch.cat((q, k, self.queue.clone().detach()), dim=0)
        target = torch.cat((labels, labels, self.queue_l.clone().detach()), dim=0)

        self._dequeue_and_enqueue(k, labels)

        # compute logits 
        logits_q = self.linear(self.feat_after_avg_q)

        return features, target, logits_q 

    def _inference(self, image):
        q = self.encoder_q(image)
        q = nn.functional.normalize(q, dim=1)
        encoder_q_logits = self.linear(self.feat_after_avg_q)

        return encoder_q_logits

    def forward(self, im_q, im_k=None, labels=None):
        if self.training:
           return self._train(im_q, im_k, labels) 
        else:
           return self._inference(im_q)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
