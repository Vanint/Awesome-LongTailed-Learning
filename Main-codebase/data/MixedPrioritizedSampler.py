"""Copyright (c) Facebook, Inc. and its affiliates.
Custmoized by Yifan Zhang.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


import random
import numpy as np
from torch.utils.data.sampler import Sampler


class PriorityTree(object):
    def __init__(self, capacity, fixed_weights=None, fixed_scale=1.0, 
                 init_weight=1.0):
        """
        fixed_weights: weights that wont be updated by self.update()
        """
        assert fixed_weights is None or len(fixed_weights) == capacity
        self._capacity = capacity
        self._tree_size = 2 * capacity - 1
        self.fixed_scale = fixed_scale
        self.fixed_weights = np.zeros(self._capacity) if fixed_weights is None \
                             else fixed_weights
        self.tree = np.zeros(self._tree_size)
        self._initialized = False
        self.initialize(init_weight)

    def initialize(self, init_weight):
        """Initialize the tree."""

        # Rescale the fixed_weights if it is not zero
        if self.fixed_weights.sum() > 0 and init_weight > 0:
            self.fixed_weights *= self.fixed_scale * init_weight * self.capacity \
                                / self.fixed_weights.sum()
        print('FixedWeights: {}'.format(self.fixed_weights.sum()))

        self.update_whole(init_weight + self.fixed_weights)
        self._initialized = True
    
    def reset_fixed_weights(self, fixed_weights, rescale=False):
        """ Reset the manually designed weights and 
            update the whole tree accordingly.

            @rescale: rescale the fixed_weights such that 
            fixed_weights.sum() = self.fixed_scale * adaptive_weights.sum()
        """

        adaptive_weights = self.get_adaptive_weights()
        fixed_sum = fixed_weights.sum()
        if rescale and fixed_sum > 0:
            scale = self.fixed_scale * adaptive_weights.sum() / fixed_sum
            self.fixed_weights = fixed_weights * scale
        else:
            self.fixed_weights = fixed_weights
        self.update_whole(self.fixed_weights + adaptive_weights)
    
    def update_whole(self, total_weights):
        """ Update the whole tree based on per-example sampling weights """
        lefti = self.pointer_to_treeidx(0)
        righti = self.pointer_to_treeidx(self.capacity-1)
        self.tree[lefti:righti+1] = total_weights

        # Iteratively find a parent layer
        while lefti != 0 and righti != 0:
            lefti = (lefti - 1) // 2 if lefti != 0 else 0
            righti = (righti - 1) // 2 if righti != 0 else 0
            
            # Assign paraent weights from right to left
            for i in range(righti, lefti-1, -1):
                self.tree[i] = self.tree[2*i+1] + self.tree[2*i+2]
    
    def get_adaptive_weights(self):
        """ Get the instance-aware weights, that are not mannually designed"""
        return self.get_total_weights() - self.fixed_weights
    
    def get_total_weights(self):
        """ Get the per-example sampling weights
            return shape: [capacity]
        """
        lefti = self.pointer_to_treeidx(0)
        righti = self.pointer_to_treeidx(self.capacity-1)
        return self.tree[lefti:righti+1]

    @property
    def size(self):
        return self._tree_size

    @property
    def capacity(self):
        return self._capacity

    def __len__(self):
        return self.capacity

    def pointer_to_treeidx(self, pointer):
        assert pointer < self.capacity
        return int(pointer + self.capacity - 1)

    def update(self, pointer, priority):
        assert pointer < self.capacity
        tree_idx = self.pointer_to_treeidx(pointer)
        priority += self.fixed_weights[pointer]
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += delta

    def get_leaf(self, value):
        assert self._initialized, 'PriorityTree not initialized!!!!'
        assert self.total > 0, 'No priority weights setted!!'
        parent = 0
        while True:
            left_child = 2 * parent + 1
            right_child = 2 * parent + 2
            if left_child >= len(self.tree):
                tgt_leaf = parent
                break
            if value < self.tree[left_child]:
                parent = left_child
            else:
                value -= self.tree[left_child]
                parent = right_child
        data_idx = tgt_leaf - self.capacity + 1
        return data_idx, self.tree[tgt_leaf]        # data idx, priority

    @property
    def total(self):
        assert self._initialized, 'PriorityTree not initialized!!!!'
        return self.tree[0]

    @property
    def max(self):
        return np.max(self.tree[-self.capacity:])

    @property
    def min(self):
        assert self._initialized, 'PriorityTree not initialized!!!!'
        return np.min(self.tree[-self.capacity:])
    
    def get_weights(self):
        return {'fixed_weights': self.fixed_weights, 
                'total_weights': self.get_total_weights()}


class MixedPrioritizedSampler(Sampler):
    """
    A sampler combining manually designed sampling strategy and prioritized 
    sampling strategy.

    Manually disigned strategy contains two parts:

        $$ manual_weights = lam * balanced_weights + (1-lam) uniform_weights
    
        Here we use a generalized version of balanced weights as follows,
        when n limits to infinity, balanced_weights = real_balanced_weights
    
        $$ balanced_weights = uniform_weights ^  (1/n)
    
        Then the balanced weights are scaled such that 
    
        $$ balanced_weights.sum() =  balance_scale * uniform_weights.sum()

        Note: above weights are per-class weights

    Overall sampling weights are given as 
        $$ sampling_weights = manual_weights * fixed_scale + priority_weights

    Arguments:
        @dataset: A dataset 
        @balance_scale: The scale of balanced_weights
        @lam: A weight to combine balanced weights and uniform weights
            - None for shifting sampling
            - 0 for uniform sampling
            - 1 for balanced sampling
        @fixed_scale: The scale of manually designed weights
        @cycle: shifting strategy
            - 0 for linear shifting: 3 -> 2 - > 1
            - 1 for periodic shifting: 
                3 -> 2 - > 1 -> 3 -> 2 - > 1 -> 3 -> 2 - > 1
            - 2 for cosine-like periodic shifting:
                3 -> 2 - > 1 -> 1 -> 2 - > 3 -> 3 -> 2 - > 1
        @nroot:
            - None for truly balanced weights
            - >= 2 for pseudo-balanced weights 
        @rescale: whether to rebalance the manual weights and priority weights
            every epoch
        @root_decay:
            - 'exp': for exponential decay 
            - 'linear': for linear decay 
    """
    def __init__(self, dataset, balance_scale=1.0, fixed_scale=1.0,
                 lam=None, epochs=90, cycle=0, nroot=None, manual_only=False,
                 rescale=False, root_decay=None, decay_gap=30, ptype='score',
                 alpha=1.0):
        """
        """
        self.dataset = dataset
        self.balance_scale = balance_scale
        self.fixed_scale = fixed_scale
        self.epochs = epochs
        self.lam = lam
        self.cycle = cycle
        self.nroot = nroot
        self.rescale = rescale
        self.manual_only = manual_only
        self.root_decay = root_decay
        self.decay_gap = decay_gap
        self.ptype = ptype
        self.num_samples = len(dataset)
        self.alpha = alpha

        # If using root_decay, reset relevent parameters
        if self.root_decay in ['exp', 'linear', 'autoexp']:
            self.lam = 1
            self.manual_only = True
            self.nroot = 1
            if self.root_decay == 'autoexp':
                self.decay_gap = 1
                self.decay_factor = np.power(nroot, 1/(self.epochs-1))
        else:
            assert self.root_decay is None
            assert self.nroot is None or self.nroot >= 2
        print("====> Decay GAP: {}".format(self.decay_gap))

        # Take care of lambdas
        if self.lam is None:
            self.freeze = False
            if cycle == 0:
                self.lams = np.linspace(0, 1, epochs)
            elif cycle == 1:
                self.lams = np.concatenate([np.linspace(0,1,epochs//3)] * 3)
            elif cycle == 2:
                self.lams = np.concatenate([np.linspace(0,1,epochs//3),
                                            np.linspace(0,1,epochs//3)[::-1],
                                            np.linspace(0,1,epochs//3)])
            else:
                raise NotImplementedError(
                    'cycle = {} not implemented'.format(cycle))
        else:
            self.lams = [self.lam]
            self.freeze = True

        # Get num of samples per class
        self.cls_cnts = []
        self.labels = labels = np.array(self.dataset.labels)
        for l in np.unique(labels):
            self.cls_cnts.append(np.sum(labels==l))
        self.num_classes = len(self.cls_cnts)
        self.cnts = np.array(self.cls_cnts).astype(float)
        
        # Get per-class image indexes
        self.cls_idxs = [[] for _ in range(self.num_classes)]
        for i, label in enumerate(self.dataset.labels):
            self.cls_idxs[label].append(i)
        for ci in range(self.num_classes):
            self.cls_idxs[ci] = np.array(self.cls_idxs[ci])
        
        # Build balanced weights based on class counts 
        self.balanced_weights = self.get_balanced_weights(self.nroot)
        self.manual_weights = self.get_manual_weights(self.lams[0])

        # Setup priority tree
        if self.ptype == 'score':
            self.init_weight = 1.
        elif self.ptype in ['CE', 'entropy']:
            self.init_weight = 6.9
        else:
            raise NotImplementedError('ptype {} not implemented'.format(self.ptype))
        if self.manual_only:
            self.init_weight = 0.
        self.init_weight = np.power(self.init_weight, self.alpha)
        self.ptree = PriorityTree(self.num_samples, self.manual_weights,
                                  fixed_scale=self.fixed_scale,
                                  init_weight=self.init_weight)
    
    def get_manual_weights(self, lam):
        # Merge balanced weights and uniform weights 
        if lam == 1:
            manual_weights = self.balanced_weights
        elif lam == 0:
            manual_weights = np.ones(len(self.balanced_weights))
        else:
            manual_weights = self.balanced_weights * lam + (1-lam)
        return manual_weights        
    
    def get_balanced_weights(self, nroot):
        """ Calculate normalized generalized balanced weights """

        cnts = self.cnts
        if nroot is None:
            # Real balanced sampling weights
            cls_ws = cnts.min() / cnts
        elif nroot >= 1:
            # Generalized balanced weights
            cls_ws = cnts / cnts.sum()
            cls_ws = np.power(cls_ws, 1./nroot) * cnts.sum()
            cls_ws = cls_ws / cnts
        else:
            raise NotImplementedError('root:{} not implemented'.format(nroot))

        # Get un-normalized weights
        balanced_weights = np.zeros(self.num_samples)    
        for ci in range(self.num_classes):
            balanced_weights[self.cls_idxs[ci]] = cls_ws[ci]

        # Normalization and rescale
        balanced_weights *= self.num_samples / balanced_weights.sum() * \
                            self.balance_scale
        return balanced_weights

    def __iter__(self):
        for _ in range(self.num_samples):
            w = random.random() * self.ptree.total
            i, pri = self.ptree.get_leaf(w)
            yield i

    def __len__(self):
        return self.num_samples

    def reset_weights(self, epoch):
        if not self.freeze and self.fixed_scale > 0:
            if epoch >= self.epochs:
                e = self.epochs - 1
            elif epoch < 1:
                e = 0
            else:
                e = epoch
            self.manual_weights = self.get_manual_weights(self.lams[e])
            self.ptree.reset_fixed_weights(self.manual_weights, self.rescale)
        
        if self.root_decay in ['exp', 'linear', 'autoexp'] and epoch % self.decay_gap == 0:
            if self.root_decay == 'exp':
                self.nroot *= 2
            elif self.root_decay == 'linear':
                self.nroot += 1
            elif self.root_decay == 'autoexp':
                # self.nroot *= self.decay_factor
                self.nroot = np.power(self.decay_factor, epoch)

            bw = self.get_balanced_weights(self.nroot)
            self.ptree.reset_fixed_weights(bw)

    def update_weights(self, inds, weights):
        """ Update priority weights """
        if not self.manual_only:
            weights = np.clip(weights, 0, self.init_weight)
            weights = np.power(weights, self.alpha)
            for i, w in zip(inds, weights):
                self.ptree.update(i, w)
    
    def get_weights(self):
        return self.ptree.get_weights()


def get_sampler():
    return MixedPrioritizedSampler
