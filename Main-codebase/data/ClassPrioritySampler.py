"""Copyright (c) Facebook, Inc. and its affiliates.
Custmoized by Yifan Zhang.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


import random
import numpy as np
from torch.utils.data.sampler import Sampler


class RandomCycleIter:
    
    def __init__ (self, data, test_mode=False):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.test_mode = test_mode
        
    def __iter__ (self):
        return self
    
    def __next__ (self):
        self.i += 1
        
        if self.i == self.length:
            self.i = 0
            if not self.test_mode:
                random.shuffle(self.data_list)
            
        return self.data_list[self.i]


class PriorityTree(object):
    def __init__(self, capacity, init_weights, fixed_weights=None, fixed_scale=1.0,
                 alpha=1.0):
        """
        fixed_weights: weights that wont be updated by self.update()
        """
        assert fixed_weights is None or len(fixed_weights) == capacity
        assert len(init_weights) == capacity
        self.alpha = alpha
        self._capacity = capacity
        self._tree_size = 2 * capacity - 1
        self.fixed_scale = fixed_scale
        self.fixed_weights = np.zeros(self._capacity) if fixed_weights is None \
                             else fixed_weights
        self.tree = np.zeros(self._tree_size)
        self._initialized = False
        self.initialize(init_weights)

    def initialize(self, init_weights):
        """Initialize the tree."""

        # Rescale the fixed_weights if it is not zero
        self.fixed_scale_init = self.fixed_scale
        if self.fixed_weights.sum() > 0 and init_weights.sum() > 0:
            self.fixed_scale_init *= init_weights.sum() / self.fixed_weights.sum()
            self.fixed_weights *= self.fixed_scale * init_weights.sum() \
                                / self.fixed_weights.sum()
        print('FixedWeights: {}'.format(self.fixed_weights.sum()))

        self.update_whole(init_weights + self.fixed_weights)
        self._initialized = True
    
    def reset_adaptive_weights(self, adaptive_weights):
        self.update_whole(self.fixed_weights + adaptive_weights)
    
    def reset_fixed_weights(self, fixed_weights, rescale=False):
        """ Reset the manually designed weights and 
            update the whole tree accordingly.

            @rescale: rescale the fixed_weights such that 
            fixed_weights.sum() = self.fixed_scale * adaptive_weights.sum()
        """

        adaptive_weights = self.get_adaptive_weights()
        fixed_sum = fixed_weights.sum()
        if rescale and fixed_sum > 0:
            # Rescale fixedweight based on adaptive weights
            scale = self.fixed_scale * adaptive_weights.sum() / fixed_sum
        else:
            # Rescale fixedweight based on previous fixedweight
            scale = self.fixed_weights.sum() / fixed_sum
        self.fixed_weights = fixed_weights * scale
        self.update_whole(self.fixed_weights + adaptive_weights)
    
    def update_whole(self, total_weights):
        """ Update the whole tree based on per-example sampling weights """
        if self.alpha != 1:
            total_weights = np.power(total_weights, self.alpha)
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
        if self.alpha == 1:
            return self.get_total_weights() - self.fixed_weights
        else:
            return self.get_raw_total_weights() - self.fixed_weights
    
    def get_total_weights(self):
        """ Get the per-example sampling weights
            return shape: [capacity]
        """
        lefti = self.pointer_to_treeidx(0)
        righti = self.pointer_to_treeidx(self.capacity-1)
        return self.tree[lefti:righti+1]

    def get_raw_total_weights(self):
        """ Get the per-example sampling weights
            return shape: [capacity]
        """
        lefti = self.pointer_to_treeidx(0)
        righti = self.pointer_to_treeidx(self.capacity-1)
        return np.power(self.tree[lefti:righti+1], 1/self.alpha)

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
        if self.alpha != 1:
            priority = np.power(priority, self.alpha)
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += delta
    
    def update_delta(self, pointer, delta):
        assert pointer < self.capacity
        tree_idx = self.pointer_to_treeidx(pointer)
        ratio = 1- self.fixed_weights[pointer] / self.tree[tree_idx]
        # delta *= ratio
        if self.alpha != 1:
            # Update delta
            if self.tree[tree_idx] < 0 or \
                np.power(self.tree[tree_idx], 1/self.alpha) + delta < 0:
                import pdb; pdb.set_trace()
            delta = np.power(np.power(self.tree[tree_idx], 1/self.alpha) + delta,
                             self.alpha) \
                  - self.tree[tree_idx]
        self.tree[tree_idx] += delta
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
        wdict = {'fixed_weights': self.fixed_weights, 
                 'total_weights': self.get_total_weights()}
        if self.alpha != 1:
            wdict.update({'raw_total_weights': self.get_raw_total_weights(),
                          'alpha': self.alpha})

        return wdict

class ClassPrioritySampler(Sampler):
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
            - fixed_scale < 0 means, the manually designed distribution will 
              be used as the backend distribution of priorities. 
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
                 pri_mode='train', momentum=0., alpha=1.0):
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
        self.pri_mode = pri_mode
        self.num_samples = len(dataset)
        self.manual_as_backend = False
        self.momentum = momentum
        self.alpha = alpha

        assert 0. <= self.momentum <= 1.0
        assert 0. <= self.alpha

        # Change the backend distribution of priority if needed
        if self.fixed_scale < 0:
            self.fixed_scale = 0
            self.manual_as_backend = True

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
            assert self.nroot is None or self.nroot > 1
        print("====> Decay GAP: {}".format(self.decay_gap))

        # Take care of lambdas
        self.freeze = True
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
        self.data_iter_list = [RandomCycleIter(x) for x in self.cls_idxs]
        for ci in range(self.num_classes):
            self.cls_idxs[ci] = np.array(self.cls_idxs[ci])
        
        # Build balanced weights based on class counts 
        self.balanced_weights = self.get_balanced_weights(self.nroot)
        self.uniform_weights = self.get_uniform_weights()
        self.manual_weights = self.get_manual_weights(self.lams[0])

        # back_weights = self.get_balanced_weights(1.5)
        back_weights = self.uniform_weights

        # Calculate priority ratios that reshape priority into target distribution
        self.per_cls_ratios = self.get_cls_ratios(
            self.manual_weights if self.manual_as_backend else back_weights)
        self.per_example_ratios = self.broadcast(self.per_cls_ratios)

        # Setup priority tree
        if self.ptype == 'score':
            self.init_weight = 1.
        elif self.ptype in ['CE', 'entropy']:
            self.init_weight = 6.9
        else:
            raise NotImplementedError('ptype {} not implemented'.format(self.ptype))
        if self.manual_only:
            self.init_weight = 0.
        self.per_example_uni_weights = np.ones(self.num_samples) * self.init_weight
        self.per_example_velocities = np.zeros(self.num_samples)
        # init_priorities = np.power(self.init_weight, self.alpha) \
        #                 * self.uniform_weights * self.per_cls_ratios
        init_priorities = self.init_weight * self.uniform_weights * self.per_cls_ratios
        self.ptree = PriorityTree(self.num_classes, init_priorities,
                                  self.manual_weights.copy(), fixed_scale=self.fixed_scale,
                                  alpha=self.alpha)
    
    def get_cls_ratios(self, tgt_weights):
        if tgt_weights is self.uniform_weights:
            return np.ones_like(self.uniform_weights)
        per_cls_ratios = tgt_weights / self.uniform_weights
        per_cls_ratios *= self.uniform_weights.sum() / tgt_weights.sum()
        return per_cls_ratios
    
    def get_cls_weights(self):
        ratioed_ws = self.per_example_uni_weights * self.per_example_ratios
        return self.debroadcast_sum(ratioed_ws)

    def broadcast(self, per_cls_info):
        per_exmaple_info = np.zeros(self.num_samples)
        # Braodcast per-cls info to each example 
        for ci in range(self.num_classes):
            per_exmaple_info[self.cls_idxs[ci]] = per_cls_info[ci]
        return per_exmaple_info
    
    def debroadcast_sum(self, per_example_info):
        per_cls_info = np.zeros(self.num_classes)
        # DeBraodcast per-example info to each cls by summation 
        for ci in range(self.num_classes):
            per_cls_info[ci] = per_example_info[self.cls_idxs[ci]].sum()
        return per_cls_info
    
    def get_manual_weights(self, lam):
        # Merge balanced weights and uniform weights 
        if lam == 1:
            manual_weights = self.balanced_weights.copy()
        elif lam == 0:
            manual_weights = self.uniform_weights.copy()
        else:
            manual_weights = self.balanced_weights * lam + (1-lam) * self.uniform_weights
        return manual_weights

    def get_uniform_weights(self):
        return self.cnts.copy()
    
    def get_balanced_weights(self, nroot):
        """ Calculate normalized generalized balanced weights """

        cnts = self.cnts
        if nroot is None:
            # Real balanced sampling weights, each class has the same weights
            # Un-normalized !!!
            cls_ws = np.ones(len(cnts))
        elif nroot >= 1:
            # Generalized balanced weights
            # Un-normalized !!!
            cls_ws = cnts / cnts.sum()
            cls_ws = np.power(cls_ws, 1./nroot) * cnts.sum()
            cls_ws = cls_ws
        else:
            raise NotImplementedError('root:{} not implemented'.format(nroot))

        # Get un-normalized weights
        balanced_weights = cls_ws

        # Normalization and rescale
        balanced_weights *= self.num_samples / balanced_weights.sum() * \
                            self.balance_scale
        return balanced_weights

    def __iter__(self):
        for _ in range(self.num_samples):
            w = random.random() * self.ptree.total
            ci, pri = self.ptree.get_leaf(w)
            yield next(self.data_iter_list[ci])

    def __len__(self):
        return self.num_samples

    def reset_weights(self, epoch):
        # If it is linear shifting 
        if not self.freeze:
            e = np.clip(epoch, 0, self.epochs-1)
            self.manual_weights = self.get_manual_weights(self.lams[e])
            # make sure 'self.fixed_scale > 0' and 'self.manual_as_backend = True' are 
            # mutually exclusive 
            if self.fixed_scale > 0:
                self.ptree.reset_fixed_weights(self.manual_weights, self.rescale)
            if self.manual_as_backend:
                self.update_backend_distribution(self.manual_weights)
        
        # If it is root decay
        if self.root_decay in ['exp', 'linear', 'autoexp'] and epoch % self.decay_gap == 0:
            if self.root_decay == 'exp':
                self.nroot *= 2
            elif self.root_decay == 'linear':
                self.nroot += 1
            elif self.root_decay == 'autoexp':
                # self.nroot *= self.decay_factor
                self.nroot = np.power(self.decay_factor, epoch)

            bw = self.get_balanced_weights(self.nroot)
            if self.manual_as_backend:
                self.update_backend_distribution(bw)
            else:
                self.ptree.reset_fixed_weights(bw)

    def update_backend_distribution(self, tgt_weights):
        # Recalculate the cls ratios based on the given target distribution
        self.per_cls_ratios = self.get_cls_ratios(tgt_weights)
        self.per_example_ratios = self.broadcast(self.per_cls_ratios)

        # Recalculate the new per-class weights based on the new ratios
        # new_backend_weights = self.init_weight * self.uniform_weights * self.per_cls_ratios
        new_cls_weights = self.get_cls_weights()
        self.ptree.reset_adaptive_weights(new_cls_weights)

    def update_weights(self, inds, weights, labels):
        """ Update priority weights """
        if not self.manual_only and self.pri_mode == 'train':
            weights = np.clip(weights, 0, self.init_weight)

            # Iterate over all classes in the batch
            for l in np.unique(labels):
                # Calculate per-class delta weights
                example_inds = inds[labels==l]
                last_weights = self.per_example_uni_weights[example_inds]
                # delta = np.power(weights[labels==l], self.alpha) - \
                #         np.power(last_weights, self.alpha)
                delta = weights[labels==l] - last_weights
                delta = self.momentum * self.per_example_velocities[example_inds] + \
                        (1-self.momentum) * delta
                
                # Update velocities 
                self.per_example_velocities[example_inds] = delta
                # Update per-example weights 
                # self.per_example_uni_weights[example_inds] = weights[labels==l]
                self.per_example_uni_weights[example_inds] += delta

                # Sacle the delta 
                # (ie, the per-example weights both before and after update)
                delta *= self.per_example_ratios[example_inds]

                # Update tree
                if self.alpha == 1:
                    self.ptree.update_delta(l, delta.sum())
                else:
                    self.ptree.update(l, self.per_example_uni_weights[self.cls_idxs[l]].sum())
                    

    def reset_priority(self, weights, labels):
        if self.pri_mode == 'valid':
            assert len(np.unique(labels)) == self.num_classes
            weights = np.clip(weights, 0, self.init_weight)
            cls_weights = np.zeros(self.num_classes)
            for c in np.unique(labels):
                cls_weights[c] = weights[labels==c].mean()
            cls_weights *= self.cnts
            cls_weights *= self.per_cls_ratios
            self.ptree.reset_adaptive_weights(cls_weights)
    
    def get_weights(self):
        return self.ptree.get_weights()


def get_sampler():
    return ClassPrioritySampler
