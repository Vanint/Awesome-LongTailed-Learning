import torch
import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict

# WARNING: 
# There is no guarantee that it will work or be used on a model. Please do use it with caution unless you make sure everything is working.
use_fp16 = False

if use_fp16:
    from torch.cuda.amp import autocast
else:
    class Autocast(): # This is a dummy autocast class
        def __init__(self):
            pass
        def __enter__(self, *args, **kwargs):
            pass
        def __call__(self, arg=None):
            if arg is None:
                return self
            return arg
        def __exit__(self, *args, **kwargs):
            pass

    autocast = Autocast()

def rename_parallel_state_dict(state_dict):
    count = 0
    for k in list(state_dict.keys()):
        if k.startswith('module.'):
            v = state_dict.pop(k)
            renamed = k[7:]
            state_dict[renamed] = v
            count += 1
    if count > 0:
        print("Detected DataParallel: Renamed {} parameters".format(count))
    return count

def load_state_dict(model, state_dict, no_ignore=False):
    own_state = model.state_dict()
    count = 0
    for name, param in state_dict.items():
        if name not in own_state: # ignore
            print("Warning: {} ignored because it does not exist in state_dict".format(name))
            assert not no_ignore, "Ignoring param that does not exist in model's own state dict is not allowed."
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        try:
            own_state[name].copy_(param)
        except RuntimeError as e:
            print("Error in copying parameter {}, source shape: {}, destination shape: {}".format(name, param.shape, own_state[name].shape))
            raise e
        count += 1
    if count != len(own_state):
        print("Warning: Model has {} parameters, copied {} from state dict".format(len(own_state), count))
    return count

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if isinstance(value, tuple) and len(value) == 2:
            value, n = value
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)
