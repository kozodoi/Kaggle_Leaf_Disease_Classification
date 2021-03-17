####### UTILITIES

import os
import numpy as np
import random
import torch

# random sequences
def randomly(seq):
    shuffled = list(seq)
    random.shuffle(shuffled)
    return iter(shuffled)

# voting ensemble
def convert_to_10(a): 
    idx = a.argmax(axis = 1)
    out = np.zeros_like(a,dtype = float)
    out[np.arange(a.shape[0]), idx] = 1
    return out

# device-aware printing
def smart_print(expression, CFG):
    if CFG['device'] != 'TPU':
        print(expression)
    else:
        xm.master_print(expression)

# device-aware model save
def smart_save(weights, path, CFG):
    if CFG['device'] != 'TPU':
        torch.save(weights, path)    
    else:
        xm.save(weights, path) 

# randomness
def seed_everything(seed, CFG):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    smart_print('- setting random seed to {}...'.format(seed), CFG)