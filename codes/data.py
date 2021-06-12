from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch

import cv2
import numpy as np
import pandas as pd
import numpy as np

import os
import time

from utilities import *
from augmentations import *


####### DATASET

class LeafData(Dataset):
    
    # initialization
    def __init__(self, 
                 data, 
                 directory, 
                 transform = None, 
                 labeled   = False):
        self.data      = data
        self.directory = directory
        self.transform = transform
        self.labeled   = labeled
        
    # length
    def __len__(self):
        return len(self.data)
    
    # get item  
    def __getitem__(self, idx):
                
        # import
        path  = os.path.join(self.directory, self.data.iloc[idx]['image_id'])
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # augmentations
        if self.transform is not None:
            image = self.transform(image = image)['image']
        
        # output
        if self.labeled:
            labels = torch.tensor(self.data.iloc[idx]['label']).long()
            return image, labels
        else:
            return image



####### DATA PREP

def get_data(df, fold, CFG,
             df_2019    = None,
             df_no      = None,
             df_pl      = None,
             df_ext     = None,
             epoch      = None,
             list_dupl  = [],
             list_noise = []):

    ##### EPOCH-BASED PARAMS

    # image size
    if (CFG['step_size']) and (epoch is not None):
        image_size = CFG['step_size'][epoch]
    else:
        image_size = CFG['image_size']

    # augmentation probability
    if (CFG['step_p_aug']) and (epoch is not None):
        p_augment = CFG['step_p_aug'][epoch]
    else:
        p_augment = CFG['p_augment']


    ##### PARTITIONING

    # load splits
    df_train = df.loc[df.fold != fold].reset_index(drop = True)
    df_valid = df.loc[df.fold == fold].reset_index(drop = True)     
    if epoch is None:  
        smart_print('-- no. images: train - {}, valid - {}'.format(len(df_train), len(df_valid)), CFG)
        
    # flip labels
    if epoch is not None:
        if CFG['flip_prob']:
            list_noise = [img for img in df_no['image_id'].values if img in df_train['image_id'].values]
            flip_idx    = (np.random.binomial(1, CFG['flip_prob'], len(list_noise)) == 1)
            for img_idx, img in enumerate(list_noise):
                if flip_idx[img_idx]:
                    df_train.loc[df_train.image_id == img, 'label'] = df_no.loc[df_no.image_id == img, 'pred'].astype('int').values
            smart_print('- flipping {} labels...'.format(np.sum(flip_idx)), CFG)

    # 2019 labeled data
    if CFG['data_2019']:
        df_train = pd.concat([df_train, df_2019], axis = 0).reset_index(drop = True)
        if epoch is None:  
            smart_print('- appending 2019 labeled data to train...', CFG)
            smart_print('-- no. images: train - {}, valid - {}'.format(len(df_train), len(df_valid)), CFG)

    # 2019 psueudo-labeled data
    if CFG['data_pl']:
        df_train = pd.concat([df_train, df_pl], axis = 0).reset_index(drop = True)
        if epoch is None:  
            smart_print('- appending 2019 pseudo-labeled data to train...', CFG)
            smart_print('-- no. images: train - {}, valid - {}'.format(len(df_train), len(df_valid)), CFG)

    # external data
    if CFG['data_ext']:
        df_train = pd.concat([df_train, df_ext], axis = 0).reset_index(drop = True)
        if epoch is None:  
            smart_print('- appending external data to train...', CFG)
            smart_print('-- no. images: train - {}, valid - {}'.format(len(df_train), len(df_valid)), CFG)


    ##### SUBSETTING

    # removing bad examples
    if CFG['drop_dupl']:
        df_train = df_train.loc[~df_train.image_id.isin(list_dupl)].reset_index(drop = True)
    if CFG['drop_outs']:
        df_train = df_train.loc[~df_train.image_id.isin(list_outs)].reset_index(drop = True)
    if CFG['drop_noise']:
        list_noise = list(df_no['image_id'].values)
        df_train   = df_train.loc[~df_train.image_id.isin(list_noise)].reset_index(drop = True)
    if CFG['flip_noise']:
        list_noise = [img for img in df_no['image_id'].values if img in df_train['image_id'].values]
        for img in list_noise:
            df_train.loc[df_train.image_id == img, 'label'] = df_no.loc[df_no.image_id == img, 'pred'].astype('int').values
    if epoch is None:  
        smart_print('- dealing with bad images from train...', CFG)
        smart_print('-- no. images: train - {}, valid - {}'.format(len(df_train), len(df_valid)), CFG)

    # subset for debug mode
    if CFG['debug']:
        df_train = df_train.sample(CFG['batch_size'] * 5, random_state = CFG['seed']).reset_index(drop = True)
        df_valid = df_valid.sample(CFG['batch_size'] * 5, random_state = CFG['seed']).reset_index(drop = True)


    ##### DATASETS
        
    # augmentations
    train_augs, test_augs = get_augs(CFG, image_size, p_augment)

    # datasets
    train_dataset = LeafData(data      = df_train, 
                             directory = CFG['data_path'] + 'train_images/',
                             transform = train_augs,
                             labeled   = True)
    valid_dataset = LeafData(data      = df_valid, 
                             directory = CFG['data_path'] + 'train_images/',
                             transform = test_augs,
                             labeled   = True)
    
    
    ##### DATA SAMPLERS
    
    ### GPU SAMPLERS
    if CFG['device'] != 'TPU':
    
        # with oversampling 
        if CFG['oversample']:
            weights        = 1. / torch.tensor(df_train['label'].value_counts(sort = False).values, dtype = torch.float)
            sample_weights = weights[df_train['label']]
            train_sampler  = WeightedRandomSampler(weights     = sample_weights,
                                                   num_samples = len(sample_weights),
                                                   replacement = True)
            valid_sampler = SequentialSampler(valid_dataset)

        # ordinary samplers
        else:
            train_sampler = RandomSampler(train_dataset)
            valid_sampler = SequentialSampler(valid_dataset)
        
    ### TPU SAMPLERS  
    if CFG['device'] == 'TPU':
        
        # distributed samplers
        train_sampler = DistributedSampler(train_dataset,
                                           num_replicas = xm.xrt_world_size(),
                                           rank         = xm.get_ordinal(),
                                           shuffle        = True)
        valid_sampler = DistributedSampler(valid_dataset,
                                           num_replicas = xm.xrt_world_size(),
                                           rank         = xm.get_ordinal(),
                                           shuffle        = False)
        
    ##### DATA LOADERS
       
    # data loaders
    train_loader = DataLoader(dataset     = train_dataset, 
                              batch_size  = CFG['batch_size'], 
                              sampler     = train_sampler,
                              num_workers = CFG['num_workers'],
                              pin_memory  = True)
    valid_loader = DataLoader(dataset     = valid_dataset, 
                              batch_size  = CFG['batch_size'], 
                              sampler     = valid_sampler, 
                              num_workers = CFG['num_workers'],
                              pin_memory  = True)
    
    # feedback
    smart_print('- image size: {}x{}, p(augment): {}'.format(image_size, image_size, p_augment), CFG)
    if epoch is None:
        smart_print('-' * 55, CFG)

    # output
    return train_loader, valid_loader, df_train, df_valid