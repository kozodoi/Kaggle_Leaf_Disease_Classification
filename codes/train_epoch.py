import torch
import numpy as np
import timm
from timm.utils import *
from utilities import *
from augmentations import *
import gc
from tqdm import tqdm


####### TRAINING

def train_epoch(loader, model, optimizer, scheduler, criterion, epoch, CFG, device):
       
    # switch regime
    model.train()

    # running loss
    trn_loss = AverageMeter()

    # update scheduler on epoch
    if not CFG['update_on_batch']:
        scheduler.step() 
        if epoch == CFG['warmup']:
            scheduler.step() 

    # loop through batches
    for batch_idx, (inputs, labels) in (tqdm(enumerate(loader), total = len(loader)) if CFG['device'] != 'TPU' \
                                        else enumerate(loader)):

        # extract inputs and labels
        inputs = inputs.to(device)
        labels = labels.to(device)

        # apply cutmix augmentation
        if CFG['cutmix'][0] > 0:
            mix_decision = np.random.rand(1)
            if mix_decision < CFG['cutmix'][0]:
                inputs, labels = cutmix_fn(data   = inputs, 
                                           target = labels, 
                                           alpha  = CFG['cutmix'][1])
        else:
            mix_decision = 0

        # update scheduler on batch
        if CFG['update_on_batch']:
            scheduler.step(epoch + 1 + batch_idx / len(loader))

        # passes and weight updates
        with torch.set_grad_enabled(True):
            
            # forward pass 
            with amp_autocast():
                preds = model(inputs)
                if (CFG['cutmix'][0] > 0) and (mix_decision < CFG['cutmix'][0]):
                    loss = criterion(preds, labels[0]) * labels[2] + criterion(preds, labels[1]) * (1. - labels[2])
                else:
                    loss = criterion(preds, labels)
                    
            # backward pass
            if CFG['use_amp']:
                scaler.scale(loss).backward()   
            else:
                loss.backward() 

            # update weights
            if ((batch_idx + 1) % CFG['accum_iter'] == 0) or ((batch_idx + 1) == len(loader)):
                if CFG['device'] == 'TPU':
                    xm.optimizer_step(optimizer, barrier = True)
                else:
                    if CFG['use_amp']:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                optimizer.zero_grad()

        # update loss
        trn_loss.update(loss.item(), inputs.size(0))

        # clear memory
        del inputs, labels, preds, loss
        gc.collect()

    # output
    return trn_loss.sum