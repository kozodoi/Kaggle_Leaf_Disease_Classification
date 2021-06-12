import torch
import numpy as np
import timm
from timm.utils import *
from utilities import *
from augmentations import *
import gc
from tqdm import tqdm


####### INFERENCE

def valid_epoch(loader, model, criterion, CFG, device):

    # switch regime
    model.eval()

    # running loss
    val_loss = AverageMeter()

    # preds placeholders
    PROBS = []
       
    # loop through batches
    with torch.no_grad():
        for batch_idx, (inputs, labels) in (tqdm(enumerate(loader), total = len(loader)) if CFG['device'] != 'TPU' \
                                            else enumerate(loader)):

            # extract inputs and labels
            inputs = inputs.to(device)
            labels = labels.to(device)

            # preds placeholders
            logits = torch.zeros((inputs.shape[0], CFG['num_classes']), device = device)
            probs  = torch.zeros((inputs.shape[0], CFG['num_classes']), device = device)

            # compute predictions
            for tta_idx in range(CFG['num_tta']): 
                preds   = model(get_tta_flips(inputs, tta_idx))
                logits += preds / CFG['num_tta']
                probs  += preds.softmax(axis = 1) / CFG['num_tta']

            # compute loss
            loss = criterion(logits, labels)
            val_loss.update(loss.item(), inputs.size(0))

            # store predictions
            PROBS.append(probs.detach().cpu())

            # clear memory
            del inputs, labels, preds, probs, logits, loss
            gc.collect()

    # transform predictions
    PROBS = torch.cat(PROBS).numpy()

    # output
    return val_loss.sum, PROBS