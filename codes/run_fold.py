from utilities import *
from data import *
from model import *
from losses import *
from train_epoch import *
from valid_epoch import *

import os
import time



####### WRAPPER FUNCTION

def run_fold(fold, trn_loader, val_loader, 
             df, df_2019, df_pl, df_ext, df_no, list_dupl, list_noise,
             df_trn, df_val, 
             CFG, device):

    ##### PREPARATIONS
    
    # reset seed
    seed_everything(CFG['seed'] + fold, CFG)

    # get model
    model, optimizer, scheduler = get_model(CFG, device)
        
    # placeholders
    trn_losses  = []
    val_losses  = []
    val_metrics = []
    lrs         = []

    
    ##### TRAINING AND INFERENCE

    for epoch in range(CFG['num_epochs'] + CFG['fine_tune']):
                
        ### PREPARATIONS

        # timer
        epoch_start = time.time()

        # update data loaders if needed
        if (CFG['step_size']) or (CFG['step_p_aug']) or (CFG['flip_prob']):
            trn_loader, val_loader, df_train, df_valid = get_data(df, fold, CFG,
                                                                  df_2019    = df_2019 if CFG['data_2019'] else None,
                                                                  df_pl      = df_pl if CFG['data_pl'] else None,
                                                                  df_ext     = df_ext if CFG['data_ext'] else None,
                                                                  df_no      = df_no,
                                                                  list_dupl  = list_dupl if CFG['drop_dupl'] else [],
                                                                  list_noise = list_noise if CFG['drop_noise'] else [])  
            
        # update freezing for normal training if needed
        if (CFG['warmup_freeze']) and (epoch == CFG['warmup']):
            model, optimizer, scheduler = get_model(CFG, device, epoch)
            model.load_state_dict(torch.load(CFG['out_path'] + 'weights_fold{}.pth'.format(fold),
                                             map_location = device))

        # update freezing for fine-tuning if needed
        if (CFG['fine_tune']) and (epoch == CFG['num_epochs']):
            model, optimizer, scheduler = get_model(CFG, device, epoch)
            model.load_state_dict(torch.load(CFG['out_path'] + 'weights_fold{}.pth'.format(fold),
                                             map_location = device))
            
        # get losses            
        trn_criterion, val_criterion = get_losses(CFG, device, epoch)


        ### MODELING

        # training
        gc.collect()
        if CFG['device'] == 'TPU':
            pl_loader = pl.ParallelLoader(trn_loader, [device])
        trn_loss = train_epoch(loader     = trn_loader if CFG['device'] != 'TPU' else pl_loader.per_device_loader(device), 
                               model      = model, 
                               optimizer  = optimizer, 
                               scheduler  = scheduler,
                               criterion  = trn_criterion, 
                               epoch      = epoch,
                               CFG        = CFG,
                               device     = device)

        # inference
        gc.collect()
        if CFG['device'] == 'TPU':
            pl_loader = pl.ParallelLoader(val_loader, [device])
        val_loss, val_preds = valid_epoch(loader    = val_loader if CFG['device'] != 'TPU' else pl_loader.per_device_loader(device), 
                                          model     = model, 
                                          criterion = val_criterion, 
                                          CFG       = CFG,
                                          device    = device)
        

        ### EVALUATION
        
        # reduce losses
        if CFG['device'] == 'TPU':
            trn_loss = xm.mesh_reduce('loss', trn_loss, lambda x: sum(x) / (len(df_trn) * xm.xrt_world_size()))
            val_loss = xm.mesh_reduce('loss', val_loss, lambda x: sum(x) / (len(df_val) * xm.xrt_world_size()))
            lr       = scheduler.state_dict()['_last_lr'][0] / xm.xrt_world_size()
        else:
            trn_loss = trn_loss / len(df_trn)
            val_loss = val_loss / len(df_val)
            lr       = scheduler.state_dict()['_last_lr'][0]
            
        # save LR and losses
        lrs.append(lr)
        trn_losses.append(trn_loss)
        val_losses.append(val_loss)
        val_metrics.append((np.argmax(val_preds, axis = 1) == df_val['label']).sum() / len(df_val))
        
        # feedback
        smart_print('-- epoch {}/{} | lr = {:.6f} | trn_loss = {:.4f} | val_loss = {:.4f} | val_acc = {:.4f} | {:.2f} min'.format(
            epoch + 1, CFG['num_epochs'] + CFG['fine_tune'], lrs[epoch],
            trn_losses[epoch], val_losses[epoch], val_metrics[epoch],
            (time.time() - epoch_start) / 60), CFG)
        
        # export weights and save preds
        if val_metrics[epoch] >= max(val_metrics):
            smart_save(model.state_dict(), CFG['out_path'] + 'weights_fold{}.pth'.format(fold), CFG)
            val_preds_best = val_preds.copy()
        if CFG['save_all']:
            smart_save(model.state_dict(), CFG['out_path'] + 'weights_fold{}_epoch{}.pth'.format(fold, epoch), CFG)      
            
    
    return trn_losses, val_losses, val_metrics, val_preds_best