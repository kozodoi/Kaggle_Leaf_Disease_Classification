import torch
import torch.nn as nn
import torch.nn.functional as F


##### CROSSENTROPY WITH LABEL SMOOTHING

class LabelSmoothingCrossEntropy(nn.Module):

    def __init__(self, smoothing = 0.1, reduction = 'mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing  = smoothing
        self.confidence = 1. - smoothing
        self.reduction  = reduction

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


##### OHEM LOSS

class OhemCrossEntropy(nn.Module):

    def __init__(self, top_k = 0.7, smoothing = 0, reduction = 'none'):
        super(OhemCrossEntropy, self).__init__()
        self.reduction     = reduction
        self.top_k         = top_k
        self.smoothing     = smoothing
        self.ce_lab_smooth = LabelSmoothingCrossEntropy(smoothing = self.smoothing, reduction = self.reduction)

    def forward(self, pred, target):
        log_probs = F.log_softmax(pred, dim=1)
        loss      = self.ce_lab_smooth(log_probs, target)   
        if self.top_k == 1:
            return torch.mean(loss)
        else:
            valid_loss, idxs = torch.topk(loss, int(self.top_k * loss.size()[0]))    
            return torch.mean(valid_loss)


##### SYMMETRIC CROSSENTROPY

class SymmetricCrossEntropy(nn.Module):

    def __init__(self, alpha = 0.1, beta = 1.0, num_classes = 5, smoothing = 0, reduction = 'mean'):
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha         = alpha
        self.beta          = beta
        self.num_classes   = num_classes
        self.reduction     = reduction
        self.smoothing     = smoothing
        self.ce_lab_smooth = LabelSmoothingCrossEntropy(smoothing = self.smoothing, reduction = self.reduction)
        
    def forward(self, logits, targets):
        onehot_targets = torch.eye(self.num_classes)[targets].to(device)
        ce_loss  = self.ce_lab_smooth(logits, targets)
        rce_loss = (-onehot_targets*logits.softmax(1).clamp(1e-7, 1.0).log()).sum(1)
        if self.reduction == 'mean':
            rce_loss = rce_loss.mean()
        elif self.reduction == 'sum':
            rce_loss = rce_loss.sum()
        return self.alpha * ce_loss + self.beta * rce_loss


##### COMPLEMENT CROSSENTROPY

class ComplementEntropy(nn.Module):

    def __init__(self, num_classes = 5):
        super(ComplementEntropy, self).__init__()
        self.classes = num_classes
        self.batch_size = None

    def forward(self, y_hat, y):
        self.batch_size = len(y)
        y_hat = F.softmax(y_hat, dim=1)
        Yg = torch.gather(y_hat, 1, torch.unsqueeze(y, 1))
        Yg_ = (1 - Yg) + 1e-7
        Px = y_hat / Yg_.view(len(y_hat), 1)
        Px_log = torch.log(Px + 1e-10)
        y_zerohot = torch.ones(self.batch_size, self.classes).scatter_\
            (1, y.view(self.batch_size, 1).data.cpu(), 0)
        output = Px * Px_log * y_zerohot.to(device)
        entropy = torch.sum(output)
        entropy /= float(self.batch_size)
        entropy /= float(self.classes)
        return entropy

class ComplementCrossEntropy(nn.Module):

    def __init__(self, num_classes = 5, gamma = 5, smoothing = 0, reduction = 'mean'):
        super(ComplementCrossEntropy, self).__init__()
        self.gamma              = gamma
        self.smoothing          = smoothing
        self.reduction          = reduction
        self.complement_entropy = ComplementEntropy(num_classes)
        self.ce_lab_smooth      = LabelSmoothingCrossEntropy(smoothing = self.smoothing, reduction = self.reduction)

    def forward(self, y_hat, y):
        l1 = self.ce_lab_smooth(y_hat, y)
        l2 = self.complement_entropy(y_hat, y)
        return l1 + self.gamma * l2


##### FOCAL LOSS

class FocalLoss(nn.Module):
    def __init__(self, alpha = 1, gamma = 2, smoothing = 0, reduction = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha         = alpha
        self.gamma         = gamma
        self.reduction     = reduction
        self.smoothing     = smoothing
        self.ce_lab_smooth = LabelSmoothingCrossEntropy(smoothing = self.smoothing, reduction = self.reduction)

    def forward(self, inputs, targets):
        ce_loss = self.ce_lab_smooth(inputs, targets)
        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        return F_loss
        

##### FOCAL COSINE LOSS

class FocalCosineLoss(nn.Module):

    def __init__(self, alpha = 1, gamma = 2, xent = 0.1, smoothing = 0, reduction = 'mean'):
        super(FocalCosineLoss, self).__init__()
        self.alpha         = alpha
        self.gamma         = gamma
        self.xent          = xent
        self.y             = torch.Tensor([1]).to(device)
        self.reduction     = reduction
        self.smoothing     = smoothing
        self.ce_lab_smooth = LabelSmoothingCrossEntropy(smoothing = self.smoothing, reduction = 'none')

    def forward(self, input, target):
        cosine_loss = F.cosine_embedding_loss(input, F.one_hot(target, num_classes=input.size(-1)), self.y, reduction=self.reduction)
        cent_loss   = self.ce_lab_smooth(F.normalize(input), target)
        pt = torch.exp(-cent_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * cent_loss
        if self.reduction == 'mean':
            focal_loss = torch.mean(focal_loss)
        elif self.reduction == 'sum':
            focal_loss = torch.mean(focal_loss)
        return cosine_loss + self.xent * focal_loss
  
    
##### TAYLOR LOSS

class TaylorSoftmax(nn.Module):

    def __init__(self, dim=1, n=2):
        super(TaylorSoftmax, self).__init__()
        assert n % 2 == 0
        self.dim = dim
        self.n = n

    def forward(self, x):
        fn = torch.ones_like(x)
        denor = 1.
        for i in range(1, self.n+1):
            denor *= i
            fn = fn + x.pow(i) / denor
        out = fn / fn.sum(dim=self.dim, keepdims=True)
        return out

class LabelSmoothingLoss(nn.Module):

    def __init__(self, classes, smoothing=0.0, dim=-1): 
        super(LabelSmoothingLoss, self).__init__() 
        self.confidence = 1.0 - smoothing 
        self.smoothing = smoothing 
        self.cls       = classes 
        self.dim       = dim 

    def forward(self, pred, target): 
        with torch.no_grad(): 
            true_dist = torch.zeros_like(pred) 
            true_dist.fill_(self.smoothing / (self.cls - 1)) 
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence) 
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class TaylorCrossEntropyLoss(nn.Module):

    def __init__(self, CFG, n = 2, ignore_index = -1, reduction = 'mean', smoothing = 0):
        super(TaylorCrossEntropyLoss, self).__init__()
        assert n % 2 == 0
        self.taylor_softmax = TaylorSoftmax(dim=1, n=n)
        self.reduction      = reduction
        self.ignore_index   = ignore_index
        self.lab_smooth     = LabelSmoothingLoss(CFG['num_classes'], smoothing = smoothing)
        self.smoothing      = smoothing

    def forward(self, logits, labels):
        log_probs = self.taylor_softmax(logits).log()
        if self.smoothing == 0:
            loss = F.nll_loss(log_probs, labels, reduction=self.reduction,
                    ignore_index=self.ignore_index)
        else:
            loss = self.lab_smooth(log_probs, labels)
        return loss
    
    
    
####### LOSS PREP

def get_losses(CFG, device, epoch = None):

    # look up training loss
    if CFG['step_loss'] and epoch is not None:
        loss_fn = CFG['step_loss'][epoch]
        if epoch >= 1:
            if loss_fn != CFG['step_loss'][epoch - 1]:
                smart_print('- switching loss to {}...'.format(CFG['step_loss'][epoch]), CFG)
    else:
        loss_fn = CFG['loss_fn']

    # define training loss
    if loss_fn == 'CE':
        trn_criterion = LabelSmoothingCrossEntropy(smoothing = CFG['smoothing']).to(device)

    elif loss_fn == 'OHEM':
        trn_criterion = OhemCrossEntropy(top_k     = CFG['ohem'], 
                                         smoothing = CFG['smoothing']).to(device)

    elif loss_fn == 'SCE':
        trn_criterion = SymmetricCrossEntropy(alpha       = CFG['sce'][0],
                                              beta        = CFG['sce'][1],
                                              num_classes = CFG['num_classes'],
                                              smoothing   = CFG['smoothing']).to(device)

    elif loss_fn == 'CCE':
        trn_criterion = ComplementCrossEntropy(gamma       = CFG['cce'],
                                               num_classes = CFG['num_classes'],
                                               smoothing   = CFG['smoothing']).to(device)

    elif loss_fn == 'Focal':
        trn_criterion = FocalLoss(alpha     = CFG['focal'][0],
                                  gamma     = CFG['focal'][1],
                                  smoothing = CFG['smoothing']).to(device)

    elif loss_fn == 'FocalCosine':
        trn_criterion = FocalCosineLoss(alpha     = CFG['focalcosine'][0],
                                        gamma     = CFG['focalcosine'][1],
                                        xent      = CFG['focalcosine'][2],
                                        smoothing = CFG['smoothing']).to(device)

    elif loss_fn == 'Taylor':
        trn_criterion = TaylorCrossEntropyLoss(CFG       = CFG,
                                               n         = CFG['taylor'], 
                                               smoothing = CFG['smoothing']).to(device)

    elif loss_fn == 'BiTempered':
        trn_criterion = BiTemperedLogisticLoss(t1        = CFG['bitempered'][0], 
                                               t2        = CFG['bitempered'][1], 
                                               smoothing = CFG['smoothing'])

    # define validation loss
    val_criterion = nn.CrossEntropyLoss().to(device)

    return trn_criterion, val_criterion