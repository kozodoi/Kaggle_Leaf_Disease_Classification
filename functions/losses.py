####### LOSSES LIBRARY

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

    def __init__(self, n = 2, ignore_index = -1, reduction = 'mean', smoothing = 0):
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
       
    
##### BI-TEMPERED LOSS

def log_t(u, t):
    if t==1.0:
        return u.log()
    else:
        return (u.pow(1.0 - t) - 1.0) / (1.0 - t)

def exp_t(u, t):
    if t==1:
        return u.exp()
    else:
        return (1.0 + (1.0-t)*u).relu().pow(1.0 / (1.0 - t))

def compute_normalization_fixed_point(activations, t, num_iters):
    mu, _ = torch.max(activations, -1, keepdim=True)
    normalized_activations_step_0 = activations - mu

    normalized_activations = normalized_activations_step_0

    for _ in range(num_iters):
        logt_partition = torch.sum(
                exp_t(normalized_activations, t), -1, keepdim=True)
        normalized_activations = normalized_activations_step_0 * \
                logt_partition.pow(1.0-t)

    logt_partition = torch.sum(
            exp_t(normalized_activations, t), -1, keepdim=True)
    normalization_constants = - log_t(1.0 / logt_partition, t) + mu

    return normalization_constants

def compute_normalization_binary_search(activations, t, num_iters):
    mu, _ = torch.max(activations, -1, keepdim=True)
    normalized_activations = activations - mu

    effective_dim = \
        torch.sum(
                (normalized_activations > -1.0 / (1.0-t)).to(torch.int32),
            dim=-1, keepdim=True).to(activations.dtype)

    shape_partition = activations.shape[:-1] + (1,)
    lower = torch.zeros(shape_partition, dtype=activations.dtype, device=activations.device)
    upper = -log_t(1.0/effective_dim, t) * torch.ones_like(lower)

    for _ in range(num_iters):
        logt_partition = (upper + lower)/2.0
        sum_probs = torch.sum(
                exp_t(normalized_activations - logt_partition, t),
                dim=-1, keepdim=True)
        update = (sum_probs < 1.0).to(activations.dtype)
        lower = torch.reshape(
                lower * update + (1.0-update) * logt_partition,
                shape_partition)
        upper = torch.reshape(
                upper * (1.0 - update) + update * logt_partition,
                shape_partition)

    logt_partition = (upper + lower)/2.0
    return logt_partition + mu

class ComputeNormalization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, activations, t, num_iters):
        if t < 1.0:
            normalization_constants = compute_normalization_binary_search(activations, t, num_iters)
        else:
            normalization_constants = compute_normalization_fixed_point(activations, t, num_iters)

        ctx.save_for_backward(activations, normalization_constants)
        ctx.t=t
        return normalization_constants

    @staticmethod
    def backward(ctx, grad_output):
        activations, normalization_constants = ctx.saved_tensors
        t = ctx.t
        normalized_activations = activations - normalization_constants 
        probabilities = exp_t(normalized_activations, t)
        escorts = probabilities.pow(t)
        escorts = escorts / escorts.sum(dim=-1, keepdim=True)
        grad_input = escorts * grad_output
        
        return grad_input, None, None

def compute_normalization(activations, t, num_iters=5):
    return ComputeNormalization.apply(activations, t, num_iters)

def tempered_sigmoid(activations, t, num_iters = 5):
    internal_activations = torch.stack([activations,
        torch.zeros_like(activations)],
        dim=-1)
    internal_probabilities = tempered_softmax(internal_activations, t, num_iters)
    return internal_probabilities[..., 0]


def tempered_softmax(activations, t, num_iters=5):
    if t == 1.0:
        return activations.softmax(dim=-1)

    normalization_constants = compute_normalization(activations, t, num_iters)
    return exp_t(activations - normalization_constants, t)

def bi_tempered_binary_logistic_loss(activations,
        labels,
        t1,
        t2,
        label_smoothing = 0.0,
        num_iters=5,
        reduction='mean'):

    internal_activations = torch.stack([activations,
        torch.zeros_like(activations)],
        dim=-1)
    internal_labels = torch.stack([labels.to(activations.dtype),
        1.0 - labels.to(activations.dtype)],
        dim=-1)
    return bi_tempered_logistic_loss(internal_activations, 
            internal_labels,
            t1,
            t2,
            label_smoothing = label_smoothing,
            num_iters = num_iters,
            reduction = reduction)

def bi_tempered_logistic_loss(activations,
        labels,
        t1,
        t2,
        label_smoothing=0.0,
        num_iters=5,
        reduction = 'mean'):

    if len(labels.shape)<len(activations.shape): #not one-hot
        labels_onehot = torch.zeros_like(activations)
        labels_onehot.scatter_(1, labels[..., None], 1)
    else:
        labels_onehot = labels

    if label_smoothing > 0:
        num_classes = labels_onehot.shape[-1]
        labels_onehot = ( 1 - label_smoothing * num_classes / (num_classes - 1) ) \
                * labels_onehot + \
                label_smoothing / (num_classes - 1)

    probabilities = tempered_softmax(activations, t2, num_iters)

    loss_values = labels_onehot * log_t(labels_onehot + 1e-10, t1) \
            - labels_onehot * log_t(probabilities, t1) \
            - labels_onehot.pow(2.0 - t1) / (2.0 - t1) \
            + probabilities.pow(2.0 - t1) / (2.0 - t1)
    loss_values = loss_values.sum(dim = -1) #sum over classes

    if reduction == 'none':
        return loss_values
    if reduction == 'sum':
        return loss_values.sum()
    if reduction == 'mean':
        return loss_values.mean()

class BiTemperedLogisticLoss(nn.Module): 
    def __init__(self, t1, t2, smoothing=0.0): 
        super(BiTemperedLogisticLoss, self).__init__() 
        self.t1 = t1
        self.t2 = t2
        self.smoothing = smoothing
    def forward(self, logit_label, truth_label):
        loss_label = bi_tempered_logistic_loss(
            logit_label, truth_label,
            t1=self.t1, t2=self.t2,
            label_smoothing=self.smoothing,
            reduction='none'
        )
        
        loss_label = loss_label.mean()
        return loss_label



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
        trn_criterion = TaylorCrossEntropyLoss(n         = CFG['taylor'], 
                                               smoothing = CFG['smoothing']).to(device)

    elif loss_fn == 'BiTempered':
        trn_criterion = BiTemperedLogisticLoss(t1        = CFG['bitempered'][0], 
                                               t2        = CFG['bitempered'][1], 
                                               smoothing = CFG['smoothing'])

    # define validation loss
    val_criterion = nn.CrossEntropyLoss().to(device)

    # output
    return trn_criterion, val_criterion