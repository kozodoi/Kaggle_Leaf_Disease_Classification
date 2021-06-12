import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from warmup_scheduler import GradualWarmupScheduler  


####### MODEL ARCHITECTURE

def init_model(CFG, device, path = None):

    ##### CONVOLUTIONAL PART

    if 'deit' in CFG['backbone']: 
        model = torch.hub.load(repo_or_dir = 'facebookresearch/deit:main', 
                               model       = CFG['backbone'], 
                               pretrained  = False if (CFG['weights'] == 'empty') or (CFG['weights'] == 'custom') else True)
                 
    else:
        model = timm.create_model(model_name = CFG['backbone'], 
                                  pretrained = False if (CFG['weights'] == 'empty') or (CFG['weights'] == 'custom') else True)
        

    ##### CUSTOM WEIGHTS

    if CFG['weights'] == 'custom':

        if 'efficient' in CFG['backbone']:
            model.classifier = nn.Linear(model.classifier.in_features, CFG['pr_num_classes'])
        elif ('vit' in CFG['backbone']) or ('deit' in CFG['backbone']):
            model.head = nn.Linear(model.head.in_features, CFG['pr_num_classes'])
        else:
            model.fc = nn.Linear(model.fc.in_features, CFG['pr_num_classes'])
        
        if path is None:
            path = 'weights_{}_pretrain.pth'.format(CFG['backbone'])
        model.load_state_dict(torch.load(CFG['model_path'] + path, map_location = device))

        
    ##### CLASSIFIER PART

    if (CFG['weights'] != 'custom') or ((CFG['weights'] == 'custom') and (CFG['num_classes'] != CFG['pr_num_classes'])):
    
        if 'efficient' in CFG['backbone']:
            model.classifier = nn.Linear(model.classifier.in_features, CFG['num_classes'])
        elif ('vit' in CFG['backbone']) or ('deit' in CFG['backbone']):
            model.head = nn.Linear(model.head.in_features, CFG['num_classes'])
        else:
            model.fc = nn.Linear(model.fc.in_features, CFG['num_classes'])
            
            
    ##### MODEL WITH ATTENTION [EFFICIENTNET ONLY]
    
    if CFG['attention']:
        
        class model_with_attention(nn.Module):
            
            def __init__(self, CFG):
                super().__init__()
                self.backbone           = timm.create_model(model_name = CFG['backbone'], pretrained = False if (CFG['weights'] == 'empty') or (CFG['weights'] == 'custom') else True)
                self.backbone._dropout  = nn.Dropout(0.1)
                n_features              = self.backbone.classifier.in_features
                self.backbone.classifier = nn.Linear(n_features, CFG['num_classes'])
                self.local_fe           = CBAM(n_features)
                self.dropout            = nn.Dropout(0.1)
                self.classifier          = nn.Sequential(nn.Linear(n_features + n_features, n_features),
                                                        nn.BatchNorm1d(n_features),
                                                        nn.Dropout(0.1),
                                                        nn.ReLU(),
                                                        nn.Linear(n_features, CFG['num_classes']))

            def forward(self, image):
                enc_feas    = self.backbone.forward_features(image)
                global_feas = self.backbone.global_pool(enc_feas)
                global_feas = global_feas.flatten(start_dim = 1)
                global_feas = self.dropout(global_feas)
                local_feas  = self.local_fe(enc_feas)
                local_feas  = torch.sum(local_feas, dim = [2, 3])
                local_feas  = self.dropout(local_feas)
                all_feas    = torch.cat([global_feas, local_feas], dim = 1)
                outputs     = self.classifier(all_feas)
                return outputs
            
        model = model_with_attention(CFG)
    

    return model



####### MODEL PREP

def get_model(CFG, device, epoch = None):

    ##### MODEL

    # initialize model
    model = init_model(CFG, device)

    # send to device
    if CFG['device'] != 'TPU':
        model = model.to(device)
    else:
        mx    = xmp.MpModelWrapper(model)
        model = mx.to(device)
        
        
    ##### FREEZING
    
    # freezing deep layers for warmup
    if (epoch is None) and (CFG['warmup_freeze']):
        smart_print('- freezing deep layers...', CFG)
        for name, child in model.named_children():
            if name not in ['classifier', 'fc', 'head']:
                for param in child.parameters():
                    param.requires_grad = False
                    
    # unfreezing deep layers for normal training
    if epoch is not None: 
        if (CFG['warmup_freeze']) and (epoch == CFG['warmup']):
            smart_print('- unfreezing deep layers...', CFG)
            for name, child in model.named_children():
                for param in child.parameters():
                    param.requires_grad = True

    # freezing deep layers for fine-tuning
    if epoch is not None:
        if (CFG['fine_tune']) and (epoch >= CFG['num_epochs']):
            smart_print('- freezing deep layers...', CFG)
            for name, child in model.named_children():
                if name not in ['classifier', 'fc', 'head']:
                    for param in child.parameters():
                        param.requires_grad = False
            eta = CFG['eta_min']

                    
    ##### OPTIMIZER

    # scale learning rates
    if CFG['device'] == 'TPU':
        eta     = CFG['eta']     * xm.xrt_world_size()
        eta_min = CFG['eta_min'] * xm.xrt_world_size()
    else:
        eta     = CFG['eta']
        eta_min = CFG['eta_min']

    # optimizer
    if CFG['optim'] == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                               lr           = eta, 
                               weight_decay = CFG['decay'])
    elif CFG['optim'] == 'AdamW':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                lr           = eta, 
                                weight_decay = CFG['decay'])
    elif CFG['optim'] == 'AdamP':
        optimizer = AdamP(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr           = eta, 
                          weight_decay = CFG['decay'])


    ##### SCHEDULER

    # scheduler after warmup
    if CFG['schedule'] == 'CosineAnnealing':
        after_scheduler = CosineAnnealingWarmRestarts(optimizer = optimizer,
                                                      T_0       = CFG['num_epochs'] - CFG['warmup'] if CFG['num_epochs'] > 1 else 1,
                                                      eta_min   = eta_min)
        
    # warmup
    scheduler = GradualWarmupScheduler(optimizer       = optimizer, 
                                       multiplier      = 1, 
                                       total_epoch     = CFG['warmup'] + 1, 
                                       after_scheduler = after_scheduler)

    # output
    return model, optimizer, scheduler



####### ATTENTION MODULES

class PAM_Module(nn.Module):
    ''' Position attention module'''
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in  = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma      = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        '''
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        '''
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key   = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy     = torch.bmm(proj_query, proj_key)
        attention  = torch.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    ''' Channel attention module'''
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        '''
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        '''
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key   = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy     = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention  = torch.softmax(energy_new, dim=-1)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        
        return out


class CBAM(nn.Module):
    def __init__(self, in_channels):
        # def __init__(self):
        super(CBAM, self).__init__()
        inter_channels = in_channels // 4
        self.conv1_c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(inter_channels),
                                     nn.ReLU())
        
        self.conv1_s = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(inter_channels),
                                     nn.ReLU())

        self.channel_gate = CAM_Module(inter_channels)
        self.spatial_gate = PAM_Module(inter_channels)

        self.conv2_c = nn.Sequential(nn.Conv2d(inter_channels, in_channels, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(in_channels),
                                     nn.ReLU())
        self.conv2_a = nn.Sequential(nn.Conv2d(inter_channels, in_channels, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(in_channels),
                                     nn.ReLU())

    def forward(self, x):
        feat1    = self.conv1_c(x)
        chnl_att = self.channel_gate(feat1)
        chnl_att = self.conv2_c(chnl_att)

        feat2    = self.conv1_s(x)
        spat_att = self.spatial_gate(feat2)
        spat_att = self.conv2_a(spat_att)

        x_out = chnl_att + spat_att

        return x_out



####### GRADCAM MODULES

'''
Based on https://www.kaggle.com/debarshichanda/gradcam-visualize-your-cnn
'''

class FeatureExtractor():
    ''' Class for extracting activations and
    registering gradients from targetted intermediate layers '''

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    ''' Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. '''

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif 'avgpool' in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                x = module(x)

        return target_activations, x


class GradCam():

    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        features, output = self.extractor(input_img)

        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()
        
        one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis = (2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_img.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * (1 - mask)), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = np.float32(img) + 0.5 * heatmap
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)