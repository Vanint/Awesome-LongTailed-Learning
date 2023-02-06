import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from .fb_resnets import ResNet
from .fb_resnets import ResNeXt
from .fb_resnets import Expert_ResNet
from .fb_resnets import Expert_ResNeXt 
from .ldam_drw_resnets import resnet_cifar
from .ldam_drw_resnets import expert_resnet_cifar 


class Model(BaseModel):
    requires_target = False

    def __init__(self, num_classes, backbone_class=None):
        super().__init__()
        if backbone_class is not None: # Do not init backbone here if None
            self.backbone = backbone_class(num_classes)

    def _hook_before_iter(self):
        self.backbone._hook_before_iter()

    def forward(self, x, mode=None):
        x = self.backbone(x)

        assert mode is None
        return x

class EAModel(BaseModel):
    requires_target = True
    confidence_model = True

    def __init__(self, num_classes, backbone_class=None):
        super().__init__()
        if backbone_class is not None: # Do not init backbone here if None
            self.backbone = backbone_class(num_classes)

    def _hook_before_iter(self):
        self.backbone._hook_before_iter()

    def forward(self, x, mode=None, target=None):
        x = self.backbone(x, target=target)

        assert isinstance(x, tuple) # logits, extra_info
        assert mode is None
        
        return x

class ResNet10Model(Model):
    def __init__(self, num_classes, reduce_dimension=False, layer3_output_dim=None, layer4_output_dim=None, use_norm=False, num_experts=1, **kwargs):
        super().__init__(num_classes, None)
        if num_experts == 1:
            self.backbone = ResNet.ResNet(ResNet.BasicBlock, [1, 1, 1, 1], dropout=None, num_classes=num_classes, use_norm=use_norm, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim, **kwargs)
        else: 
            self.backbone = Expert_ResNet.ResNet(ResNet.BasicBlock, [1, 1, 1, 1], dropout=None, num_classes=num_classes, use_norm=use_norm, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim, num_experts=num_experts, **kwargs)
 
class ResNet32Model(Model): # From LDAM_DRW
    def __init__(self, num_classes, reduce_dimension=False, layer2_output_dim=None, layer3_output_dim=None, use_norm=False, num_experts=1, **kwargs):
        super().__init__(num_classes, None)
        if num_experts == 1:
            self.backbone = resnet_cifar.ResNet_s(resnet_cifar.BasicBlock, [5, 5, 5], num_classes=num_classes, reduce_dimension=reduce_dimension, layer2_output_dim=layer2_output_dim, layer3_output_dim=layer3_output_dim, use_norm=use_norm, **kwargs)
        else:
            self.backbone = expert_resnet_cifar.ResNet_s(expert_resnet_cifar.BasicBlock, [5, 5, 5], num_classes=num_classes, reduce_dimension=reduce_dimension, layer2_output_dim=layer2_output_dim, layer3_output_dim=layer3_output_dim, use_norm=use_norm, num_experts=num_experts, **kwargs)
 
class ResNet50Model(Model):
    def __init__(self, num_classes, reduce_dimension=False, layer3_output_dim=None, layer4_output_dim=None, use_norm=False, num_experts=1, **kwargs):
        super().__init__(num_classes, None)
        if num_experts == 1:
            self.backbone = ResNet.ResNet(ResNet.Bottleneck, [3, 4, 6, 3], dropout=None, num_classes=num_classes, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim, use_norm=use_norm, **kwargs)
        else:
            self.backbone = Expert_ResNet.ResNet(Expert_ResNet.Bottleneck, [3, 4, 6, 3], dropout=None, num_classes=num_classes, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim, use_norm=use_norm, num_experts=num_experts, **kwargs)
            self.backbone =  init_weights_R50(model = self.backbone, weights_path="./model/moco_ckpt.pth.tar")
 
  
class ResNeXt50Model(Model):
    def __init__(self, num_classes, reduce_dimension=False, layer3_output_dim=None, layer4_output_dim=None, num_experts=1, **kwargs):
        super().__init__(num_classes, None)
        if num_experts == 1:
            self.backbone = ResNeXt.ResNext(ResNeXt.Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4, dropout=None, num_classes=num_classes, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim, **kwargs)
        else:
            self.backbone = Expert_ResNeXt.ResNext(Expert_ResNeXt.Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4, dropout=None, num_classes=num_classes, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim, num_experts=num_experts, **kwargs)
 
class ResNet101Model(Model):
    def __init__(self, num_classes, reduce_dimension=False, layer3_output_dim=None, layer4_output_dim=None, use_norm=False, num_experts=1, **kwargs):
        super().__init__(num_classes, None)
        if num_experts == 1:
            self.backbone = ResNet.ResNet(ResNet.Bottleneck, [3, 4, 23, 3], dropout=None, num_classes=num_classes, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim, use_norm=use_norm, **kwargs)
        else:
            self.backbone = Expert_ResNet.ResNet(Expert_ResNet.Bottleneck, [3, 4, 23, 3], dropout=None, num_classes=num_classes, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim, use_norm=use_norm, num_experts=num_experts, **kwargs)


def init_weights_R50(model, weights_path="./model/pretrained_model_places/resnet152.pth"):
    """Initialize weights""" 
    checkpoint = torch.load(weights_path)
    model_state = checkpoint['state_dict']
    #weights = {k: weights['module.' + k] if 'module.' + k in weights else model.state_dict()[k]
    #            for k in model.state_dict()}
 
    for k in list(model_state.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            model_state[f"{k[len('module.encoder_q.'):]}"] = model_state[k]
        # delete renamed or unused k
        del model_state[k]
    weights = model_state  
    weights1 = {} 
            # lower layers are the shared backbones
    for k in model.state_dict():
        if 'layer3s' not in k and 'layer4s' not in k:
            weights1[k] =  weights[k] if k in weights else model.state_dict()[k]
        elif 'num_batches_tracked' in k:
            weights1[k] =  weights[k] if k in weights else model.state_dict()[k]
            
        elif 'layer3s.0.' in k and 'num_batches_tracked' not in k:
            weights1[k] = weights[k.replace('layer3s.0.','layer3.')]
        elif 'layer3s.1.' in k and 'num_batches_tracked' not in k:
            weights1[k] = weights[k.replace('layer3s.1.','layer3.')]
        elif 'layer3s.2.' in k and 'num_batches_tracked' not in k:
            weights1[k] = weights[k.replace('layer3s.2.','layer3.')]                       
        elif 'layer4s.0.' in k and 'num_batches_tracked' not in k:
            weights1[k] = weights[k.replace('layer4s.0.','layer4.')]
        elif 'layer4s.1.' in k and 'num_batches_tracked' not in k:
            weights1[k] = weights[k.replace('layer4s.1.','layer4.')]
        elif 'layer4s.2.' in k and 'num_batches_tracked' not in k:
            weights1[k] = weights[k.replace('layer4s.2.','layer4.')] 
    
    model.load_state_dict(weights1, strict=True)
    return model

def init_weights(model, weights_path="./model/pretrained_model_places/resnet152.pth", caffe=False, classifier=False):
    """Initialize weights"""
    print('Pretrained %s weights path: %s' % ('classifier' if classifier else 'feature model',  weights_path))
    weights = torch.load(weights_path)
    weights1 = {}
    if not classifier:
        if caffe: 
            # lower layers are the shared backbones
            for k in model.state_dict():
                if 'layer3s' not in k and 'layer4s' not in k:
                    weights1[k] =  weights[k] if k in weights else model.state_dict()[k]
                elif 'num_batches_tracked' in k:
                    weights1[k] =  weights[k] if k in weights else model.state_dict()[k]
                    
                elif 'layer3s.0.' in k and 'num_batches_tracked' not in k:
                    weights1[k] = weights[k.replace('layer3s.0.','layer3.')]
                elif 'layer3s.1.' in k and 'num_batches_tracked' not in k:
                    weights1[k] = weights[k.replace('layer3s.1.','layer3.')]
                elif 'layer3s.2.' in k and 'num_batches_tracked' not in k:
                    weights1[k] = weights[k.replace('layer3s.2.','layer3.')]                       
                elif 'layer4s.0.' in k and 'num_batches_tracked' not in k:
                    weights1[k] = weights[k.replace('layer4s.0.','layer4.')]
                elif 'layer4s.1.' in k and 'num_batches_tracked' not in k:
                    weights1[k] = weights[k.replace('layer4s.1.','layer4.')]
                elif 'layer4s.2.' in k and 'num_batches_tracked' not in k:
                    weights1[k] = weights[k.replace('layer4s.2.','layer4.')]
 
        else:
            weights = weights['state_dict_best']['feat_model']
            weights = {k: weights['module.' + k] if 'module.' + k in weights else model.state_dict()[k]
                       for k in model.state_dict()}
    else:
        weights = weights['state_dict_best']['classifier']
        weights = {k: weights['module.fc.' + k] if 'module.fc.' + k in weights else model.state_dict()[k]
                   for k in model.state_dict()}
    model.load_state_dict(weights1)
    return model

class ResNet152Model(Model):
    def __init__(self, num_classes, reduce_dimension=False, layer3_output_dim=None, layer4_output_dim=None, share_layer3=False, use_norm=False, num_experts=1, **kwargs):
        super().__init__(num_classes, None)
        if num_experts == 1:
            self.backbone = ResNet.ResNet(ResNet.Bottleneck, [3, 8, 36, 3], dropout=None, num_classes=num_classes, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim, use_norm=use_norm, **kwargs)
        else:
            self.backbone = Expert_ResNet.ResNet(Expert_ResNet.Bottleneck, [3, 8, 36, 3], dropout=None, num_classes=num_classes, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim, share_layer3=share_layer3, use_norm=use_norm, num_experts=num_experts, **kwargs)
            self.backbone =  init_weights(model = self.backbone, weights_path="./model/pretrained_model_places/resnet152.pth", caffe=True)
 
class ResNeXt152Model(Model):
    def __init__(self, num_classes, reduce_dimension=False, layer3_output_dim=None, layer4_output_dim=None, use_norm=False, num_experts=1, **kwargs):
        super().__init__(num_classes, None)
        if num_experts == 1:
            self.backbone = ResNeXt.ResNext(ResNeXt.Bottleneck, [3, 8, 36, 3], groups=32, width_per_group=4, dropout=None, num_classes=num_classes, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim)
        else:
            self.backbone = Expert_ResNeXt.ResNext(Expert_ResNeXt.Bottleneck, [3, 8, 36, 3], groups=32, width_per_group=4, dropout=None, num_classes=num_classes, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim, num_experts=num_experts)
