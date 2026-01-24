from .resnet_big import *
from .otfusion_model import *
from .lightweight_model import *

def get_train_models(model_name, num_classes, mode, use_pretrain=False, **kwargs):
    if mode == 'unsupervised':
        train_model = SupConResNet(model_name, head=kwargs['head'])
        if kwargs['classifier'] == 'linear':
            classifier = LinearClassifier(model_name, num_classes=num_classes)
        elif kwargs['classifier'] == 'mlp':
            classifier = MLPClassifier(model_name, num_classes=num_classes)
        return train_model, classifier
    elif mode == 'ot':
        model = get_model_for_ot(model_name, n_c=num_classes)
        return model
    elif mode == 'etf':
        model = ETFCEResNet(model_name, num_classes=num_classes)
        return model
    elif mode == 'our':
        if 'mobilenet' in model_name:
            # MobileNet implementation usually does not support automatic weight loading here yet, 
            # or requires similar changes. Assuming ResNet18 for now as per user request.
            model = LearnableProtoMobileNet(model_name, num_classes=num_classes)
        else:
            if model_name == 'resnet18' and use_pretrain:
                # Load standard ResNet18 with ImageNet weights
                import torchvision.models as models
                # weights='DEFAULT' corresponds to the best available weights (ImageNet1K_V1)
                # Note: This requires torchvision >= 0.13. For older versions, use pretrained=True
                try:
                    base_model = models.resnet18(weights='DEFAULT')
                except:
                    # Fallback for older torchvision
                    base_model = models.resnet18(pretrained=True)
                
                # The LearnableProtoResNet expects a fresh initialization usually, 
                # but here we want to inject the pretrained weights.
                # Since LearnableProtoResNet (in resnet_big.py) instantiates its own encoder inside its __init__,
                # we have two options:
                # 1. Pass weights to LearnableProtoResNet (requires modifying resnet_big.py)
                # 2. Hack: Instantiate LearnableProtoResNet, then overwrite its encoder state_dict.
                
                # Let's go with option 2 for minimal invasiveness if the architecture matches.
                # However, resnet_big.py uses a custom ResNet class implementation (not torchvision's).
                # Checking resnet_big.py content... 
                # It defines `class ResNet(nn.Module)` and `def resnet18`.
                # This is NOT torchvision.models.ResNet. Keys might not match 100% (e.g. 'layer1.0.conv1.weight').
                
                # Wait, if `models_lib.resnet_big` defines its own ResNet, we cannot simply load torchvision weights 
                # if the naming convention or block structure differs slightly.
                # `resnet_big.py` seems to follow standard naming (layer1...layer4, conv1, bn1).
                # Let's try loading state_dict.
                
                model = LearnableProtoResNet(model_name, num_classes=num_classes)
                
                # Filter out fc layer from pretrained state_dict (since we have num_classes mismatch)
                pretrained_dict = base_model.state_dict()
                model_dict = model.encoder.state_dict()
                
                # 1. filter out unnecessary keys
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k != 'fc.weight' and k != 'fc.bias'}
                # 2. overwrite entries in the existing state_dict
                model_dict.update(pretrained_dict) 
                # 3. load the new state_dict
                model.encoder.load_state_dict(model_dict)
                
            else:
                model = LearnableProtoResNet(model_name, num_classes=num_classes)
        return model
    elif mode == 'our_projector':
        # New mode for V15
        if model_name == 'resnet18' and use_pretrain:
            import torchvision.models as models
            try:
                base_model = models.resnet18(weights='DEFAULT')
            except:
                base_model = models.resnet18(pretrained=True)
            
            model = LearnableProtoResNetWithProjector(model_name, num_classes=num_classes)
            
            # For LearnableProtoResNetWithProjector, the structure is model.encoder = Sequential(backbone, projector)
            # So the backbone is model.encoder[0]
            # however, resnet_big.py's ResNet keys might mismatch with torchvision's.
            # But earlier in mode='our', we found that `resnet_big.py` follows standard naming mostly?
            # Actually, `resnet_big.py` defines `class ResNet` which has `self.conv1`, `self.bn1`, etc.
            # Torchvision ResNet also has `self.conv1`, `self.bn1`.
            # So loading state_dict should work if keys match.
            
            pretrained_dict = base_model.state_dict()
            # The target backbone is at model.encoder[0]
            target_backbone = model.encoder[0]
            backbone_dict = target_backbone.state_dict()
            
            # Filter and update
            # We filter out fc keys. Note: LearnableProtoResNetWithProjector's backbone (from resnet_big) 
            # might have an 'fc' layer initialized but unused (resnet18() creates it). 
            # We just overwrite excluding fc.
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in backbone_dict and 'fc' not in k}
            
            backbone_dict.update(pretrained_dict)
            target_backbone.load_state_dict(backbone_dict)
            
        else:
            model = LearnableProtoResNetWithProjector(model_name, num_classes=num_classes)
        return model
    else:
        if 'mobilenetv2' in model_name:
            model = SupConMobileNet(model_name, feat_dim=num_classes)
        else:
            model = SupCEResNet(model_name, num_classes=num_classes)
        return model