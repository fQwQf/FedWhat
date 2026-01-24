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
                
            # If use_pretrain is a STRING, we treat it as a path to a checkpoint
            if isinstance(use_pretrain, str):
                print(f"[get_train_models] Loading custom weights from {use_pretrain}")
                state_dict = torch.load(use_pretrain, map_location='cpu')
                # If loaded from our pretrain script, it might be the whole model state dict
                # We need to filter out the final FC layer because LearnableProtoResNet handles classification differently (via proto)
                # The keys in resnet_big.py are like 'conv1.weight', 'layer1...', 'fc...'
                
                model_state_dict = model.encoder.state_dict()
                
                # Check if state_dict keys match model.encoder keys (which are also 'conv1...', 'layer1...')
                # Be careful: LearnableProtoResNet (in resnet_big.py) has self.encoder = model_fun() which IS the ResNet.
                # So model.encoder IS the ResNet instance.
                # Our saved checkpoint IS also the ResNet instance state_dict.
                # So keys should match perfectly, except maybe 'fc'.
                
                new_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and 'fc' not in k}
                model_state_dict.update(new_state_dict)
                model.encoder.load_state_dict(model_state_dict)
                
                model_state_dict.update(new_state_dict)
                model.encoder.load_state_dict(model_state_dict)
                
            elif use_pretrain:
                # User explicitly requested NO ImageNet weights.
                # If we are here, it means use_pretrain is True (bool) but not a string path.
                # We should NOT load ImageNet.
                print("[get_train_models] Warning: use_pretrain=True but no path provided. ImageNet weights are DISABLED by user request. Using random initialization.")
                model = LearnableProtoResNet(model_name, num_classes=num_classes)
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
        if isinstance(use_pretrain, str):
             # Load custom weights for projector mode
             print(f"[get_train_models] Loading custom weights from {use_pretrain} for Projector Model")
             state_dict = torch.load(use_pretrain, map_location='cpu')
             
             model = LearnableProtoResNetWithProjector(model_name, num_classes=num_classes) # Initialize model before loading state_dict
             # model.encoder is Sequential(backbone, projector)
             # Our pretrain script saves ResNet (backbone + fc).
             # We want to load this into backbone (model.encoder[0]).
             
             target_backbone = model.encoder[0]
             backbone_dict = target_backbone.state_dict()
             
             new_state_dict = {k: v for k, v in state_dict.items() if k in backbone_dict and 'fc' not in k}
             backbone_dict.update(new_state_dict)
             target_backbone.load_state_dict(backbone_dict)
        
        elif use_pretrain:
             # Disabling ImageNet fallback for Projector mode too
             print("[get_train_models] Warning: use_pretrain=True but no path provided for Projector Model. ImageNet weights are DISABLED. Using random initialization.")
             model = LearnableProtoResNetWithProjector(model_name, num_classes=num_classes)

        else:
            model = LearnableProtoResNetWithProjector(model_name, num_classes=num_classes)
        return model
    else:
        if 'mobilenetv2' in model_name:
            model = SupConMobileNet(model_name, feat_dim=num_classes)
        else:
            model = SupCEResNet(model_name, num_classes=num_classes)
        return model