from .resnet_big import *
from .otfusion_model import *
from .lightweight_model import *
from .vit import *

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
        elif model_name == 'vit':
            model = LearnableProtoViT(num_classes=num_classes)
        else:
                
            # If use_pretrain is a STRING, we treat it as a path to a checkpoint
            if isinstance(use_pretrain, str):
                model = LearnableProtoResNet(model_name, num_classes=num_classes)
                print(f"[get_train_models] Loading custom weights from {use_pretrain}")
                state_dict = torch.load(use_pretrain, map_location='cpu')
                
                # Our pretrain_centralized.py uses SupCEResNet, which has structure:
                #   self.encoder = ...
                #   self.fc = ...
                # So state_dict keys are 'encoder.conv1...', 'fc...', etc.
                
                # LearnableProtoResNet has structure:
                #   self.encoder = ...
                #   self.learnable_proto = ...
                # So its keys are 'encoder.conv1...', 'learnable_proto...'
                
                # The 'encoder.' keys match perfectly.
                # We can use strict=False to ignore 'fc' (in ckpt) and 'learnable_proto' (in model)
                
                # Check consistency just in case
                if any(k.startswith('encoder.') for k in state_dict.keys()):
                    print("[get_train_models] Detected 'encoder.' prefix in checkpoint. Using top-level selective load.")
                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
                    print(f"[get_train_models] Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
                else:
                    # Fallback for checkpoints without 'encoder.' prefix (e.g. if we saved just the backbone)
                    print("[get_train_models] No 'encoder.' prefix. Trying to load into model.encoder directly.")
                    model.encoder.load_state_dict(state_dict, strict=False)
                
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
             model = LearnableProtoResNetWithProjector(model_name, num_classes=num_classes) 
             
             # LearnableProtoResNetWithProjector structure:
             # self.encoder = nn.Sequential(backbone, projector)
             # Keys: 'encoder.0.conv1...', 'encoder.1.weight'...
             
             # Checkpoint (SupCEResNet) structure:
             # keys: 'encoder.conv1...', 'fc...'
             
             # Mismatch: Checkpoint says 'encoder.conv1...', Model says 'encoder.0.conv1...'
             # We need to map 'encoder.' -> 'encoder.0.'
             
             new_state_dict = {}
             for k, v in state_dict.items():
                 if k.startswith('encoder.'):
                     # encoder.conv1... -> encoder.0.conv1...
                     new_k = k.replace('encoder.', 'encoder.0.')
                     new_state_dict[new_k] = v
                 # Ignore fc
            
             if len(new_state_dict) > 0:
                 print("[get_train_models] Remapping 'encoder.' -> 'encoder.0.' for Projector architecture.")
                 model.load_state_dict(new_state_dict, strict=False)
             else:
                 # Fallback: maybe exact match?
                 model.load_state_dict(state_dict, strict=False)
        
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
        elif model_name == 'vit':
            # Standard supervised training for ViT (FedAvg)
            # Use the defaults defined in vit.py (Tiny scale)
            model = SimpleViT(num_classes=num_classes)
        else:
            model = SupCEResNet(model_name, num_classes=num_classes)
        return model