
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import logging
import numpy as np

# Add FedWhat to path

from models_lib.resnet_big import BottleneckProtoResNet

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
NORMALIZE_DICT = {
    'CIFAR100': {'mean': (0.5071, 0.4867, 0.4408), 'std': (0.2675, 0.2565, 0.2761)},
}

def generate_etf_anchors(num_classes, feature_dim, device):
    """
    Generate Equiangular Tight Frame (ETF) anchors.
    """
    # Check constraint
    if feature_dim < num_classes - 1:
        logger.warning(f"[ETF Failure Mode] Feature dim ({feature_dim}) < C-1 ({num_classes-1}). ETF cannot be theoretically formed.")
        # Fallback to random initialization for comparison (simulating failed alignment or suboptimal)
        return torch.randn(num_classes, feature_dim).to(device)

    # Standard ETF generation
    I = torch.eye(num_classes)
    J = torch.ones(num_classes, num_classes)
    M = I - (1 / num_classes) * J
    try:
        L = torch.linalg.cholesky(M + 1e-6 * I)
    except torch.linalg.LinAlgError:
        eigvals, eigvecs = torch.linalg.eigh(M)
        eigvals[eigvals < 0] = 0
        L = eigvecs @ torch.diag(torch.sqrt(eigvals))
        
    H_ortho = torch.randn(feature_dim, num_classes)
    Q, _ = torch.linalg.qr(H_ortho)
    W = Q @ L.T
    etf_anchors = W.T.to(device)
    etf_anchors = torch.nn.functional.normalize(etf_anchors, dim=1)
    return etf_anchors

def get_loaders(batch_size=128):
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(**NORMALIZE_DICT['CIFAR100'])
    ])
    
    transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(size=32, scale=(0.2, 1.), antialias=False),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomApply([
                torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(**NORMALIZE_DICT['CIFAR100'])
    ])

    # Download to a common data folder
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
    
    # Create a subset for quick experiment (simulate one client)
    # Use 5000 samples (50 per class)
    subset_indices = list(range(5000))
    train_subset = torch.utils.data.Subset(trainset, subset_indices)
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    return train_loader, test_loader

def eval_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits, _ = model(data) # Expect tuple (logits, features)
            _, predicted = torch.max(logits.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total

def run_condition(condition_name, feature_dim, projector_dim, device, epochs=20):
    logger.info(f"\n{'='*20}\nRunning Condition: {condition_name}\n{'='*20}")
    
    # 1. Setup Model
    # Note: ResNet18 default out is 512. BottleneckProtoResNet will force it to feature_dim.
    # If projector_dim is set, it adds MLP feature_dim -> projector_dim.
    model = BottleneckProtoResNet(name='resnet18', num_classes=100, feature_dim=feature_dim, projector_dim=projector_dim)
    model.to(device)
    
    # 2. Setup Analyical Target (ETF)
    # The ETF is generated for the FINAL dimension used for alignment.
    # If projector is used, final dim is projector_dim. Else it is feature_dim.
    final_dim = projector_dim if projector_dim else feature_dim
    fixed_anchors = generate_etf_anchors(100, final_dim, device)
    
    # 3. Data
    train_loader, test_loader = get_loaders()
    
    # 4. Training
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    
    # Constants for loss
    lambda_align = 5.0 # Strong alignment pressure
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            logits, feature_norm = model(data)
            
            # CE Loss
            loss_ce = criterion_ce(logits, target)
            
            # ETF Alignment Loss
            # Align normalized features to the fixed ETF anchors for their class
            target_anchors = fixed_anchors[target]
            loss_align = criterion_mse(feature_norm, target_anchors)
            
            loss = loss_ce + lambda_align * loss_align
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
        train_acc = correct / total
        if (epoch + 1) % 5 == 0:
            test_acc = eval_model(model, test_loader, device)
            logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
            
    final_acc = eval_model(model, test_loader, device)
    return final_acc

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Condition 1: Direct Alignment on Bottleneck (d=32)
    # This violates d >= C-1 (32 << 99). Expect SEVERE FAILUE.
    acc_1 = run_condition("1. Bottleneck d=32 (No Projector)", feature_dim=32, projector_dim=None, device=device, epochs=50)
    
    # Condition 2: Projector (d=32 -> 128)
    # The bottleneck is 32 (carrying info), but we align in 128 (where ETF is valid).
    acc_2 = run_condition("2. Bottleneck d=32 + Projector -> 128", feature_dim=32, projector_dim=128, device=device, epochs=50)
    
    print("\n" + "="*40)
    print("FINAL RESULTS SUMMARY")
    print("="*40)
    print(f"Condition 1 (d=32, Violates ETF): {acc_1:.4f}")
    print(f"Condition 2 (d=32->128, Valid ETF): {acc_2:.4f}")
    print(f"Improvement: {(acc_2 - acc_1)*100:.2f}%")
