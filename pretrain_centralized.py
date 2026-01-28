import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset_helper import get_fl_dataset, get_supervised_transform, load_dataset, build_dataset_idxs
import torchvision.transforms as transforms
import torchvision

# Import models
from models_lib import model_dict

def train(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        if epoch == 1 and batch_idx == 0:
            print("\n--- DEBUG INFO ---")
            print(f"Model Conv1: {model.encoder.conv1}") 
            print(f"Data Range: min={data.min().item():.2f}, max={data.max().item():.2f}")
            print(f"Target Min: {target.min().item()}, Max: {target.max().item()}")
            print("------------------\n")

        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        pbar.set_postfix({'Loss': running_loss / (batch_idx + 1), 'Acc': 100. * correct / total})

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    acc = 100. * correct / total
    print(f'\nTest set: Average loss: {test_loss/len(test_loader):.4f}, Accuracy: {correct}/{total} ({acc:.2f}%)\n')
    return acc

def main():
    parser = argparse.ArgumentParser(description='Centralized Pretraining')
    parser.add_argument('--config', type=str, default='configs/Tiny_alpha0.5.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Using device: {device}')

    # Dataset setup
    dataset_name = config['dataset']['data_name']
    root_path = os.path.expanduser(config['dataset']['root_path'])
    num_classes = config['dataset']['num_classes']
    batch_size = config['pretrain'].get('batch_size', 128) # Use config or default

    print(f'Loading {dataset_name} from {root_path}')
    
    # We use the standard load_dataset from dataset_helper
    # But for centralized, we just need train and test loaders, not partitioned
    train_dataset, test_dataset = load_dataset(dataset_name, root_path, normalize_train=True, normalize_test=True)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model setup
    model_name = config['server']['model_name']
    print(f'Initializing {model_name} (Centralized) using SupCEResNet wrapper')
    
    from models_lib.resnet_big import SupCEResNet
    
    # We use SupCEResNet which encapsulates the backbone and adds a proper FC layer
    model = SupCEResNet(model_name, num_classes=num_classes)
    
    # SupCEResNet definition in resnet_big.py:
    # class SupCEResNet(nn.Module):
    #     def __init__(self, name='resnet50', num_classes=10):
    #         model_fun, dim_in = model_dict[name]
    #         self.encoder = model_fun()
    #         self.fc = nn.Linear(dim_in, num_classes)
    #     def forward(self, x):
    #         return self.fc(self.encoder(x))
    
    # This ensures we have a trainable classifier.
    
    model = model.to(device)

    # Optimization
    lr = config['pretrain'].get('lr', 0.01)
    momentum = config['pretrain'].get('momentum', 0.9)
    weight_decay = config['pretrain'].get('weight_decay', 5e-4)
    epochs = config['pretrain'].get('epoch', 50)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Save path
    save_dir = config['pretrain'].get('model_path', './pretrain')
    os.makedirs(save_dir, exist_ok=True)
    save_name = f"{dataset_name.lower()}_{model_name}_centralized.pth"
    save_path = os.path.join(save_dir, save_name)

    best_acc = 0

    print(f'Starting training for {epochs} epochs...')
    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, criterion, device, epoch)
        acc = test(model, test_loader, criterion, device)
        scheduler.step()

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
            print(f'Model saved to {save_path}')

    print(f'Training finished. Best Accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    main()
