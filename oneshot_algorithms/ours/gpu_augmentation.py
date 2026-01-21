"""
GPU-based data augmentation using Kornia for improved training performance.
This module replaces CPU-based torchvision transforms with GPU-accelerated operations.
"""

import torch
import torch.nn as nn
import kornia.augmentation as K
from dataset_helper import NORMALIZE_DICT


class GPUSupConAugmentation(nn.Module):
    """
    GPU-based augmentation module for supervised contrastive learning.
    Replicates the CPU transforms from get_supcon_transform() using Kornia.
    """
    
    def __init__(self, dataset_name='CIFAR10'):
        super(GPUSupConAugmentation, self).__init__()
        
        # Determine image size based on dataset
        if dataset_name in ['CIFAR10', 'CIFAR100', 'SVHN']:
            img_size = 32
        else:
            img_size = 64
        
        # Get normalization parameters
        if dataset_name in NORMALIZE_DICT:
            mean = NORMALIZE_DICT[dataset_name]['mean']
            std = NORMALIZE_DICT[dataset_name]['std']
        else:
            # Default to 0.5 if not found (safer fallback)
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)

        # Build GPU augmentation pipeline
        # Kornia augmentations are applied sequentially
        self.augmentation = nn.Sequential(
            # RandomResizedCrop equivalent
            K.RandomResizedCrop(size=(img_size, img_size), scale=(0.2, 1.0), p=1.0),
            
            # RandomHorizontalFlip
            K.RandomHorizontalFlip(p=0.5),
            
            # ColorJitter with p=0.8 (applied directly, no RandomApply needed)
            K.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1,
                p=0.8  # 80% probability of applying color jitter
            ),
            
            # RandomGrayscale
            K.RandomGrayscale(p=0.2),
            
            # Normalize at the END of the pipeline (data enters as [0, 1])
            K.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        )
    
    def forward(self, x):
        """
        Apply augmentation to input batch.
        
        Args:
            x: Input tensor of shape (B, C, H, W), already on GPU
            
        Returns:
            Augmented tensor of same shape
        """
        return self.augmentation(x)


def get_gpu_augmentation(dataset_name, device):
    """
    Factory function to create GPU augmentation module.
    
    Args:
        dataset_name: Name of the dataset ('CIFAR10', 'CIFAR100', 'SVHN', etc.)
        device: torch.device to place the augmentation module on
        
    Returns:
        GPUSupConAugmentation module on specified device
    """
    aug_module = GPUSupConAugmentation(dataset_name)
    return aug_module.to(device)
