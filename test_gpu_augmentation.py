"""
Quick verification script to test GPU augmentation implementation.
This script verifies that the GPU augmentation module works correctly.
"""

import torch
import sys
import os

# Direct imports to avoid loading full package
import importlib.util

# Load gpu_augmentation module directly
spec = importlib.util.spec_from_file_location(
    "gpu_augmentation",
    os.path.join(os.path.dirname(__file__), "oneshot_algorithms", "ours", "gpu_augmentation.py")
)
gpu_aug_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu_aug_module)
get_gpu_augmentation = gpu_aug_module.get_gpu_augmentation

# Manually define NORMALIZE_DICT to avoid importing dataset_helper
NORMALIZE_DICT = {
    'CIFAR10': {'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5)},
    'SVHN': {'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5)},
}

def test_gpu_augmentation():
    print("=" * 60)
    print("Testing GPU Augmentation Module")
    print("=" * 60)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("[WARNING] CUDA not available, using CPU for testing")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        print(f"[OK] Using device: {device}")
        print(f"     GPU: {torch.cuda.get_device_name(0)}")
    
    # Test with CIFAR10 parameters
    dataset_name = 'CIFAR10'
    print(f"\n[TEST] Dataset: {dataset_name}")
    
    # Create GPU augmentation module
    try:
        aug_module = get_gpu_augmentation(
            dataset_name=dataset_name,
            device=device
        )
        print("[OK] GPU augmentation module created successfully")
    except Exception as e:
        print(f"[FAIL] Failed to create augmentation module: {e}")
        return False
    
    # Create a dummy batch (batch_size=4, channels=3, height=32, width=32)
    batch_size = 4
    print(f"\n[TEST] Testing with batch size: {batch_size}")
    
    try:
        # Create random input on the correct device
        dummy_input = torch.rand(batch_size, 3, 32, 32, device=device)
        print(f"       Input shape: {dummy_input.shape}")
        
        # Apply augmentation
        with torch.no_grad():
            augmented_output1 = aug_module(dummy_input)
            augmented_output2 = aug_module(dummy_input)
        
        print(f"       Output1 shape: {augmented_output1.shape}")
        print(f"       Output2 shape: {augmented_output2.shape}")
        
        # Verify outputs are different (randomness check)
        if torch.allclose(augmented_output1, augmented_output2):
            print("[WARNING] Augmentations produced identical outputs (randomness may not be working)")
        else:
            print("[OK] Augmentations produce different outputs (randomness working)")
        
        # Verify output is on correct device
        if augmented_output1.device == device:
            print(f"[OK] Output is on correct device: {device}")
        else:
            print(f"[FAIL] Output device mismatch: expected {device}, got {augmented_output1.device}")
            return False
            
        # Verify shape is preserved
        if augmented_output1.shape == dummy_input.shape:
            print("[OK] Output shape matches input shape")
        else:
            print(f"[FAIL] Shape mismatch: input {dummy_input.shape}, output {augmented_output1.shape}")
            return False
        
    except Exception as e:
        print(f"[FAIL] Augmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test with SVHN (another dataset)
    print(f"\n[TEST] Testing with SVHN dataset...")
    try:
        svhn_aug = get_gpu_augmentation('SVHN', device)
        svhn_output = svhn_aug(dummy_input)
        print(f"[OK] SVHN augmentation works, output shape: {svhn_output.shape}")
    except Exception as e:
        print(f"[FAIL] SVHN test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("=" * 60)
    print("\nNext steps:")
    print("   1. Run a short training experiment to verify GPU utilization")
    print("   2. Monitor GPU usage with: nvidia-smi -l 1")
    print("   3. Compare training speed with previous CPU-based version")
    
    return True

if __name__ == "__main__":
    success = test_gpu_augmentation()
    sys.exit(0 if success else 1)
