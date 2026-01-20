#!/bin/bash

# Configuration
# MAX_GPUS=8

echo "Starting distributed experiments on up to 8 GPUs..."

# Define experiment commands
declare -a commands=(
    "python test.py --cfp configs/CIFAR100_alpha0.05_sigma_lr_0.001.yaml --algo OursV14"
    "python test.py --cfp configs/CIFAR100_alpha0.05.yaml --algo OursV14" # Default sigma_lr=0.005
    "python test.py --cfp configs/CIFAR100_alpha0.05_sigma_lr_0.01.yaml --algo OursV14"
    "python test.py --cfp configs/CIFAR100_alpha0.05.yaml --algo OursV14 --lambda_max 20"
    "python test.py --cfp configs/CIFAR100_alpha0.05.yaml --algo OursV14 --lambda_max 50" # Default lambda_max=50
    "python test.py --cfp configs/CIFAR100_alpha0.05.yaml --algo OursV14 --lambda_max 100"
)

# Loop through commands and distribute across GPUs
for i in "${!commands[@]}"; do
    cmd="${commands[$i]}"
    gpu_id=$((i % 8))  # Modulo 8 to distribute across GPUs 0-7
    
    echo "---------------------------------------------------"
    echo "Launching Job $i on GPU $gpu_id"
    echo "Command: $cmd"
    
    # Run in background with nohup, redirecting output
    CUDA_VISIBLE_DEVICES=$gpu_id nohup $cmd > "logs/exp_job_${i}.log" 2>&1 &
    
    # Optional: small sleep to avoid race conditions on file creation
    sleep 2
done

echo "---------------------------------------------------"
echo "All 6 experiments have been launched in the background."
echo "Check 'logs/exp_job_*.log' for progress."
echo "Use 'nvidia-smi' to monitor GPU usage and 'ps aux | grep test.py' to monitor processes."
