#!/bin/bash

# ================================================================================
# Lambda_max Sensitivity Analysis for AURORA Paper Review Response
# ================================================================================
# Purpose: Test sensitivity of λ_max threshold parameter
# Target: SVHN α=0.05 (extreme non-IID)
# Lambda_max values: [20, 30, 50, 75, 100, 150]
# Fixed: γ=0.001 (default)
# ================================================================================

echo "========================================================"
echo "Starting Lambda_max Sensitivity Analysis on SVHN α=0.05"
echo "========================================================"

# Create logs directory
mkdir -p logs/lambda_max_sensitivity

# Define lambda_max values to test
declare -a lambda_max_values=(
    "20"
    "30"
    "50"
    "75"
    "100"
    "150"
)

# Base configuration
CONFIG="configs/SVHN_alpha0.05.yaml"
ALGO="OursV14"
GAMMA="0.001"  # Default gamma value

job_id=0

for lambda_max in "${lambda_max_values[@]}"; do
    gpu_id=$((job_id % 8))
    
    echo "---------------------------------------------------"
    echo "Job ${job_id}: SVHN α=0.05 - AURORA - λ_max=${lambda_max}"
    echo "GPU: ${gpu_id}"
    
    # Run experiment
    CUDA_VISIBLE_DEVICES=$gpu_id nohup python test.py \
        --cfp "$CONFIG" \
        --algo "$ALGO" \
        --gamma_reg "$GAMMA" \
        --lambda_max "$lambda_max" \
        > "logs/lambda_max_sensitivity/SVHN_a005_lambdamax${lambda_max}.log" 2>&1 &
    
    job_id=$((job_id + 1))
    sleep 2
done

echo "========================================================"
echo "All ${job_id} lambda_max sensitivity experiments launched!"
echo "Check logs/lambda_max_sensitivity/*.log for progress"
echo "========================================================"
