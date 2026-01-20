#!/bin/bash

# ================================================================================
# GPU-Aware Experiment Launcher for Shared Server
# ================================================================================
# This script uses the Python GPU scheduler to run experiments efficiently
# on a shared server without interfering with other users' tasks
# ================================================================================

echo "========================================================"
echo "GPU-Aware Experiment Scheduler for Shared Server"
echo "========================================================"

# Create necessary directories
mkdir -p logs/multi_seed
mkdir -p logs/gamma_sensitivity
mkdir -p logs/lambda_max_sensitivity
mkdir -p logs/sigma_lr_sensitivity

# Configuration parameters
MIN_MEMORY=8000        # Minimum free GPU memory in MB (8GB)
MAX_UTIL=90            # Maximum GPU utilization percentage
MAX_CONCURRENT=16       # Maximum concurrent experiments (adjust based on server)

echo ""
echo "Configuration:"
echo "  Minimum free GPU memory: ${MIN_MEMORY} MB"
echo "  Maximum GPU utilization: ${MAX_UTIL}%"
echo "  Maximum concurrent jobs: ${MAX_CONCURRENT}"
echo ""

# Function to run experiment set
run_experiment_set() {
    local job_file=$1
    local description=$2
    
    echo "========================================================"
    echo "Starting: $description"
    echo "Job file: $job_file"
    echo "========================================================"
    
    python gpu_scheduler.py \
        --job-file "$job_file" \
        --min-memory "$MIN_MEMORY" \
        --max-util "$MAX_UTIL" \
        --max-concurrent "$MAX_CONCURRENT"
    
    echo ""
    echo "✓ Completed: $description"
    echo ""
}

# Parse command line arguments
EXPERIMENT_TYPE=${1:-"all"}

case $EXPERIMENT_TYPE in
    "multi_seed")
        run_experiment_set "jobs_multi_seed.yaml" "Multi-Seed Statistical Experiments"
        ;;
    
    "gamma")
        run_experiment_set "jobs_gamma_sensitivity.yaml" "Gamma Sensitivity Analysis"
        ;;
    
    "lambda_max")
        run_experiment_set "jobs_lambda_max_sensitivity.yaml" "Lambda_max Sensitivity Analysis"
        ;;
    
    "all")
        echo "Running all experiments sequentially..."
        echo ""
        
        # Run multi-seed experiments first (most important)
        run_experiment_set "jobs_multi_seed.yaml" "Multi-Seed Statistical Experiments"
        
        # Run gamma sensitivity (important for reviewer response)
        run_experiment_set "jobs_gamma_sensitivity.yaml" "Gamma Sensitivity Analysis"
        
        # Run lambda_max sensitivity
        run_experiment_set "jobs_lambda_max_sensitivity.yaml" "Lambda_max Sensitivity Analysis"
        
        echo "========================================================"
        echo "All experiments completed!"
        echo "========================================================"
        ;;
    
    *)
        echo "Usage: $0 [multi_seed|gamma|lambda_max|all]"
        echo ""
        echo "Options:"
        echo "  multi_seed    - Run multi-seed experiments only"
        echo "  gamma         - Run gamma sensitivity analysis only"
        echo "  lambda_max    - Run lambda_max sensitivity analysis only"
        echo "  all           - Run all experiments (default)"
        exit 1
        ;;
esac

echo ""
echo "Next steps:"
echo "1. Check experiment logs in logs/ directories"
echo "2. Run: python scripts/aggregate_multi_seed_results.py"
echo "3. Update paper with results"
