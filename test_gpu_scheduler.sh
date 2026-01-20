#!/bin/bash

# Quick test of GPU scheduler with a single job
# This validates the seed handling mechanism

echo "Testing GPU Scheduler with single multi-seed job..."

# Create minimal test job file
cat > test_job.yaml << 'EOF'
jobs:
  - name: "Test_CIFAR100_seed42"
    config: "configs/CIFAR100_alpha0.05.yaml"
    algorithm: "FedAvg"
    log_path: "logs/test_seed42.log"
    extra_args:
      seed: 42
EOF

echo "Running test job..."
python gpu_scheduler.py \
    --job-file test_job.yaml \
    --min-memory 7000 \
    --max-util 50 \
    --max-concurrent 1

echo ""
echo "Check the log: tail logs/test_seed42.log"
echo "Check temp config created: ls -la configs/temp/"
