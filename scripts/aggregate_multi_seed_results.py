#!/usr/bin/env python3
"""
Aggregate Multi-Seed Experiment Results for Statistical Reporting
Generates Mean ± Std tables for the AURORA paper
"""

import os
import re
import yaml
import numpy as np
from pathlib import Path
from collections import defaultdict

def parse_log_file(log_path):
    """Extract final test accuracy from log file"""
    try:
        with open(log_path, 'r') as f:
            content = f.read()
            
        # Try multiple patterns to find accuracy
        patterns = [
            r'The test accuracy of .+?: ([\d.]+)',
            r'Test Accuracy: ([\d.]+)',
            r'Final Accuracy: ([\d.]+)%',
            r'Accuracy: ([\d.]+)%'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            if matches:
                # Return the last (final) accuracy
                return float(matches[-1])
        
        print(f"Warning: Could not find accuracy in {log_path}")
        return None
        
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return None

def main():
    log_dir = Path("logs/multi_seed")
    
    if not log_dir.exists():
        print(f"Log directory {log_dir} does not exist!")
        return
    
    # Structure: results[dataset][algorithm] = [acc1, acc2, acc3]
    results = defaultdict(lambda: defaultdict(list))
    
    # Parse all log files
    for log_file in log_dir.glob("*.log"):
        filename = log_file.stem  # e.g., CIFAR100_a005_OursV14_seed42
        
        # Parse filename: {dataset}_{algo}_seed{seed}
        match = re.match(r'(.+?)_([A-Za-z0-9]+)_seed(\d+)', filename)
        if not match:
            print(f"Skipping unrecognized filename: {filename}")
            continue
        
        dataset, algo, seed = match.groups()
        
        accuracy = parse_log_file(log_file)
        if accuracy is not None:
            results[dataset][algo].append(accuracy)
            print(f"✓ {dataset} - {algo} - Seed {seed}: {accuracy:.2f}%")
    
    # Generate summary tables
    print("\n" + "="*80)
    print("SUMMARY: Mean ± Std Accuracy (%)")
    print("="*80)
    
    for dataset in sorted(results.keys()):
        print(f"\n{dataset}:")
        print("-" * 60)
        print(f"{'Algorithm':<20} {'Mean ± Std':<20} {'Individual Runs'}")
        print("-" * 60)
        
        for algo in sorted(results[dataset].keys()):
            accs = np.array(results[dataset][algo])
            
            if len(accs) == 0:
                print(f"{algo:<20} NO DATA")
                continue
            
            mean = np.mean(accs)
            std = np.std(accs, ddof=1) if len(accs) > 1 else 0.0
            
            accs_str = ", ".join([f"{a:.2f}" for a in accs])
            print(f"{algo:<20} {mean:.2f} ± {std:.2f}      [{accs_str}]")
    
    # Generate LaTeX table
    print("\n" + "="*80)
    print("LaTeX Table Format:")
    print("="*80)
    
    for dataset in sorted(results.keys()):
        print(f"\n% {dataset}")
        print("\\begin{tabular}{lc}")
        print("\\toprule")
        print("\\textbf{Method} & \\textbf{Accuracy (\\%)} \\\\")
        print("\\midrule")
        
        for algo in ['FedAvg', 'FedETF', 'OursV7', 'OursV14']:
            if algo not in results[dataset]:
                continue
            
            accs = np.array(results[dataset][algo])
            mean = np.mean(accs)
            std = np.std(accs, ddof=1) if len(accs) > 1 else 0.0
            
            # Map algorithm names to paper names
            algo_map = {
                'FedAvg': 'FedAvg',
                'FedETF': 'FAFI',
                'OursV7': 'FAFI+Ann.',
                'OursV14': 'AURORA'
            }
            
            display_name = algo_map.get(algo, algo)
            print(f"{display_name} & ${mean:.2f} \\pm {std:.2f}$ \\\\")
        
        print("\\bottomrule")
        print("\\end{tabular}")
    
    print("\n" + "="*80)
    print("Aggregation Complete!")
    print("="*80)

if __name__ == "__main__":
    main()
