# AURORA

This repository contains the official implementation for the paper: **"AURORA: Autonomous Regularization for One-shot Representation Alignment"**.

## Code Structure

- `configs/`: Configuration files for experiments.
- `models_lib/`: Model definitions (ResNet, ViT, etc.).
- `oneshot_algorithms/`: Implementation of AURORA and baseline algorithms.
- `data/`: Dataset handling scripts.
- `examples.sh`: Example commands for single-run experiments.
- `run_gpu_aware_experiments.sh`: Main script for distributed/batch experiments.

## Installation

Please install the required dependencies using the provided environment file:

```bash
conda env create -f environment.yml
```

## Usage

```bash
python test.py --cfp ./configs/CIFAR10.yaml --algo OursV14
```

### Arguments

The `test.py` script accepts the following arguments:

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--cfp` | `str` | **Required** | Path to the experiment configuration file (YAML). |
| `--algo` | `str` | **Required** | Algorithm to run. <br> **Key Options**: `OursV14` (AURORA), `FedAvg`, `FedProto`, `Ours_FeatureCollapse_Ablation`. |
| `--lambdaval` | `float` | `0` | Alignment loss weight ($\lambda$). |
| `--annealing_strategy` | `str` | `none` | Strategy for annealing $\lambda$ during training. |
| `--gamma_reg` | `float` | `1e-5` | Regularization factor ($\gamma$) for the alignment loss weight. |
| `--lambda_max` | `float` | `50.0` | Maximum threshold for effective $\lambda$ in stability regularization. |