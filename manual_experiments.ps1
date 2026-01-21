# FedWhat 手动实验执行命令 - Windows PowerShell 版本
# 共24个实验：CIFAR-100 (12) + SVHN (12)

# ============================================================
# 准备工作
# ============================================================

# 创建日志目录
New-Item -ItemType Directory -Force -Path "logs\multi_seed"

# 查看GPU状态
nvidia-smi


# ============================================================
# CIFAR-100 实验 (12个)
# ============================================================

# --- Seed 42 ---
$env:CUDA_VISIBLE_DEVICES="0"; python test.py --cfp configs/CIFAR100_alpha0.05_seed42.yaml --algo FedAvg > logs/multi_seed/CIFAR100_a005_FedAvg_seed42.log 2>&1

$env:CUDA_VISIBLE_DEVICES="0"; python test.py --cfp configs/CIFAR100_alpha0.05_seed42.yaml --algo FedETF > logs/multi_seed/CIFAR100_a005_FedETF_seed42.log 2>&1

$env:CUDA_VISIBLE_DEVICES="0"; python test.py --cfp configs/CIFAR100_alpha0.05_seed42.yaml --algo OursV7 > logs/multi_seed/CIFAR100_a005_OursV7_seed42.log 2>&1

$env:CUDA_VISIBLE_DEVICES="0"; python test.py --cfp configs/CIFAR100_alpha0.05_seed42.yaml --algo OursV14 > logs/multi_seed/CIFAR100_a005_OursV14_seed42.log 2>&1


# --- Seed 114514 ---
$env:CUDA_VISIBLE_DEVICES="0"; python test.py --cfp configs/CIFAR100_alpha0.05_seed114514.yaml --algo FedAvg > logs/multi_seed/CIFAR100_a005_FedAvg_seed114514.log 2>&1

$env:CUDA_VISIBLE_DEVICES="0"; python test.py --cfp configs/CIFAR100_alpha0.05_seed114514.yaml --algo FedETF > logs/multi_seed/CIFAR100_a005_FedETF_seed114514.log 2>&1

$env:CUDA_VISIBLE_DEVICES="0"; python test.py --cfp configs/CIFAR100_alpha0.05_seed114514.yaml --algo OursV7 > logs/multi_seed/CIFAR100_a005_OursV7_seed114514.log 2>&1

$env:CUDA_VISIBLE_DEVICES="0"; python test.py --cfp configs/CIFAR100_alpha0.05_seed114514.yaml --algo OursV14 > logs/multi_seed/CIFAR100_a005_OursV14_seed114514.log 2>&1


# --- Seed 350234 ---
$env:CUDA_VISIBLE_DEVICES="0"; python test.py --cfp configs/CIFAR100_alpha0.05_seed350234.yaml --algo FedAvg > logs/multi_seed/CIFAR100_a005_FedAvg_seed350234.log 2>&1

$env:CUDA_VISIBLE_DEVICES="0"; python test.py --cfp configs/CIFAR100_alpha0.05_seed350234.yaml --algo FedETF > logs/multi_seed/CIFAR100_a005_FedETF_seed350234.log 2>&1

$env:CUDA_VISIBLE_DEVICES="0"; python test.py --cfp configs/CIFAR100_alpha0.05_seed350234.yaml --algo OursV7 > logs/multi_seed/CIFAR100_a005_OursV7_seed350234.log 2>&1

$env:CUDA_VISIBLE_DEVICES="0"; python test.py --cfp configs/CIFAR100_alpha0.05_seed350234.yaml --algo OursV14 > logs/multi_seed/CIFAR100_a005_OursV14_seed350234.log 2>&1


# ============================================================
# SVHN 实验 (12个)
# ============================================================

# --- Seed 42 ---
$env:CUDA_VISIBLE_DEVICES="0"; python test.py --cfp configs/SVHN_alpha0.05_seed42.yaml --algo FedAvg > logs/multi_seed/SVHN_a005_FedAvg_seed42.log 2>&1

$env:CUDA_VISIBLE_DEVICES="0"; python test.py --cfp configs/SVHN_alpha0.05_seed42.yaml --algo FedETF > logs/multi_seed/SVHN_a005_FedETF_seed42.log 2>&1

$env:CUDA_VISIBLE_DEVICES="0"; python test.py --cfp configs/SVHN_alpha0.05_seed42.yaml --algo OursV7 > logs/multi_seed/SVHN_a005_OursV7_seed42.log 2>&1

$env:CUDA_VISIBLE_DEVICES="0"; python test.py --cfp configs/SVHN_alpha0.05_seed42.yaml --algo OursV14 > logs/multi_seed/SVHN_a005_OursV14_seed42.log 2>&1


# --- Seed 114514 ---
$env:CUDA_VISIBLE_DEVICES="0"; python test.py --cfp configs/SVHN_alpha0.05_seed114514.yaml --algo FedAvg > logs/multi_seed/SVHN_a005_FedAvg_seed114514.log 2>&1

$env:CUDA_VISIBLE_DEVICES="0"; python test.py --cfp configs/SVHN_alpha0.05_seed114514.yaml --algo FedETF > logs/multi_seed/SVHN_a005_FedETF_seed114514.log 2>&1

$env:CUDA_VISIBLE_DEVICES="0"; python test.py --cfp configs/SVHN_alpha0.05_seed114514.yaml --algo OursV7 > logs/multi_seed/SVHN_a005_OursV7_seed114514.log 2>&1

$env:CUDA_VISIBLE_DEVICES="0"; python test.py --cfp configs/SVHN_alpha0.05_seed114514.yaml --algo OursV14 > logs/multi_seed/SVHN_a005_OursV14_seed114514.log 2>&1


# --- Seed 350234 ---
$env:CUDA_VISIBLE_DEVICES="0"; python test.py --cfp configs/SVHN_alpha0.05_seed350234.yaml --algo FedAvg > logs/multi_seed/SVHN_a005_FedAvg_seed350234.log 2>&1

$env:CUDA_VISIBLE_DEVICES="0"; python test.py --cfp configs/SVHN_alpha0.05_seed350234.yaml --algo FedETF > logs/multi_seed/SVHN_a005_FedETF_seed350234.log 2>&1

$env:CUDA_VISIBLE_DEVICES="0"; python test.py --cfp configs/SVHN_alpha0.05_seed350234.yaml --algo OursV7 > logs/multi_seed/SVHN_a005_OursV7_seed350234.log 2>&1

$env:CUDA_VISIBLE_DEVICES="0"; python test.py --cfp configs/SVHN_alpha0.05_seed350234.yaml --algo OursV14 > logs/multi_seed/SVHN_a005_OursV14_seed350234.log 2>&1


# ============================================================
# 实用命令
# ============================================================

# 实时监控GPU (每秒刷新)
# nvidia-smi -l 1

# 查看运行中的Python进程
# Get-Process python

# 实时查看日志文件
# Get-Content logs/multi_seed/CIFAR100_a005_FedAvg_seed42.log -Wait -Tail 50

# 如果要使用不同GPU，修改数字：
# $env:CUDA_VISIBLE_DEVICES="1"  # 使用GPU 1
# $env:CUDA_VISIBLE_DEVICES="2"  # 使用GPU 2
