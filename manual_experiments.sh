# FedWhat 手动实验执行命令清单
# 基于 jobs_multi_seed.yaml 的24个实验

## 使用说明
# 1. 手动选择一个空闲的GPU（用 nvidia-smi 查看）
# 2. 设置 CUDA_VISIBLE_DEVICES 环境变量
# 3. 复制对应实验的命令执行
# 4. 每次只运行一个实验，避免抢占资源

## 查看GPU状态命令
# nvidia-smi

## 创建日志目录
# mkdir -p logs/multi_seed

# ============================================================
# CIFAR-100 α=0.05 实验 (12个)
# ============================================================

## Seed 42 (4个算法)
# 1. CIFAR100_a005_FedAvg_seed42
CUDA_VISIBLE_DEVICES=0 python test.py --cfp configs/CIFAR100_alpha0.05_seed42.yaml --algo FedAvg > logs/multi_seed/CIFAR100_a005_FedAvg_seed42.log 2>&1

# 2. CIFAR100_a005_FedETF_seed42
CUDA_VISIBLE_DEVICES=0 python test.py --cfp configs/CIFAR100_alpha0.05_seed42.yaml --algo FedETF > logs/multi_seed/CIFAR100_a005_FedETF_seed42.log 2>&1

# 3. CIFAR100_a005_OursV7_seed42
CUDA_VISIBLE_DEVICES=0 python test.py --cfp configs/CIFAR100_alpha0.05_seed42.yaml --algo OursV7 > logs/multi_seed/CIFAR100_a005_OursV7_seed42.log 2>&1

# 4. CIFAR100_a005_OursV14_seed42
CUDA_VISIBLE_DEVICES=0 python test.py --cfp configs/CIFAR100_alpha0.05_seed42.yaml --algo OursV14 > logs/multi_seed/CIFAR100_a005_OursV14_seed42.log 2>&1


## Seed 114514 (4个算法)
# 5. CIFAR100_a005_FedAvg_seed114514
CUDA_VISIBLE_DEVICES=0 python test.py --cfp configs/CIFAR100_alpha0.05_seed114514.yaml --algo FedAvg > logs/multi_seed/CIFAR100_a005_FedAvg_seed114514.log 2>&1

# 6. CIFAR100_a005_FedETF_seed114514
CUDA_VISIBLE_DEVICES=0 python test.py --cfp configs/CIFAR100_alpha0.05_seed114514.yaml --algo FedETF > logs/multi_seed/CIFAR100_a005_FedETF_seed114514.log 2>&1

# 7. CIFAR100_a005_OursV7_seed114514
CUDA_VISIBLE_DEVICES=0 python test.py --cfp configs/CIFAR100_alpha0.05_seed114514.yaml --algo OursV7 > logs/multi_seed/CIFAR100_a005_OursV7_seed114514.log 2>&1

# 8. CIFAR100_a005_OursV14_seed114514
CUDA_VISIBLE_DEVICES=0 python test.py --cfp configs/CIFAR100_alpha0.05_seed114514.yaml --algo OursV14 > logs/multi_seed/CIFAR100_a005_OursV14_seed114514.log 2>&1


## Seed 350234 (4个算法)
# 9. CIFAR100_a005_FedAvg_seed350234
CUDA_VISIBLE_DEVICES=0 python test.py --cfp configs/CIFAR100_alpha0.05_seed350234.yaml --algo FedAvg > logs/multi_seed/CIFAR100_a005_FedAvg_seed350234.log 2>&1

# 10. CIFAR100_a005_FedETF_seed350234
CUDA_VISIBLE_DEVICES=0 python test.py --cfp configs/CIFAR100_alpha0.05_seed350234.yaml --algo FedETF > logs/multi_seed/CIFAR100_a005_FedETF_seed350234.log 2>&1

# 11. CIFAR100_a005_OursV7_seed350234
CUDA_VISIBLE_DEVICES=0 python test.py --cfp configs/CIFAR100_alpha0.05_seed350234.yaml --algo OursV7 > logs/multi_seed/CIFAR100_a005_OursV7_seed350234.log 2>&1

# 12. CIFAR100_a005_OursV14_seed350234
CUDA_VISIBLE_DEVICES=0 python test.py --cfp configs/CIFAR100_alpha0.05_seed350234.yaml --algo OursV14 > logs/multi_seed/CIFAR100_a005_OursV14_seed350234.log 2>&1


# ============================================================
# SVHN α=0.05 实验 (12个)
# ============================================================

## Seed 42 (4个算法)
# 13. SVHN_a005_FedAvg_seed42
CUDA_VISIBLE_DEVICES=0 python test.py --cfp configs/SVHN_alpha0.05_seed42.yaml --algo FedAvg > logs/multi_seed/SVHN_a005_FedAvg_seed42.log 2>&1

# 14. SVHN_a005_FedETF_seed42
CUDA_VISIBLE_DEVICES=0 python test.py --cfp configs/SVHN_alpha0.05_seed42.yaml --algo FedETF > logs/multi_seed/SVHN_a005_FedETF_seed42.log 2>&1

# 15. SVHN_a005_OursV7_seed42
CUDA_VISIBLE_DEVICES=0 python test.py --cfp configs/SVHN_alpha0.05_seed42.yaml --algo OursV7 > logs/multi_seed/SVHN_a005_OursV7_seed42.log 2>&1

# 16. SVHN_a005_OursV14_seed42
CUDA_VISIBLE_DEVICES=0 python test.py --cfp configs/SVHN_alpha0.05_seed42.yaml --algo OursV14 > logs/multi_seed/SVHN_a005_OursV14_seed42.log 2>&1


## Seed 114514 (4个算法)
# 17. SVHN_a005_FedAvg_seed114514
CUDA_VISIBLE_DEVICES=0 python test.py --cfp configs/SVHN_alpha0.05_seed114514.yaml --algo FedAvg > logs/multi_seed/SVHN_a005_FedAvg_seed114514.log 2>&1

# 18. SVHN_a005_FedETF_seed114514
CUDA_VISIBLE_DEVICES=0 python test.py --cfp configs/SVHN_alpha0.05_seed114514.yaml --algo FedETF > logs/multi_seed/SVHN_a005_FedETF_seed114514.log 2>&1

# 19. SVHN_a005_OursV7_seed114514
CUDA_VISIBLE_DEVICES=0 python test.py --cfp configs/SVHN_alpha0.05_seed114514.yaml --algo OursV7 > logs/multi_seed/SVHN_a005_OursV7_seed114514.log 2>&1

# 20. SVHN_a005_OursV14_seed114514
CUDA_VISIBLE_DEVICES=0 python test.py --cfp configs/SVHN_alpha0.05_seed114514.yaml --algo OursV14 > logs/multi_seed/SVHN_a005_OursV14_seed114514.log 2>&1


## Seed 350234 (4个算法)
# 21. SVHN_a005_FedAvg_seed350234
CUDA_VISIBLE_DEVICES=0 python test.py --cfp configs/SVHN_alpha0.05_seed350234.yaml --algo FedAvg > logs/multi_seed/SVHN_a005_FedAvg_seed350234.log 2>&1

# 22. SVHN_a005_FedETF_seed350234
CUDA_VISIBLE_DEVICES=0 python test.py --cfp configs/SVHN_alpha0.05_seed350234.yaml --algo FedETF > logs/multi_seed/SVHN_a005_FedETF_seed350234.log 2>&1

# 23. SVHN_a005_OursV7_seed350234
CUDA_VISIBLE_DEVICES=0 python test.py --cfp configs/SVHN_alpha0.05_seed350234.yaml --algo OursV7 > logs/multi_seed/SVHN_a005_OursV7_seed350234.log 2>&1

# 24. SVHN_a005_OursV14_seed350234
CUDA_VISIBLE_DEVICES=0 python test.py --cfp configs/SVHN_alpha0.05_seed350234.yaml --algo OursV14 > logs/multi_seed/SVHN_a005_OursV14_seed350234.log 2>&1


# ============================================================
# 使用技巧
# ============================================================

# 1. 如果想在后台运行并保留终端（推荐）：
# nohup bash -c "CUDA_VISIBLE_DEVICES=0 python test.py --cfp configs/CIFAR100_alpha0.05_seed42.yaml --algo FedAvg" > logs/multi_seed/CIFAR100_a005_FedAvg_seed42.log 2>&1 &

# 2. 查看运行中的实验进程：
# ps aux | grep test.py

# 3. 实时查看日志：
# tail -f logs/multi_seed/CIFAR100_a005_FedAvg_seed42.log

# 4. 监控GPU使用情况：
# watch -n 1 nvidia-smi

# 5. 如果需要指定多个GPU（例如GPU 1和2）：
# CUDA_VISIBLE_DEVICES=1,2 python test.py --cfp configs/XXX.yaml --algo YYY

# ============================================================
# 建议执行顺序（避免资源冲突）
# ============================================================
# 
# 第一批（较短的SVHN实验）：
#   - 先运行 SVHN seed42 的4个算法
#   - 等全部完成后，运行 SVHN seed114514 的4个算法
#   - 最后运行 SVHN seed350234 的4个算法
#
# 第二批（较长的CIFAR100实验）：
#   - 同样按 seed 分批运行
#
# 或者：
#   - 如果有多个GPU，可以在不同GPU上并行运行不同seed的实验
#   - 例如：GPU0运行seed42，GPU1运行seed114514
