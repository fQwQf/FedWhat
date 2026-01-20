# GPU-Aware Experiment Scheduler for Shared Servers

## 概述

这个智能调度系统专为**共享GPU服务器**设计，能够：
- ✅ **动态检测GPU可用性** - 实时监控显存和利用率
- ✅ **队列管理** - 自动排队等待空闲GPU
- ✅ **避免冲突** - 不干扰其他用户的任务
- ✅ **资源优化** - 最大化GPU利用率

---

## 核心机制

### 1. GPU可用性检测
使用 `nvidia-smi` 实时检查：
- **显存占用**: 必须有≥7GB空闲显存 (由用户调整)
- **GPU利用率**: 必须≤30%利用率

### 2. 智能队列与启动冷却 (Launch Cooldown)
- 任务提交到队列后，依次启动。
- **启动冷却**: 每次成功启动一个任务后，调度器会进入 **10秒冷却期**。
- **理由**: 深度学习框架（如 PyTorch）启动时需要时间完成 GPU 显存分配。10秒冷却确保显存占用能真实反映在 `nvidia-smi` 中，防止调度器因为“信息滞后”在同一张卡上连续启动多个任务导致 OOM。

### 3. 参数调优
```bash
MIN_MEMORY=12000       # 提高到12GB (更保守)
MAX_UTIL=20            # 降低到20% (避免与其他任务冲突)
MAX_CONCURRENT=2       # 减少并发数
LAUNCH_DELAY=15        # 增加启动冷却时间 (单位:秒)
```

---

## 使用方法

### 基本用法

```bash
# 1. 上传所有文件到服务器
cd FedWhat/

# 2. 给脚本执行权限
chmod +x run_gpu_aware_experiments.sh

# 3. 运行实验
bash run_gpu_aware_experiments.sh all           # 运行所有实验
bash run_gpu_aware_experiments.sh multi_seed    # 只运行多种子实验
bash run_gpu_aware_experiments.sh gamma         # 只运行gamma敏感性
```

---

## 实验配置文件

### 1. Multi-Seed Experiments (`jobs_multi_seed.yaml`)
- **目的**: 运行3个随机种子 (42, 114514, 350234) 以报告 Mean ± Std
- 24个任务 (2 datasets × 4 algorithms × 3 seeds)
- 自动管理seed参数 (通过临时配置文件)
- 结果保存到 `logs/multi_seed/`

### 2. Gamma Sensitivity (`jobs_gamma_sensitivity.yaml`)
- 6个任务 (γ ∈ {0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1})
- 固定 λ_max=50
- 结果保存到 `logs/gamma_sensitivity/`

### 3. Lambda_max Sensitivity (`jobs_lambda_max_sensitivity.yaml`)
- 6个任务 (λ_max ∈ {20, 30, 50, 75, 100, 150})
- 固定 γ=0.001
- 结果保存到 `logs/lambda_max_sensitivity/`

---

## 监控和调试

### 查看实时日志
```bash
# 查看各个任务日志
tail -f logs/multi_seed/*.log

# 查看GPU状态
watch -n 5 nvidia-smi
```

---

## 文件清单
```
FedWhat/
├── gpu_scheduler.py                    # 核心调度器 (含Seed处理逻辑)
├── run_gpu_aware_experiments.sh        # Bash包装脚本
├── jobs_multi_seed.yaml                # 多种子配置 (42, 114514, 350234)
├── jobs_gamma_sensitivity.yaml         # Gamma敏感性配置
├── jobs_lambda_max_sensitivity.yaml    # Lambda_max敏感性配置
└── scripts/
    └── aggregate_multi_seed_results.py # 结果聚合脚本
```
