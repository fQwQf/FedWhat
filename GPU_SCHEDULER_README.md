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
- **显存占用**: 必须有≥8GB空闲显存
- **GPU利用率**: 必须≤30%利用率

### 2. 智能队列
- 任务提交到队列
- 自动等待GPU可用
- 完成后释放GPU给下一个任务

### 3. 参数调优
```python
MIN_MEMORY = 8000      # 最小空闲显存(MB) - 根据实验需求调整
MAX_UTIL = 30          # 最大GPU利用率(%) - 更保守可设为20%
MAX_CONCURRENT = 4     # 最大并发任务数 - 根据服务器GPU数量调整
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

### 直接使用Python调度器

```bash
# 运行多种子实验
python gpu_scheduler.py \
    --job-file jobs_multi_seed.yaml \
    --min-memory 8000 \
    --max-util 30 \
    --max-concurrent 4

# 运行gamma敏感性分析
python gpu_scheduler.py \
    --job-file jobs_gamma_sensitivity.yaml \
    --min-memory 8000 \
    --max-util 30
```

---

## 工作流程示意

```
[Job Queue]               [GPU Monitor]              [GPU Pool]
   ├─ Job1                     │                      GPU0: Free (16GB)
   ├─ Job2          ──────────►│◄────────────────     GPU1: Busy (2GB)
   ├─ Job3                     │                      GPU2: Free (12GB)
   └─ Job4                     │                      GPU3: Busy (0.5GB)
                               │
                               ▼
                      [Find available GPU]
                       (Free memory ≥ 8GB
                        Utilization ≤ 30%)
                               │
                               ▼
                        Job1 → GPU0 ✓
                        Job2 → Wait...
                        Job3 → GPU2 ✓
                        Job4 → Wait...
```

---

## 实验配置文件

### 1. Multi-Seed Experiments (`jobs_multi_seed.yaml`)
- 24个任务 (2 datasets × 4 algorithms × 3 seeds)
- 自动管理seed参数
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
# 查看调度器日志
tail -f logs/multi_seed/*.log

# 查看GPU状态
watch -n 5 nvidia-smi

# 查看正在运行的Python进程
ps aux | grep test.py
```

### 调整参数应对高负载服务器

如果服务器非常繁忙，可以修改 `run_gpu_aware_experiments.sh`:

```bash
MIN_MEMORY=12000       # 提高到12GB (更保守)
MAX_UTIL=20            # 降低到20% (避免与其他任务冲突)
MAX_CONCURRENT=2       # 减少并发数
```

---

## 常见问题

### Q1: 所有GPU都被占用怎么办？
**A**: 调度器会自动等待，每60秒检查一次。默认最长等待1小时，如需延长：

修改 `gpu_scheduler.py`:
```python
def wait_for_available_gpu(..., max_wait_time: int = 7200):  # 延长到2小时
```

### Q2: 如何优先运行某些实验？
**A**: 修改 `.yaml` 文件中任务的顺序，队列按从上到下的顺序执行。

### Q3: 实验失败了会重试吗？
**A**: 当前版本不会自动重试。如果需要重试，可以：

1. 检查日志找出失败原因
2. 删除对应的日志文件
3. 重新运行调度器（会跳过已完成的实验）

### Q4: 如何修改seed参数的传递方式？
**A**: 如果 `test.py` 需要从config读取seed而不是命令行参数，需要修改 `gpu_scheduler.py` 的 `run_experiment()` 函数，在运行前修改config文件。

---

## 性能评估

### 传统Bash脚本 vs GPU调度器

| 方面 | 传统Bash | GPU调度器 |
|------|----------|-----------|
| GPU冲突 | ❌ 可能占用他人GPU | ✅ 自动检测避免冲突 |
| 资源利用 | ⚠️ 固定分配，可能浪费 | ✅ 动态分配，最大化利用 |
| 容错性 | ❌ 一个失败全部中断 | ✅ 独立任务，单个失败不影响 |
| 监控 | ⚠️ 需手动检查 | ✅ 自动记录状态日志 |
| 扩展性 | ❌ 难以修改 | ✅ 配置文件驱动，易扩展 |

---

## 文件清单

```
FedWhat/
├── gpu_scheduler.py                    # 核心调度器
├── run_gpu_aware_experiments.sh        # Bash包装脚本
├── jobs_multi_seed.yaml                # 多种子实验配置
├── jobs_gamma_sensitivity.yaml         # Gamma敏感性配置
├── jobs_lambda_max_sensitivity.yaml    # Lambda_max敏感性配置
└── scripts/
    └── aggregate_multi_seed_results.py # 结果聚合脚本
```

---

## 进阶配置

### 自定义GPU选择策略

修改 `find_available_gpu()` 函数可以实现：
- 优先选择显存最多的GPU
- 优先选择利用率最低的GPU
- 轮询策略避免总是选择GPU0

示例（优先选择显存最多）:
```python
def find_available_gpu(min_free_memory_mb: int = 8000, max_utilization: int = 30):
    gpu_info = get_gpu_memory_usage()
    
    # Sort by free memory (descending)
    valid_gpus = [
        gpu for gpu in gpu_info
        if (gpu['memory_total'] - gpu['memory_used']) >= min_free_memory_mb
        and gpu['utilization'] <= max_utilization
    ]
    
    if not valid_gpus:
        return -1
    
    # Return GPU with most free memory
    best_gpu = max(valid_gpus, key=lambda g: g['memory_total'] - g['memory_used'])
    return best_gpu['gpu_id']
```

---

## 总结

这个GPU调度系统通过**动态资源检测**和**智能队列管理**，确保：
1. ✅ 不干扰其他用户任务（检查利用率和显存）
2. ✅ 最大化自己的实验吞吐（自动填充空闲GPU）
3. ✅ 容错和可扩展（任务独立，配置文件驱动）

适合在**多用户共享服务器**上运行大规模实验。
