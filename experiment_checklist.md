# FedWhat 实验执行追踪表
# 使用方法：在完成每个实验后，在状态列标记 [完成] 或 [运行中]

## 实验总览
- 总实验数：24
- CIFAR-100：12个实验
- SVHN：12个实验
- 每组：4个算法 × 3个种子

---

## CIFAR-100 实验 (12个)

### Seed 42
- [ ] 1. FedAvg   - configs/CIFAR100_alpha0.05_seed42.yaml
- [ ] 2. FedETF   - configs/CIFAR100_alpha0.05_seed42.yaml
- [ ] 3. OursV7   - configs/CIFAR100_alpha0.05_seed42.yaml
- [ ] 4. OursV14  - configs/CIFAR100_alpha0.05_seed42.yaml

### Seed 114514
- [ ] 5. FedAvg   - configs/CIFAR100_alpha0.05_seed114514.yaml
- [ ] 6. FedETF   - configs/CIFAR100_alpha0.05_seed114514.yaml
- [ ] 7. OursV7   - configs/CIFAR100_alpha0.05_seed114514.yaml
- [ ] 8. OursV14  - configs/CIFAR100_alpha0.05_seed114514.yaml

### Seed 350234
- [ ] 9. FedAvg   - configs/CIFAR100_alpha0.05_seed350234.yaml
- [ ] 10. FedETF  - configs/CIFAR100_alpha0.05_seed350234.yaml
- [ ] 11. OursV7  - configs/CIFAR100_alpha0.05_seed350234.yaml
- [ ] 12. OursV14 - configs/CIFAR100_alpha0.05_seed350234.yaml

---

## SVHN 实验 (12个)

### Seed 42
- [ ] 13. FedAvg  - configs/SVHN_alpha0.05_seed42.yaml
- [ ] 14. FedETF  - configs/SVHN_alpha0.05_seed42.yaml
- [ ] 15. OursV7  - configs/SVHN_alpha0.05_seed42.yaml
- [ ] 16. OursV14 - configs/SVHN_alpha0.05_seed42.yaml

### Seed 114514
- [ ] 17. FedAvg  - configs/SVHN_alpha0.05_seed114514.yaml
- [ ] 18. FedETF  - configs/SVHN_alpha0.05_seed114514.yaml
- [ ] 19. OursV7  - configs/SVHN_alpha0.05_seed114514.yaml
- [ ] 20. OursV14 - configs/SVHN_alpha0.05_seed114514.yaml

### Seed 350234
- [ ] 21. FedAvg  - configs/SVHN_alpha0.05_seed350234.yaml
- [ ] 22. FedETF  - configs/SVHN_alpha0.05_seed350234.yaml
- [ ] 23. OursV7  - configs/SVHN_alpha0.05_seed350234.yaml
- [ ] 24. OursV14 - configs/SVHN_alpha0.05_seed350234.yaml

---

## 执行笔记

### 建议执行策略
1. **按数据集分批**：先完成SVHN（较快），再做CIFAR100（较慢）
2. **按seed分组**：同一seed的4个算法一起运行，便于对比
3. **避免高峰期**：在服务器使用率低时运行实验

### 预估时间（参考）
- SVHN 单个实验：~30-60分钟
- CIFAR-100 单个实验：~2-4小时
- 总计时间：约40-80小时（顺序执行）

### GPU分配记录
| GPU ID | 当前实验 | 状态 | 开始时间 |
|--------|---------|------|---------|
| GPU 0  |         |      |         |
| GPU 1  |         |      |         |
| GPU 2  |         |      |         |
| GPU 3  |         |      |         |
