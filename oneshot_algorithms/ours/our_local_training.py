from oneshot_algorithms.utils import init_optimizer, init_loss_fn, test_acc, save_best_local_model
from oneshot_algorithms.ours.unsupervised_loss import SupConLoss, Contrastive_proto_feature_loss, Contrastive_proto_loss

from common_libs import *



def ours_local_training(model, training_data, test_dataloader, start_epoch, local_epochs, optim_name, lr, momentum, loss_name, device, num_classes, sample_per_class, aug_transformer, client_model_dir, total_rounds, save_freq=1, use_drcl=False, fixed_anchors=None, lambda_align=1.0, use_progressive_alignment=False, initial_protos=None, use_uncertainty_weighting=False, sigma_lr=None, annealing_factor=1.0, use_dynamic_task_attenuation=False, gamma_reg=0, lambda_max=50.0, force_feature_alignment=False):
   
    model.train()
    model.to(device)

    if sigma_lr is None:
        sigma_lr = 0.05 * lr # sigma 的学习率设为基础学习率的 0.05 倍

    if use_uncertainty_weighting:
        # For V10, we create a special optimizer with two parameter groups.
        # This allows us to set a much smaller learning rate for the sigma parameters.
        
        # 1. Identify the sigma parameters
        sigma_params = [
            model.log_sigma_sq_local,
            model.log_sigma_sq_align
        ]
        sigma_param_ids = {id(p) for p in sigma_params}

        # 2. Identify all other model parameters (the "base" parameters)
        base_params = [p for p in model.parameters() if id(p) not in sigma_param_ids]
        
        # 3. Define the two parameter groups with different learning rates
        param_groups = [
            {'params': base_params},  # Uses the default learning rate `lr`
            {'params': sigma_params, 'lr': sigma_lr}  # Uses the special, smaller `sigma_lr`
        ]
        
        # 4. Create the optimizer
        # We assume 'sgd' as per your config, but you can adapt this if needed.
        if optim_name.lower() == 'sgd':
            optimizer = torch.optim.SGD(param_groups, lr=lr, momentum=momentum)
        else:
            # Fallback for other optimizers like Adam
            logger.warning(f"Creating Adam optimizer for V10 with custom sigma_lr. Check if this is intended.")
            optimizer = torch.optim.Adam(param_groups, lr=lr)
            
        logger.info(f"V10 mode: Optimizer created with base_lr={lr} and sigma_lr={sigma_lr}")

    else:
        # For all other versions (V4-V9), use the original optimizer.
        optimizer = init_optimizer(model, optim_name, lr, momentum)

    cls_loss_fn = torch.nn.CrossEntropyLoss()
    contrastive_loss_fn = SupConLoss(temperature=0.07)
    con_proto_feat_loss_fn = Contrastive_proto_feature_loss(temperature=1.0)
    con_proto_loss_fn = Contrastive_proto_loss(temperature=1.0)

    # 如果使用DRCL，定义对齐损失函数
    if use_drcl or use_progressive_alignment:
        alignment_loss_fn = torch.nn.MSELoss()

    initial_lambda = lambda_align

    total_training_steps = total_rounds * local_epochs

    for e in range(start_epoch, start_epoch + local_epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(training_data):
            
            
            data, target = data.to(device), target.to(device)
            aug_data1, aug_data2 = aug_transformer(data), aug_transformer(data)
            aug_data = torch.cat([aug_data1, aug_data2], dim=0)
            bsz = target.shape[0]

            optimizer.zero_grad()
            
            logits, feature_norm = model(aug_data)

            # classification loss
            aug_labels = torch.cat([target, target], dim=0).to(device)            
            cls_loss = cls_loss_fn(logits, aug_labels)
            
            # contrastive loss
            f1, f2 = torch.split(feature_norm, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            contrastive_loss = contrastive_loss_fn(features, target)

            # prototype <--> feature contrastive loss
            pro_feat_con_loss = con_proto_feat_loss_fn(feature_norm, model.learnable_proto, aug_labels)
            
            # prototype self constrastive 
            pro_con_loss = con_proto_loss_fn(model.learnable_proto)

            # 计算基础损失，并根据开关决定是否加入对齐损失
            base_loss = cls_loss + contrastive_loss + pro_con_loss + pro_feat_con_loss

            align_loss = 0

            # 选择对齐策略
            if use_progressive_alignment and initial_protos is not None and fixed_anchors is not None:
                # OursV8 逻辑: 渐进式对齐
                progress = (e - start_epoch) / local_epochs
                # 动态计算插值目标
                target_anchor = (1 - progress) * initial_protos + progress * fixed_anchors
                align_loss = alignment_loss_fn(model.learnable_proto, target_anchor)
            elif use_drcl and fixed_anchors is not None:
                # OursV5, V6, V7 逻辑: 对齐到固定目标
                # 只对当前batch中出现的类计算对齐损失 (class mask)
                unique_classes = torch.unique(target)
                if len(unique_classes) > 0:
                    if force_feature_alignment:
                         # [ABLATION] Force features to align directly with anchors
                         # This bypasses the prototype layer to prove "Feature Collapse"
                         # Extract features for unique classes (using mask/indexing is tricky for features,
                         # so we align features to their corresponding class anchors for ALL samples in batch)
                         
                         # Get anchors for each sample in the batch
                         batch_anchors = fixed_anchors[target] # [batch_size, feature_dim]
                         
                         # Directly minimize distance between normalized features and anchors
                         # feature_norm is already normalized
                         align_loss = alignment_loss_fn(feature_norm, batch_anchors)
                         
                    else:
                        # Standard AURORA: Align Prototypes <-> Anchors
                        proto_subset = model.learnable_proto[unique_classes]
                        anchor_subset = fixed_anchors[unique_classes]
                        align_loss = alignment_loss_fn(proto_subset, anchor_subset)
                else:
                    align_loss = 0

            if use_uncertainty_weighting:
                # V10: 动态学习权重
                sigma_sq_local = torch.exp(model.log_sigma_sq_local)
                sigma_sq_align = torch.exp(model.log_sigma_sq_align)

                ## V12: 新的内部退火逻辑
                if use_dynamic_task_attenuation:
                    # 计算全局训练进度
                    current_step = e # 使用epoch级别进度
                    progress = current_step / total_training_steps
                    # 使用余弦衰减函数，从1平滑降到0
                    schedule_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                    schedule_factor = max(0.0, schedule_factor)

                    # 稳定正则: ReLU-hinge 形式 L_reg = γ·ReLU(λ_eff - λ_max)
                    # 注意：这里用非detach的λ_eff，让梯度可以流向σ参数
                    # Modified to use Linear ReLU to match User's V14 logic, replacing the duplicate block.
                    lambda_eff_for_reg = sigma_sq_local / sigma_sq_align
                    stability_reg = gamma_reg * torch.relu(lambda_eff_for_reg - lambda_max)

                    # 修正版: schedule_factor 放在 log 正则项上，而非数据项
                    # 这保证 s(p)->0 时, log正则项消失, σ²_align->∞, λ_eff->0
                    loss_sigma_main  = (0.5 / sigma_sq_local) * base_loss.detach() + \
                                    (0.5 / sigma_sq_align) * align_loss.detach()
                                        
                    # loss_for_weights 不再需要外部的 annealing_factor
                    effective_lambda = (sigma_sq_local / sigma_sq_align).detach()
                    loss_for_weights = base_loss + effective_lambda * align_loss

                # V11: 保留原有的外部退火逻辑以供对比
                else:
                    schedule_factor = 1.0  # V11不使用内部退火，保持log正则项完整
                    stability_reg = 0  # 在旧版本中关闭此功能

                    loss_sigma_main = (0.5 / sigma_sq_local) * base_loss.detach() + \
                                    (0.5 / sigma_sq_align) * align_loss.detach()
                    
                    effective_lambda = (sigma_sq_local / sigma_sq_align).detach()
                    lambda_annealed = effective_lambda * annealing_factor
                    loss_for_weights = base_loss + lambda_annealed * align_loss

                # 将所有与 sigma 相关的项组合在一起
                # 关键修正: schedule_factor 乘在 log(sigma_sq_align) 上
                # 当 s(p)->0 时，对齐任务的正则项消失，σ²_align 会趋向无穷大
                loss_for_sigma_total = loss_sigma_main + \
                           0.5 * (torch.log(sigma_sq_local) + schedule_factor * torch.log(sigma_sq_align)) + \
                           stability_reg

                loss = loss_for_weights + loss_for_sigma_total

                
            elif use_drcl: # 兼容 V7, V8, V9
                # 固定的或自适应的lambda + 全局退火
                global_progress = e / total_training_steps
                lambda_annealed = lambda_align * (1 - global_progress)
                loss = base_loss + lambda_annealed * align_loss
            else: # 兼容 V4
                loss = base_loss


            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()

            total_loss += loss.item()

        train_test_acc = test_acc(copy.deepcopy(model), test_dataloader, device, mode='etf')
        train_set_acc = test_acc(copy.deepcopy(model), training_data, device, mode='etf')

        logger.info(f'Epoch {e} loss: {total_loss}; train accuracy: {train_set_acc}; test accuracy: {train_test_acc}')

        if e % save_freq == 0:
            save_best_local_model(client_model_dir, model, f'epoch_{e}.pth')


    return model