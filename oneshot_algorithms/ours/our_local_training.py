from oneshot_algorithms.utils import init_optimizer, init_loss_fn, test_acc, save_best_local_model
from oneshot_algorithms.ours.unsupervised_loss import SupConLoss, Contrastive_proto_feature_loss, Contrastive_proto_loss

from common_libs import *



def ours_local_training(model, training_data, test_dataloader, start_epoch, local_epochs, optim_name, lr, momentum, loss_name, device, num_classes, sample_per_class, aug_transformer, client_model_dir, total_rounds, save_freq=1, use_drcl=False, fixed_anchors=None, lambda_align=1.0, use_progressive_alignment=False, initial_protos=None, use_uncertainty_weighting=False):
    model.train()
    model.to(device)

    sigma_lr = 0.01 * lr # sigma 的学习率设为基础学习率的 0.01 倍

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
            
            
            aug_data1, aug_data2 = aug_transformer(data), aug_transformer(data)
            aug_data = torch.cat([aug_data1, aug_data2], dim=0)
            
            aug_data, target = aug_data.to(device), target.to(device)
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
                align_loss = alignment_loss_fn(model.learnable_proto, fixed_anchors)

            if use_uncertainty_weighting:
                # V10: 动态学习权重
                sigma_sq_local = torch.exp(model.log_sigma_sq_local)
                sigma_sq_align = torch.exp(model.log_sigma_sq_align)
                
                loss_local_weighted = (0.5 / sigma_sq_local) * base_loss
                loss_align_weighted = (0.5 / sigma_sq_align) * align_loss
                
                # 正则化项，防止sigma无限增大
                loss_reg = 0.5 * (torch.log(sigma_sq_local) + torch.log(sigma_sq_align))
                
                loss = loss_local_weighted + loss_align_weighted + loss_reg
                
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