from oneshot_algorithms.utils import init_optimizer, init_loss_fn, test_acc, save_best_local_model
from oneshot_algorithms.ours.unsupervised_loss import SupConLoss, Contrastive_proto_feature_loss, Contrastive_proto_loss

from common_libs import *



def ours_local_training(model, training_data, test_dataloader, start_epoch, local_epochs, optim_name, lr, momentum, loss_name, device, num_classes, sample_per_class, aug_transformer, client_model_dir, total_rounds, save_freq=1, use_drcl=False, fixed_anchors=None, lambda_align=1.0, use_progressive_alignment=False, initial_protos=None, use_uncertainty_weighting=False, sigma_lr=None, annealing_factor=1.0, use_dynamic_task_attenuation=False, gamma_reg=0, lambda_max=50.0, force_feature_alignment=False):
   
    model.train()
    model.to(device)

    if sigma_lr is None:
        sigma_lr = 0.05 * lr # Set sigma learning rate to 0.05 times the base learning rate

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

    # If using DRCL, define alignment loss function
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

            # Calculate base loss, and decide whether to add alignment loss based on switch
            base_loss = cls_loss + contrastive_loss + pro_con_loss + pro_feat_con_loss

            align_loss = 0

            # Select alignment strategy
            if use_progressive_alignment and initial_protos is not None and fixed_anchors is not None:
                # OursV8 Logic: Progressive Alignment
                progress = (e - start_epoch) / local_epochs
                # Dynamically calculate interpolation target
                target_anchor = (1 - progress) * initial_protos + progress * fixed_anchors
                align_loss = alignment_loss_fn(model.learnable_proto, target_anchor)
            elif use_drcl and fixed_anchors is not None:
                # OursV5, V6, V7 Logic: Align to fixed targets
                # Only calculate alignment loss for classes present in the current batch (class mask)
                unique_classes = torch.unique(target)
                if len(unique_classes) > 0:
                    if force_feature_alignment:
                         # [ABLATION] Force features to align directly with anchors
                         # This bypasses the prototype layer to prove "Feature Collapse"
                         # Extract features for unique classes (using mask/indexing is tricky for features,
                         # so we align features to their corresponding class anchors for ALL samples in batch)
                         
                         # Get anchors for each sample in the batch
                         # Note: feature_norm corresponds to aug_data which is [aug1, aug2], so it has 2*batch_size
                         # We need to duplicate targets/anchors to match
                         aug_targets = torch.cat([target, target], dim=0)
                         batch_anchors = fixed_anchors[aug_targets] # [2*batch_size, feature_dim]
                         
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
                # V10: Dynamic weight learning
                sigma_sq_local = torch.exp(model.log_sigma_sq_local)
                sigma_sq_align = torch.exp(model.log_sigma_sq_align)

                ## V12: New internal annealing logic
                if use_dynamic_task_attenuation:
                    # Calculate global training progress
                    current_step = e # Use epoch-level progress
                    progress = current_step / total_training_steps
                    # Use cosine decay function, smoothly dropping from 1 to 0
                    schedule_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                    schedule_factor = max(0.0, schedule_factor)

                    # Stability Regularization: ReLU-hinge form L_reg = gamma * ReLU(lambda_eff - lambda_max)
                    # Note: Using non-detached lambda_eff here, allowing gradients to flow to sigma parameters
                    # Modified to use Linear ReLU to match User's V14 logic, replacing the duplicate block.
                    lambda_eff_for_reg = sigma_sq_local / sigma_sq_align
                    stability_reg = gamma_reg * torch.relu(lambda_eff_for_reg - lambda_max)

                    # Corrected version: schedule_factor is placed on the log regularization term, not the data term
                    # This ensures when s(p)->0, log regularization term vanishes, sigma_sq_align->inf, lambda_eff->0
                    loss_sigma_main  = (0.5 / sigma_sq_local) * base_loss.detach() + \
                                    (0.5 / sigma_sq_align) * align_loss.detach()
                                        
                    # loss_for_weights no longer needs external annealing_factor
                    effective_lambda = (sigma_sq_local / sigma_sq_align).detach()
                    loss_for_weights = base_loss + effective_lambda * align_loss

                # V11: Keep original external annealing logic for comparison
                else:
                    schedule_factor = 1.0  # V11 does not use internal annealing, keeps log regularization term separate
                    stability_reg = 0  # Disable this feature in old versions

                    loss_sigma_main = (0.5 / sigma_sq_local) * base_loss.detach() + \
                                    (0.5 / sigma_sq_align) * align_loss.detach()
                    
                    effective_lambda = (sigma_sq_local / sigma_sq_align).detach()
                    lambda_annealed = effective_lambda * annealing_factor
                    loss_for_weights = base_loss + lambda_annealed * align_loss

                # Combine all sigma-related terms
                # Key correction: schedule_factor multiplied on log(sigma_sq_align)
                # When s(p)->0, alignment task regularization vanishes, sigma_sq_align tends to infinity
                loss_for_sigma_total = loss_sigma_main + \
                           0.5 * (torch.log(sigma_sq_local) + schedule_factor * torch.log(sigma_sq_align)) + \
                           stability_reg

                loss = loss_for_weights + loss_for_sigma_total

                
            elif use_drcl: # Compatible with V7, V8, V9
                # Fixed or adaptive lambda + global annealing
                global_progress = e / total_training_steps
                lambda_annealed = lambda_align * (1 - global_progress)
                loss = base_loss + lambda_annealed * align_loss
            else: # Compatible with V4
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