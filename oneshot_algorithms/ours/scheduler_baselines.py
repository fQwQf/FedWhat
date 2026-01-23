import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class GradNormScheduler:
    def __init__(self, device, num_tasks=2, alpha=1.5, lr=0.001):
        """
        GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks.
        """
        self.num_tasks = num_tasks
        self.device = device
        self.alpha = alpha
        self.weights = torch.ones(num_tasks, device=device, requires_grad=True)
        self.optimizer = optim.Adam([self.weights], lr=lr)
        self.initial_losses = None

    def step(self, loss_list, shared_layer_weights):
        """
        loss_list: list of losses [loss_1, loss_2, ...]
        shared_layer_weights: weights of the last shared layer (usually just before task-specific heads)
        """
        if self.initial_losses is None:
            self.initial_losses = [l.item() for l in loss_list]

        # 1. Update weights (standard GradNorm procedure uses a separate backward pass)
        # However, to avoid double backward on the main graph, we usually detach losses for the weight update.
        
        # Calculate L_grad
        # This part is tricky in PyTorch without retaining graph. 
        # Standard implementation:
        # L = \sum w_i * L_i
        # G^(i)_W = || \nabla_W (w_i * L_i) ||_2
        # \bar{G}_W = average(G^(i)_W)
        # r_i = L_i / L_i(0)
        # \bar{r} = average(r_i)
        # L_grad = \sum | G^(i)_W - \bar{G}_W * (r_i / \bar{r})^\alpha |_1
        
        # We need gradients of each task loss w.r.t shared weights.
        grads = []
        for i, loss in enumerate(loss_list):
            # retain_graph=True because we need to backward multiple times on shared_layer_weights
            # create_graph=True allows differentiating through the gradient itself (needed for L_grad backward?)
            # Actually L_grad update is only for w_i. w_i does not affect G^(i)_W magnitude directly in the simplified view 
            # BUT in standard GradNorm, G^(i)_W = w_i * || \nabla L_i ||. 
            # We want G^(i)_W to match target.
            
            # Simplified per-task gradient norm:
            g = torch.autograd.grad(loss, shared_layer_weights, retain_graph=True)[0]
            grads.append(torch.norm(g))
            
        grads = torch.stack(grads) # [G_1, G_2] (norms)
        
        # Inverse relative loss rates
        loss_ratios = []
        for i, loss in enumerate(loss_list):
            loss_ratios.append(loss.item() / max(self.initial_losses[i], 1e-8))
        loss_ratios = torch.tensor(loss_ratios, device=self.device)
        avg_ratio = loss_ratios.mean()
        inverse_ratios = (loss_ratios / avg_ratio) ** self.alpha
        
        target_grads = grads.mean() * inverse_ratios
        
        # L_grad: loss for the weights
        # We treat 'grads' as constant targets for the weights? 
        # Wait, the paper says w_i controls the gradient magnitude.
        # G_i(w) = w_i * || \nabla_W L_i ||. 
        # We calculated grads above without w_i. Let's call those raw_grads.
        # So actual G_i(w) = w_i * raw_grads[i].
        
        # L_grad = sum | w_i * raw_grad[i] - target_grad[i] |_1
        
        l_grad = sum(torch.abs(self.weights[i] * grads[i].detach() - target_grads[i].detach()))
        
        self.optimizer.zero_grad()
        l_grad.backward()
        
        # Renormalize weights to sum to num_tasks
        with torch.no_grad():
            # Standard GradNorm renormalizes
            self.weights.data = self.weights.data / self.weights.data.sum() * self.num_tasks
            
        self.optimizer.step()
        
        return self.weights.detach()


class DWAScheduler:
    def __init__(self, num_tasks=2, temp=2.0, device='cuda'):
        self.num_tasks = num_tasks
        self.temp = temp
        self.device = device
        self.loss_history = [] # List of [loss_1, loss_2]
        self.weights = torch.ones(num_tasks, device=device) # Initial weights

    def step(self, epoch_losses):
        """
        epoch_losses: list/tensor of losses for this epoch [L_1, L_2]
        Should be called at the end of each epoch.
        """
        self.loss_history.append(epoch_losses)
        
        if len(self.loss_history) < 2:
            return self.weights # Keep default 1.0 for first two epochs
            
        # DWA formula:
        # r_k(t-1) = L_k(t-1) / L_k(t-2)
        # w_k(t) = K * exp(r_k(t-1) / T) / sum(exp(r_i(t-1) / T))
        
        last_losses = np.array(self.loss_history[-1])
        prev_losses = np.array(self.loss_history[-2])
        
        # Avoid division by zero
        ratios = last_losses / (prev_losses + 1e-8)
        
        exp_ratios = np.exp(ratios / self.temp)
        total_exp = np.sum(exp_ratios)
        
        new_weights = self.num_tasks * exp_ratios / total_exp
        
        self.weights = torch.tensor(new_weights, device=self.device, dtype=torch.float)
        
        return self.weights
