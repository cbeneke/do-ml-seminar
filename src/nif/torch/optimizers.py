import math
import torch
from torch.optim.optimizer import Optimizer


class AdaBeliefOptimizer(Optimizer):
    """
    Implementation of AdaBelief optimizer.
    
    AdaBelief Optimizer: Adapting stepsizes by the belief in observed gradients
    (https://arxiv.org/abs/2010.07468)
    
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing running averages
            of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve numerical stability
            (default: 1e-14)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        rectify (bool, optional): whether to enable rectification as in RectifiedAdam (default: True)
        amsgrad (bool, optional): whether to use the AMSGrad variant (default: False)
        sma_threshold (float, optional): threshold for simple mean average (default: 5.0)
        total_steps (int, optional): total number of training steps for warmup (default: 0)
        warmup_proportion (float, optional): proportion of warmup steps (default: 0.1)
        min_lr (float, optional): minimum learning rate after warmup (default: 0.0)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-14,
                 weight_decay=0, rectify=True, amsgrad=False, sma_threshold=5.0,
                 total_steps=0, warmup_proportion=0.1, min_lr=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= warmup_proportion <= 1.0:
            raise ValueError(f"Invalid warmup_proportion value: {warmup_proportion}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       rectify=rectify, amsgrad=amsgrad, sma_threshold=sma_threshold,
                       total_steps=total_steps, warmup_proportion=warmup_proportion,
                       min_lr=min_lr, step=0)
        super(AdaBeliefOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdaBeliefOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('rectify', True)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                
                if grad.is_sparse:
                    raise RuntimeError('AdaBelief does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_var'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_var'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # Get parameters
                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']
                if group['amsgrad']:
                    max_exp_avg_var = state['max_exp_avg_var']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Apply warmup if enabled
                if group['total_steps'] > 0:
                    warmup_steps = group['total_steps'] * group['warmup_proportion']
                    decay_steps = max(group['total_steps'] - warmup_steps, 1)
                    decay_rate = (group['min_lr'] - group['lr']) / decay_steps
                    current_step = state['step']
                    
                    if current_step <= warmup_steps:
                        lr_t = group['lr'] * (current_step / warmup_steps)
                    else:
                        lr_t = group['lr'] + decay_rate * min(current_step - warmup_steps, decay_steps)
                else:
                    lr_t = group['lr']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2).add_(group['eps'])

                if group['amsgrad']:
                    torch.max(max_exp_avg_var, exp_avg_var, out=max_exp_avg_var)
                    denom = (max_exp_avg_var.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_var.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                # Compute SMA
                beta2_t = beta2 ** state['step']
                sma_inf = 2.0 / (1.0 - beta2) - 1.0
                sma_t = sma_inf - 2.0 * state['step'] * beta2_t / (1.0 - beta2_t)

                # Compute r_t (using torch operations for numerical stability)
                sma_t_tensor = torch.tensor(sma_t, dtype=torch.float32, device=p.device)
                sma_inf_tensor = torch.tensor(sma_inf, dtype=torch.float32, device=p.device)
                
                r_t_numerator = (sma_t_tensor - 4.0).clamp(min=0.0)
                r_t_denominator = (sma_inf_tensor - 4.0).clamp(min=group['eps'])
                r_t_a = r_t_numerator / r_t_denominator
                
                r_t_numerator = (sma_t_tensor - 2.0).clamp(min=0.0)
                r_t_denominator = (sma_inf_tensor - 2.0).clamp(min=group['eps'])
                r_t_b = r_t_numerator / r_t_denominator
                
                r_t = torch.sqrt(r_t_a * r_t_b * sma_inf_tensor / sma_t_tensor.clamp(min=group['eps']))

                # Update parameters
                step_size = lr_t / bias_correction1
                if group['rectify']:
                    if sma_t >= group['sma_threshold']:
                        r_t = r_t * exp_avg / denom
                    else:
                        r_t = exp_avg
                else:
                    r_t = exp_avg / denom

                if group['weight_decay'] != 0:
                    r_t.add_(p, alpha=group['weight_decay'])

                p.add_(r_t, alpha=-step_size)

        return loss 