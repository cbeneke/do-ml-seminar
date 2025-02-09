import math
import torch
from torch.optim.optimizer import Optimizer


def centralize_gradient(x, gc_axis=0):
    """Centralizes the gradient.
    
    Args:
        x: A gradient tensor
        gc_axis: The axis to compute mean over for gradient centralization
    """
    return x - x.mean(dim=tuple(range(gc_axis)) + tuple(range(gc_axis + 1, len(x.shape))), keepdim=True)


class AdaBeliefOptimizer(Optimizer):
    """
    Implementation of AdaBelief optimizer with gradient centralization.
    
    AdaBelief Optimizer: Adapting stepsizes by the belief in observed gradients
    (https://arxiv.org/abs/2010.07468)
    
    This implementation includes:
    - Gradient centralization
    - Proper bias correction
    - AMSGrad variant
    - Rectification option
    - Learning rate warmup and decay
    
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
        centralize_gradients (bool, optional): whether to use gradient centralization (default: True)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-14,
                 weight_decay=0, rectify=True, amsgrad=False, sma_threshold=5.0,
                 total_steps=0, warmup_proportion=0.1, min_lr=0.0,
                 centralize_gradients=True):
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
                       min_lr=min_lr, centralize_gradients=centralize_gradients)
        super(AdaBeliefOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdaBeliefOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('rectify', True)
            group.setdefault('centralize_gradients', True)

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

                # Apply gradient centralization for Conv/FC layers
                if group['centralize_gradients'] and len(grad.shape) > 1:
                    grad = centralize_gradient(grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient differences
                    state['exp_avg_var'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        # Maintains max of all exp. moving avg. of sq. grad. differences
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

                # Update first moment estimate (momentum)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update second moment estimate (variance)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2).add_(group['eps'])

                # Bias correction
                exp_avg_corrected = exp_avg.div(bias_correction1)
                
                if group['amsgrad']:
                    torch.maximum(max_exp_avg_var, exp_avg_var, out=max_exp_avg_var)
                    denom = max_exp_avg_var.sqrt().div_(math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = exp_avg_var.sqrt().div_(math.sqrt(bias_correction2)).add_(group['eps'])

                # Compute SMA
                beta2_t = beta2 ** state['step']
                sma_inf = 2.0 / (1.0 - beta2) - 1.0
                sma_t = sma_inf - 2.0 * state['step'] * beta2_t / (1.0 - beta2_t)

                # Compute rectification term
                if sma_t > 4.0:  # Prevents taking sqrt of negative number
                    r_t = math.sqrt(
                        ((sma_t - 4.0) * (sma_t - 2.0) * sma_inf)
                        / ((sma_inf - 4.0) * (sma_inf - 2.0) * sma_t)
                    )
                else:
                    r_t = 1.0

                # Update parameters
                if group['rectify']:
                    if sma_t >= group['sma_threshold']:
                        update = r_t * exp_avg_corrected / denom
                    else:
                        update = exp_avg_corrected
                else:
                    update = exp_avg_corrected / denom

                if group['weight_decay'] != 0:
                    update.add_(p, alpha=group['weight_decay'])

                p.add_(update, alpha=-lr_t)

        return loss 