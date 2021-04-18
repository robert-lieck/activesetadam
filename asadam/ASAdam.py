import math
import torch
import numpy as np
from torch.optim import Optimizer


class ASAdam(Optimizer):
    r"""Adapted from the Pytorch Adam implementation. Adds features for L1-regularisation based on uncertainty
    estimates. Maintains an active set of dimensions, activation and deactivation is coupled to the L1 regularisation.
     For `l1` and `glue` both None (default) it behaves like Adam and can be used as a drop-in replacement.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.9), eps=1e-8,
                 l1=None, glue=None, safety=1., active=None, sub_uncertain=False, log_scale=None,
                 max_activation=None, min_steps=0,
                 weight_decay=0, amsgrad=False, debug=False):
        """
        :param params: Same as for Adam
        :param lr: Same as for Adam
        :param betas: Same as for Adam, but three values (third vor variance estimation)
        :param eps: Same as for Adam
        :param l1: (None or float >= 0.; default None) L1 regularisation strength, which will be added to the gradient.
        Effectively corresponds to an additional term l1 * |x_i| in the loss function, which corresponds to adding a
        gradient of l1 * sign(x_i).
        :param glue: (None or float >= 0.; default None) Like an L1 regularisation but applies only to inactive
        dimensions (no change to the gradient)
        :param safety: (float >= 0.; default: 1.) Multiplier to get the uncertainty from the standard error of the mean
        gradient (computed via exponential average). The uncertainty is subtracted from the gradient to determine,
         whether a dimension should be activated.
        :param active: (None/True/False; default: None) Whether dimensions are initially active or not. For None, any
        dimension is inactive [active] if its initial value is zero [non-zero].
        :param sub_uncertain: (bool; default: False) Whether to subtract the uncertainty from the gradient
        :param log_scale: (None or float > 0; default: None) instead of a plain L1 regularisation add a term
        l1 * s * log(1 + |x_i|/s), where s is the log_scale. This corresponds to a gradient of
        l1 * sign(x_i)/(1 + |x_i|/s). That is for |x_i| << s it behaves like a normal L1 regularisation with strength
        l1 but then flattens out. For log_scale --> inf, it is approaching a standard L1 regularisation, which exactly
        realised with the default value of None. This also adds a concave component to the loss function, which may help
        increasing sparsity in case of redundant parameters. For instance, if the loss only depends on the sum x_i + x_j
        of two parameters, both have the exact same gradient everywhere. While only one would need to be non-zero, they
        will be update jointly and never separate. Even if they have slightly different values there is not component of
        the gradient pushing one down and the other up. The log_scale parameter is effectively adding such a component.
        To break the symmetry and induce an initial disbalance, the max_activation and min_steps parameters can be used.
        :param max_activation: (None or int > 0; default: None) Maximal number of dimensions activated per iteration. If
        more dimensions than allowed surpass the threshold of l1 + glue, the ones with the highest gradient will be
        activated. Ties are broken randomly.
        :param min_steps: Minimal number of steps to take between activating dimensions. That is, if a dimension is
        activated in iteration n, at least min_steps steps (including the current step) are taken before activating the
        next dimension(s), i.e. earliest in iteration n + min_steps.
        :param weight_decay: Same as for Adam
        :param amsgrad: Same as for Adam
        :param debug: (False/True; default: False) If True, collect additional debug information that is available after
        a call to step().
        """
        self.debug = debug
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if l1 is not None and not torch.all(0.0 <= torch.tensor(l1)):
            raise ValueError("Invalid L1 value: {}".format(l1))
        if glue is not None and not torch.all(0.0 <= torch.tensor(glue)):
            raise ValueError("Invalid glue value: {}".format(glue))
        if log_scale is not None and not torch.all(0.0 <= torch.tensor(log_scale)):
            raise ValueError("Invalid log_scale value: {}".format(log_scale))
        if max_activation is not None and not 0.0 <= max_activation:
            raise ValueError("Invalid max_activation value: {}".format(max_activation))
        if not 0 <= min_steps:
            raise ValueError("Invalid min_steps value: {}".format(min_steps))
        if not 0.0 <= safety:
            raise ValueError("Invalid safety value: {}".format(safety))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        l1=l1,  glue=glue, safety=safety, active=active, sub_uncertain=sub_uncertain, log_scale=log_scale,
                        max_activation=max_activation, min_steps=min_steps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(ASAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ASAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

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
                    raise RuntimeError('ASAdam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']
                l1, glue, log_scale = group['l1'], group['glue'], group['log_scale']
                max_activation, min_steps = group['max_activation'], group['min_steps']
                beta1, beta2, beta3 = group['betas']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['activation_step'] = np.inf
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of the variance of the gradient around the average
                    state['exp_avg_var'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Active set for each dimension
                    if group['active'] is None:
                        state['active'] = p != 0.
                    elif group['active']:
                        state['active'] = torch.ones_like(p, dtype=torch.bool, memory_format=torch.preserve_format)
                    else:
                        state['active'] = torch.zeros_like(p, dtype=torch.bool, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq, exp_avg_var = state['exp_avg'], state['exp_avg_sq'], state['exp_avg_var']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']

                state['step'] += 1
                state['activation_step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                bias_correction3 = 1 - beta3 ** state['step']

                # add L2
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                # add L1
                if l1 is not None:
                    if log_scale is None:
                        # add normal L1 term
                        grad = grad + group['l1'] * p.sign()
                    else:
                        # add hyperbolic L1 term
                        grad = grad + log_scale * group['l1'] * p.sign() / (p.abs() + log_scale)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                # active set feature
                if l1 is not None or glue is not None:
                    # rename
                    avg_grad = exp_avg
                    avg_sq_grad = exp_avg_sq
                    # correct bias
                    avg_grad_corr = avg_grad / bias_correction1
                    avg_sq_grad_corr = avg_sq_grad / bias_correction2
                    # decay variance, rename, correct bias
                    dgrad = avg_grad_corr - grad
                    exp_avg_var.mul_(beta3).addcmul_(dgrad, dgrad, value=1 - beta3)
                    avg_var = exp_avg_var
                    avg_var_corr = avg_var / bias_correction3
                    # add bias towards magnitude of current gradient estimate
                    bias_weight = beta3 ** state['step']
                    eff_var = bias_weight * avg_grad_corr ** 2 + avg_var_corr * bias_correction3
                    # uncertainty of mean estimator is
                    # (sqrt of) variance divided by effective number of samples: (1 - beta3**n) / (1 - beta3)
                    uncertainty = eff_var.sqrt() * math.sqrt((1 - beta3) / bias_correction3)
                    # reduced abs gradient: subtract uncertainty (results may also be negative)
                    reduced_abs_grad = avg_grad_corr.abs() - uncertainty * group['safety']
                    # only activate if minimal number of steps was taken
                    active = state['active']
                    if min_steps <= state['activation_step']:
                        # activate based on reduced gradient
                        sticky = 0.
                        if l1 is not None:
                            sticky += l1
                        if glue is not None:
                            sticky += glue
                        new_active = torch.logical_and(reduced_abs_grad > sticky, torch.logical_not(active))
                        if new_active.any():
                            state['activation_step'] = 0
                            if max_activation is not None:
                                # find the dimension with highest gradient (setting already active to zero)
                                linear_sorted = np.lexsort((torch.rand_like(reduced_abs_grad).numpy().flatten(),
                                                            reduced_abs_grad.mul(new_active).numpy().flatten()))
                                selected = np.unravel_index(indices=linear_sorted[-max_activation:],
                                                            shape=reduced_abs_grad.shape)
                                # if we were to activate all top dimensions
                                all_activated = active.clone()
                                all_activated[selected] = True
                                # only activate those that surpass the threshold from aboce
                                active.logical_or_(torch.logical_and(new_active, all_activated))
                            else:
                                # activate all
                                active.logical_or_(new_active)
                    # use full or reduced gradient
                    if group['sub_uncertain']:
                        safe_grad = avg_grad_corr.sign() * reduced_abs_grad.clamp(min=0)
                    else:
                        safe_grad = avg_grad_corr
                    # effective gradient
                    eff_grad = safe_grad / (avg_sq_grad_corr.sqrt() + group['eps'])
                    if self.debug:
                        self.direction = eff_grad.clone()
                    eff_grad.mul_(active)  # zero inactive components
                    # remember old location
                    old_p = p.detach().clone()
                    # perform step
                    p.add_(eff_grad, alpha=-group['lr'])
                    # check for sign switches
                    switch = old_p.sign() * p.sign() < 0
                    # deactivate
                    active.logical_and_(torch.logical_not(switch))
                    # reset switching dimensions
                    p[switch] = 0

                    # remember values
                    if self.debug:

                        self.grad = grad.detach().clone()

                        self.avg_grad = avg_grad.detach().clone()
                        self.avg_sq_grad = avg_sq_grad.detach().clone()
                        self.avg_var = avg_var.detach().clone()

                        self.avg_grad_corr = avg_grad_corr.detach().clone()
                        self.avg_sq_grad_corr = avg_sq_grad_corr.detach().clone()
                        self.avg_var_corr = avg_var_corr.detach().clone()
                        self.eff_var = eff_var.detach().clone()
                        self.uncertainty = uncertainty.detach().clone()

                        self.safe_grad = safe_grad.detach().clone()
                else:
                    # normal Adam update
                    step_size = group['lr'] / bias_correction1
                    p.addcdiv_(exp_avg, denom, value=-step_size)

                    # remember values
                    if self.debug:
                        self.grad = grad.detach().clone()
                        self.avg_grad = exp_avg.detach().clone()
                        self.avg_sq_grad = exp_avg_sq.detach().clone()
                        self.avg_grad_corr = (exp_avg / bias_correction1).detach().clone()
                        self.avg_sq_grad_corr = (exp_avg_sq / bias_correction2).detach().clone()

                        self.direction = (exp_avg / denom / bias_correction1).detach().clone()


        return loss
