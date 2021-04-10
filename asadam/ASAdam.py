import math
import torch
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
                 l1=None, glue=None, safety=1., active=None,
                 weight_decay=0, amsgrad=False, debug=False):
        """
        :param params: Same as for Adam
        :param lr: Same as for Adam
        :param betas: Same as for Adam, but three values (third vor variance estimation)
        :param eps: Same as for Adam
        :param l1: (None or float >= 0.; default None) L1 regularisation strength (will be added to the gradient)
        :param glue: (None or float >= 0.; default None) Like an L1 regularisation but applies only to inactive
        dimensions (no change to the gradient)
        :param safety: (float >= 0.; default: 1.) Multiplier to get the uncertainty from the standard error of the mean
        gradient (computed via exponential average). The uncertainty is subtracted from the gradient to determine,
         whether a dimension should be activated.
        :param active: (None/True/False; default: None) Whether dimensions are initially active or not. For None, any
        dimension is inactive [active] if its initial value is zero [non-zero].
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
        if l1 is not None and not 0.0 <= l1:
            raise ValueError("Invalid L1 value: {}".format(l1))
        if glue is not None and not 0.0 <= glue:
            raise ValueError("Invalid glue value: {}".format(glue))
        if not 0.0 <= safety:
            raise ValueError("Invalid safety value: {}".format(safety))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        l1=l1,  glue=glue, safety=safety, active=active,
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

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
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
                beta1, beta2, beta3 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                bias_correction3 = 1 - beta3 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment, and variance running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # correct biases
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                # active set feature
                l1, glue = group['l1'], group['glue']
                if l1 is not None or glue is not None:
                    # to determine activation we use the biased averages
                    active = state['active']
                    # deviation of gradient from (biased) average
                    dgrad = exp_avg - grad
                    # decay variance
                    exp_avg_var.mul_(beta3).addcmul_(dgrad, dgrad, value=1 - beta3)
                    #  correct bias in variance estimate
                    exp_avg_var_corr = exp_avg_var / bias_correction3
                    # uncertainty of mean estimator is
                    # (sqrt of) variance divided by effective number of samples: (1 - beta3**n) / (1 - beta3)
                    uncertainty = exp_avg_var_corr.sqrt() * math.sqrt((1 - beta3) / bias_correction3)
                    # activate (based on biased average)
                    sticky = 0.
                    if l1 is not None:
                        sticky += l1
                    if glue is not None:
                        sticky += glue
                    active.logical_or_(exp_avg.abs() - uncertainty * group['safety'] > sticky)
                    # effective step (based on corrected average)
                    eff_grad = (exp_avg / denom) / bias_correction1  # adapted gradient like in Adam
                    if l1 is not None:
                        eff_grad.add_(p.sign(), alpha=group['l1'])  # add L1 term
                    if self.debug:
                        self.eff_grad = eff_grad.detach().clone()
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
                        self.true_grad = grad.detach().clone()
                        self.exp_avg = exp_avg.detach().clone()
                        self.exp_avg_corr = (exp_avg / bias_correction1).detach().clone()
                        self.exp_avg_std = exp_avg_sq.sqrt().detach().clone()
                        self.exp_avg_std_corr = denom.detach().clone()
                        self.exp_avg_var = exp_avg_var.detach().clone()
                        self.exp_avg_var_corr = exp_avg_var_corr.detach().clone()
                        self.uncertainty = uncertainty.detach().clone()
                else:
                    # normal Adam update
                    step_size = group['lr'] / bias_correction1
                    p.addcdiv_(exp_avg, denom, value=-step_size)

                    # remember values
                    if self.debug:
                        self.true_grad = grad.detach().clone()
                        self.exp_avg = exp_avg.detach().clone()
                        self.exp_avg_corr = (exp_avg / bias_correction1).detach().clone()
                        self.exp_avg_std = exp_avg_sq.sqrt().detach().clone()
                        self.exp_avg_std_corr = denom.detach().clone()
                        self.eff_grad = (exp_avg / denom / bias_correction1).detach().clone()

        return loss
