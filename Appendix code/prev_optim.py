import torch as t
from numpy.ma.core import zeros_like
from torch import optim
import math
from typing import Optional
from collections.abc import Callable, Iterable

def gradient_clipping(parameters: Iterable[t.nn.Parameter], max_l2_norm: float) -> None:
    params = list(parameters)

    with t.no_grad():
        norms = [
            t.linalg.vector_norm(p.grad)
            for p in params
            if p.grad is not None
        ]

        if not norms:
            return

        total_norm = t.linalg.vector_norm(t.stack(norms))

        if total_norm > max_l2_norm:
            scale = max_l2_norm / (total_norm + 1e-6)

            for p in params:
                if p.grad is not None:
                    p.grad *= scale


def learning_rate_schedule(
    it,
    max_learning_rate,
    min_learning_rate,
    warmup_iters,
    cosine_cycle_final,
):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate

    elif it <= cosine_cycle_final:
        cosine_ratio = (it - warmup_iters) / (cosine_cycle_final - warmup_iters)

        return min_learning_rate + 0.5 * (
            1 + math.cos(math.pi * cosine_ratio)
        ) * (max_learning_rate - min_learning_rate)

    return min_learning_rate

class AdamW(optim.Optimizer):
    """Implements Adam algorithm."""
    def __init__(self,
                 params,
                 lr,
                 betas,
                 eps,
                 weight_decay):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            with t.no_grad():
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    #extract parameter state
                    state = self.state[p]

                    # extract step count, gradient, adjusted learning rate
                    if len(state) == 0:
                        step = 1
                        state["step"] = step
                        state['exp_avg'] = t.zeros_like(p)
                        state['exp_avg_sq'] = t.zeros_like(p)
                    else:
                        step = state["step"]

                    grad = p.grad
                    adj_lr = lr * math.sqrt(1 - beta2 ** step) / (1 - beta1 ** step)

                    # weight decay
                    p -= lr * weight_decay * p

                    # update first moment and second moment (exponential averages - exponentially weighted moving average if expanded)
                    state['exp_avg'] = beta1 * state['exp_avg'] + (1 - beta1) * grad
                    state['exp_avg_sq'] = beta2 * state['exp_avg_sq'] + (1 - beta2) * t.square(grad)

                    # update the parameter
                    p -= adj_lr * state['exp_avg'] / (t.sqrt(state['exp_avg_sq']) + eps)

                    #increment step count
                    state["step"] = step + 1





class SGD(optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.

        return loss


def cross_entropy_loss(logits, targets):
    #p(xi+1 |x1:i) = softmax(logiti)[xi+1] = e^(logiti)[xi+1] / sum to vocab size (e^(logiti)[a])
    # individual cross entropy = -log( prev ) = -(log(num) - log(denom)) = log(sum (e^(logit i)[a]) - logit i [xi+1]
    # however, exp can be unstable so we scale by a factor of M = max logit. then:
    # = M + log(sum (e^((logit i)[a] - M)) ) - logit i [xi+1]
    max_logits = t.max(logits, dim=-1, keepdim=True).values
    shifted = logits - max_logits

    log_sum_exp = max_logits.squeeze(-1) + t.log(
        t.exp(shifted).sum(dim=-1)
    )

    target_logits = logits.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    loss = log_sum_exp - target_logits

    return loss.mean()

#perplexity = exp( 1/m sum(m) li) for each cross entropy loss in a sequence length m.
