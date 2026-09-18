import math

import torch as t
from torch.optim import Optimizer
import torch.nn.functional as F
from torch_llm.kernals.muon_kernel_wrapper import muon_ns_kernal_wrapper



class Muon(Optimizer):

    def __init__(
            self,
            params,
            lr,
            momentum=0.95,
            weight_decay=0.0,
            ns_steps=5,
            nesterov=True,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            nesterov=nesterov,
        )

        super().__init__(params, defaults)

    @t.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            ns_steps = group['ns_steps']
            nesterov = group['nesterov']
            params = group['params']

            for param in params:
                state = self.state[param]
                if 'momentum_matrix' not in state:
                    state['momentum_matrix'] = t.zeros_like(param).to(dtype=t.float32)

                state['momentum_matrix'].mul_(momentum).add_(param.grad)

            if nesterov:
                Bs = [momentum * self.state[param]['momentum_matrix'] + param.grad for param in params]
            else:
                Bs = [self.state[param]['momentum_matrix'] for param in params]

            Os = muon_ns_step_process(
                Bs,
                ns_steps
            )

            for idx, param in enumerate(params):
                fan_out, fan_in = param.grad.shape[-2:]
                scale = math.sqrt(max(1, (fan_out / fan_in)))
                scaled_O = Os[idx] * scale

                # 7. decoupled weight decay if desired
                param.mul_(1 - lr * weight_decay)

                # 8. update parameter
                param.add_(scaled_O, alpha=-lr)

def muon_ns_step_process(
        Bs, ns_steps
) -> list[t.Tensor]:

    flattened_dims = []

    for B in Bs:
        if B.dim() == 3:
            num_experts, rows, cols = B.shape
            flattened_dims.extend([(rows, cols)] * num_experts)
        else:
            flattened_dims.append((B.shape[0], B.shape[1]))

    B_dims_flattened = t.tensor(flattened_dims, dtype=t.long, device='cuda')



    # transpose each B such that is longer ie width > height for memory intermediate tensor and pack
    b_buffer = t.cat([
        B.transpose(-1, -2).reshape(-1) if B.shape[-2] > B.shape[-1]
        else B.reshape(-1)
        for B in Bs
    ])

    singular_direction_buffer = muon_ns_kernal_wrapper(
        b_buffer,
        B_dims_flattened,
        ns_steps
    )  # shape is same as input Buffer

    resolved_directions = []

    start = 0

    for B in Bs:
        if B.dim() == 2:
            rows, cols = B.shape
            length = rows * cols

            buffer_seq = singular_direction_buffer[start:start + length]

            if rows > cols:
                O = buffer_seq.reshape(cols, rows).transpose(-1, -2)
            else:
                O = buffer_seq.reshape(rows, cols)

            resolved_directions.append(O)


        elif B.dim() == 3:
            num_experts, rows, cols = B.shape

            length = num_experts * rows * cols

            buffer_seq = singular_direction_buffer[start:start + length]

            if rows > cols:
                O = buffer_seq.reshape(num_experts, cols, rows).transpose(-1, -2)
            else:
                O = buffer_seq.reshape(num_experts, rows, cols)

            resolved_directions.append(O)

        start += length

    return resolved_directions


    '''
    def step(self):

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            ns_steps = group['ns_steps']
            nesterov = group['nesterov']

            for p in group['params']:

                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]

                # 1. initialize momentum state
                if 'momentum_matrix' not in state:
                    state['momentum_matrix'] = t.zeros_like(p).to(dtype=t.float32)


                # 2. update momentum
                state['momentum_matrix'] = momentum * state['momentum_matrix'] + p.grad

                # 3. optionally construct Nesterov update

                if nesterov is True:
                    B = momentum * state['momentum_matrix'] + p.grad
                else:
                    B = state['momentum_matrix']

                # 4. convert update to FP32 if appropriate
                B = B.to(t.float32)
                # 5. orthogonalize matrix update
                #    Newton-Schulz and combine scale to reduce memory
                #    scale transformed update - s(m,n) scales O
                #  #original  muon shape scaling
                fan_out, fan_in = p.grad.shape[-2:]
                scale = math.sqrt(max(1, (fan_out / fan_in)))
                scaled_O = zeropower_via_newton_schulz(B, steps=ns_steps) * scale

                # 7. decoupled weight decay if desired
                p.mul_(1-lr*weight_decay)

                # 8. update parameter
                p.add_(scaled_O, alpha=-lr)

                pass



def zeropower_via_newton_schulz(
    B: t.Tensor,
    steps: int,
) -> t.Tensor:
    """
    Approximate the polar / zeroth-power transform
    of a matrix or batch of matrices.

    Input:
        (..., M, N)

    Output:
        (..., M, N)
    """

    # handle tall vs wide orientation
    transposed = B.shape[-2] > B.shape[-1] # make matrix wide (M <= N) for Newton-Schulz

    if transposed:
        B = B.transpose(-1, -2)

    # normalize B
    x = B / (t.linalg.norm(B, ord='fro', dim=(-2, -1), keepdim=True) + 1e-8)

    # tuned Newton-Schulz polynomial used by modern Muon
    # X = U Sigma V^T
    #
    # sigma <- a*sigma + b*sigma^3 + c*sigma^5
    #
    # tuned to rapidly push nonzero singular values toward ~1
    a = 3.4445
    b = -4.7750
    c = 2.0315

    for _ in range(steps):
        A = x @ x.transpose(-1, -2)
        Y = A @ x
        Z = A @ Y
        x = a * x + b * Y + c * Z


    # restore orientation if needed
    if transposed:
        x = x.transpose(-1, -2)

    # 5. x now approximates U V^T
    #    same singular-vector directions as B, but singular values ~1
    #    this is the orthogonalized / polar Muon update

    return x'''

