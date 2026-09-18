from torch_llm.kernals.muon_kernel_wrapper import muon_ns_kernal_wrapper
import torch as t
import math

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
            scaled_O = Os[idx, ...] * scale

            # 7. decoupled weight decay if desired
            param.mul_(1 - lr * weight_decay)

            # 8. update parameter
            param.add_(scaled_O, alpha=-lr)


def muon_ns_step_process(
        Bs, ns_steps
) -> list[t.Tensor]:

    #preserve dimensions
    orig_B_dims = t.tensor([
        (B.shape[0], B.shape[1]) for B in Bs
    ]).to(dtype=t.long, device='cuda')

    #transpose each B such that is longer ie width > height for memory intermediate tensor and pack
    b_buffer = t.cat([
        B.transpose(-1, -2).reshape(-1) if B.shape[-2] > B.shape[-1]
        else B.reshape(-1)
        for B in Bs
    ])

    singular_direction_buffer =  muon_ns_kernal_wrapper(
        b_buffer,
        orig_B_dims,
        ns_steps
    ) #shape is same as input Buffer

    resolved_directions = []

    start = 0

    for B in Bs:
        rows, cols = B.shape
        length = rows * cols

        buffer_seq = singular_direction_buffer[start:start + length]

        if rows > cols:
            O = buffer_seq.reshape(cols, rows).transpose(-1, -2)
        else:
            O = buffer_seq.reshape(rows, cols)

        resolved_directions.append(O)

        start += length

    return resolved_directions










'''
pass parameters through as buffer: ie 1 dimensional long tensor
must also pass original tensor metadata to resolve dimensions in kernals.
convert to floating point 32

spawn worker threads in kernal for stacked parameter matrics
Kernals: -- must use masking for appropriate passage
frob normalization kernal for Bs
A = XXT
Y = AX
Xnew = aX + bY +cAY


'''