
def step(self):
    params = t.empty()


    for group in self.param_groups:
        lr = group['lr']
        momentum = group['momentum']
        weight_decay = group['weight_decay']
        ns_steps = group['ns_steps']
        nesterov = group['nesterov']
        params = group['params']

        Bs = t.empty((params.shape[0], ))

        for param in params:
            if 'momentum_matrix' not in param.state:
                param.state['momentum_matrix'] = t.zeros_like(param).to(dtype=t.float32)

            param.state['momentum_matrix'] = momentum * param.state['momentum_matrix'] + param.grad

        if nesterov: ######### CANNOT DO THIS, MUST PRESERVE SHAPE METADATA AND PAD
            Bs = t.stack([
                momentum * param.state['momentum_matrix'] + param.grad
                for param in params
            ], dim=0)
        else:
            Bs = t.tensor((params.shape[0],), momentum)

        Bs = Bs.to(t.float32)


        Os = muon_ns_step_wrapper(
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


def muon_ns_step_wrapper(Bs, ns_steps):
    '''
    prepare kernel
          ↓
    grouped A = X Xᵀ
          ↓
    grouped Y = A X
          ↓
    grouped Z = A Y
          ↓
    grouped X update
          ↓
    repeat NS iterations
          ↓
    parameter update kernel
    '''