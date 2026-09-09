import math
from torch.optim.lr_scheduler import LambdaLR


def build_warmup_cosine_scheduler(
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float,
):
    assert warmup_steps >= 0
    assert total_steps > warmup_steps
    assert 0.0 <= min_lr_ratio <= 1.0

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step / warmup_steps)

        prog = float(
            (current_step - warmup_steps)
            / (total_steps - warmup_steps)
        )

        prog = min(prog, 1.0)

        return min_lr_ratio + (1 - min_lr_ratio) * (
                1 + math.cos(math.pi * prog)
        ) / 2

    return LambdaLR(optimizer, lr_lambda)