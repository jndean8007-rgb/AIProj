from dataclasses import dataclass


@dataclass
class TrainConfig:
    total_steps: int = 2_000
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1

    muon_lr: float = 5e-3
    adamw_lr: float = 3e-4
    weight_decay: float = 0.01

    log_interval: int = 100
    checkpoint_dir: str = r"C:\Users\rosie\PycharmProjects\AIBigProject\checkpoints\tinyshake"
    checkpoint_interval: int = 500
    eval_interval: int = 100

    aux_loss_weight: float = 0.01
    max_grad_norm: float | None = 1.0