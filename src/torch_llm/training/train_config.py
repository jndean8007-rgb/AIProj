from dataclasses import dataclass


@dataclass
class TrainConfig:
    total_steps: int
    scheduler_total_steps: int
    scheduler_warmup_steps: int

    total_steps: int
    warmup_steps: int
    min_lr_ratio: float

    muon_lr: float
    adamw_lr: float
    weight_decay: float

    log_inteval: int
    checkpoint_dir: str
    checkpoint_interval: int
    eval_interval: int

    aux_loss_weight: float
    max_grad_norm: float | None = None
