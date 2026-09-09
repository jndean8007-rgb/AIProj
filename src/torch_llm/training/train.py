import torch as t

from torch_llm.model.model_config import ModelConfig
from torch_llm.model.model import TransformerLM
from torch_llm.training.optim.build import build_optimizers
from torch_llm.training.scheduler import build_warmup_cosine_scheduler
from torch_llm.training.train_config import TrainConfig
from torch_llm.training.train_step import train_step


def train(
        model_config: ModelConfig,
        train_config: TrainConfig,
        train_loader,
        *,
        device='cuda',
        dtype=t.bfloat16,
):

    model = TransformerLM(model_config).to(
        device=device,
        dtype=dtype
    )

    model.train()

    muon, adamw, muon_names, adamw_names = build_optimizers(
        model,
        train_config.muon_lr,
        train_config.adamw_lr,
        train_config.weight_decay,
    )

    # Optional:
    # inspect / log parameter grouping once

    muon_scheduler = build_warmup_cosine_scheduler(
        muon,
        train_config.warmup_steps,
        train_config.total_steps,
        train_config.min_lr_ratio
    )

    adamw_scheduler = build_warmup_cosine_scheduler(
        adamw,
        train_config.warmup_steps,
        train_config.total_steps,
        train_config.min_lr_ratio
    )

    step = 0

    while step < train_config.total_steps:

        for batch in train_loader:
            if step > train_config.total_steps:
                break

            batch =batch.to(
                device,
                non_blocking=True
            )

            metrics = train_step(
                model,
                batch,
                muon,
                adamw,
                aux_loss_weight=train_config.aux_loss_weight,
                max_grad_norm=train_config.max_grad_norm,
                muon_scheduler=muon_scheduler,
                adamw_scheduler=adamw_scheduler
            )

            # -----------------------------------------
            # logging / diagnostics
            # -----------------------------------------

            if step % train_config.log_inteval == 0:
                print(metrics.aux_loss)
                print(metrics.lm_loss)

            # loss
            # lm_loss
            # aux_loss
            # grad_norm
            # current Muon LR
            # current AdamW LR
            # MoE load stats later
            # tokens/sec later
            # memory later

            # later:
            # checkpointing
            # validation
            # tokens/sec
            # memory/profiling

            step += 1

            if step >= train_config.total_steps:
                break
    return model
