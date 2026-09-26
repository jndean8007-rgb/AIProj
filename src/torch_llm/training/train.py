import time

import torch as t

from torch_llm.data_pipeline.bpe_tokenizer_config import BPETokenizerConfig
from torch_llm.model.model_config import ModelConfig
from torch_llm.training.checkpoint import save_checkpoint
from torch_llm.training.evaluate import evaluate
from torch_llm.training.train_config import TrainConfig
from torch_llm.training.train_step import train_step


def train(
        model_config: ModelConfig,
        train_config: TrainConfig,
        tokenizer_config: BPETokenizerConfig,
        training_logger,
        train_loader,
        eval_loader,
        model,
        adamw,
        muon,
        adamw_scheduler,
        muon_scheduler,
        *,
        device='cuda',
        dtype = t.bfloat16,
        start_step = 0,
):
    model.train()

    step = start_step

    while step < train_config.total_steps:
        epoch_start_step = step

        for batch in train_loader:
            batch = batch.to(
                device,
                non_blocking=True
            )

            t.cuda.synchronize()
            start = time.perf_counter()

            model_metrics = train_step(
                model,
                batch,
                muon,
                adamw,
                aux_loss_weight=train_config.aux_loss_weight,
                max_grad_norm=train_config.max_grad_norm,
                muon_scheduler=muon_scheduler,
                adamw_scheduler=adamw_scheduler
            )

            t.cuda.synchronize()
            step_time = time.perf_counter() - start

            # -----------------------------------------
            # logging / diagnostics
            # -----------------------------------------

            if step % train_config.log_interval == 0:
                other_metrics = {
                    "Step": step,
                    "Muon lr": muon.param_groups[0]["lr"],
                    "AdamW lr": adamw.param_groups[0]["lr"],
                    "Step time (s)": step_time,
                    "Validation loss": evaluate(model, eval_loader, device=device),
                    "VRAM GB": t.cuda.memory_allocated() / (1024 ** 3),
                    "Tokens / sec": batch.token_positions.numel() / step_time,
                }

                training_logger.log(model_metrics=model_metrics, other_metrics=other_metrics)


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

            if step % train_config.checkpoint_interval == 0:
                save_checkpoint(
                    model,
                    adamw,
                    muon,
                    adamw_scheduler,
                    muon_scheduler,
                    step,
                    model_config,
                    train_config,
                    tokenizer_config,
                    train_config.checkpoint_dir,
                )

            if step % train_config.eval_interval == 0:
                loss = evaluate(model, eval_loader, device=device)
                print("Validation set loss: ", loss)

            step += 1
            if step >= train_config.total_steps:
                break

        if step == epoch_start_step:
            raise ValueError("training dataset is empty or exhausted")

    return model
