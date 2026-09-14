import torch as t
from torch_llm.training.train import train
from torch_llm.training.checkpoint import load_config
from torch_llm.training.setup import setup, build_or_load_tokenizer
from torch_llm.training import checkpoint
from torch_llm.training.training_logger import TrainingLogger


def run_training(
        model_config,
        train_config,
        tokenizer_config,
        train_path,
        eval_path,
        training_log_path,
        from_checkpoint = False,
        config_load_path = None,
        model_load_path = None,
        tokenizer_load_path = None
):

    if from_checkpoint and config_load_path is not None:
        model_config, train_config, tokenizer_config = load_config(
            config_load_path,
        )


    tokenizer = build_or_load_tokenizer(
        train_path,
        tokenizer_config,
        from_checkpoint,
        tokenizer_load_path,
    )
    if not from_checkpoint:
        tokenizer.save(tokenizer_config.tokenizer_save_path)


    model, train_loader, eval_loader, adamw, muon, adamw_scheduler, muon_scheduler = setup(
        model_config,
        train_config,
        tokenizer,
        train_path,
        eval_path,
        device='cuda',
        dtype=t.bfloat16,
    )

    step = 0

    if from_checkpoint:
        step = checkpoint.load_checkpoint(
            load_path=model_load_path,
            model=model,
            AdamW=adamw,
            Muon=muon,
            adamw_scheduler=adamw_scheduler,
            muon_scheduler=muon_scheduler
        )


    training_logger = TrainingLogger(
        training_log_path
    )


    model = train(
        model_config,
        train_config,
        tokenizer_config,
        training_logger,
        train_loader,
        eval_loader,
        model,
        adamw,
        muon,
        adamw_scheduler,
        muon_scheduler,
        device='cuda',
        dtype=t.bfloat16,
        start_step=step
    )
