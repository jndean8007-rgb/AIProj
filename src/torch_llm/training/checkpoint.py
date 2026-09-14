import torch as t
import json

from torch_llm.data_pipeline.bpe_tokenizer import BPETokenizer
from torch_llm.model.model_config import ModelConfig
from torch_llm.training.train_config import TrainConfig


def save_checkpoint(
        model,
        adamw,
        muon,
        adamw_scheduler,
        muon_scheduler,
        step,
        model_config,
        train_config,
        tokenizer_config,
        config_path,
):
    t.save({
        "model": model.state_dict(),
        "AdamW": adamw.state_dict(),
        "Muon": muon.state_dict(),
        'step': step,
        'Adamw_Scheduler': adamw_scheduler.state_dict(),
        'Muon_scheduler': muon_scheduler.state_dict(),
    }, f"{train_config.checkpoint_dir}/{step}.pth")

    with open(f"{config_path}.json", "w") as f:
        json.dump({
            "ModelConfig": model_config.__dict__,
            "TrainConfig": train_config.__dict__,
            "TokenizerConfig": tokenizer_config.__dict__,
        }, f)

def load_checkpoint(
        load_path,
        model,
        AdamW,
        Muon,
        adamw_scheduler,
        muon_scheduler
):
    checkpoint = t.load(load_path)

    model.load_state_dict(checkpoint["model"])
    AdamW.load_state_dict(checkpoint["AdamW"])
    Muon.load_state_dict(checkpoint["Muon"])
    adamw_scheduler.load_state_dict(checkpoint["Adamw_Scheduler"])
    muon_scheduler.load_state_dict(checkpoint["Muon_scheduler"])
    step = checkpoint["step"]

    return step


def load_config(
        config_load_path,
):
    with open(config_load_path, "r") as f:
        config = json.load(f)
        model_config = config["ModelConfig"]
        train_config = config["TrainConfig"]
        tokenizer_config = config["TokenizerConfig"]

        return (
            ModelConfig(**model_config),
            TrainConfig(**train_config),
            BPETokenizer(**tokenizer_config)
        )