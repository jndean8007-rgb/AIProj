from dataclasses import dataclass
from pathlib import Path

@dataclass
class PathConfig:
    train_path: Path | None = Path(r'C:\Users\rosie\PycharmProjects\AIBigProject\data\processed\tiny_shake\train')
    eval_path: Path | None = Path(r'C:\Users\rosie\PycharmProjects\AIBigProject\data\processed\tiny_shake\eval')
    training_log_path: Path | None = Path(r"C:\Users\rosie\PycharmProjects\AIBigProject\checkpoints\tinyshake\trainingstats")
    model_load_path: Path| None = Path(r"C:\Users\rosie\PycharmProjects\AIBigProject\checkpoints\tinyshake\2000.pth")
    tokenizer_load_path: Path | None = Path(r"C:\Users\rosie\PycharmProjects\AIBigProject\checkpoints\tinyshake\BPETokenizer")
    config_load_path: Path | None = None

    def __post_init__(self):
        assert type(self.train_path) == type(self.eval_path) # upon distribution no longer valid
        assert type(self.training_log_path) == type(self.model_load_path) == type(self.tokenizer_load_path)