from dataclasses import dataclass
from pathlib import Path

@dataclass
class PathConfig:
    train_path: Path | None = Path(r'C:\Users\rosie\PycharmProjects\AIBigProject\data\processed\HuggingFaceFW/fineweb-edu\train')
    eval_path: Path | None = Path(r'C:\Users\rosie\PycharmProjects\AIBigProject\data\processed\HuggingFaceFW/fineweb-edu\eval')
    training_log_path: Path | None = Path(r"C:\Users\rosie\PycharmProjects\AIBigProject\checkpoints\HuggingFaceFW/fineweb-edu\trainingstats")
    model_load_path: Path| None = Path(r"C:\Users\rosie\PycharmProjects\AIBigProject\checkpoints\HuggingFaceFW\fineweb-edu\13000.pth")
    tokenizer_load_path: Path | None = Path(r'C:\Users\rosie\PycharmProjects\AIBigProject\checkpoints\tinyshake\BPETokenizer')
    config_load_path: Path | None = None

    def __post_init__(self):
        assert type(self.train_path) == type(self.eval_path) # upon distribution no longer valid
        assert type(self.training_log_path) == type(self.model_load_path) == type(self.tokenizer_load_path)