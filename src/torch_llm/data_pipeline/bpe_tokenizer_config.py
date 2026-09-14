from dataclasses import dataclass, field


@dataclass
class BPETokenizerConfig:
    vocab_size: int = 4096
    min_frequency: int = 2
    tokenizer_save_path: str = r'C:\Users\rosie\PycharmProjects\AIBigProject\checkpoints\tinyshake\BPETokenizer'

    unk_token: str = "<unk>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    pad_token: str = "<pad>"

    special_tokens: list[str] = field(init=False)

    def __post_init__(self):
        self.special_tokens = [
            self.unk_token,
            self.bos_token,
            self.eos_token,
            self.pad_token,
        ]