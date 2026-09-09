from dataclasses import dataclass


@dataclass
class BPETokenizerConfig:
    vocab_size: int
    min_frequency: int
    special_tokens: list[str]