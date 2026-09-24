from os import PathLike

from torch_llm.data_pipeline.bpe_tokenizer_config import BPETokenizerConfig
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from pathlib import Path

class BPETokenizer:
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config

    @classmethod
    def from_config(cls, config):
        tokenizer = Tokenizer(
            BPE(unk_token=config.unk_token),
        )

        tokenizer.pre_tokenizer = ByteLevel(
            add_prefix_space=False
        )
        tokenizer.decoder = ByteLevelDecoder()

        return cls(tokenizer, config)

    def train(self, config: BPETokenizerConfig, *paths):
        trainer = BpeTrainer(
            vocab_size=config.vocab_size,
            min_frequency=config.min_frequency,
            special_tokens=config.special_tokens
        )

        self.tokenizer.train(
            files=[str(path) for path in paths],
            trainer=trainer
        )

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode([token_ids])

    def save(self, path):
        self.tokenizer.save(path)

    @classmethod
    def load(cls, path: str | PathLike, config):
        return cls(Tokenizer.from_file(str(path)), config)

    @property
    def eos_token_id(self):
        token_id = self.tokenizer.token_to_id(
            self.config.eos_token
        )

        if token_id is None:
            raise ValueError("EOS token is not in the vocabulary")

        return token_id