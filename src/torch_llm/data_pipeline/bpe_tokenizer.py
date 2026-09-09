from torch_llm.data_pipeline.bpe_tokenizer_config import BPETokenizerConfig
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


class BPETokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_config(cls, config):
        tokenizer = Tokenizer(
            BPE(unk_token=config.unk_token),
        )
        tokenizer.pre_tokenizer = ByteLevel(
            add_prefix_space=False
        )
        tokenizer.decoder = ByteLevelDecoder()

        return cls(tokenizer)

    def train(self, config: BPETokenizerConfig, *paths):
        trainer = BpeTrainer(
            vocab_size=config.vocab_size,
            min_frequency=config.min_frequency,
            special_tokens=config.special_tokens
        )

        self.tokenizer.train(
            files=list(paths),
            trainer=trainer
        )

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def save(self, path):
        self.tokenizer.save(path)

    @classmethod
    def load(cls, path):
        return cls(Tokenizer.from_file(path))

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id