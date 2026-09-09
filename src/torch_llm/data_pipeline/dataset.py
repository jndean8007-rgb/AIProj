import torch as t
from typing import Iterable

class TokenSequenceDataset(t.utils.data.Dataset):
    def __init__(
        self,
        token_sequences,
        sequence_length_limit,
    ):
        assert isinstance(token_sequences, list)
        assert len(token_sequences) > 0

        self.token_sequences = token_sequences
        self.sequence_length_limit = sequence_length_limit

    def __len__(self):
        return len(self.token_sequences)

    def __getitem__(self, index):
        seq = self.token_sequences[index]

        assert seq is not None
        assert seq.ndim == 1
        assert len(seq) <= self.sequence_length_limit

        return seq

    @classmethod
    def from_training_text(
            cls,
            texts,
            tokenizer,
            model_max_seq_len,
    ):
        sequences = []

        for text in texts:
            sequences.extend(tokenizer.encode(text))
            sequences.append(tokenizer.eos_token_id)


        chunks = cls.chunks_of(
            sequences,
            model_max_seq_len + 1
        )

        return cls(
            [t.tensor(chunk, dtype=t.long) for chunk in chunks],
            model_max_seq_len + 1,
        )

    @staticmethod
    def chunks_of(token_ids, chunk_size):
        for start in range(0, len(token_ids)-1, chunk_size):
            yield token_ids[start:start + chunk_size + 1]



