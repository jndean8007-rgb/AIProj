import torch as t


class StreamingDataset(t.utils.data.IterableDataset):
    def __init__(self, text_iterator, seq_len, tokenizer=None):
        assert seq_len > 0
        self.text_iterator = text_iterator
        self.seq_len = seq_len
        self.tokenizer = tokenizer

    def __iter__(self):
        if self.tokenizer is None:
            raise ValueError("tokenizer required before model training")

        buffer = []

        for text in self.tokenizer_training_iterator():
            token_ids = self.tokenizer.encode(text)
            buffer.extend(token_ids)
            buffer.append(self.tokenizer.eos_token_id)

            while len(buffer) >= self.seq_len + 1:
                # extra token for the training shift
                yield t.tensor(buffer[:self.seq_len + 1], dtype=t.long)
                buffer = buffer[self.seq_len:]

        if len(buffer) > 1:
            yield t.tensor(buffer, dtype=t.long)

    def tokenizer_training_iterator(self):
        for text in self.text_iterator:
            yield text["text"] if isinstance(text, dict) else text
