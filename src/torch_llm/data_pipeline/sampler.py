import math

import torch as t

class TokenBudgetBatchSampler(t.utils.data.Sampler):
    def __init__(
            self,
            dataset,
            max_tokens_per_batch:int,
            shuffle:bool = True
    ):
        self.dataset = dataset
        self.max_tokens_per_batch = max_tokens_per_batch
        self.shuffle = shuffle

    def __iter__(self):

        # optionally shuffle indices here
        if self.shuffle:
            indices = t.randperm(len(self.dataset)).tolist()
        else:
            indices = list(range(len(self.dataset)))

        batch = []
        batch_tokens = 0

        for idx in indices:
            seq = self.dataset[idx]

            seq_tokens = len(seq) - 1  # because training shift

            assert seq_tokens <= self.max_tokens_per_batch

            if batch and batch_tokens + seq_tokens > self.max_tokens_per_batch:
                yield batch
                batch = []
                batch_tokens = 0

            batch.append(idx)
            batch_tokens += seq_tokens

        if batch:
            yield batch

    def __len__(self):
        total_tokens = sum(
            len(seq) - 1
            for seq in self.dataset
        )

        return math.ceil(total_tokens / self.max_tokens_per_batch)