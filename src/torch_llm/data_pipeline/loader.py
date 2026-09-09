import torch as t

from .sampler import TokenBudgetBatchSampler
from .collate import collate_training_sequences

def loader(
        dataset,
        model_max_seq_len,
        max_tokens_per_batch,
        shuffle: bool = True,
):

    batch_sampler = TokenBudgetBatchSampler(
        dataset,
        max_tokens_per_batch,
        shuffle=shuffle,
    )

    def collate_fn(sequences):
        return collate_training_sequences(
            sequences,
            model_max_seq_len
        )

    train_loader = t.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader