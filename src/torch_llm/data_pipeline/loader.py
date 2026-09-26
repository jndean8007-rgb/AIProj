import torch as t

from .sampler import TokenBudgetBatchSampler
from .collate import collate_training_sequences

def loader(
        dataset,
        model_max_seq_len,
        max_tokens_per_batch,
        shuffle: bool = True,
):

    def collate_fn(sequences):
        return collate_training_sequences(
            sequences,
            model_max_seq_len
        )

    if isinstance(dataset, t.utils.data.IterableDataset):
        assert model_max_seq_len <= max_tokens_per_batch

        # shuffle the source stream before wrapping it
        return t.utils.data.DataLoader(
            dataset,
            batch_size=max_tokens_per_batch // model_max_seq_len,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    batch_sampler = TokenBudgetBatchSampler(
        dataset,
        max_tokens_per_batch,
        shuffle=shuffle,
    )

    train_loader = t.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader
