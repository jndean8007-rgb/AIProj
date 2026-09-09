from dataclasses import dataclass
import torch as t


def pack_training_sequences(
    sequences: list[t.Tensor],
    model_max_seq_len: int,
):
    assert len(sequences) > 0
    assert all(
        2 <= len(seq) <= model_max_seq_len + 1
        for seq in sequences
    )
    assert all(sequence.ndim == 1 for sequence in sequences)

    device = sequences[0].device
    assert all(sequence.device == device for sequence in sequences)

    token_ids = t.cat([
        seq[:-1] for seq in sequences
    ], dim=0)

    targets = t.cat([
        seq[1:] for seq in sequences
    ], dim=0)

    lengths = [len(seq) - 1 for seq in sequences]

    cu_seqlens = t.cat([
        t.zeros(1, dtype=t.int32, device=device),
        t.tensor(lengths, dtype=t.int32, device=device).cumsum(dim=0),
    ])

    token_positions = t.cat([
        t.arange(length, device=device, dtype=t.int32)
        for length in lengths
    ])

    batch_max_seq_len = max(lengths)

    return PackedTrainingBatch(
        token_ids,
        targets,
        cu_seqlens,
        token_positions,
        batch_max_seq_len,
    )


@dataclass
class PackedTrainingBatch:
    token_ids: t.Tensor
    targets: t.Tensor
    cu_seqlens: t.Tensor
    token_positions: t.Tensor
    batch_max_seq_len: int

    def to(self, device, non_blocking=False):
        return PackedTrainingBatch(
            token_ids=self.token_ids.to(device),
            targets=self.targets.to(device),
            cu_seqlens=self.cu_seqlens.to(device),
            token_positions=self.token_positions.to(device),
            batch_max_seq_len=self.batch_max_seq_len,
        )