
def build_prefill_batch(
        token_sequences: list[t.Tensor],
) -> PrefillBatch:
    device = token_sequences[0].device

    lengths = [len(seq) for seq in token_sequences]

    token_ids = t.cat(token_sequences)

    cu_seqlens = t.cat([
        t.zeros(1, dtype=t.int32, device=device),
        t.tensor(lengths, dtype=t.int32, device=device).cumsum(dim=0),
    ])

    token_positions = t.cat([
        t.arange(length, dtype=t.int32, device=device)
        for length in lengths
    ])

    batch_max_seq_len = max(lengths)

    return PrefillBatch(
        token_ids=token_ids,
        cu_seqlens=cu_seqlens,
        token_positions=token_positions,
        batch_max_seq_len=batch_max_seq_len,
    )