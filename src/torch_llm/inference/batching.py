from dataclasses import dataclass
import torch as t

from torch_llm.data_pipeline.bpe_tokenizer import BPETokenizer


def build_decode_batch(
        current_batch: PrefillBatch | DecodeBatch,
        active_indices: t.Tensor
) -> DecodeBatch:
    #check each active_indice. if exists, add to new structure, with cseqlens =
    #difference in input + previous of current
    device = current_batch.token_ids.device
    decode_ids = t.empty(
        (len(active_indices), current_batch.token_ids.shape[-1]),
        dtype=current_batch.token_ids.dtype,
        device=device,
    )

    lengths = []
    i = 0

    for seq, index in enumerate(current_batch.token_ids):

        if index in active_indices:
            lengths.append(len(seq))
            decode_ids[i] = seq

        i += 1

    cu_seqlens = t.cat([
        t.zeros(1, dtype=t.int32, device=device),
        t.tensor(lengths, dtype=t.int32, device=device).cumsum(dim=0),
    ])

    token_positions = t.cat([
        t.arange(length, device=device, dtype=t.int32)
        for length in lengths
    ])


@dataclass
class DecodeBatch:
    token_ids: t.Tensor
    cu_seqlens: t.Tensor
    token_positions: t.Tensor
    batch_max_seq_len: int






def batch_tokenize_prompts(
        prompts: list[str], # will be changed later to accomodate
        tokenizer: BPETokenizer,
) -> list[t.Tensor]:

    return [
        t.tensor(tokenizer.encode(prompt), dtype=t.int32)
        for prompt in prompts
    ]



def build_prefill_batch(
        token_sequences: list[t.Tensor],
) -> PrefillBatch:
    device = token_sequences[0].device

    token_ids = t.cat([
        seq for seq in token_sequences
    ])

    lengths = [len(seq) for seq in token_ids]

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



@dataclass
class PrefillBatch:
    token_ids: t.Tensor
    cu_seqlens: t.Tensor
    token_positions: t.Tensor
    batch_max_seq_len: int