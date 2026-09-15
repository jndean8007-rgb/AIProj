

from torch_llm.data_pipeline.bpe_tokenizer import BPETokenizer
from dataclasses import dataclass

import torch as t



@dataclass
class DecodeBatch:
    token_ids: t.Tensor
    cu_seqlens: t.Tensor
    token_positions: t.Tensor
    batch_max_seq_len: int


def build_decode_batch(
    current_batch: PrefillBatch | DecodeBatch,
    next_tokens: list[int] | t.Tensor,
    active_mask: list[bool] | t.Tensor,
) -> DecodeBatch | None:

    device = current_batch.token_ids.device

    # -----------------------------------------
    # 1. Current number of active sequences
    #    + position of each newly sampled token
    # -----------------------------------------

    if isinstance(current_batch, PrefillBatch):
        current_batch_size = current_batch.cu_seqlens.numel() - 1

        # Prompt lengths.
        #
        # If a prompt currently contains positions:
        #     0, 1, 2, 3
        #
        # its newly generated token belongs at position:
        #     4
        next_positions = (
            current_batch.cu_seqlens[1:]
            - current_batch.cu_seqlens[:-1]
        )

    else:
        current_batch_size = current_batch.token_ids.numel()

        # Each decode step advances each surviving stream
        # by exactly one position.
        next_positions = current_batch.token_positions + 1

    # -----------------------------------------
    # 2. Convert sampled tokens + active mask
    # -----------------------------------------

    next_tokens = t.as_tensor(
        next_tokens,
        dtype=t.long,
        device=device,
    )

    active_mask = t.as_tensor(
        active_mask,
        dtype=t.bool,
        device=device,
    )

    assert next_tokens.ndim == 1
    assert active_mask.ndim == 1

    assert next_tokens.numel() == current_batch_size
    assert active_mask.numel() == current_batch_size

    # -----------------------------------------
    # 3. If every stream finished, there is no
    #    next decode batch.
    # -----------------------------------------

    if not active_mask.any().item():
        return None

    # -----------------------------------------
    # 4. Compact to surviving sequences
    # -----------------------------------------

    decode_ids = next_tokens[active_mask]

    token_positions = next_positions[active_mask].to(
        dtype=t.int32
    )

    # -----------------------------------------
    # 5. Every decode sequence contributes
    #    exactly ONE current input token.
    #
    #    For B_active = 3:
    #
    #    token_ids   = [a, b, c]
    #    cu_seqlens  = [0, 1, 2, 3]
    # -----------------------------------------

    active_batch_size = decode_ids.numel()

    cu_seqlens = t.arange(
        active_batch_size + 1,
        dtype=t.int32,
        device=device,
    )

    return DecodeBatch(
        token_ids=decode_ids,
        cu_seqlens=cu_seqlens,
        token_positions=token_positions,
        batch_max_seq_len=1,
    )






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