from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import accumulate

import torch as t

from torch_llm.inference import InferenceTokenizer
from torch_llm.inference.request_state import RequestState


@dataclass
class _PackedBatch:
    token_ids: t.Tensor
    cu_seqlens: t.Tensor
    token_positions: t.Tensor
    batch_max_seq_len: int

    def to(self, device, non_blocking=False):
        return type(self)(
            token_ids=self.token_ids.to(device, non_blocking=non_blocking),
            cu_seqlens=self.cu_seqlens.to(device, non_blocking=non_blocking),
            token_positions=self.token_positions.to(device, non_blocking=non_blocking),
            batch_max_seq_len=self.batch_max_seq_len,
        )


@dataclass
class PrefillBatch(_PackedBatch):
    """A packed batch of complete prompts."""


@dataclass
class DecodeBatch(_PackedBatch):
    """A packed batch containing exactly one input token per request."""


def build_prefill_batch(
    requests: Iterable[RequestState], device="cpu",
) -> PrefillBatch:
    requests = list(requests)
    if not requests:
        raise ValueError("Cannot build a prefill batch without requests")
    lengths = [len(request.prompt_tokens) for request in requests]
    if min(lengths) <= 0:
        raise ValueError("Prefill requires nonempty prompts")
    # Construct each packed tensor once, rather than launching one copy per prompt.
    return PrefillBatch(
        token_ids=t.tensor(
            [token for request in requests for token in request.prompt_tokens],
            dtype=t.long, device=device,
        ),
        cu_seqlens=t.tensor([0, *accumulate(lengths)], dtype=t.int32, device=device),
        token_positions=t.tensor(
            [position for length in lengths for position in range(length)],
            dtype=t.int32, device=device,
        ),
        batch_max_seq_len=max(lengths),
    )


def build_decode_batch(
    current_batch: PrefillBatch | DecodeBatch,
    next_tokens: Sequence[int] | t.Tensor,
    active_mask: Sequence[bool] | t.Tensor,
) -> DecodeBatch | None:
    device = current_batch.token_ids.device
    batch_size = current_batch.cu_seqlens.numel() - 1
    next_tokens = t.as_tensor(next_tokens, dtype=t.long, device=device)
    if next_tokens.shape != (batch_size,):
        raise ValueError("Tokens must have one entry per request")
    # Index the last actual position, so offset prefill metadata works too.
    next_positions = current_batch.token_positions[current_batch.cu_seqlens[1:] - 1] + 1
    if isinstance(active_mask, t.Tensor):
        mask = active_mask.to(device=device, dtype=t.bool)
        if mask.shape != (batch_size,):
            raise ValueError("The active mask must have one entry per request")
        indices = mask.nonzero(as_tuple=True)[0]
    else:
        if len(active_mask) != batch_size:
            raise ValueError("The active mask must have one entry per request")
        # CPU masks already have a known size, avoiding a CUDA scalar read.
        indices = t.tensor([i for i, keep in enumerate(active_mask) if keep], device=device, dtype=t.long)
    if indices.numel() == 0:
        return None
    return DecodeBatch(
        token_ids=next_tokens[indices],
        cu_seqlens=t.arange(indices.numel() + 1, dtype=t.int32, device=device),
        token_positions=next_positions[indices].to(t.int32),
        batch_max_seq_len=1,
    )


def merge_decode_batches(decode_batches: Iterable[DecodeBatch | None]) -> DecodeBatch | None:
    batches = [batch for batch in decode_batches if batch is not None]
    if not batches:
        return None
    if len(batches) == 1:
        return batches[0]
    device = batches[0].token_ids.device
    if any(batch.token_ids.device != device for batch in batches):
        raise ValueError("Decode batches must be on the same device")
    token_ids = t.cat([batch.token_ids for batch in batches])
    return DecodeBatch(
        token_ids=token_ids,
        cu_seqlens=t.arange(token_ids.numel() + 1, dtype=t.int32, device=device),
        token_positions=t.cat([batch.token_positions for batch in batches]),
        batch_max_seq_len=1,
    )


def batch_tokenize_prompts(prompts: Iterable[str], tokenizer: InferenceTokenizer) -> list[t.Tensor]:
    return [t.tensor(tokenizer.encode(prompt), dtype=t.long) for prompt in prompts]
