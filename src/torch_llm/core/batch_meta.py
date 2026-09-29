from __future__ import annotations

from dataclasses import dataclass, replace
import torch as t

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from torch_llm.inference.cache_manager import CacheContainer

@dataclass(frozen=True)
class BatchMeta: #need to annotate actual shapes
    token_ids: t.Tensor
    cu_seqlens: t.Tensor
    token_positions: t.Tensor
    max_seqlen: int
    mode: Literal['train', 'prefill', 'decode']

    cache_context: CacheContainer | None

    def __post_init__(self):
        if self.cu_seqlens.ndim != 1:
            raise ValueError('cu_seqlens.ndim must be equal to 1')
        if self.token_ids.ndim != 1:
            raise ValueError('token_ids.ndim must be equal to 1')
        if self.token_ids.dtype not in (t.int8, t.int16, t.int32, t.int64, t.uint8):
            raise ValueError('token_ids.dtype must be an torch integer')
        if self.token_positions.dtype not in (t.int8, t.int16, t.int32, t.int64, t.uint8):
            raise ValueError('token_positions.dtype must be an torch integer')
        if self.cu_seqlens.dtype != t.int32:
            raise ValueError('cu_seqlens.dtype must be t.int32')
        if self.token_positions.shape != self.token_ids.shape:
            raise ValueError('token_positions.shape must be equal to token_ids.shape')
        if not isinstance(self.max_seqlen, int):
            raise ValueError('max_seqlen must be an integer')
        if isinstance(self.max_seqlen, bool):
            raise ValueError('max_seqlen cannot be a bool')
        if self.max_seqlen <= 0:
            raise ValueError('max_seqlen must be positive')
        if not self.token_ids.device == self.cu_seqlens.device == self.token_positions.device:
            raise ValueError("Tensors must be on the same device")
        if self.mode not in ("train", "prefill", "decode"):
            raise ValueError('mode must be one of ("train", "prefill", "decode")')
        if self.mode in ("prefill", "decode") and self.cache_context is None:
            raise ValueError('cache_context must be set when mode is decode or prefill')
        if self.mode == 'train' and self.cache_context is not None:
            raise ValueError("cache_context must be none when mode is train")
        if self.mode == 'decode' and self.num_tokens != self.num_sequences:
            raise ValueError("num_tokens must be equal to num_sequences in decode")



    @property
    def num_tokens(self) -> int:
        return self.token_ids.size(-1)


    @property
    def num_sequences(self) -> int:
        return self.cu_seqlens.size(-1) - 1


    def to(self, device) -> BatchMeta:
        return replace(
            self,
            token_ids=self.token_ids.to(device),
            cu_seqlens=self.cu_seqlens.to(device),
            token_positions=self.token_positions.to(device)
        )