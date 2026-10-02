from typing import Protocol, runtime_checkable
from dataclasses import dataclass
from torch import t
from aux_outputs import AuxOutputs
from batch_meta import BatchMeta
from inference.paged_kv_cache import PagedKVCache


@dataclass(frozen=True)
class MixerCaps:
    needs_positions: bool = False #mixer uses token positions
    chunked_prefill: bool = False #prompt can be processed in several pieces with the same result
    prefix_snapshot: bool = False #state after a shared prefix can be saved and reused by other requests
    rollback: bool = False #the state can be cut back after speculative decoding guesses wrong
    context_parallel: bool = False #one sequence can be split across GPUs

    def __post_init__(self):
        if not isinstance(self.needs_positions, bool):
            raise ValueError("needs_positions must be a bool")
        if not isinstance(self.chunked_prefill, bool):
            raise ValueError("chunked_prefill must be a bool")
        if not isinstance(self.prefix_snapshot, bool):
            raise ValueError("prefix_snapshot must be a bool")
        if not isinstance(self.rollback, bool):
            raise ValueError("rollback must be a bool")
        if not isinstance(self.context_parallel, bool):
            raise ValueError("context_parallel must be a bool")

@runtime_checkable
class SequenceMixer(Protocol):
    capabilities: MixerCaps

    def forward(
            self,
            x: t.Tensor, #[T, D]
            meta: BatchMeta,
            state: PagedKVCache, #state to change to stateview
    ) -> tuple[t.Tensor, AuxOutputs]:
        ...


@runtime_checkable
class FeedForward(Protocol):
    ...


@runtime_checkable
class TokenMemory(Protocol):
    ...


@runtime_checkable
class Residual(Protocol):
    ...


@runtime_checkable
class OutputHead(Protocol):
    ...


