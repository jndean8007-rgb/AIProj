from typing import Protocol, runtime_checkable, Any
from dataclasses import dataclass
import torch as t
from torch_llm.core.aux_outputs import AuxOutputs
from torch_llm.core.batch_meta import BatchMeta


@dataclass(frozen=True)
class MixerCaps:
    """
    describes how engine can treat memory of ealier tokens, only needed for SequenceMixer
    can it chunk, snapshot, roll back or split it across GPUs. Only a mixer has memory across tokens.
    """
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
            state: Any, #state to change to stateview, currently PagedKVCache
    ) -> tuple[t.Tensor, AuxOutputs]:
        ...


@runtime_checkable
class FeedForward(Protocol):
    def forward(
            self,
            x: t.Tensor,  # [T, D]
            meta: BatchMeta,
    ) -> tuple[t.Tensor, AuxOutputs]: ...

    def post_step(self) -> None: ...



@runtime_checkable
class TokenMemory(Protocol):
    def forward(
            self,
            x: t.Tensor,  # [T, D]
            meta: BatchMeta,
    ) -> tuple[t.Tensor, AuxOutputs]: ...



@runtime_checkable
class Residual(Protocol):
    def init(
            self,
            x0: t.Tensor, #[T, D]
    ) -> Any: ... #start new residual stream, NOT a new residual object - built when model is built, returns state
    def read(self, state: Any, i: int) -> t.Tensor: ... #state is

    def write(self, state: Any, i: int, update: t.Tensor) -> Any: ... #returns state

    def final(self, state: Any) -> t.Tensor: ... #final input into head


@runtime_checkable
class OutputHead(Protocol):
    def forward(
            self,
            h: t.Tensor, #[T, D] head input
            meta: BatchMeta,
    ) -> tuple[t.Tensor, AuxOutputs]: ... #logits is [T, V]


