from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch as t
from jaxtyping import Shaped


@dataclass(frozen=True)
class FullKVSpec:


@dataclass(frozen=True)
class PagedKVRead:
    k: Shaped[t.Tensor, "num_blocks block_size H_kv head_dim"]
    v: Shaped[t.Tensor, "num_blocks block_size H_kv head_dim"]
    k_scales:
    v_scales:



@runtime_checkable
class FullKVView(Protocol):
    def append(self, k: Shaped[t.Tensor, 'T H_kv head_dim'], v: Shaped[t.Tensor, 'T H_kv head_dim']) -> None: ...

    def read(self) -> PagedKVRead: ...
