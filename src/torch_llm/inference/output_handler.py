from collections.abc import Sequence
from typing import Protocol

import torch as t


class OutputSink(Protocol):
    def handle_output(self, token_ids: list[int], request_ids: list[int]) -> None: ...


class OutputHandler:
    """Console token diagnostics; implement OutputSink for application streaming."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def handle_output(self, token_ids: Sequence[int] | t.Tensor, request_ids: Sequence[int]):
        if isinstance(token_ids, t.Tensor):
            token_ids = token_ids.tolist()
        for token, request_id in zip(token_ids, request_ids, strict=True):
            #print(f"Stream {request_id} -> {self.tokenizer.decode([token])}")
            print(self.tokenizer.decode([token]), end='')
