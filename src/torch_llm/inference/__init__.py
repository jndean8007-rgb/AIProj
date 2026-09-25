"""Packed batching, paged cache management, and single-owner inference scheduling."""

from typing import Protocol


class InferenceTokenizer(Protocol):
    """The runtime needs prompt encoding and an optional EOS ID."""

    @property
    def eos_token_id(self) -> int | None: ...

    def encode(self, text: str) -> list[int]: ...
