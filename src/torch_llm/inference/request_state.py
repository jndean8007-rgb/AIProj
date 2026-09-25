from dataclasses import dataclass, field
from typing import Literal


@dataclass
class RequestState:
    """CPU-side state for one request, independent of its current batch row."""

    request_id: int
    prompt_tokens: list[int]
    max_new_tokens: int
    finished: bool = False
    generated_tokens: list[int] = field(default_factory=list)
    finish_reason: Literal["eos", "length", "context_length", "error"] | None = None

    def __post_init__(self):
        if not isinstance(self.request_id, int) or isinstance(self.request_id, bool):
            raise ValueError("request_id must be an integer")
        if not isinstance(self.max_new_tokens, int) or isinstance(self.max_new_tokens, bool) or self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        self.prompt_tokens = list(self.prompt_tokens)
        if not self.prompt_tokens:
            raise ValueError("A request must contain at least one prompt token")
        if any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in self.prompt_tokens):
            raise ValueError("prompt_tokens must contain nonnegative integer token IDs")
