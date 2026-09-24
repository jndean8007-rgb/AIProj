from dataclasses import dataclass

@dataclass
class RequestState:
    request_id: int

    prompt_tokens: list[int]
    generated_tokens: list[int] | list[None]

    cache_slot: int | None

    next_token: int | None
    next_position: int | None

    max_new_tokens: int
    finished: bool