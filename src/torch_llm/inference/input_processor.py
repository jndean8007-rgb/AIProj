from collections.abc import Iterable

from torch_llm.inference.batching import PrefillBatch, build_prefill_batch
from torch_llm.inference.request_state import RequestState


class InferenceInputProcessor:
    """Prepare packed prompts for callers that run model forwards directly."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def prepare(self, prompts: Iterable[str], device="cpu") -> PrefillBatch:
        requests = [
            RequestState(request_id, self.tokenizer.encode(prompt), max_new_tokens=0)
            for request_id, prompt in enumerate(prompts)
        ]
        return build_prefill_batch(requests, device)
