from torch_llm.inference.batching import build_prefill_batch, batch_tokenize_prompts, PrefillBatch


class InferenceInputProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def prepare(self, prompts: list[str]) -> PrefillBatch:

        token_sequences = batch_tokenize_prompts(prompts, self.tokenizer)

        return build_prefill_batch(token_sequences)