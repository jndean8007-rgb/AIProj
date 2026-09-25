import math

import torch as t


def sample(
    logits: t.Tensor, tokenizer=None, *, temperature: float = 0.0,
    top_k: int | None = None, top_p: float = 1.0, generator: t.Generator | None = None,
) -> t.Tensor:
    """Sample one token per row; temperature=0 preserves greedy decoding.

    tokenizer is retained for compatibility and is not used. A configured sampler
    (for example functools.partial(sample, temperature=0.8)) can be passed to the
    runtime. The generator must match the logits device.
    """
    if logits.ndim != 2 or logits.shape[-1] == 0 or not logits.is_floating_point():
        raise ValueError("logits must be a floating-point [batch, vocabulary] tensor")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0):
        raise ValueError("top_k must be a positive integer")
    if temperature == 0:
        return logits.argmax(dim=-1)
    scores = logits.float() / temperature
    if top_k is not None:
        threshold = scores.topk(min(top_k, scores.shape[-1]), dim=-1).values[:, -1:]
        scores = scores.masked_fill(scores < threshold, -t.inf)
    if top_p < 1:
        sorted_scores, sorted_indices = scores.sort(dim=-1, descending=True)
        cumulative = sorted_scores.softmax(dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        scores = t.full_like(scores, -t.inf).scatter(
            -1, sorted_indices, sorted_scores.masked_fill(remove, -t.inf),
        )
    return t.multinomial(scores.softmax(dim=-1), 1, generator=generator).squeeze(-1)
