import torch as t
from jaxtyping import Shaped
from collections import Counter

def dispatch(
        expert_indices: Shaped[t.Tensor, 'T top_k'],
        routing_weights: Shaped[t.Tensor, 'T top_k'],
        num_experts,
):
    # every token creates k assignments, so T * k total
    # flatten expert indices into (T*k), and for each k create k tokens at that position in T
    # ie 0, 0, 0, 1, 1, 1 = token positions
    #flatten routing weights similarly
    T, top_k = routing_weights.shape
    tokens = t.repeat_interleave(t.arange(0, T, device=expert_indices.device), top_k)
    flat_expert = expert_indices.flatten(0)
    flat_routing = routing_weights.flatten(0)

    sort_order = t.argsort(flat_expert)

    flat_exp_sorted = flat_expert[sort_order]
    flat_routing_sorted = flat_routing[sort_order]
    tokens_sorted = tokens[sort_order]

    expert_counts = t.bincount(
        flat_exp_sorted,
        minlength=num_experts,
    )
    expert_offsets = t.cat([
        t.zeros(1, device=expert_indices.device, dtype= expert_indices.dtype),
        t.cumsum(expert_counts, dim=0),
    ])
    return flat_exp_sorted, tokens_sorted, flat_routing_sorted, expert_counts, expert_offsets


