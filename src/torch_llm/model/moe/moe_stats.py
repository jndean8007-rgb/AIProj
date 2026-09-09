from dataclasses import dataclass
import torch as t

@dataclass
class MoeStats:
    expert_counts: t.Tensor
    expert_fractions: t.Tensor
    mean_router_probs: t.Tensor

    total_assignments: int
    target_load_per_expert: float

    aux_loss: t.Tensor

    entropy_per_token: t.Tensor
    mean_entropy: t.Tensor
    routing_margin_per_token: t.Tensor
    mean_routing_margin: t.Tensor

    max_expert_fraction: t.Tensor
    load_std: t.Tensor
    load_cv: t.Tensor
