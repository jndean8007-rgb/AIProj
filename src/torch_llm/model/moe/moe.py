import torch as t
import torch.nn as nn
from jaxtyping import Shaped

from core.batch_meta import BatchMeta
from torch_llm.core.aux_outputs import AuxOutputs
from torch_llm.kernals.moe.grouped_swiglu_up import grouped_swiglu_up
from torch_llm.kernals.moe.grouped_down import grouped_down
from torch_llm.model.moe.expert_weights import ExpertWeights
from torch_llm.model.moe.moe_autograd import MoeAutograd
from torch_llm.model.moe.router import Router
from torch_llm.model.moe.dispatch import dispatch
from torch_llm.model.moe.moe_stats import MoeStats

from torch_llm.core.registry import FFNS

@FFNS.register("moe")
class MoE(nn.Module):
    def __init__(
            self,
            config,
            layer_idx
    ):
        super().__init__()
        num_experts = config.num_experts
        top_k = config.top_k
        d_model = config.d_model
        d_ff = config.d_ff
        bias_lr = config.router_bias_lr
        beta = config.router_beta
        self.layer_idx = layer_idx

        self.num_experts = num_experts
        self.top_k = top_k

        self.router = Router(
            num_experts=num_experts,
            top_k=top_k,
            d_model=d_model,
            bias_lr=bias_lr,
            beta=beta,
        )

        self.expert_weights = ExpertWeights(
            num_experts=num_experts,
            d_model=d_model,
            d_ff=d_ff,
        )


    def forward(self, x: Shaped[t.Tensor, 'T d_model'], meta: BatchMeta) -> tuple[t.Tensor, AuxOutputs]:

        T = x.shape[0]
        expert_indices, routing_weights, router_probs = self.router(x)
        flat_exp_sorted, tokens_sorted, flat_routing_sorted, expert_counts, expert_offsets = \
        dispatch(expert_indices, routing_weights, self.num_experts)


        #compute balancing statistics
        expert_fractions = expert_counts / (T * self.top_k)
        mean_router_probs = router_probs.mean(dim=0)

        #compute auxiliary balancing loss
        aux_loss = self.num_experts * t.sum(
            expert_fractions * mean_router_probs
        )
        #utilization statistics
        total_assignments = T * self.top_k
        target_load_per_expert = total_assignments / self.num_experts

        #router diagnostics
        entropy_per_token = -(router_probs * router_probs.clamp_min(1e-9).log()).sum(dim=-1)
        mean_entropy = entropy_per_token.mean()
        top2 = t.topk(router_probs, k=2, dim=-1).values
        routing_margin_per_token = top2[..., 0] - top2[..., 1]
        mean_routing_margin = t.mean(routing_margin_per_token, dim=-1)

        max_expert_fraction = t.max(expert_fractions, dim=-1)[0]
        load_std = t.std(expert_counts.float(), dim=-1, correction=0)
        load_cv = load_std / target_load_per_expert

        stats = MoeStats(
            expert_counts=expert_counts,
            expert_fractions=expert_fractions,
            mean_router_probs=mean_router_probs,

            total_assignments=total_assignments,
            target_load_per_expert=target_load_per_expert,

            aux_loss=aux_loss,

            entropy_per_token=entropy_per_token,
            mean_entropy=mean_entropy,
            routing_margin_per_token=routing_margin_per_token,
            mean_routing_margin=mean_routing_margin,

            max_expert_fraction=max_expert_fraction,
            load_std=load_std,
            load_cv=load_cv,
        )

        if self.training:
            self.router.update_expert_bias(expert_fractions)



        expert_inputs = x[tokens_sorted]


        expert_outputs = MoeAutograd.apply(
            expert_inputs,
            self.expert_weights.gate_weight,
            self.expert_weights.up_weight,
            self.expert_weights.down_weight,
            expert_offsets
        )


        weighted_outputs = expert_outputs * flat_routing_sorted[:, None]
        output = t.zeros_like(x)
        output.index_add_(0, tokens_sorted, weighted_outputs)

        aux_outputs = AuxOutputs(
            losses={"balance": stats.aux_loss},
            metrics={
                "expert_fractions": stats.expert_fractions.detach(),
                "load_cv": stats.load_cv.detach(),
                "max_expert_fraction": stats.max_expert_fraction.detach(),
                "mean_entropy": stats.mean_entropy.detach(),
                "mean_routing_margin": stats.mean_routing_margin.detach(),
            },
        ) #can change if want more metrics

        return output, aux_outputs
