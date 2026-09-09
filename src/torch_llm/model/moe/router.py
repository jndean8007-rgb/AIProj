import torch as t
import torch.nn as nn
from jaxtyping import Shaped

class Router(nn.Module):
    def __init__(self,
                 num_experts,
                 top_k,
                 d_model,
                 bias_lr,
                 beta
                 ):
        super().__init__()
        assert 1 <= top_k <= num_experts
        assert 0 <= beta < 1
        assert bias_lr > 0
        self.top_k = top_k
        self.num_experts = num_experts
        self.d_model = d_model
        self.bias_lr = bias_lr
        self.beta = beta

        self.routing_weights = nn.Linear(d_model, num_experts, bias=False)

        self.register_buffer("ema",
             t.full((num_experts,), 1.0 / num_experts, dtype=t.float32)
        )
        self.register_buffer("expert_bias", t.zeros(self.num_experts)) # statistics derived update this

    def forward(self, x: Shaped[t.Tensor, 'T d_model']):

        router_logits = self.routing_weights(x) #called router_logits

        router_probs = t.softmax(router_logits, dim=-1)

        selection_scores = router_logits + self.expert_bias[None, :] #add expert bias to router logits for selection

        top_k_selections, expert_indices = t.topk(
            selection_scores,
            self.top_k,
            dim=-1,
        )

        top_k_logits = router_logits.gather(-1, expert_indices)
        #get back original top k logits but un exper biased

        routing_weights = t.softmax(top_k_logits, dim=-1)

        return expert_indices, routing_weights, router_probs

    @t.no_grad()
    def update_expert_bias( #working on currently
            self,
            expert_fractions,
    ):

        self.ema = (
                self.beta * self.ema
                + (1 - self.beta) * expert_fractions
        )

        load_error = 1 - self.ema * self.num_experts

        self.expert_bias += self.bias_lr * load_error

