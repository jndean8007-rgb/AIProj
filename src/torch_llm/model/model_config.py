from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 50257

    d_model: int = 513
    num_layers: int = 6

    num_q_heads: int = 64
    num_kv_heads: int = 8
    head_dim: int = 16

    model_max_seq_len: int = 512
    rope_theta: float = 10_000.0
    rms_eps: float = 1e-8

    d_ff: int = 1368
    num_experts: int = 15
    top_k: int = 3

    router_beta: float = 0.9
    router_bias_lr: float = 0.01

    def __post_init__(self):

        assert self.d_model % self.num_q_heads == 0
        assert self.head_dim == self.d_model // self.num_q_heads

        assert self.num_q_heads % self.num_kv_heads == 0


        assert 1 <= self.top_k <= self.num_experts

        assert self.d_model > 0
        assert self.d_ff > 0
        assert self.num_layers > 0

