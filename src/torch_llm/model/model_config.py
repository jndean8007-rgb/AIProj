from dataclasses import dataclass

@dataclass(frozen=True)
class LayerSpec:


@dataclass
class ModelConfig:
    vocab_size: int = 50257

    d_model: int = 512
    num_layers: int = 6

    num_q_heads: int = 8
    num_kv_heads: int = 4
    head_dim: int = 64

    model_max_seq_len: int = 512
    rope_theta: float = 10_000.0
    rms_eps: float = 1e-8

    d_ff: int = 1408
    num_experts: int = 10
    top_k: int = 3

    router_beta: float = 0.9
    router_bias_lr: float = 0.01

    layers: list[LayerSpec] | None = None
    residual: str = "standard"
    output_head: str = "lm"

    def __post_init__(self):
        if not self.num_q_heads % self.num_kv_heads == 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")

        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError("top_k must be between 1 and num_experts")

        if not self.d_model > 0:
            raise ValueError("d_model must be greater than 0")
        if not self.d_ff > 0:
            raise ValueError("d_ff must be greater than 0")
        if not self.num_layers > 0:
            raise ValueError("num_layers must be greater than 0")

        if self.layers is not None:
            if not len(self.layers) == self.num_layers:
                raise ValueError("Number of layers must be equal to num_layers")
            self.layers = [LayerSpec(**e) if isinstance(e, dict) else e for e in self.layers]
        else:
            self.layers = [LayerSpec() for _ in range(self.num_layers)]

