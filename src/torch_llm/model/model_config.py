from dataclasses import dataclass, field, fields


@dataclass(frozen=True)
class LayerSpec:
    mixer: str = 'gqa'
    ffn: str = 'moe'
    memory: str | None = None #will be tpye of memory entries

    @property
    def num_sublayers(self) -> int:
        return sum(
            getattr(self, field.name) is not None
            for field in fields(self)
        )

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

    layers: list[LayerSpec] = field(default_factory=list)
    residual: str = "standard"
    output_head: str = "lm"

    def __post_init__(self):
        for name in ("d_model", "d_ff", "num_layers", "num_q_heads", "num_kv_heads", "num_experts"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than 0")

        if not self.num_q_heads % self.num_kv_heads == 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")

        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError("top_k must be between 1 and num_experts")

        if self.layers is not None:
            if not len(self.layers) == self.num_layers:
                raise ValueError("Number of layers must be equal to num_layers")
            self.layers = [LayerSpec(**e) if isinstance(e, dict) else e for e in self.layers]
        else:
            self.layers = [LayerSpec() for _ in range(self.num_layers)]


    @property
    def sublayer_indices(self) -> list[int]:
        cumidx = [0]
        for layer in self.layers:
            cumidx.append(cumidx[-1] + layer.num_sublayers)
        return cumidx.copy()
