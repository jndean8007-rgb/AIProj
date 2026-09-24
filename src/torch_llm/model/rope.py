import torch as t
from jaxtyping import Shaped

class RoPE(t.nn.Module):
    def __init__(self,
                 d_q: int, #dimension of query
                 model_max_seq_len: int, #maximum length of token sequence inputted
                 theta: float = 10000.0,
                 ):
        super().__init__()
        assert d_q % 2 == 0, "RoPE requires an even d_k"
        self.d_q = d_q
        self.model_max_seq_len = model_max_seq_len
        self.theta = theta

        #create sin and cosine buffer tables
        pair_indices = t.arange(d_q // 2)

        #exponents depend on position in sequence of pair relative to total size of query/key vector
        exponents = 2 * pair_indices / d_q
        denominator = theta ** exponents

        #positions of each token embedding according to position n in sequence
        positions = t.arange(model_max_seq_len) #also can be thought of as inverse frequencies

        #produces shape of (model_max_seq_len, d_k / 2) -- because positions and denominator are each 1d we use this syntax to allow
        #each position element to be divided by each denominator element
        angles = positions[:, None] / denominator[None, :]

        cos_table = t.cos(angles)
        sin_table = t.sin(angles)

        self.register_buffer('cos_table', cos_table, persistent=False,)
        self.register_buffer('sin_table', sin_table, persistent=False,)


    def forward(self,
                x: Shaped[t.Tensor, 't num_heads head_dim'], #t represents packed sequence length aggregation over b
                token_pos: Shaped[t.Tensor, 't'],
    ) -> Shaped[t.Tensor, 't num_heads head_dim']:

        #resolve token positions to correctly retrieve trig tables
        cos = self.cos_table[token_pos].unsqueeze(-2).to(x.dtype)
        sin = self.sin_table[token_pos].unsqueeze(-2).to(x.dtype)

        x_first = x[..., 0::2]
        x_second = x[..., 1::2]

        # rotate according to rotation matrix of corresponding positions
        rotated_first = cos * x_first - sin * x_second
        rotated_second = sin * x_first + cos * x_second
        # dimensions are each 'T num heads head_dim / 2' need to restore

        # stack pairs corresponding elements along a new dimension, flatten(-2) removes temporary pair dimension, flattening last 2 dims
        return t.stack([rotated_first, rotated_second], dim=-1).flatten(-2)




