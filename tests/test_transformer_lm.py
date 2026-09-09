import torch as t

from torch_llm.model.model_config import ModelConfig
from torch_llm.model.model import TransformerLM
from torch_llm.inference.kv_cache import KVCache


def make_layer_caches(
    config,
    batch_size,
    device,
    dtype,
):
    """
    One independent KV cache for every decoder layer.
    """

    return [
        KVCache(
            batch_size=batch_size,
            cache_max_seq_len=config.model_max_seq_len,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            device=device,
            dtype=dtype,
        )
        for _ in range(config.num_layers)
    ]


@t.no_grad()
def test_transformer_prefill_decode_matches_full_forward():
    t.manual_seed(0)

    device = "cuda"
    dtype = t.float16

    # -------------------------------------------------
    # Small deterministic model
    # -------------------------------------------------

    config = ModelConfig(
        vocab_size=256,

        d_model=64,
        num_layers=2,

        num_q_heads=4,
        num_kv_heads=2,
        head_dim=16,

        d_ff=128,
        num_experts=4,
        top_k=2,

        model_max_seq_len=32,

        rms_eps=1e-6,
        rope_theta=10000.0,

        router_beta=0.9,
        router_bias_lr=0.01,
    )

    model = TransformerLM(config).to(
        device=device,
        dtype=dtype,
    )

    model.eval()

    # -------------------------------------------------
    # One sequence of length 7
    #
    # Full path:
    #     [0 1 2 3 4 5 6]
    #
    # Cached path:
    #     prefill [0 1 2 3]
    #     decode  [4]
    #     decode  [5]
    #     decode  [6]
    # -------------------------------------------------

    tokens = t.tensor(
        [17, 42, 8, 91, 33, 7, 124],
        device=device,
        dtype=t.long,
    )

    seq_len = tokens.numel()

    # =================================================
    # PATH A:
    # full causal computation
    # =================================================

    full_cu_seqlens = t.tensor(
        [0, seq_len],
        device=device,
        dtype=t.int32,
    )

    full_positions = t.arange(
        seq_len,
        device=device,
        dtype=t.int32,
    )

    full_out = model(
        tokens,
        cu_seqlens=full_cu_seqlens,
        batch_max_seq_len=seq_len,
        token_positions=full_positions,
        mode="train",
        kv_caches=None,
    )

    logits_full = full_out.logits.detach()

    assert logits_full.shape == (
        seq_len,
        config.vocab_size,
    )

    # =================================================
    # PATH B:
    # prefill + iterative decode
    # =================================================

    kv_caches = make_layer_caches(
        config=config,
        batch_size=1,
        device=device,
        dtype=dtype,
    )

    # -------------------------
    # Prefill first four tokens
    # -------------------------

    prefill_len = 4

    prefill_tokens = tokens[:prefill_len]

    prefill_cu_seqlens = t.tensor(
        [0, prefill_len],
        device=device,
        dtype=t.int32,
    )

    prefill_positions = t.arange(
        prefill_len,
        device=device,
        dtype=t.int32,
    )

    prefill_out = model(
        prefill_tokens,
        cu_seqlens=prefill_cu_seqlens,
        batch_max_seq_len=prefill_len,
        token_positions=prefill_positions,
        mode="prefill",
        kv_caches=kv_caches,
    )

    cached_logits = [
        prefill_out.logits.detach()
    ]

    # -------------------------
    # Decode remaining tokens
    # -------------------------

    for position in range(prefill_len, seq_len):

        decode_token = tokens[position:position + 1]

        # One active sequence containing one new token.
        decode_cu_seqlens = t.tensor(
            [0, 1],
            device=device,
            dtype=t.int32,
        )

        # Crucial:
        # RoPE position is the GLOBAL sequence position,
        # not zero just because this call contains one token.
        decode_position = t.tensor(
            [position],
            device=device,
            dtype=t.int32,
        )

        decode_out = model(
            decode_token,
            cu_seqlens=decode_cu_seqlens,
            batch_max_seq_len=1,
            token_positions=decode_position,
            mode="decode",
            kv_caches=kv_caches,
        )

        cached_logits.append(
            decode_out.logits.detach()
        )

    logits_cached = t.cat(
        cached_logits,
        dim=0,
    )

    # =================================================
    # Compare complete sequence
    # =================================================

    assert logits_cached.shape == logits_full.shape

    max_error = (
        logits_cached.float()
        - logits_full.float()
    ).abs().max()

    print(
        "maximum full-vs-cached logit error:",
        max_error.item(),
    )

    t.testing.assert_close(
        logits_cached.float(),
        logits_full.float(),
        rtol=3e-2,
        atol=3e-2,
    )
