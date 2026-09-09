import torch as t
import torch.nn.functional as F

from torch_llm.model.model_config import ModelConfig
from torch_llm.model.model import TransformerLM


def test_transformer_lm_train_backward():
    t.manual_seed(0)

    device = "cuda"
    dtype = t.float16

    # ----------------------------
    # Small model
    # ----------------------------

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

    model.train()

    # ----------------------------
    # Ragged packed batch
    # ----------------------------

    lengths = [5, 3, 7]

    # B = 3
    # T = 15
    cu_seqlens = t.tensor(
        [0, 5, 8, 15],
        device=device,
        dtype=t.int32,
    )

    token_positions = t.cat([
        t.arange(
            length,
            device=device,
            dtype=t.int32,
        )
        for length in lengths
    ])

    T = sum(lengths)
    batch_max_seq_len = max(lengths)

    token_ids = t.randint(
        0,
        config.vocab_size,
        (T,),
        device=device,
    )

    # Arbitrary targets are enough for the integration test.
    targets = t.randint(
        0,
        config.vocab_size,
        (T,),
        device=device,
    )

    # ----------------------------
    # Forward
    # ----------------------------

    out = model(
        token_ids,
        cu_seqlens=cu_seqlens,
        batch_max_seq_len=batch_max_seq_len,
        token_positions=token_positions,
        mode="train",
    )

    # ----------------------------
    # Output contract
    # ----------------------------

    assert out.logits.shape == (
        T,
        config.vocab_size,
    )

    assert len(out.moe_stats) == config.num_layers

    assert out.aux_loss.ndim == 0

    assert t.isfinite(out.logits).all()
    assert t.isfinite(out.aux_loss)

    # ----------------------------
    # Training loss
    # ----------------------------

    lm_loss = F.cross_entropy(
        out.logits.float(),
        targets,
    )

    loss = lm_loss + 0.01 * out.aux_loss

    assert t.isfinite(loss)

    loss.backward()

    # ----------------------------
    # General gradient checks
    # ----------------------------

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue

        assert parameter.grad is not None, (
            f"Missing gradient: {name}"
        )

        assert t.isfinite(parameter.grad).all(), (
            f"Non-finite gradient: {name}"
        )

    # ----------------------------
    # Representative subsystem checks
    # ----------------------------

    # Embedding
    assert model.embedding.weight.grad is not None

    # First decoder block
    block = model.blocks[0]

    # Norms
    assert block.attn_norm.weight.grad is not None
    assert block.moe_norm.weight.grad is not None

    # Attention
    assert block.attention.q_proj.weight.grad is not None
    assert block.attention.k_proj.weight.grad is not None
    assert block.attention.v_proj.weight.grad is not None
    assert block.attention.o_proj.weight.grad is not None

    # Expert parameters
    ew = block.moe.expert_weights

    assert ew.gate_weight.grad is not None
    assert ew.up_weight.grad is not None
    assert ew.down_weight.grad is not None

    assert ew.gate_weight.grad.shape == (
        config.num_experts,
        config.d_ff,
        config.d_model,
    )

    assert ew.up_weight.grad.shape == (
        config.num_experts,
        config.d_ff,
        config.d_model,
    )

    assert ew.down_weight.grad.shape == (
        config.num_experts,
        config.d_model,
        config.d_ff,
    )

    # Router
    assert (
        block.moe.router.routing_weights.weight.grad
        is not None
    )

    # Final norm
    assert model.final_norm.weight.grad is not None

    # LM head
    assert model.lm_head.weight.grad is not None
