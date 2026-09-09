import torch as t
import torch.nn.functional as F

from torch_llm.model.model_config import ModelConfig
from torch_llm.model.model import TransformerLM
from torch_llm.training.optim.build import build_optimizers
from torch_llm.training.scheduler import build_warmup_cosine_scheduler
from torch_llm.training.train_step import train_step


def test_overfit_tiny_batch():
    t.manual_seed(0)

    device = "cuda"
    dtype = t.bfloat16

    # -------------------------------------------------
    # 1. Tiny model
    # -------------------------------------------------

    config = ModelConfig(
        vocab_size=128,

        d_model=64,
        num_layers=2,

        num_q_heads=4,
        num_kv_heads=2,
        head_dim=16,

        d_ff=128,
        num_experts=4,
        top_k=2,

        model_max_seq_len=16,

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

    # -------------------------------------------------
    # 2. Fixed packed batch
    #
    # seq A: [10, 20, 30, 40, 50]
    # seq B: [7,  8,  9,  10]
    #
    # packed T = 9
    # -------------------------------------------------

    token_ids = t.tensor(
        [10, 20, 30, 40, 50,
         7, 8, 9, 10],
        device=device,
        dtype=t.long,
    )

    lengths = [5, 4]

    cu_seqlens = t.tensor(
        [0, 5, 9],
        device=device,
        dtype=t.int32,
    )

    token_positions = t.tensor(
        [0, 1, 2, 3, 4,
         0, 1, 2, 3],
        device=device,
        dtype=t.int32,
    )

    batch_max_seq_len = max(lengths)

    # -------------------------------------------------
    # 3. Next-token targets
    #
    # Last token of each packed sequence gets ignored.
    # -------------------------------------------------

    ignore_index = -100

    targets = t.tensor(
        [
            20, 30, 40, 50, ignore_index,
            8,  9,  10, ignore_index,
        ],
        device=device,
        dtype=t.long,
    )

    batch = {
        "token_ids": token_ids,
        "targets": targets,
        "cu_seqlens": cu_seqlens,
        "token_positions": token_positions,
        "batch_max_seq_len": batch_max_seq_len,
    }

    # -------------------------------------------------
    # 4. Optimizers
    # -------------------------------------------------

    muon, adamw, _, _ = build_optimizers(
        model,
        muon_lr=0.01,
        adamw_lr=3e-4,
        weight_decay=0.0,
    )

    # -------------------------------------------------
    # 5. Short scheduler
    # -------------------------------------------------

    num_steps = 200

    muon_scheduler = build_warmup_cosine_scheduler(
        muon,
        warmup_steps=10,
        total_steps=num_steps,
        min_lr_ratio=0.1,
    )

    adamw_scheduler = build_warmup_cosine_scheduler(
        adamw,
        warmup_steps=10,
        total_steps=num_steps,
        min_lr_ratio=0.1,
    )

    # -------------------------------------------------
    # 6. Initial loss
    # -------------------------------------------------

    with t.no_grad():
        out = model(
            token_ids=token_ids,
            cu_seqlens=cu_seqlens,
            token_positions=token_positions,
            batch_max_seq_len=batch_max_seq_len,
            mode="train",
        )

        initial_lm_loss = F.cross_entropy(
            out.logits.float(),
            targets,
            ignore_index=ignore_index,
        ).item()

    # -------------------------------------------------
    # 7. Repeatedly train on SAME batch
    # -------------------------------------------------

    final_lm_loss = None

    for step in range(num_steps):

        metrics = train_step(
            model,
            batch,
            muon,
            adamw,
            aux_loss_weight=0.01,
            max_grad_norm=1.0,
            muon_scheduler=muon_scheduler,
            adamw_scheduler=adamw_scheduler,
        )


        final_lm_loss = metrics.lm_loss.item()

        if step % 20 == 0:
            print(
                f"step={step:3d} "
                f"lm_loss={final_lm_loss:.4f} "
                f"aux={metrics.aux_loss.item():.4f}"
            )

    print(
        step,
        metrics.lm_loss.item(),
        metrics.aux_loss.item(),
    )

    # -------------------------------------------------
    # 8. Sanity checks
    # -------------------------------------------------

    assert t.isfinite(t.tensor(initial_lm_loss))
    assert t.isfinite(t.tensor(final_lm_loss))

    print("initial LM loss:", initial_lm_loss)
    print("final LM loss:", final_lm_loss)

    assert final_lm_loss < initial_lm_loss
