"""Contract tests for Phase 0b step 2: interfaces, registries, LayerSpec, generic block (handoff D15).

Where things live
  torch_llm/core/interfaces.py  Protocols (runtime_checkable) + MixerCaps (frozen dataclass)
  torch_llm/core/registry.py    Registry class + MIXERS, FFNS, RESIDUALS, TOKEN_MEMORIES, OUTPUT_HEADS
  torch_llm/model/components.py imports every implementation module, so registration happens
                                 (a registry is only populated once the defining module is imported)
  torch_llm/model/model_config.py  LayerSpec + ModelConfig.layers / residual / output_head
  torch_llm/model/checkpoint_migration.py  migrate_state_dict for pre-refactor checkpoints

Interfaces (every sublayer returns an update plus AuxOutputs, so any of them can add
losses later without an interface change):
  SequenceMixer.forward(x [T,D], meta, state) -> (update [T,D], AuxOutputs); .capabilities: MixerCaps
  FeedForward.forward(x [T,D], meta) -> (update [T,D], AuxOutputs); .post_step() -> None
  TokenMemory.forward(x [T,D], meta) -> (update [T,D], AuxOutputs)   (no implementation yet)
  Residual.init(x0) -> state; .read(state, i) -> [T,D]; .write(state, i, update) -> state;
           .final(state) -> [T,D]      (i counts sublayers across the whole model, in order)
  OutputHead.forward(h [T,D], meta) -> (logits [T,V], AuxOutputs)

Construction: components are built only through registries, never by importing classes
  MIXERS/FFNS/TOKEN_MEMORIES: cls(config, layer_idx); RESIDUALS: cls(config);
  OUTPUT_HEADS: cls(config, embedding)  (the LM head ties its weight to the embedding)

Registered names in this step: MIXERS "gqa", FFNS "moe", RESIDUALS "standard", OUTPUT_HEADS "lm".

LayerSpec(mixer="gqa", ffn="moe", memory=None). ModelConfig.layers defaults to num_layers copies;
if given, its length must equal num_layers. ModelConfig.residual="standard", .output_head="lm".

MixerCaps flags are claims the engine relies on, so a mixer only sets a flag once a contract
test proves it: for GQA today only needs_positions is True.

Stateful non-gradient updates (MoE router bias balancing) never happen inside forward: forward
may run 0..n times per optimizer step (evaluation, gradient accumulation, activation
recomputation). The FFN accumulates what it needs during training forwards and applies it
once in post_step(); TransformerLM.post_step() calls post_step() on every FFN.

The LM head owns the final norm (head = norm + tied projection), because each future head
(e.g. MTP depths) normalizes its own input.

Checkpoints: blocks.{i}.attention / attn_norm / moe / moe_norm become
blocks.{i}.mixer / mixer_norm / ffn / ffn_norm, and top-level final_norm / lm_head become
head.norm / head.proj. migrate_state_dict maps old keys to new ones, is a no-op on
new-format dicts, and generate_setup applies it when loading.
"""

import pytest
import torch as t

from torch_llm.core.aux_outputs import AuxOutputs
from torch_llm.core.batch_meta import BatchMeta
from torch_llm.core.interfaces import FeedForward, MixerCaps, OutputHead, Residual, SequenceMixer
from torch_llm.core.registry import FFNS, MIXERS, OUTPUT_HEADS, RESIDUALS, Registry
from torch_llm.model.model_config import LayerSpec, ModelConfig

import torch_llm.model.model  # noqa: F401  (building the model must populate the registries)

requires_cuda = pytest.mark.skipif(not t.cuda.is_available(), reason="attention and MoE kernels require CUDA")


def tiny_config(**overrides):
    fields = dict(
        vocab_size=128,
        d_model=64,
        num_layers=2,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=16,
        d_ff=128,
        num_experts=4,
        top_k=2,
        model_max_seq_len=64,
    )
    fields.update(overrides)
    return ModelConfig(**fields)


def train_meta(lengths, device="cuda"):
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)

    return BatchMeta(
        token_ids=t.zeros(sum(lengths), dtype=t.long, device=device),
        cu_seqlens=t.tensor(cu, dtype=t.int32, device=device),
        token_positions=t.cat([t.arange(n) for n in lengths]).to(device),
        max_seqlen=max(lengths),
        mode="train",
        cache_context=None,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_register_and_get():
    registry = Registry("widgets")

    @registry.register("a")
    class A:
        pass

    assert registry.get("a") is A
    assert "a" in registry.names()


def test_registry_rejects_duplicates():
    registry = Registry("widgets")
    registry.register("a")(type("A", (), {}))

    with pytest.raises(ValueError):
        registry.register("a")(type("B", (), {}))


def test_registry_unknown_name_lists_available_names():
    registry = Registry("widgets")
    registry.register("alpha")(type("A", (), {}))

    with pytest.raises(KeyError, match="alpha"):
        registry.get("beta")


def test_current_components_are_registered():
    assert "gqa" in MIXERS.names()
    assert "moe" in FFNS.names()
    assert "standard" in RESIDUALS.names()
    assert "lm" in OUTPUT_HEADS.names()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_layers_default_to_current_architecture():
    config = tiny_config(num_layers=3)
    assert config.layers == [LayerSpec(mixer="gqa", ffn="moe", memory=None)] * 3
    assert config.residual == "standard" and config.output_head == "lm"


def test_explicit_layers_must_match_num_layers():
    with pytest.raises(ValueError):
        tiny_config(num_layers=3, layers=[LayerSpec()] * 2)


# ---------------------------------------------------------------------------
# Interface conformance (construction only; runs on CPU)
# ---------------------------------------------------------------------------

def test_components_built_from_registries_satisfy_interfaces():
    config = tiny_config()
    embedding = t.nn.Embedding(config.vocab_size, config.d_model)

    mixer = MIXERS.get("gqa")(config, 0)
    ffn = FFNS.get("moe")(config, 0)
    residual = RESIDUALS.get("standard")(config)
    head = OUTPUT_HEADS.get("lm")(config, embedding)

    assert isinstance(mixer, SequenceMixer)
    assert isinstance(ffn, FeedForward)
    assert isinstance(residual, Residual)
    assert isinstance(head, OutputHead)


def test_gqa_capabilities_claim_only_what_is_tested():
    caps = MIXERS.get("gqa")(tiny_config(), 0).capabilities
    assert isinstance(caps, MixerCaps)
    assert caps.needs_positions is True
    assert not any([caps.chunked_prefill, caps.prefix_snapshot, caps.rollback, caps.context_parallel])


def test_mixer_caps_is_frozen():
    caps = MixerCaps(needs_positions=True)
    with pytest.raises(Exception):
        caps.rollback = True


def test_lm_head_ties_weight_to_embedding():
    config = tiny_config()
    embedding = t.nn.Embedding(config.vocab_size, config.d_model)
    head = OUTPUT_HEADS.get("lm")(config, embedding)
    tied = [p for p in head.parameters() if p is embedding.weight]
    assert tied, "the LM head must share the embedding weight"


def test_model_blocks_follow_layer_specs():
    from torch_llm.model.model import TransformerLM

    model = TransformerLM(tiny_config())
    for block in model.blocks:
        assert isinstance(block.mixer, SequenceMixer)
        assert isinstance(block.ffn, FeedForward)


# ---------------------------------------------------------------------------
# Standard residual algebra (CPU)
# ---------------------------------------------------------------------------

def test_standard_residual_accumulates_updates():
    residual = RESIDUALS.get("standard")(tiny_config())
    x0 = t.randn(5, 64)
    updates = [t.randn(5, 64) for _ in range(4)]

    state = residual.init(x0)
    running = x0.clone()
    for i, update in enumerate(updates):
        t.testing.assert_close(residual.read(state, i), running)
        state = residual.write(state, i, update)
        running = running + update

    t.testing.assert_close(residual.final(state), running)


def test_standard_residual_write_is_not_in_place():
    # Autograd needs the pre-write stream (the next norm's backward reads it), so write
    # must return a new state rather than modifying the old one.
    residual = RESIDUALS.get("standard")(tiny_config())
    x0 = t.randn(5, 64)
    state = residual.init(x0)
    before = residual.read(state, 0).clone()

    residual.write(state, 0, t.ones(5, 64))

    t.testing.assert_close(residual.read(state, 0), before)
    t.testing.assert_close(x0, before)


# ---------------------------------------------------------------------------
# Checkpoint migration (CPU)
# ---------------------------------------------------------------------------

BLOCK_RENAMES = {".attention.": ".mixer.", ".attn_norm.": ".mixer_norm.", ".moe.": ".ffn.", ".moe_norm.": ".ffn_norm."}
TOP_LEVEL_RENAMES = {"final_norm.": "head.norm.", "lm_head.": "head.proj."}


def to_old_format(state_dict):
    renamed = {}
    for key, value in state_dict.items():
        for old, new in BLOCK_RENAMES.items():
            key = key.replace(new, old)

        for old, new in TOP_LEVEL_RENAMES.items():
            if key.startswith(new):
                key = old + key[len(new):]

        renamed[key] = value
    return renamed


def test_old_checkpoint_keys_migrate_and_load():
    from torch_llm.model.checkpoint_migration import migrate_state_dict
    from torch_llm.model.model import TransformerLM

    source, target = TransformerLM(tiny_config()), TransformerLM(tiny_config())
    old = to_old_format(source.state_dict())
    assert any(".attention." in key for key in old)
    assert "final_norm.weight" in old and "lm_head.weight" in old

    target.load_state_dict(migrate_state_dict(old))

    for key, value in source.state_dict().items():
        t.testing.assert_close(target.state_dict()[key], value)


def test_migration_is_a_no_op_on_new_format():
    from torch_llm.model.checkpoint_migration import migrate_state_dict
    from torch_llm.model.model import TransformerLM

    state_dict = TransformerLM(tiny_config()).state_dict()
    assert list(migrate_state_dict(dict(state_dict))) == list(state_dict)


# ---------------------------------------------------------------------------
# Generic mixer / FFN contracts (CUDA)
# ---------------------------------------------------------------------------

@requires_cuda
@t.inference_mode()
@pytest.mark.parametrize("name", ["gqa"])
def test_mixer_shape_and_aux(name):
    mixer = MIXERS.get(name)(tiny_config(), 0).cuda().eval()
    x = t.randn(12, 64, device="cuda")

    update, aux = mixer(x, train_meta([5, 7]), None)

    assert update.shape == x.shape
    assert isinstance(aux, AuxOutputs)


@requires_cuda
@t.inference_mode()
@pytest.mark.parametrize("name", ["gqa"])
def test_mixer_isolates_sequences_and_is_causal(name):
    # Packed varlen batches: no information may cross a cu_seqlens boundary, and within
    # a sequence a token may only depend on earlier tokens.
    mixer = MIXERS.get(name)(tiny_config(), 0).cuda().eval()
    meta = train_meta([5, 7])
    x = t.randn(12, 64, device="cuda")
    base, _ = mixer(x, meta, None)

    changed = x.clone()
    changed[5:] = t.randn(7, 64, device="cuda")  # all of sequence 1
    out, _ = mixer(changed, meta, None)
    t.testing.assert_close(out[:5], base[:5], rtol=0, atol=0)

    changed = x.clone()
    changed[4] = t.randn(64, device="cuda")  # last token of sequence 0
    out, _ = mixer(changed, meta, None)
    t.testing.assert_close(out[:4], base[:4], rtol=0, atol=0)
    t.testing.assert_close(out[5:], base[5:], rtol=0, atol=0)


@requires_cuda
@t.inference_mode()
@pytest.mark.parametrize("name", ["moe"])
def test_ffn_is_per_token(name):
    # An FFN maps each token independently, so permuting tokens permutes the output.
    # (Eval mode: no training-time routing state.)
    ffn = FFNS.get(name)(tiny_config(), 0).cuda().eval()
    meta = train_meta([5, 7])
    x = t.randn(12, 64, device="cuda")
    permutation = t.randperm(12, device="cuda")

    base, aux = ffn(x, meta)
    permuted, _ = ffn(x[permutation], meta)

    assert base.shape == x.shape and isinstance(aux, AuxOutputs)
    t.testing.assert_close(permuted, base[permutation], rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Stateful updates happen in post_step, never in forward (CUDA)
# ---------------------------------------------------------------------------

def expert_biases(model):
    return [buffer.clone() for name, buffer in model.named_buffers() if name.endswith("expert_bias")]


@requires_cuda
def test_router_bias_updates_only_in_post_step():
    from torch_llm.model.model import TransformerLM

    t.manual_seed(0)
    model = TransformerLM(tiny_config()).cuda().train()
    meta = train_meta([9, 11])
    meta = BatchMeta(
        token_ids=t.randint(0, 128, (20,), device="cuda"),
        cu_seqlens=meta.cu_seqlens,
        token_positions=meta.token_positions,
        max_seqlen=meta.max_seqlen,
        mode="train",
        cache_context=None,
    )
    initial = expert_biases(model)
    assert initial, "expected expert_bias buffers in the MoE routers"

    model(meta)
    for before, after in zip(initial, expert_biases(model)):
        t.testing.assert_close(after, before, rtol=0, atol=0)

    model.post_step()
    updated = expert_biases(model)
    assert any(not t.equal(a, b) for a, b in zip(initial, updated)), "post_step must apply the bias update"

    model.post_step()  # nothing new accumulated since the last post_step
    for before, after in zip(updated, expert_biases(model)):
        t.testing.assert_close(after, before, rtol=0, atol=0)

    model.eval()
    with t.inference_mode():
        model(meta)
    model.post_step()  # eval forwards must not accumulate routing statistics
    for before, after in zip(updated, expert_biases(model)):
        t.testing.assert_close(after, before, rtol=0, atol=0)
