"""Contract tests for Phase 0b step 1: BatchMeta and AuxOutputs (handoff D14).

Both live in `torch_llm/core/` because the data pipeline, the inference engine and
the model all use them; putting them under `model/` would make the engine and the
data pipeline depend on model internals.

BatchMeta (torch_llm/core/batch_meta.py): one frozen description of a forward pass.
  fields: token_ids [T] int, positions [T] int, cu_seqlens [B+1] int32,
          max_seqlen: int (a Python int, never a tensor), mode: "train" | "prefill" | "decode",
          cache_context: CacheContainer | None  (transitional; replaced by state views in step 3)
  properties: num_tokens (T), num_sequences (B)  -- from shapes only
  method: to(device) -> new BatchMeta
  validation happens at construction and must never read tensor *values*
  (that would force a device->host sync on every forward); only shapes, dtypes,
  devices and Python values are checked. Tests enforce this with tensors on the
  "meta" device, where any value read raises.

AuxOutputs (torch_llm/core/aux_outputs.py): the open channel for losses and metrics.
  fields: losses: dict[str, Tensor] (0-dim), metrics: dict[str, Any] (no autograd graph)
  merged(other, prefix) -> new AuxOutputs with other's keys prefixed "prefix.key"; collisions raise
  total_loss(weights) -> weighted sum where each loss is grouped by its base name (the
                         last dotted component), averaged within the group, then weighted.
                         A loss with no weight raises, so a new loss is never silently ignored.
"""

import dataclasses

import pytest
import torch as t

from torch_llm.core.aux_outputs import AuxOutputs
from torch_llm.core.batch_meta import BatchMeta


def make_meta(device="cpu", mode="train", lengths=(3, 5), cache_context=None, **overrides):
    total = sum(lengths)
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    fields = dict(
        token_ids=t.zeros(total, dtype=t.long, device=device),
        positions=t.cat([t.arange(n) for n in lengths]).to(device) if device != "meta" else t.empty(total, dtype=t.long, device="meta"),
        cu_seqlens=t.tensor(cu, dtype=t.int32).to(device) if device != "meta" else t.empty(len(cu), dtype=t.int32, device="meta"),
        max_seqlen=max(lengths),
        mode=mode,
        cache_context=cache_context,
    )
    fields.update(overrides)
    return BatchMeta(**fields)


CACHE = object()  # stands in for a CacheContainer; BatchMeta only checks presence


# ---------------------------------------------------------------------------
# BatchMeta
# ---------------------------------------------------------------------------

def test_valid_train_meta_exposes_fields_and_sizes():
    meta = make_meta()
    assert meta.mode == "train" and meta.max_seqlen == 5
    assert meta.num_tokens == 8 and meta.num_sequences == 2
    assert meta.cache_context is None


def test_meta_is_frozen():
    meta = make_meta()
    with pytest.raises(dataclasses.FrozenInstanceError):
        meta.mode = "decode"


@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_inference_modes_require_cache_context(mode):
    lengths = (1, 1) if mode == "decode" else (3, 5)
    with pytest.raises(ValueError):
        make_meta(mode=mode, lengths=lengths, cache_context=None)
    assert make_meta(mode=mode, lengths=lengths, cache_context=CACHE).mode == mode


def test_train_mode_rejects_cache_context():
    with pytest.raises(ValueError):
        make_meta(mode="train", cache_context=CACHE)


def test_decode_requires_one_token_per_sequence():
    # T == B is checkable from shapes alone, so it is checked without a sync.
    with pytest.raises(ValueError):
        make_meta(mode="decode", lengths=(1, 2), cache_context=CACHE)


@pytest.mark.parametrize("overrides", [
    {"mode": "verify"},                                          # not a supported mode yet
    {"positions": t.zeros(7, dtype=t.long)},                     # length != T
    {"token_ids": t.zeros(8, dtype=t.float32)},                  # not integer
    {"token_ids": t.zeros(2, 4, dtype=t.long)},                  # not 1-D
    {"cu_seqlens": t.tensor([0, 3, 8], dtype=t.int64)},          # kernels expect int32
    {"cu_seqlens": t.tensor([[0, 3, 8]], dtype=t.int32)},        # not 1-D
    {"max_seqlen": t.tensor(5)},                                 # tensor would force a sync to use
    {"max_seqlen": 0},
    {"max_seqlen": True},
])
def test_invalid_meta_is_rejected(overrides):
    with pytest.raises(ValueError):
        make_meta(**overrides)


def test_mixed_devices_are_rejected():
    with pytest.raises(ValueError):
        make_meta(positions=t.empty(8, dtype=t.long, device="meta"))


@pytest.mark.parametrize("mode,lengths,cache", [("train", (3, 5), None), ("prefill", (3, 5), CACHE), ("decode", (1, 1, 1), CACHE)])
def test_validation_never_reads_tensor_values(mode, lengths, cache):
    # Tensors on the "meta" device have shapes and dtypes but no data: reading a
    # value (.item(), .tolist(), bool(tensor), comparisons used in `if`) raises.
    meta = make_meta(device="meta", mode=mode, lengths=lengths, cache_context=cache)
    assert meta.num_tokens == sum(lengths) and meta.num_sequences == len(lengths)


def test_to_moves_tensors_and_keeps_everything_else():
    meta = make_meta(mode="prefill", cache_context=CACHE)
    moved = meta.to("meta")
    assert moved is not meta
    for name in ("token_ids", "positions", "cu_seqlens"):
        assert getattr(moved, name).device.type == "meta"
        assert getattr(moved, name).shape == getattr(meta, name).shape
        assert getattr(moved, name).dtype == getattr(meta, name).dtype
    assert (moved.max_seqlen, moved.mode, moved.cache_context) == (meta.max_seqlen, meta.mode, meta.cache_context)
    assert meta.token_ids.device.type == "cpu"  # original untouched


# ---------------------------------------------------------------------------
# AuxOutputs
# ---------------------------------------------------------------------------

def test_empty_aux_outputs():
    aux = AuxOutputs()
    assert aux.losses == {} and aux.metrics == {}
    assert aux.total_loss({}).item() == 0.0


def test_merged_prefixes_keys_and_leaves_inputs_unchanged():
    parent = AuxOutputs(losses={"mtp": t.tensor(1.0)})
    child = AuxOutputs(losses={"balance": t.tensor(2.0)}, metrics={"expert_counts": t.tensor([3, 1])})
    merged = parent.merged(child, prefix="layers.0.ffn")
    assert set(merged.losses) == {"mtp", "layers.0.ffn.balance"}
    assert set(merged.metrics) == {"layers.0.ffn.expert_counts"}
    assert set(parent.losses) == {"mtp"} and set(child.losses) == {"balance"}


def test_merge_collision_raises():
    a = AuxOutputs(losses={"layers.0.ffn.balance": t.tensor(1.0)})
    b = AuxOutputs(losses={"balance": t.tensor(2.0)})
    with pytest.raises(ValueError):
        a.merged(b, prefix="layers.0.ffn")


def test_losses_must_be_scalars():
    with pytest.raises(ValueError):
        AuxOutputs(losses={"balance": t.tensor([1.0, 2.0])})


def test_metrics_must_not_carry_autograd_graphs():
    # A metric holding a graph keeps activations alive and can be backpropagated by accident.
    x = t.tensor(2.0, requires_grad=True)
    with pytest.raises(ValueError):
        AuxOutputs(metrics={"router_entropy": x * 3})
    AuxOutputs(metrics={"router_entropy": (x * 3).detach(), "note": "anything non-tensor is fine"})


def test_total_loss_averages_within_a_type_then_weights():
    # Averaging per type keeps a weight's meaning independent of model depth: the
    # current model already takes the mean of per-layer MoE aux losses.
    aux = AuxOutputs()
    for layer, value in enumerate([1.0, 3.0]):
        aux = aux.merged(AuxOutputs(losses={"balance": t.tensor(value)}), prefix=f"layers.{layer}.ffn")
    aux = aux.merged(AuxOutputs(losses={"mtp": t.tensor(10.0)}), prefix="head")
    total = aux.total_loss({"balance": 0.5, "mtp": 0.1})
    t.testing.assert_close(total, t.tensor(0.5 * 2.0 + 0.1 * 10.0))


def test_total_loss_rejects_unweighted_losses():
    aux = AuxOutputs(losses={"indexer_kl": t.tensor(1.0)})
    with pytest.raises(ValueError):
        aux.total_loss({"balance": 0.01})


def test_total_loss_is_differentiable():
    w = t.tensor(2.0, requires_grad=True)
    AuxOutputs(losses={"balance": w * 3}).total_loss({"balance": 1.0}).backward()
    assert w.grad.item() == 3.0
