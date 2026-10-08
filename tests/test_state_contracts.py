"""Contract tests for Phase 0b step 3: state specs, StateManager, layer state views (handoff D18).

Everything here runs on CPU (float32 storage, no Triton), so it works on any machine.

Where things live
  torch_llm/core/state.py              FullKVSpec, PagedKVRead, FullKVView (Protocol)
  torch_llm/inference/page_allocator.py PageAllocator (was PageAllocator), CacheCapacityError
  torch_llm/inference/state_manager.py StateManager, StateSnapshot, PagedKVView

Ownership: a mixer declares what state it needs (state_specs()); the engine's StateManager
allocates it, tracks every request's length, and hands each layer a view for the current
batch. The mixer only ever calls view.append(k, v) and view.read().

StateManager API (request-level; lengths are in tokens):
  can_reserve(capacity) / fits_in_pool(capacity) -> bool
  reserve(request_id, capacity)       first call takes a slot; later calls grow capacity; atomic
  commit(request_ids, num_tokens)     the batch's tokens are now part of each request's state
  seq_len(request_id) -> int
  truncate(request_id, length)        forget tokens after `length` (append-only kinds: no copy)
  snapshot(request_id) -> StateSnapshot;  restore(snapshot)
  free(request_id);  request_ids() -> list[int];  clear()
  layer_states(request_ids, cu_seqlens, token_positions) -> list[tuple[view, ...]]
      one tuple per layer, one view per spec of that layer, for this batch; call it BEFORE
      commit, so each view's context is committed tokens + this batch's tokens.
Unknown request IDs raise KeyError; impossible lengths raise ValueError; lack of capacity
raises CacheCapacityError.
"""

import dataclasses

import pytest
import torch as t

from torch_llm.core.state import FullKVSpec, FullKVView, PagedKVRead
from torch_llm.inference.kvcache_config import KVCacheConfig
from torch_llm.inference.page_allocator import CacheCapacityError
from torch_llm.inference.state_manager import StateManager, StateSnapshot

H, D = 2, 4


def make_manager(num_layers=2, num_blocks=8, block_size=4, slots=3, max_len=16):
    config = KVCacheConfig(
        num_blocks=num_blocks,
        block_size=block_size,
        max_cache_slots=slots,
        cache_max_seq_len=max_len,
        kv_cache_dtype=t.float32,
    )
    layer_specs = [[FullKVSpec(num_kv_heads=H, head_dim=D)] for _ in range(num_layers)]
    return StateManager(layer_specs, config, "cpu")


def packed(lengths, starts):
    """cu_seqlens and token_positions for sequences of `lengths` tokens starting at `starts`."""
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    positions = t.cat([t.arange(s, s + n) for s, n in zip(starts, lengths)])
    return t.tensor(cu, dtype=t.int32), positions


def stored_rows(read: PagedKVRead, slot: int, positions):
    """Gather the K/V rows a kernel would read for `positions` of the request in `slot`."""
    positions = t.as_tensor(positions)
    blocks = read.block_table[slot, positions // read.block_size].long()
    offsets = positions % read.block_size
    return read.k[blocks, offsets], read.v[blocks, offsets]


# ---------------------------------------------------------------------------
# FullKVSpec
# ---------------------------------------------------------------------------

def test_full_kv_spec_is_frozen_and_comparable():
    spec = FullKVSpec(num_kv_heads=H, head_dim=D)
    assert spec == FullKVSpec(num_kv_heads=H, head_dim=D)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.head_dim = 8


@pytest.mark.parametrize("bad", [0, -1, True, 2.0])
def test_full_kv_spec_rejects_invalid_sizes(bad):
    with pytest.raises(ValueError):
        FullKVSpec(num_kv_heads=bad, head_dim=D)
    with pytest.raises(ValueError):
        FullKVSpec(num_kv_heads=H, head_dim=bad)


# ---------------------------------------------------------------------------
# Capacity: reserve / commit / free
# ---------------------------------------------------------------------------

def test_reserve_then_commit_tracks_length():
    manager = make_manager()
    manager.reserve(7, 5)
    assert manager.seq_len(7) == 0 and manager.request_ids() == [7]

    manager.commit([7], [3])
    assert manager.seq_len(7) == 3

    with pytest.raises(CacheCapacityError):
        manager.commit([7], [6])  # 3 + 6 = 9 tokens, but only 5 were reserved (2 pages = 8 tokens)


def test_reserve_grows_capacity_and_failure_takes_nothing():
    manager = make_manager(num_blocks=8, block_size=4, max_len=32)  # 8 pages of 4 tokens
    manager.reserve(1, 16)  # 4 pages
    manager.reserve(2, 12)  # 3 pages -> 1 page left

    assert manager.can_reserve(4) and not manager.can_reserve(5)
    with pytest.raises(CacheCapacityError):
        manager.reserve(3, 8)  # needs 2 pages
    assert manager.request_ids() == [1, 2]  # the failed request holds no slot or pages

    manager.reserve(1, 16)  # same capacity again: no new pages
    manager.reserve(1, 20)  # grows by one page
    assert not manager.can_reserve(1)


def test_fits_in_pool_checks_max_length_and_total_pages():
    assert make_manager(max_len=16).fits_in_pool(16)
    assert not make_manager(max_len=16).fits_in_pool(17)
    small_pool = make_manager(num_blocks=2, block_size=4, max_len=16)
    assert small_pool.fits_in_pool(8) and not small_pool.fits_in_pool(9)


def test_no_free_slot_means_no_reservation():
    manager = make_manager(slots=2)
    manager.reserve(1, 4)
    manager.reserve(2, 4)
    assert not manager.can_reserve(1)
    with pytest.raises(CacheCapacityError):
        manager.reserve(3, 1)


def test_free_and_clear_release_everything():
    manager = make_manager(num_blocks=8, block_size=4, slots=2)
    manager.reserve(1, 16)
    manager.reserve(2, 16)
    assert not manager.can_reserve(1)

    manager.free(1)
    assert manager.request_ids() == [2] and manager.can_reserve(16)
    with pytest.raises(KeyError):
        manager.seq_len(1)

    manager.clear()
    assert manager.request_ids() == [] and manager.can_reserve(16)


def test_unknown_requests_raise_key_error():
    manager = make_manager()
    for call in (lambda: manager.seq_len(9), lambda: manager.free(9), lambda: manager.snapshot(9),
                 lambda: manager.commit([9], [1])):
        with pytest.raises(KeyError):
            call()


def test_duplicate_requests_in_one_commit_are_rejected():
    manager = make_manager()
    manager.reserve(1, 8)
    with pytest.raises(ValueError):
        manager.commit([1, 1], [1, 1])


# ---------------------------------------------------------------------------
# truncate / snapshot / restore
# ---------------------------------------------------------------------------

def test_truncate_snapshot_and_restore():
    manager = make_manager()
    manager.reserve(1, 12)
    manager.commit([1], [5])

    snapshot = manager.snapshot(1)
    assert isinstance(snapshot, StateSnapshot)
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.length = 0

    manager.commit([1], [3])
    assert manager.seq_len(1) == 8
    manager.restore(snapshot)
    assert manager.seq_len(1) == 5

    manager.truncate(1, 2)
    assert manager.seq_len(1) == 2
    with pytest.raises(ValueError):
        manager.truncate(1, 3)  # cannot truncate forward
    with pytest.raises(ValueError):
        manager.restore(snapshot)  # the state no longer contains the snapshot's tokens


def test_restore_after_free_fails():
    manager = make_manager()
    manager.reserve(1, 8)
    manager.commit([1], [2])
    snapshot = manager.snapshot(1)
    manager.free(1)
    with pytest.raises(KeyError):
        manager.restore(snapshot)


def test_tokens_after_a_truncate_overwrite_the_old_ones():
    manager = make_manager(num_layers=1)
    manager.reserve(1, 8)

    cu, positions = packed([4], [0])
    (view,), = manager.layer_states([1], cu, positions)
    old = t.randn(4, H, D)
    view.append(old, old)
    manager.commit([1], [4])

    manager.truncate(1, 2)
    cu, positions = packed([1], [2])
    (view,), = manager.layer_states([1], cu, positions)
    new = t.randn(1, H, D)
    view.append(new, new)

    read = view.read()
    assert read.context_lengths.tolist() == [3]
    k, _ = stored_rows(read, read.slots[0].item(), [0, 1, 2])
    t.testing.assert_close(k, t.cat([old[:2], new]))


# ---------------------------------------------------------------------------
# layer_states and views
# ---------------------------------------------------------------------------

def test_layer_states_gives_one_view_per_spec_per_layer():
    manager = make_manager(num_layers=3)
    manager.reserve(1, 4)
    states = manager.layer_states([1], *packed([2], [0]))

    assert len(states) == 3
    for layer_state in states:
        assert isinstance(layer_state, tuple) and len(layer_state) == 1
        assert isinstance(layer_state[0], FullKVView)


def test_prefill_then_decode_appends_where_read_points():
    manager = make_manager()
    manager.reserve(10, 8)
    manager.reserve(20, 8)

    # Prefill: request 10 has 3 tokens, request 20 has 2.
    cu, positions = packed([3, 2], [0, 0])
    states = manager.layer_states([10, 20], cu, positions)
    view0, view1 = states[0][0], states[1][0]
    k, v = t.randn(5, H, D), t.randn(5, H, D)
    view0.append(k, v)

    read = view0.read()
    assert isinstance(read, PagedKVRead)
    assert read.context_lengths.tolist() == [3, 2]
    assert read.k_scales is None and read.v_scales is None  # unquantized storage
    slot10, slot20 = read.slots.tolist()
    t.testing.assert_close(stored_rows(read, slot10, [0, 1, 2]), (k[:3], v[:3]))
    t.testing.assert_close(stored_rows(read, slot20, [0, 1]), (k[3:], v[3:]))
    assert view1.read().k.data_ptr() != read.k.data_ptr()  # every layer has its own storage

    manager.commit([10, 20], [3, 2])

    # Decode: one new token each, at the next positions.
    cu, positions = packed([1, 1], [3, 2])
    (view0,), _ = manager.layer_states([10, 20], cu, positions)
    k2, v2 = t.randn(2, H, D), t.randn(2, H, D)
    view0.append(k2, v2)

    read = view0.read()
    assert read.context_lengths.tolist() == [4, 3]
    t.testing.assert_close(stored_rows(read, slot10, [0, 1, 2, 3]), (t.cat([k[:3], k2[:1]]), t.cat([v[:3], v2[:1]])))
    t.testing.assert_close(stored_rows(read, slot20, [0, 1, 2]), (t.cat([k[3:], k2[1:]]), t.cat([v[3:], v2[1:]])))


def test_view_rejects_mismatched_kv():
    manager = make_manager(num_layers=1)
    manager.reserve(1, 4)
    (view,), = manager.layer_states([1], *packed([2], [0]))
    with pytest.raises(ValueError):
        view.append(t.randn(3, H, D), t.randn(3, H, D))  # 3 rows for a 2-token batch
    with pytest.raises(ValueError):
        view.append(t.randn(2, H, D + 1), t.randn(2, H, D + 1))


def test_layer_states_for_unknown_request_raises():
    manager = make_manager()
    with pytest.raises(KeyError):
        manager.layer_states([5], *packed([1], [0]))
