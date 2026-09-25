from functools import partial
from types import SimpleNamespace

import pytest
import torch as t

from torch_llm.inference.batching import build_decode_batch, build_prefill_batch, merge_decode_batches
from torch_llm.inference.cache_manager import CacheCapacityError, KVCacheManager
from torch_llm.inference.continuous_batch_scheduler import ContinuousBatchScheduler
from torch_llm.inference.input_processor import InferenceInputProcessor
from torch_llm.inference.kv_cache import KVCache
from torch_llm.inference.kvcache_config import KVCacheConfig
from torch_llm.inference.output_handler import OutputHandler
from torch_llm.inference.paged_kv_cache import PagedKVCache
from torch_llm.inference.request_state import RequestState
from torch_llm.inference.runtime import InferenceRuntime
from torch_llm.inference.sampling import sample
from torch_llm.model.model_config import ModelConfig


DEVICES = ["cpu"] + (["cuda"] if t.cuda.is_available() else [])


class Tokenizer:
    eos_token_id = 31

    def encode(self, text):
        return [int(token) for token in text.split()]

    def decode(self, tokens):
        assert isinstance(tokens, list)
        return " ".join(map(str, tokens))


class IncrementModel(t.nn.Module):
    """Predict token + 1 and verify that runtime cache metadata is coherent."""

    def __init__(self, device):
        super().__init__()
        self.weight = t.nn.Parameter(t.zeros(1, device=device))
        self.calls = []
        self.fail = False

    def forward(self, tokens, cu_seqlens, positions, max_seq_len, *, paged_kv_caches, cache_batch_context, mode):
        assert not self.training and not t.is_grad_enabled()
        if self.fail:
            raise RuntimeError("model failure")
        context = cache_batch_context
        slots = context.cache_slots.tolist()
        lengths = context.context_lengths.tolist()
        boundaries = cu_seqlens.tolist()
        assert max_seq_len == max(b - a for a, b in zip(boundaries, boundaries[1:]))
        self.calls.append((mode, slots, lengths, tokens.tolist()))
        for row, length in enumerate(lengths):
            assert positions[boundaries[row + 1] - 1].item() == length - 1
        for cache in paged_kv_caches:
            values = tokens[:, None, None].expand(-1, *cache.k.shape[2:]).float()
            cache.append_kv(context.physical_blocks, context.block_offsets, values, values)
            # Reading logical history catches stale context and reused-slot corruption.
            for row, (slot, length) in enumerate(zip(slots, lengths)):
                history = t.arange(length, device=tokens.device)
                blocks = context.block_table[slot, history // cache.block_size]
                cached = cache.k[blocks, history % cache.block_size, 0, 0]
                count = boundaries[row + 1] - boundaries[row]
                t.testing.assert_close(cached[-count:], tokens[boundaries[row]:boundaries[row + 1]].float())
        logits = t.full((tokens.numel(), 32), -100.0, device=tokens.device)
        logits.scatter_(1, ((tokens + 1) % 32).long()[:, None], 100)
        return SimpleNamespace(logits=logits)


def make_runtime(device="cpu", *, blocks=16, slots=3, max_context=16, scheduler=None, sampler=sample):
    config = ModelConfig(vocab_size=32, num_layers=2, num_kv_heads=1, head_dim=2, model_max_seq_len=max_context)
    model = IncrementModel(device)
    runtime = InferenceRuntime(
        model, config, Tokenizer(),
        KVCacheConfig(blocks, 2, slots, max_context, t.float32),
        scheduler, sampler=sampler,
    )
    return runtime


def assert_released(runtime):
    manager = runtime.cache_manager
    assert not manager.request_to_slot
    assert not runtime.continuous_batch_scheduler.active_requests
    assert sorted(manager.free_blocks) == list(range(manager.num_blocks))
    assert len(set(manager.free_slots)) == manager.block_table.shape[0]
    assert t.all(manager.block_table == -1)
    assert t.all(manager.seq_lens == 0)


@pytest.mark.parametrize("device", DEVICES)
def test_cache_reuse_clear_and_failed_reservation_are_safe(device):
    manager = KVCacheManager(4, 2, 3, 8, device)
    a = manager.allocate_request(100, 3)
    b = manager.allocate_request(200, 1)
    before = manager.block_table.clone()
    with pytest.raises(CacheCapacityError):
        manager.reserve_batch_capacity([a, b], [5, 3])
    t.testing.assert_close(manager.block_table, before)
    assert len(manager.free_blocks) == 1
    manager.advance(100, 3)
    manager.free_request(100)
    c = manager.allocate_request(300, 4)
    assert c == a
    assert manager.get_seq_len(300) == 0
    manager.clear()
    assert sorted(manager.free_blocks) == list(range(4))
    assert sorted(manager.free_slots) == list(range(3))
    assert t.all(manager.block_table == -1)


@pytest.mark.parametrize("device", DEVICES)
def test_context_uses_full_slot_table_and_committed_history(device):
    manager = KVCacheManager(8, 2, 4, 8, device)
    slot = manager.allocate_request(9, 6)
    manager.advance(9, 3)
    slots = t.tensor([slot], device=device)
    context = manager.create_container(slots, t.tensor([0, 1], dtype=t.int32, device=device), t.tensor([3], device=device))
    assert slot == 3
    assert context.block_table is manager.block_table
    assert context.context_lengths.tolist() == [4]
    cache = PagedKVCache(8, 2, 1, 2, device, t.float16)
    kv = t.ones(1, 1, 2, device=device, dtype=t.float32)
    manager.append_request(9, t.tensor([3], device=device), kv, kv, cache)
    t.testing.assert_close(cache.k[context.physical_blocks, context.block_offsets], kv.half())
    with pytest.raises(ValueError):
        manager.get_physical_location(slots, t.tensor([-1], device=device))
    with pytest.raises(ValueError):
        manager.get_physical_location(slots, t.tensor([6], device=device))


@pytest.mark.parametrize("device", DEVICES)
def test_dense_cache_updates_only_selected_slots(device):
    cache = KVCache(4, 5, 1, 2, t.float32, device)
    kv = t.arange(6, dtype=t.float32, device=device).reshape(3, 1, 2)
    cache.append_prefill(kv, kv, t.tensor([0, 2, 3], device=device), cache_slots=[3, 1])
    new = t.full((1, 1, 2), 42.0, device=device)
    cache.append_decode(new, new, cache_slots=[1])
    assert cache.seq_lens.tolist() == [0, 2, 0, 2]
    t.testing.assert_close(cache.k_cache[1, 1], new[0])
    t.testing.assert_close(cache.k_cache[3, :2], kv[:2])
    with pytest.raises(ValueError):
        cache.append_decode(new.expand(2, -1, -1), new.expand(2, -1, -1), [1, 1])
    cache.clear([1])
    assert cache.seq_lens.tolist() == [0, 0, 0, 2]


@pytest.mark.parametrize("device", DEVICES)
def test_packed_batches_compact_merge_and_preserve_positions(device):
    requests = [RequestState(10, [2, 4, 6], 2), RequestState(40, [5], 2)]
    batch = build_prefill_batch(iter(requests), device)
    assert batch.cu_seqlens.dtype == t.int32
    assert batch.token_ids.dtype == t.long
    assert batch.cu_seqlens.tolist() == [0, 3, 4]
    assert batch.token_positions.tolist() == [0, 1, 2, 0]
    first = build_decode_batch(batch, [7, 6], [True, False])
    second = build_decode_batch(batch, [7, 6], t.tensor([False, True], device=device))
    merged = merge_decode_batches([first, None, second])
    assert merged is not None
    assert merged.token_ids.tolist() == [7, 6]
    assert merged.token_positions.tolist() == [3, 1]
    advanced = build_decode_batch(merged, [8, 7], [False, True])
    assert advanced is not None
    assert advanced.token_positions.tolist() == [2]
    assert build_decode_batch(advanced, [8], [False]) is None
    assert merge_decode_batches([None]) is None
    with pytest.raises(ValueError):
        build_prefill_batch([])


def test_input_output_and_tokenizer_list_contract(capsys):
    tokenizer = Tokenizer()
    batch = InferenceInputProcessor(tokenizer).prepare(["1 2", "3"])
    assert batch.token_ids.tolist() == [1, 2, 3]
    OutputHandler(tokenizer).handle_output([4, 5], [10, 11])
    assert capsys.readouterr().out == "Stream 10 -> 4\nStream 11 -> 5\n"
    from torch_llm.data_pipeline.bpe_tokenizer import BPETokenizer
    wrapped = BPETokenizer(tokenizer, None)
    assert wrapped.decode([7, 8]) == "7 8"


def test_scheduler_fifo_capacity_and_duplicate_ids():
    scheduler = ContinuousBatchScheduler(2)
    requests = [RequestState(i, [i], 2) for i in range(3)]
    for request in requests:
        scheduler.submit(request)
    with pytest.raises(ValueError):
        scheduler.submit(requests[0])
    assert [key for key, _ in scheduler.admission_candidates(1)] == [0]
    scheduler.admit_request(0)
    scheduler.admit_request(1)
    assert scheduler.admission_candidates() == []
    with pytest.raises(RuntimeError):
        scheduler.admit_request(2)
    assert scheduler.remove(0).finished
    assert [key for key, _ in scheduler.admission_candidates()] == [2]


@pytest.mark.parametrize("device", DEVICES)
def test_runtime_per_request_limits_eos_zero_and_terminal_output(device):
    runtime = make_runtime(device)
    ids = [runtime.submit("2 3", 3), runtime.submit("30", 4), runtime.submit("9", 0), runtime.submit("5", 1)]
    emitted = []
    sink = SimpleNamespace(handle_output=lambda tokens, request_ids: emitted.extend(zip(request_ids, tokens)))
    results = {request.request_id: request for request in runtime.generate(sink)}
    assert results[ids[0]].generated_tokens == [4, 5, 6]
    assert results[ids[1]].generated_tokens == [31]
    assert results[ids[1]].finish_reason == "eos"
    assert results[ids[2]].generated_tokens == []
    assert results[ids[3]].generated_tokens == [6]
    for request_id in ids:
        assert [token for key, token in emitted if key == request_id] == results[request_id].generated_tokens
    assert runtime.generate() == []
    assert_released(runtime)


@pytest.mark.parametrize("device", DEVICES)
def test_runtime_admits_new_requests_and_reuses_slots(device):
    runtime = make_runtime(device, slots=2)
    first = runtime.submit("2", 4)
    second = runtime.submit("10 11", 1)
    results = runtime.step()
    assert [request.request_id for request in results] == [second]
    third = runtime.submit("20 21 22", 2)
    results.extend(runtime.generate())
    by_id = {request.request_id: request.generated_tokens for request in results}
    assert by_id == {first: [3, 4, 5, 6], second: [12], third: [23, 24]}
    prefills = [call for call in runtime.model.calls if call[0] == "prefill"]
    assert len(prefills) == 2
    assert prefills[1][2:] == ([3], [20, 21, 22])
    assert_released(runtime)


def test_submit_during_output_and_noncontiguous_existing_ids():
    scheduler = ContinuousBatchScheduler(2)
    scheduler.submit(RequestState(40, [3], 1))
    runtime = make_runtime(scheduler=scheduler)
    submitted = []
    def handle_output(tokens, request_ids):
        if not submitted:
            submitted.append(runtime.submit("8", 2))
    results = runtime.generate(SimpleNamespace(handle_output=handle_output))
    assert submitted == [41]
    assert {r.request_id: r.generated_tokens for r in results} == {40: [4], 41: [9, 10]}
    assert_released(runtime)


def test_memory_pressure_waits_for_active_request_and_context_stops():
    runtime = make_runtime(blocks=2, max_context=4)
    first = runtime.submit("1", 4)
    second = runtime.submit("8 9 10", 20)
    runtime.step()
    assert list(runtime.continuous_batch_scheduler.waiting_requests) == [second]
    results = {r.request_id: r for r in runtime.generate()}
    assert results[first].generated_tokens == [2, 3, 4, 5]
    assert results[second].generated_tokens == [11, 12]
    assert results[second].finish_reason == "context_length"
    assert_released(runtime)


@pytest.mark.parametrize("failure", ["model", "output", "sampler"])
def test_failure_releases_admitted_work_and_preserves_waiting(failure):
    runtime = make_runtime(slots=1)
    first = runtime.submit("2", 4)
    waiting = runtime.submit("8", 1)
    request = runtime.continuous_batch_scheduler.waiting_requests[first]
    sink = None
    if failure == "model":
        runtime.model.fail = True
    elif failure == "sampler":
        runtime.sampler = lambda logits: t.zeros((1, 1), dtype=t.long)
    else:
        def fail_output(*args):
            raise RuntimeError("output failure")
        sink = SimpleNamespace(handle_output=fail_output)
    with pytest.raises((RuntimeError, ValueError)):
        runtime.step(sink)
    assert request.finished and request.finish_reason == "error"
    assert list(runtime.continuous_batch_scheduler.waiting_requests) == [waiting]
    assert_released(runtime)
    runtime.model.fail = False
    runtime.sampler = sample
    assert runtime.generate()[0].generated_tokens == [9]
    assert_released(runtime)


@pytest.mark.parametrize("prompt,budget", [("", 1), ("32", 1), ("1", -1), ("1", 1.5), ("1 2 3 4 5", 1)])
def test_invalid_requests_do_not_mutate_queue(prompt, budget):
    runtime = make_runtime(max_context=4)
    with pytest.raises(ValueError):
        runtime.submit(prompt, budget)
    assert not runtime.continuous_batch_scheduler.has_pending_requests()
    assert_released(runtime)


def test_request_too_large_for_pool_rejected_before_admission():
    runtime = make_runtime(blocks=1)
    with pytest.raises(ValueError, match="entire KV cache"):
        runtime.submit("1 2 3", 1)
    assert_released(runtime)


def test_many_requests_remain_independent_under_page_and_slot_reuse():
    runtime = make_runtime(blocks=8, slots=3, max_context=12)
    expected = {}
    for index in range(60):
        prompt = [index % 29] * (index % 5 + 1)
        budget = index % 8
        request_id = runtime.submit(" ".join(map(str, prompt)), budget)
        count = min(budget, 31 - prompt[-1], 12 - len(prompt) + 1)
        expected[request_id] = list(range(prompt[-1] + 1, prompt[-1] + 1 + count))
    results = runtime.generate()
    assert len(results) == len(expected)
    assert {r.request_id: r.generated_tokens for r in results} == expected
    assert_released(runtime)


def test_cache_rejects_overflow_even_inside_last_partial_page():
    manager = KVCacheManager(2, 4, 1, 5, "cpu")
    manager.allocate_request(1, 5)
    manager.advance(1, 5)
    with pytest.raises(CacheCapacityError):
        manager.advance(1, 1)
    assert manager.get_seq_len(1) == 5
    with pytest.raises(ValueError):
        manager.allocate_request(1, 2)
    with pytest.raises(ValueError):
        manager.reserve_capacity(0, -1)
    manager.clear()
    assert sorted(manager.free_blocks) == [0, 1]


def test_cache_writes_do_not_retain_autograd_graphs():
    cache = PagedKVCache(1, 2, 1, 2, "cpu", t.float32)
    values = t.ones(1, 1, 2, requires_grad=True)
    cache.append_kv(t.tensor([0]), t.tensor([0]), values, values)
    assert not cache.k.requires_grad and cache.k.grad_fn is None


def test_reentrant_step_is_rejected_and_releases_active_requests():
    runtime = make_runtime()
    runtime.submit("1", 3)
    def recurse(*args):
        runtime.step()
    with pytest.raises(RuntimeError, match="not reentrant"):
        runtime.step(SimpleNamespace(handle_output=recurse))
    assert_released(runtime)


@pytest.mark.parametrize("field,value", [("num_blocks", 0), ("block_size", 3), ("max_cache_slots", -1), ("kv_cache_dtype", t.int8)])
def test_invalid_cache_config(field, value):
    with pytest.raises(ValueError):
        KVCacheConfig(**{field: value})


def test_sampling_greedy_filtered_and_seeded():
    logits = t.tensor([[0., 2., 1.], [3., 0., 1.]])
    original = logits.clone()
    assert sample(logits).tolist() == [1, 0]
    assert sample(logits, temperature=1, top_k=1).tolist() == [1, 0]
    assert sample(logits, temperature=1, top_p=0.01).tolist() == [1, 0]
    a = sample(logits, temperature=0.8, generator=t.Generator().manual_seed(5))
    b = sample(logits, temperature=0.8, generator=t.Generator().manual_seed(5))
    t.testing.assert_close(a, b)
    t.testing.assert_close(logits, original)
    runtime = make_runtime(sampler=partial(sample, temperature=1, top_k=1))
    runtime.submit("1", 1)
    assert runtime.generate()[0].generated_tokens == [2]


@pytest.mark.parametrize("kwargs", [{"temperature": -1}, {"temperature": float("nan")}, {"top_k": 0}, {"top_p": 0}])
def test_invalid_sampling_parameters(kwargs):
    with pytest.raises(ValueError):
        sample(t.zeros(1, 4), **kwargs)
