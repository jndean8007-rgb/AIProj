import pytest
import torch as t

from torch_llm.inference.batching import build_prefill_batch
from torch_llm.inference.cache_manager import initialize_cache
from torch_llm.inference.kvcache_config import KVCacheConfig
from torch_llm.inference.request_state import RequestState
from torch_llm.inference.runtime import InferenceRuntime
from torch_llm.model.model import TransformerLM
from torch_llm.model.model_config import ModelConfig


def small_config():
    return ModelConfig(
        vocab_size=64, d_model=64, num_layers=2, num_q_heads=4,
        num_kv_heads=2, head_dim=16, d_ff=128, num_experts=4,
        top_k=2, model_max_seq_len=32, rms_eps=1e-6,
    )


@pytest.mark.skipif(not t.cuda.is_available(), reason="Model kernels require CUDA")
@pytest.mark.parametrize("dtype", [t.float16, t.bfloat16])
@t.inference_mode()
def test_transformer_prefill_decode_matches_full_forward(dtype):
    t.manual_seed(0)
    config = small_config()
    model = TransformerLM(config).to(device="cuda", dtype=dtype).eval()
    tokens = [[17, 42, 8, 31, 33, 7, 24], [5, 6, 19, 25]]
    full = build_prefill_batch([RequestState(i, row, 1) for i, row in enumerate(tokens)], "cuda")
    logits_full = model(
        full.token_ids, full.cu_seqlens, full.token_positions, full.batch_max_seq_len, mode="train",
    ).logits
    manager, caches = initialize_cache(KVCacheConfig(16, 4, 4, 32, dtype), config, "cuda")
    slots = [manager.allocate_request(i, len(row)) for i, row in enumerate(tokens)]
    prompt_lengths = [4, 2]
    prefill = build_prefill_batch([
        RequestState(i, row[:length], 1) for i, (row, length) in enumerate(zip(tokens, prompt_lengths))
    ], "cuda")
    context = manager.create_container(t.tensor(slots, device="cuda"), prefill.cu_seqlens, prefill.token_positions)
    prefill_out = model(
        prefill.token_ids, prefill.cu_seqlens, prefill.token_positions, prefill.batch_max_seq_len,
        mode="prefill", paged_kv_caches=caches, cache_batch_context=context,
    ).logits
    manager.advance_batch([0, 1], prompt_lengths)
    cached_rows = [[prefill_out[:4]], [prefill_out[4:]]]
    for offset in range(3):
        ids = [i for i, row in enumerate(tokens) if prompt_lengths[i] + offset < len(row)]
        positions = t.tensor([prompt_lengths[i] + offset for i in ids], device="cuda", dtype=t.int32)
        inputs = t.tensor([tokens[i][prompt_lengths[i] + offset] for i in ids], device="cuda")
        cu = t.arange(len(ids) + 1, device="cuda", dtype=t.int32)
        context = manager.create_container(t.tensor([slots[i] for i in ids], device="cuda"), cu, positions)
        out = model(inputs, cu, positions, 1, mode="decode", paged_kv_caches=caches, cache_batch_context=context).logits
        manager.advance_batch(ids, [1] * len(ids))
        for row, request_id in enumerate(ids):
            cached_rows[request_id].append(out[row:row + 1])
    logits_cached = t.cat([t.cat(row) for row in cached_rows])
    t.testing.assert_close(logits_cached.float(), logits_full.float(), rtol=3e-2, atol=3e-2)
    manager.clear()


@pytest.mark.skipif(not t.cuda.is_available(), reason="Model kernels require CUDA")
@t.inference_mode()
def test_runtime_matches_full_recomputation_and_preserves_router_state():
    class Tokenizer:
        eos_token_id = None

        def encode(self, text):
            return list(map(int, text.split()))

    t.manual_seed(2)
    config = small_config()
    model = TransformerLM(config).to(device="cuda", dtype=t.float16).eval()
    saved_state = {name: value.clone() for name, value in model.state_dict().items()}
    prompts = [[1, 5, 8], [17, 21]]
    expected = []
    for prompt in prompts:
        history = prompt[:]
        generated = []
        for _ in range(3):
            batch = build_prefill_batch([RequestState(0, history, 1)], "cuda")
            logits = model(batch.token_ids, batch.cu_seqlens, batch.token_positions, len(history), mode="train").logits
            token = int(logits[-1].argmax())
            generated.append(token)
            history.append(token)
        expected.append(generated)
    runtime = InferenceRuntime(model, config, Tokenizer(), KVCacheConfig(16, 4, 2, 32, t.float16))
    for prompt in prompts:
        runtime.submit(" ".join(map(str, prompt)), 3)
    results = {r.request_id: r.generated_tokens for r in runtime.generate()}
    assert results == dict(enumerate(expected))
    assert not runtime.cache_manager.request_to_slot
    for name, value in model.state_dict().items():
        t.testing.assert_close(value, saved_state[name], rtol=0, atol=0)


@pytest.mark.parametrize("training_checkpoint", [True, False])
def test_generation_setup_loads_checkpoint_on_cpu(tmp_path, training_checkpoint):
    from torch_llm.data_pipeline.bpe_tokenizer import BPETokenizer
    from torch_llm.data_pipeline.bpe_tokenizer_config import BPETokenizerConfig
    from torch_llm.inference.generation_setup import generate_setup

    config = small_config()
    original = TransformerLM(config)
    tokenizer_config = BPETokenizerConfig(vocab_size=64)
    tokenizer = BPETokenizer.from_config(tokenizer_config)
    from tokenizers.trainers import BpeTrainer
    tokenizer.tokenizer.train_from_iterator(
        ["small test text"], trainer=BpeTrainer(vocab_size=64, special_tokens=tokenizer_config.special_tokens),
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    model_path = tmp_path / "model.pt"
    state = original.state_dict()
    t.save({"model": state, "step": 1} if training_checkpoint else state, model_path)
    loaded, loaded_tokenizer = generate_setup(model_path, config, tokenizer_path, tokenizer_config, "cpu", t.float32)
    assert not loaded.training and loaded.device.type == "cpu"
    for name, value in loaded.state_dict().items():
        t.testing.assert_close(value, state[name])
    assert loaded_tokenizer.decode(loaded_tokenizer.encode("small")) == "small"
