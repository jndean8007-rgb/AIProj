"""Golden-output guard for the Phase 0b refactor (GOAL_ARCHITECTURE.md §4.2, §9).

Phase 0b must not change behavior. Its exit test: greedy generation from the
current checkpoint is token-identical before and after the refactor.

1. Record once, BEFORE refactoring (writes tests/golden/phase0b_reference.pt):
       .\\.venv\\Scripts\\python.exe tests/test_phase0b_golden.py --record
   The recorder runs generation twice and refuses to save if the two runs differ,
   so the golden file is known to be reproducible.
2. After every refactor step:
       pytest tests/test_phase0b_golden.py

What is captured (a fingerprint small enough to commit, not full logits):
  - greedy tokens for 3 prompts decoded together (continuous batching), with a
    BF16 cache and with an FP8 cache;
  - the top-8 logits at every generated step, per request;
  - one training-mode forward on a packed batch: per-token next-token loss and
    the MoE aux loss (covers the flash-attention path, no cache).

Tokens must match exactly. Logits and losses get BF16-level tolerance, because a
refactor may legitimately reorder floating-point operations.

The model call in `train_forward` is the one line to update when the model's
signature changes to `model(meta, states)` during Phase 0b.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch as t

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for path in (REPOSITORY_ROOT / "src", REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

GOLDEN_PATH = REPOSITORY_ROOT / "tests" / "golden" / "phase0b_reference.pt"
PROMPTS = [
    "The history of the printing press",
    "Photosynthesis is the process by which",
    "In mathematics, a prime number",
]
MAX_NEW_TOKENS = 48
TOP_K = 8
LOGIT_TOLERANCE = dict(rtol=2e-2, atol=2e-2)


def load_main_setup():
    """The same model, tokenizer and configs `main.py` uses for generation."""
    from torch_llm.data_pipeline.bpe_tokenizer_config import BPETokenizerConfig
    from torch_llm.inference.generation_setup import generate_setup
    from torch_llm.model.model_config import ModelConfig
    from torch_llm.path_config import PathConfig

    paths, model_config = PathConfig(), ModelConfig()
    model, tokenizer = generate_setup(
        model_path=paths.model_load_path, model_config=model_config,
        tokenizer_path=paths.tokenizer_load_path, tokenizer_config=BPETokenizerConfig(),
        device="cuda", dtype=t.bfloat16,
    )
    return model, tokenizer, model_config, paths.model_load_path


def train_forward(model, token_ids, cu_seqlens, positions, max_seqlen):
    """Training-mode forward -> (logits [T, V], aux_loss). Update here when the model signature changes."""
    output = model(token_ids, cu_seqlens, positions, max_seqlen, mode="train")
    return output.logits, output.aux_loss


def capture_training_forward(model, tokenizer):
    sequences = [t.tensor(tokenizer.encode(prompt), dtype=t.long) for prompt in PROMPTS]
    lengths = [len(s) for s in sequences]
    token_ids = t.cat(sequences).cuda()
    cu_seqlens = t.tensor([0] + t.tensor(lengths).cumsum(0).tolist(), dtype=t.int32, device="cuda")
    positions = t.cat([t.arange(n) for n in lengths]).cuda()
    with t.inference_mode():
        logits, aux_loss = train_forward(model, token_ids, cu_seqlens, positions, max(lengths))
    log_probs = logits.float().log_softmax(-1)
    # Next-token loss inside each sequence (the last token of a sequence has no target).
    losses = []
    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        targets = token_ids[start + 1:end]
        losses.append(-log_probs[start:end - 1].gather(-1, targets[:, None]).squeeze(-1))
    return {"token_loss": t.cat(losses).cpu(), "aux_loss": aux_loss.float().cpu()}


def capture_generation(model, tokenizer, model_config, kv_cache_dtype):
    """Greedy-decode all prompts together; per request: tokens and top-k logits at each step."""
    from torch_llm.inference.kvcache_config import KVCacheConfig
    from torch_llm.inference.runtime import InferenceRuntime

    last_logits = {}

    def sampler(logits):
        last_logits["value"] = logits.float()
        return logits.argmax(-1)

    top_k = {}

    class Recorder:
        def handle_output(self, token_ids, request_ids):
            values, indices = last_logits["value"].topk(TOP_K, dim=-1)
            for row, request_id in enumerate(request_ids):
                top_k.setdefault(request_id, []).append((values[row].cpu(), indices[row].cpu()))

    runtime = InferenceRuntime(model, model_config, tokenizer, KVCacheConfig(kv_cache_dtype=kv_cache_dtype), sampler=sampler)
    request_ids = [runtime.submit(prompt, max_new_tokens=MAX_NEW_TOKENS) for prompt in PROMPTS]
    finished = {request.request_id: request for request in runtime.generate(Recorder())}
    return [
        {
            "prompt_tokens": list(finished[request_id].prompt_tokens),
            "generated_tokens": list(finished[request_id].generated_tokens),
            "finish_reason": finished[request_id].finish_reason,
            "top_k_values": t.stack([v for v, _ in top_k[request_id]]),
            "top_k_indices": t.stack([i for _, i in top_k[request_id]]),
        }
        for request_id in request_ids
    ]


def capture_all():
    model, tokenizer, model_config, checkpoint = load_main_setup()
    return {
        "meta": {
            "checkpoint": str(checkpoint),
            "device": t.cuda.get_device_name(),
            "torch": t.__version__,
            "prompts": PROMPTS,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        "train": capture_training_forward(model, tokenizer),
        "generation": {
            "bf16": capture_generation(model, tokenizer, model_config, t.bfloat16),
            "fp8": capture_generation(model, tokenizer, model_config, t.float8_e4m3fn),
        },
    }


def _golden_or_skip():
    if not t.cuda.is_available():
        pytest.skip("requires CUDA")
    if not GOLDEN_PATH.exists():
        pytest.skip(f"no golden file; record it before refactoring: python {Path(__file__).name} --record")
    golden = t.load(GOLDEN_PATH, weights_only=False)
    if not Path(golden["meta"]["checkpoint"]).exists():
        pytest.skip(f"checkpoint used for the golden file is missing: {golden['meta']['checkpoint']}")
    return golden


@pytest.fixture(scope="module")
def golden_and_current():
    golden = _golden_or_skip()
    return golden, capture_all()


def test_training_forward_matches_golden(golden_and_current):
    golden, current = golden_and_current
    t.testing.assert_close(current["train"]["token_loss"], golden["train"]["token_loss"], **LOGIT_TOLERANCE)
    t.testing.assert_close(current["train"]["aux_loss"], golden["train"]["aux_loss"], **LOGIT_TOLERANCE)


@pytest.mark.parametrize("cache", ["bf16", "fp8"])
def test_greedy_generation_is_token_identical(golden_and_current, cache):
    golden, current = golden_and_current
    for prompt, expected, actual in zip(PROMPTS, golden["generation"][cache], current["generation"][cache]):
        assert actual["prompt_tokens"] == expected["prompt_tokens"], f"tokenization changed for {prompt!r}"
        if actual["generated_tokens"] != expected["generated_tokens"]:
            step = next(
                (i for i, (a, b) in enumerate(zip(actual["generated_tokens"], expected["generated_tokens"])) if a != b),
                min(len(actual["generated_tokens"]), len(expected["generated_tokens"])),
            )
            margin = (expected["top_k_values"][step, 0] - expected["top_k_values"][step, 1]).item() \
                if step < len(expected["top_k_values"]) else float("nan")
            pytest.fail(
                f"{cache} {prompt!r}: first divergence at step {step} "
                f"(golden top-1 margin there {margin:.4f}; a margin near 0 suggests a near-tie, not a logic change)"
            )
        assert actual["finish_reason"] == expected["finish_reason"]


@pytest.mark.parametrize("cache", ["bf16", "fp8"])
def test_generation_logits_match_golden(golden_and_current, cache):
    golden, current = golden_and_current
    for expected, actual in zip(golden["generation"][cache], current["generation"][cache]):
        assert actual["top_k_indices"][:, 0].tolist() == expected["top_k_indices"][:, 0].tolist()
        t.testing.assert_close(actual["top_k_values"], expected["top_k_values"], **LOGIT_TOLERANCE)


def record() -> None:
    if not t.cuda.is_available():
        raise RuntimeError("CUDA is required")
    first, second = capture_all(), capture_all()
    for cache in ("bf16", "fp8"):
        for a, b in zip(first["generation"][cache], second["generation"][cache]):
            if a["generated_tokens"] != b["generated_tokens"]:
                raise RuntimeError(f"generation is not reproducible run-to-run ({cache}); not saving a golden file")
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    t.save(first, GOLDEN_PATH)
    size_kb = GOLDEN_PATH.stat().st_size / 1024
    print(f"saved {GOLDEN_PATH} ({size_kb:.0f} KB) from {first['meta']['checkpoint']}")
    for cache in ("bf16", "fp8"):
        for prompt, result in zip(PROMPTS, first["generation"][cache]):
            print(f"  {cache} {prompt!r}: {len(result['generated_tokens'])} tokens, finish={result['finish_reason']}")


if __name__ == "__main__":
    if "--record" not in sys.argv:
        raise SystemExit("usage: python tests/test_phase0b_golden.py --record")
    record()
