import importlib
from itertools import repeat
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch as t
from datasets import IterableDataset

from torch_llm.data_pipeline.bpe_tokenizer_config import BPETokenizerConfig
from torch_llm.data_pipeline.loader import loader
from torch_llm.data_pipeline.streaming_dataset import StreamingDataset
from torch_llm.model.model_config import ModelConfig
from torch_llm.training.setup import build_or_load_tokenizer, setup
from torch_llm.training.train_config import TrainConfig


class CharacterTokenizer:
    eos_token_id = 0

    def encode(self, text):
        return [ord(char) for char in text]


def text_records():
    for text in ["abcd", "e", "fghijkl", ""]:
        yield {"text": text}


@pytest.mark.parametrize("seq_len", [1, 4, 7])
def test_streaming_targets_and_reiteration(seq_len):
    source = IterableDataset.from_generator(text_records)
    dataset = StreamingDataset(source, seq_len, CharacterTokenizer())
    batches = list(loader(dataset, seq_len, 2 * seq_len))

    tokens = [97, 98, 99, 100, 0, 101, 0, 102, 103, 104, 105, 106, 107, 108, 0, 0]
    assert t.cat([batch.token_ids for batch in batches]).tolist() == tokens[:-1]
    assert t.cat([batch.targets for batch in batches]).tolist() == tokens[1:]
    assert all(batch.token_ids.numel() <= 2 * seq_len for batch in batches)
    assert all(batch.batch_max_seq_len <= seq_len for batch in batches)
    assert [seq.tolist() for seq in dataset] == [seq.tolist() for seq in dataset]


def test_streaming_loader_reads_only_the_next_batch():
    consumed = []

    def texts():
        for i in range(100):
            consumed.append(i)
            yield "abc"

    dataset = StreamingDataset(texts(), 4, CharacterTokenizer())
    train_loader = loader(dataset, 4, 8)
    assert consumed == []

    batch = next(iter(train_loader))
    assert batch.token_ids.numel() == 8
    assert consumed == [0, 1, 2]


def test_tokenizer_samples_a_fresh_stream_and_loads_checkpoint(tmp_path):
    consumed = []

    class Texts:
        def __iter__(self):
            for text in ["hello world", "more text", "held back"]:
                consumed.append(text)
                yield {"text": text}

    dataset = StreamingDataset(Texts(), 4)
    config = BPETokenizerConfig(vocab_size=64, min_frequency=1, max_training_samples=2)
    tokenizer = build_or_load_tokenizer(None, config, train_dataset=dataset)
    assert consumed == ["hello world", "more text"]

    dataset.tokenizer = tokenizer
    assert next(iter(dataset)).dtype == t.long
    assert consumed[2] == "hello world"

    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    consumed.clear()
    loaded = build_or_load_tokenizer(None, config, True, path, train_dataset=dataset)
    assert loaded.encode("hello world") == tokenizer.encode("hello world")
    assert consumed == []


def test_tokenizer_rejects_a_one_shot_training_source():
    dataset = StreamingDataset(iter(["hello world"]), 4)
    with pytest.raises(ValueError, match="re-iterable"):
        build_or_load_tokenizer(None, BPETokenizerConfig(), train_dataset=dataset)


@pytest.mark.parametrize("stream_training", [False, True])
def test_run_training_accepts_paths_and_hugging_face_streams(tmp_path, monkeypatch, stream_training):
    run_module = importlib.import_module("torch_llm.training.run_training")
    model_config = ModelConfig(
        vocab_size=128, d_model=32, num_layers=1,
        num_q_heads=2, num_kv_heads=1, head_dim=16,
        d_ff=32, num_experts=2, top_k=1, model_max_seq_len=4,
    )
    tokenizer_config = BPETokenizerConfig(
        vocab_size=64, min_frequency=1, max_training_samples=2,
        tokenizer_save_path=str(tmp_path / "tokenizer.json"),
    )
    path = tmp_path / "texts.txt"
    path.write_text("abcd e fghijkl", encoding="utf-8")
    source = IterableDataset.from_generator(text_records)
    seen = []

    def cpu_setup(*args, **kwargs):
        kwargs["device"] = "cpu"
        kwargs["dtype"] = t.float32
        return setup(*args, **kwargs)

    def check_training(*args, **kwargs):
        train_loader, eval_loader = args[4:6]
        model = args[6]
        assert isinstance(train_loader.dataset, StreamingDataset) == stream_training
        assert isinstance(eval_loader.dataset, StreamingDataset) != stream_training
        for data_loader in [train_loader, eval_loader]:
            batch = next(iter(data_loader))
            assert batch.token_ids.dtype == t.long
            assert batch.token_ids.shape == batch.targets.shape
            assert batch.batch_max_seq_len <= model_config.model_max_seq_len
        seen.append(model)
        return model

    monkeypatch.setattr(run_module, "setup", cpu_setup)
    monkeypatch.setattr(run_module, "train", check_training)
    run_module.run_training(
        model_config, TrainConfig(), tokenizer_config,
        train_path=None if stream_training else path,
        eval_path=path if stream_training else None,
        training_log_path=tmp_path / "log.jsonl",
        train_dataset=source if stream_training else None,
        eval_dataset=None if stream_training else StreamingDataset(source, 8),
    )
    assert len(seen) == 1
    assert (tmp_path / "tokenizer.json").is_file()


@pytest.mark.parametrize("empty", [False, True])
def test_training_stops_at_step_limit_or_empty_stream(monkeypatch, empty):
    train_module = importlib.import_module("torch_llm.training.train")
    train_step = Mock()
    monkeypatch.setattr(train_module, "train_step", train_step)
    monkeypatch.setattr(train_module, "evaluate", Mock(return_value=0.0))
    monkeypatch.setattr(train_module, "save_checkpoint", Mock())
    monkeypatch.setattr(train_module.t.cuda, "synchronize", Mock())
    monkeypatch.setattr(train_module.t.cuda, "memory_allocated", Mock(return_value=0))
    batch = next(iter(loader(StreamingDataset(["abcd"], 4, CharacterTokenizer()), 4, 8)))
    consumed = []

    def batches():
        for item in [] if empty else repeat(batch):
            consumed.append(item)
            yield item

    optimizer = SimpleNamespace(param_groups=[{"lr": 0.1}])
    args = (
        ModelConfig(), TrainConfig(total_steps=3), BPETokenizerConfig(), Mock(),
        batches(), [], Mock(), optimizer, optimizer, None, None,
    )
    if empty:
        with pytest.raises(ValueError, match="empty or exhausted"):
            train_module.train(*args, device="cpu")
    else:
        train_module.train(*args, device="cpu")
        assert train_step.call_count == len(consumed) == 3
