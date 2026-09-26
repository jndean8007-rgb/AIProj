from itertools import islice
from pathlib import Path

from torch_llm.data_pipeline.dataset import TokenSequenceDataset
from torch_llm.data_pipeline.loader import loader
from torch_llm.model.model import TransformerLM
from torch_llm.training.optim.build import build_optimizers
from torch_llm.training.scheduler import build_warmup_cosine_scheduler
import torch as t
from torch_llm.data_pipeline.bpe_tokenizer import BPETokenizer

def setup(
        model_config,
        train_config,
        tokenizer,
        train_path,
        eval_path,
        device='cuda',
        dtype=t.bfloat16,
        train_dataset=None,
        eval_dataset=None,
):

    model = TransformerLM(model_config).to(
        device=device,
        dtype=dtype
    )

    muon, adamw, muon_names, adamw_names = build_optimizers(
        model,
        train_config.muon_lr,
        train_config.adamw_lr,
        train_config.weight_decay,
    )


    muon_scheduler = build_warmup_cosine_scheduler(
        muon,
        train_config.warmup_steps,
        train_config.total_steps,
        train_config.min_lr_ratio
    )

    adamw_scheduler = build_warmup_cosine_scheduler(
        adamw,
        train_config.warmup_steps,
        train_config.total_steps,
        train_config.min_lr_ratio
    )


    if train_dataset is None:
        train_text = Path(train_path).read_text(encoding="utf-8")
        tsd_train = TokenSequenceDataset.from_training_texts(
            [train_text],
            tokenizer,
            model_config.model_max_seq_len
        )
    else:
        train_dataset.tokenizer = tokenizer
        train_dataset.seq_len = model_config.model_max_seq_len
        tsd_train = train_dataset

    train_loader = loader(
        tsd_train,
        model_config.model_max_seq_len,
        4096
    )

    if eval_dataset is None:
        eval_text = Path(eval_path).read_text(encoding="utf-8")
        tsd_eval = TokenSequenceDataset.from_training_texts(
            [eval_text],
            tokenizer,
            model_config.model_max_seq_len
        )
    else:
        eval_dataset.tokenizer = tokenizer
        eval_dataset.seq_len = model_config.model_max_seq_len
        tsd_eval = eval_dataset

    eval_loader = loader(
        tsd_eval,
        model_config.model_max_seq_len,
        4096
    )


    return (
        model,
        train_loader,
        eval_loader,
        adamw,
        muon,
        adamw_scheduler,
        muon_scheduler
    )


def build_or_load_tokenizer(
        train_path,
        tokenizer_config,
        from_checkpoint = False,
        tokenizer_load_path = None,
        train_dataset = None,
):

    if from_checkpoint:
        if tokenizer_load_path is None:
            raise ValueError("tokenizer_load_path required when resuming")

        tokenizer = BPETokenizer.load(tokenizer_load_path, tokenizer_config)

    else:
        tokenizer = BPETokenizer.from_config(
            tokenizer_config
        )

        if train_dataset is None:
            tokenizer.train(
                tokenizer_config,
                train_path,
            )
        else:
            if iter(train_dataset.text_iterator) is train_dataset.text_iterator:
                raise ValueError("tokenizer training requires a re-iterable dataset, not a one-shot iterator")

            # only sample the stream when fitting the tokenizer
            tokenizer.train_from_iterator(
                tokenizer_config,
                islice(
                    train_dataset.tokenizer_training_iterator(),
                    tokenizer_config.max_training_samples,
                ),
            )

    return tokenizer
