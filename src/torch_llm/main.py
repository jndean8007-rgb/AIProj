from torch_llm.data_pipeline.bpe_tokenizer import BPETokenizer
from torch_llm.data_pipeline.bpe_tokenizer_config import BPETokenizerConfig
from torch_llm.data_pipeline.dataset import TokenSequenceDataset
from torch_llm.data_pipeline.loader import loader
from torch_llm.model.model import TransformerLM
from torch_llm.model.model_config import ModelConfig
from torch_llm.training.train_config import TrainConfig
from torch_llm.training import train
import torch as t


def main():
    model_config = ModelConfig(

    )

    train_config = TrainConfig(

    )

    tokenizer_config = BPETokenizerConfig(

    )

    tokenizer = BPETokenizer.from_config(
        tokenizer_config
    )

    tokenizer.train(
        tokenizer_config,

    )

    tsd = TokenSequenceDataset.from_training_text(
        ,
        tokenizer,
        model_config.model_max_seq_len
    )

    dataloader = loader.loader(
        tsd,
        model_config.model_max_seq_len,
        #? max tokens per batch
    )
    model = train.train(
        model_config,
        train_config,
        loader,

        device='cuda',
        dtype=t.bfloat16,
    )



if __name__ == '__main__':
    main()