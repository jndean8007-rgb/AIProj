from torch_llm.data_pipeline.bpe_tokenizer import BPETokenizer
from torch_llm.model.model import TransformerLM
import torch as t


def generate_setup(model_path, model_config, tokenizer_path, tokenizer_config, device, dtype):
    model = TransformerLM(model_config).to(
        device=device,
        dtype=dtype
    )

    checkpoint = t.load(model_path)

    model.load_state_dict(checkpoint["model"]) #pull directly from model path after splitting parameter paths

    tokenizer = BPETokenizer.load(tokenizer_path, tokenizer_config)

    return model, tokenizer