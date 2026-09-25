import torch as t

from torch_llm.data_pipeline.bpe_tokenizer import BPETokenizer
from torch_llm.model.model import TransformerLM


def generate_setup(model_path, model_config, tokenizer_path, tokenizer_config, device, dtype):
    # Load on CPU so a saved CUDA device or optimizer state does not allocate
    # extra GPU memory. Accept both training checkpoints and plain state dicts.
    checkpoint = t.load(model_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model = TransformerLM(model_config)
    model.load_state_dict(state_dict)
    model = model.to(device=device, dtype=dtype).eval()
    tokenizer = BPETokenizer.load(tokenizer_path, tokenizer_config)
    return model, tokenizer
