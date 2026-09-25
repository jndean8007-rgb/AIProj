from torch_llm.inference.kvcache_config import KVCacheConfig
from torch_llm.data_pipeline.bpe_tokenizer_config import BPETokenizerConfig
from torch_llm.inference.output_handler import OutputHandler
from torch_llm.inference.generation_setup import generate_setup
from torch_llm.inference.runtime import InferenceRuntime
from torch_llm.model.model_config import ModelConfig
from torch_llm.path_config import PathConfig
import torch as t


#all these paths probably live in configs


def main():
    #REFACTOR ALL OF THIS TO A DISTRIBUTOR: TRAIN, INFER, ETC

    path_config = PathConfig()

    '''from torch_llm.training.run_training import run_training
    from torch_llm.training.train_config import TrainConfig

    model_config = ModelConfig()
    train_config = TrainConfig()
    tokenizer_config = BPETokenizerConfig()

    from_checkpoint = True #True

    run_training(
        model_config=model_config,
        train_config=train_config,
        tokenizer_config=tokenizer_config,
        train_path=path_config.train_path,
        eval_path=path_config.eval_path,
        training_log_path=path_config.training_log_path,
        from_checkpoint=from_checkpoint,
        config_load_path=path_config.config_load_path,
        model_load_path=path_config.model_load_path,
        tokenizer_load_path=path_config.tokenizer_load_path,
    )'''


    model_config = ModelConfig()
    tokenizer_config = BPETokenizerConfig()
    kvcache_config = KVCacheConfig()

    model, tokenizer = generate_setup(
        model_path=path_config.model_load_path,
        model_config=model_config,
        tokenizer_path=path_config.tokenizer_load_path,
        tokenizer_config=tokenizer_config,
        device="cuda",
        dtype=t.bfloat16,
    )

    runtime = InferenceRuntime(
        model=model,
        model_config=model_config,
        tokenizer=tokenizer,
        kvcache_config=kvcache_config,
    )

    output_handler = OutputHandler(tokenizer)

    runtime.submit(input("Prompt: "), max_new_tokens=100)
    runtime.generate(output_handler)


if __name__ == '__main__':
    main()
