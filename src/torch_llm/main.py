from pathlib import Path

from torch_llm.data_pipeline.bpe_tokenizer_config import BPETokenizerConfig
from torch_llm.inference.output_handler import OutputHandler
from torch_llm.inference.batching import PrefillBatch
from torch_llm.inference.generation_setup import generate_setup
from torch_llm.inference.input_processor import InferenceInputProcessor
from torch_llm.inference.runtime import InferenceRuntime
from torch_llm.model.model_config import ModelConfig
from torch_llm.training.run_training import run_training
from torch_llm.training.train_config import TrainConfig



#all these paths probably live in configs


def main():
    #REFACTOR ALL OF THIS TO A DISTRIBUTOR: TRAIN, INFER, ETC

    model_config = ModelConfig()
    tokenizer_config = BPETokenizerConfig()
    tokenizer_load_path = r"C:\Users\rosie\PycharmProjects\AIProj\checkpoints\tinyshake\BPETokenizer"
    model_load_path = r"C:\Users\rosie\PycharmProjects\AIBigProject\checkpoints\tinyshake\500.pth"

    model, tokenizer = generate_setup(
        model_path=model_load_path,
        model_config=model_config,
        tokenizer_path=tokenizer_load_path,
        tokenizer_config=tokenizer_config,
        device="cuda",
        dtype=t.float32,
    )

    runtime = InferenceRuntime(
        model=model,
        model_config=model_config,
        tokenizer=tokenizer,
    )

    output_handler = OutputHandler()

    inference_processor = InferenceInputProcessor(tokenizer)

    prompts = list(input("Prompt: ")) # will make actual batches later as per intended purpose

    prefill_batch = inference_processor.prepare(prompts=prompts)

    runtime.generate(prefill_batch, output_handler)



    '''train_path = Path(r'C:\Users\rosie\PycharmProjects\AIBigProject\data\processed\tiny_shake\train')

    eval_path = Path(r'C:\Users\rosie\PycharmProjects\AIBigProject\data\processed\tiny_shake\eval')

    model_config = ModelConfig()
    train_config = TrainConfig()
    tokenizer_config = BPETokenizerConfig()

    from_checkpoint = True #True
    config_load_path = None
    model_load_path = r"C:\Users\rosie\PycharmProjects\AIBigProject\checkpoints\tinyshake\0.pth"
    tokenizer_load_path = r"C:\Users\rosie\PycharmProjects\AIBigProject\checkpoints\tinyshake\BPETokenizer"
    training_log_path = r"C:\Users\rosie\PycharmProjects\AIBigProject\checkpoints\tinyshake\trainingstats"

    run_training(
        model_config=model_config,
        train_config=train_config,
        tokenizer_config=tokenizer_config,
        train_path=train_path,
        eval_path=eval_path,
        training_log_path=training_log_path,
        from_checkpoint=from_checkpoint,
        config_load_path=config_load_path,
        model_load_path=model_load_path,
        tokenizer_load_path=tokenizer_load_path,
    )'''

if __name__ == '__main__':
    main()