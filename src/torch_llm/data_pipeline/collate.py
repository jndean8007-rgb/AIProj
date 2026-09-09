from torch_llm.data_pipeline.pack_training import pack_training_sequences


def collate_training_sequences(sequences, model_max_seq_len): # for now only training mode
    return pack_training_sequences(sequences, model_max_seq_len)
