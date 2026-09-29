from torch_llm.core.batch_meta import BatchMeta
from torch_llm.data_pipeline.pack_training import pack_training_sequences


def collate_training_sequences(sequences, model_max_seq_len): # for now only training mode
    packed_training_sequences = pack_training_sequences(sequences, model_max_seq_len)

    return BatchMeta(
        token_ids=packed_training_sequences.token_ids,
        cu_seqlens=packed_training_sequences.cu_seqlens,
        token_positions=packed_training_sequences.token_positions,
        max_seqlen=packed_training_sequences.batch_max_seq_len,
        mode="train",
        cache_context=None,
    ), packed_training_sequences