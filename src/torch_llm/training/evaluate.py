import torch as t
from torch.nn.functional import cross_entropy


def evaluate(
        model,
        eval_loader,
        *,
        device,
):
    model.eval()
    batches = 0
    accum_loss = 0

    with t.no_grad():
        for batch in eval_loader:
            batches += 1

            batch = batch.to(device)

            accum_loss += cross_entropy(model(
                token_ids=batch.token_ids,
                cu_seqlens=batch.cu_seqlens,
                token_positions=batch.token_positions,
                batch_max_seq_len=batch.batch_max_seq_len,
                mode="train",
                kv_caches=None
            ).logits.float(), batch.targets).item()

    model.train()
    return accum_loss / batches
