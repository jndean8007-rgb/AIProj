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
        for batch_meta, packed_training_batch in eval_loader:
            batches += 1

            batch_meta = batch_meta.to(device)
            packed_training_batch = packed_training_batch.to(device)

            accum_loss += cross_entropy(model(
                batch_meta,
                paged_kv_caches=None,
            ).logits.float(), packed_training_batch.targets).item()

    model.train()
    return accum_loss / batches
