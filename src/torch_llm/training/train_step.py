from dataclasses import dataclass
import time

import torch as t
import torch.nn.functional as F

from torch_llm.data_pipeline.pack_training import PackedTrainingBatch
from torch_llm.inference import kv_cache


@dataclass
class TrainStepOutput:
    loss: t.Tensor
    lm_loss: t.Tensor
    aux_loss: t.Tensor
    grad_norm: t.Tensor | None


def train_step(
        model,
        batch: PackedTrainingBatch,
        muon,
        adamw,
        *,
        aux_loss_weight: float,
        max_grad_norm: float | None = None,
        muon_scheduler=None,
        adamw_scheduler=None,
):
    '''
    Batch contains:
    token_ids: (T,)
    targets: (T,)
    cu_seqlens: (B+1,)
    token_positions: (T,)
    batch_max_seq_len: int

    '''

    # 1. clear old gradients
    '''muon.zero_grad(set_to_none=True)
    adamw.zero_grad(set_to_none=True)


    # 3. model forward

    model_output = model(
        token_ids=batch.token_ids,
        cu_seqlens=batch.cu_seqlens,
        token_positions=batch.token_positions,
        batch_max_seq_len=batch.batch_max_seq_len,
        mode="train",
        kv_caches=None
    )

    moe_stats = model_output.moe_stats
    aux_loss = model_output.aux_loss


    # 4. calculate language-model loss
    lm_loss = F.cross_entropy(model_output.logits.float(), batch.targets)


    # 5. combine with MoE auxiliary loss
    loss = lm_loss + aux_loss_weight * aux_loss


    # 6. backward #for now entirety, later can introduce microbatches
    loss.backward()

    # 7. optional gradient clipping
    if max_grad_norm is not None:
        grad_norm = t.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    else:
        grad_norm = None


    # 8. optimizer updates
    #    Muon
    #    AdamW
    muon.step()
    adamw.step()

    # 9. scheduler updates
    if muon_scheduler is not None:
        muon_scheduler.step()

    if adamw_scheduler is not None:
        adamw_scheduler.step()


    # 10. return metrics
    return TrainStepOutput(
        loss=loss,
        lm_loss=lm_loss,
        aux_loss=aux_loss,
        grad_norm=None if grad_norm is None else grad_norm.detach(),
    )'''

    # 3. forward pass
    t.cuda.synchronize()
    t0 = time.perf_counter()

    model_output = model(
        token_ids=batch.token_ids,
        cu_seqlens=batch.cu_seqlens,
        token_positions=batch.token_positions,
        batch_max_seq_len=batch.batch_max_seq_len,
        mode="train",
        paged_kv_caches=None,
        cache_batch_context=None
    )

    t.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"[forward] {t1 - t0:.4f}s")

    moe_stats = model_output.moe_stats
    aux_loss = model_output.aux_loss


    # 4. calculate language-model loss
    t.cuda.synchronize()
    t2 = time.perf_counter()

    lm_loss = F.cross_entropy(model_output.logits.float(), batch.targets)

    t.cuda.synchronize()
    t3 = time.perf_counter()
    print(f"[lm_loss] {t3 - t2:.4f}s")


    # 5. combine with MoE auxiliary loss
    t4 = time.perf_counter()
    loss = lm_loss + aux_loss_weight * aux_loss
    t5 = time.perf_counter()
    print(f"[combine_loss] {t5 - t4:.4f}s")


    # 6. backward #for now entirety, later can introduce microbatches
    t.cuda.synchronize()
    t6 = time.perf_counter()

    loss.backward()

    t.cuda.synchronize()
    t7 = time.perf_counter()
    print(f"[backward] {t7 - t6:.4f}s")


    # 7. optional gradient clipping
    t.cuda.synchronize()
    t8 = time.perf_counter()

    if max_grad_norm is not None:
        grad_norm = t.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    else:
        grad_norm = None

    t.cuda.synchronize()
    t9 = time.perf_counter()
    print(f"[grad_clip] {t9 - t8:.4f}s")


    # 8. optimizer updates
    #    Muon
    #    AdamW
    t.cuda.synchronize()
    t10 = time.perf_counter()

    muon.step()

    t.cuda.synchronize()
    t11 = time.perf_counter()
    print(f"[Muon_step] {t11 - t10:.4f}s")

    t.cuda.synchronize()
    tt = time.perf_counter()

    adamw.step()

    t.cuda.synchronize()
    ttt = time.perf_counter()
    print(f"[AdamW_step] {ttt - tt:.4f}s")


    # 9. scheduler updates
    t12 = time.perf_counter()

    if muon_scheduler is not None:
        muon_scheduler.step()

    if adamw_scheduler is not None:
        adamw_scheduler.step()

    t13 = time.perf_counter()
    print(f"[scheduler_step] {t13 - t12:.4f}s")

    print(f"[TOTAL] {t13 - t0:.4f}s")

    return TrainStepOutput(
        loss=loss,
        lm_loss=lm_loss,
        aux_loss=aux_loss,
        grad_norm=None if grad_norm is None else grad_norm.detach(),
    )


'''
later return:
learning rates
tokens/sec
expert load CV
router entropy
routing margin
max allocated VRAM
step time
'''
