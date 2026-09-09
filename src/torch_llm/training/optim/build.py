import torch as t

from torch_llm.training.optim.muon import Muon


def use_muon(name, param):
    if param.ndim < 2:
        return False

    muon_names = (
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "o_proj.weight",
        "gate_weight",
        "up_weight",
        "down_weight",
    )

    return any(key in name for key in muon_names)


def build_optimizers(
    model,
    muon_lr=0.02,
    adamw_lr=3e-4,
    weight_decay=0.1,
):
    muon_params = []
    adamw_params = []

    muon_names = []
    adamw_names = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if use_muon(name, p):
            muon_params.append(p)
            muon_names.append(name)
        else:
            adamw_params.append(p)
            adamw_names.append(name)

    assert set(map(id, muon_params)).isdisjoint(
        set(map(id, adamw_params))
    )

    muon = Muon(
        muon_params,
        lr=muon_lr,
        momentum=0.95,
        weight_decay=weight_decay,
        ns_steps=5,
        nesterov=True,
    )

    adamw = t.optim.AdamW(
        adamw_params,
        lr=adamw_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay,
    )

    return muon, adamw, muon_names, adamw_names
