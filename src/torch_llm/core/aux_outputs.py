from collections import defaultdict
from dataclasses import dataclass, field

import torch as t


@dataclass
class AuxOutputs:
    losses: dict[str, t.Tensor] = field(default_factory=dict)
    metrics: dict[str, t.Tensor] = field(default_factory=dict)

    def __post_init__(self):
        if any(x.requires_grad for x in self.metrics.values() if isinstance(x, t.Tensor)):
            raise ValueError("Metrics must not have grads")

        if any(x.ndim != 0 for x in self.losses.values()):
            raise ValueError("Loss must be 0 dimensional")

    def merged(self, other, prefix: str):
        if not {f"{prefix}.{k}" for k in other.metrics}.isdisjoint(
                {k for k in self.metrics}
        ):
            raise ValueError("Child AuxOutputs contains same metrics key as parent")

        if not {f"{prefix}.{k}" for k in other.losses}.isdisjoint(
            {k for k in self.losses}
        ):
            raise ValueError("Child AuxOutputs contains same losses key as parent")


        return AuxOutputs(
            losses=self.losses
                   | {prefix + "." + name: value
                      for name, value in other.losses.items()},
            metrics=self.metrics
                    | {prefix + "." + name: value
                       for name, value in other.metrics.items()}
        )

    def total_loss(self, weights: dict[str, float]):
        if not self.losses:
            return t.zeros(())

        loss_map: defaultdict[str, list[t.Tensor]] = defaultdict(list)

        for name in self.losses.keys():
            type = name.rsplit(".", maxsplit=1)[-1]
            loss_map[type].append(self.losses[name])

        if not all(name in weights for name in loss_map.keys()):
            raise ValueError("All types of losses must have a weight")

        avg_loss_map = {key:  t.stack(values).mean() for key, values in loss_map.items()}

        return t.sum(t.stack([avg_loss_map[name] * weights[name] for name in avg_loss_map.keys()]))
