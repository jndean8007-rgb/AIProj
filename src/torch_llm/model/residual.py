from typing import Any

import torch as t
import torch.nn as nn

from torch_llm.core.registry import RESIDUALS


@RESIDUALS.register("standard")
class StandardResidual(nn.Module):
    """
    Plain residual stream: every sublayer reads the running sum and adds its update.

    The object is built once with the model and holds no per-forward data; the
    stream itself (the state) is created by init() on every forward and passed in
    and out explicitly. For this residual the state is just the [T, D] tensor.

    An nn.Module (with no parameters) so future learnable residuals are held,
    trained and saved the same way.
    """

    def __init__(self, config):
        super().__init__()

    def init(self, x0: t.Tensor) -> t.Tensor:
        """x0: [T, D] embedding output -> initial state"""
        return x0

    def read(self, state: t.Tensor, i: int) -> t.Tensor:
        """Input for sublayer i. The standard residual ignores i."""
        return state

    def write(self, state: t.Tensor, i: int, update: t.Tensor) -> t.Tensor:
        """Never in place: autograd saved the old state for the norm's backward."""
        return state + update

    def final(self, state: Any) -> t.Tensor:
        """Input for the output head."""
        return state
