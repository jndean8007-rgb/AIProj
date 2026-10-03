from typing import Any

from core.registry import RESIDUALS
import torch.nn as nn
import torch as t


@RESIDUALS.register("standard")
class Residual(nn.Module):
    """
    current simple residual layer
    designed for future flexibility of implementation
    """
    def __init__(self, config):
        super().__init__()
        self.state = None
        self.config = config

    def init(self, x0: t.Tensor) -> t.Tensor:
        self.state = x0
        return x0

    def read(self, state, i: int) -> t.Tensor:
        state

    def write(self, state: Any, i: int, update: t.Tensor) -> t.Tensor:
        self.state = state + update
        return self.state

    def final(self, state) -> t.Tensor
        return state