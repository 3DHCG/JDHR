import jittor as jt
from jittor import nn
from jdhr.engine import REGRESSORS
from jdhr.utils.base_utils import dotdict


@REGRESSORS.register_module()
class DirectRegressor(nn.Module):
    def __init__(self, in_dim: int, name='density'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = in_dim

        self.name = name

    def execute(self, feat: jt.Var, batch: dotdict = None):
        return batch.output[self.name]
