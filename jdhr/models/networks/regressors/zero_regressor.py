import jittor as jt
from jittor import nn
from jdhr.engine import REGRESSORS
from jdhr.utils.base_utils import dotdict


@REGRESSORS.register_module()
class ZeroRegressor(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int = 3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim or in_dim

    def execute(self, feat: jt.Var, batch: dotdict = None):
        return jt.zeros(feat.shape[:-1] + (self.out_dim,), dtype=feat.dtype)
