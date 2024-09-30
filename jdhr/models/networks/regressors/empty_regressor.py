import jittor as jt
from jittor import nn
from jdhr.engine import REGRESSORS
from jdhr.utils.base_utils import dotdict
from jdhr.utils.net_utils import NoopModule


@REGRESSORS.register_module()
class EmptyRegressor(NoopModule):
    def __init__(self, in_dim: int, **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = in_dim

    def execute(self, feat: jt.Var, batch: dotdict = None):
        return jt.zeros(*feat.shape[:-1], 0,  dtype=feat.dtype)
