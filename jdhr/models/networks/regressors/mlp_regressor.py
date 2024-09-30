import jittor as jt
from jittor import nn
from typing import Literal

from jdhr.engine import REGRESSORS
from jdhr.utils.console_utils import *
from jdhr.utils.net_utils import MLP, get_function
from jdhr.utils.base_utils import dotdict


@REGRESSORS.register_module()
class MlpRegressor(nn.Module):
    # Outputs occupancy (used as alpha in volume rendering)
    def __init__(self,
                 in_dim: int,
                 out_dim: int = 3,
                 width: int = 256,
                 depth: int = 8,
                 actvn: nn.Module = nn.ReLU(),
                 out_actvn: nn.Module = nn.Identity(),

                 backend: Literal['tcnn', 'torch'] = 'torch',
                 otype: str = 'FullyFusedMLP',  # might lead to performance degredation, only used with backend == tcnn
                 dtype: str = 'float',
                 ):
        # Simply an MLP wrapper
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.width = width
        self.depth = depth
        self.backend = backend
        dtype = getattr(jt, dtype) if isinstance(dtype, str) else dtype
        actvn = get_function(actvn) if isinstance(actvn, str) else actvn
        out_actvn = nn.Sigmoid() if out_actvn=="Sigmoid" else get_function(out_actvn) if isinstance(out_actvn, str) else out_actvn
        if backend == 'torch':
            self.mlp = MLP(in_dim, width, depth, out_dim, actvn=actvn, out_actvn=out_actvn, dtype=dtype)
            jt.init.gauss_(self.mlp.linears[-1].bias,0, std=1e-4)  # small displacement by default

    def execute(self, feat: jt.Var, batch: dotdict = None):
        if self.backend == 'torch':
            return self.mlp(feat)
        else:
            return self.out_actvn(self.mlp(feat))
