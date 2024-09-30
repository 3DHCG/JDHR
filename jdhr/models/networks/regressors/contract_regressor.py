# mipnerf360 space contraction
import numpy as np
import jittor as jt
from jdhr.engine import REGRESSORS
from jdhr.utils.console_utils import *
from jdhr.utils.bound_utils import contract, get_bounds
from jdhr.models.cameras.optimizable_camera import OptimizableCamera
from jittor import nn

@REGRESSORS.register_module()
class ContractRegressor(nn.Module):
    radius = (OptimizableCamera.bounds[1] - OptimizableCamera.bounds[0]).max()  # make it easier on object-centric datasets

    def __init__(self,
                 in_dim: int = 3,
                 radius: float = radius,  # -> 10.0m?, bad convergence if radius too small
                 p: float = float('inf'),
                 normalize: bool = False,
                 ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = in_dim
        self.radius = radius
        self.p = p
        self.normalize = normalize

    def execute(self, xyz: jt.Var, batch: dotdict = None):
        xyz = contract(xyz, self.radius, self.p)
        if self.normalize:
            xyz = xyz / self.radius
        return xyz
