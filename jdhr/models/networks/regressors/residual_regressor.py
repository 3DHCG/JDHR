import jittor as jt
from jittor import nn

from jdhr.engine import REGRESSORS
from jdhr.utils.console_utils import *
from jdhr.utils.base_utils import dotdict
from jdhr.engine.registry import call_from_cfg
from jdhr.models.networks.regressors.mlp_regressor import MlpRegressor


@REGRESSORS.register_module()
class ResidualRegressor(MlpRegressor):
    # Outputs `torch.cat([x, self.mlp(x)], dim=-1)`, maybe used somewhere, eg. `IbrRegressor`
    def __init__(self,
                 out_actvn: nn.Module = nn.ReLU(),  # This probably is not the final regressor
                 **kwargs,
                 ):
        # Inherit from `MlpRegressor`
        call_from_cfg(super().__init__, kwargs, out_actvn=out_actvn)

    def execute(self, feat: jt.Var, batch: dotdict = None):
        out = super().execute(feat, batch)
        assert out.shape[:-1] == feat.shape[:-1], "Adjust the configuration of the MLP so that the input and output are in the same shape"
        return jt.cat([feat, out], dim=-1)
