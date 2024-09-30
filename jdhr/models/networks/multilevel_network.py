import copy
import jittor as jt
from jittor import nn
from typing import List, Tuple
from jdhr.engine import NETWORKS
from jdhr.utils.base_utils import dotdict
from jdhr.models.networks.volumetric_video_network import VolumetricVideoNetwork
from jdhr.models.networks.regressors.mlp_regressor import MlpRegressor
from jdhr.models.networks.regressors.empty_regressor import EmptyRegressor
from jdhr.models.networks.regressors.split_regressor import SplitRegressor


@NETWORKS.register_module()
class MultilevelNetwork(nn.Module):
    def __init__(self,
                 network_cfgs: List[dotdict] = [
                     dotdict(
                         type=VolumetricVideoNetwork.__name__,
                         geometry_cfg=dotdict(
                             type=SplitRegressor.__name__,
                             width=128, depth=4
                         ),
                         appearance_cfg=dotdict(
                             type=EmptyRegressor.__name__,
                         ),
                     ),  # coarse
                     dotdict(
                         type=VolumetricVideoNetwork,
                         geometry_cfg=dotdict(
                             type=SplitRegressor.__name__,
                             width=512, depth=8
                         ),
                         appearance_cfg=dotdict(
                             type=MlpRegressor.__name__,
                             width=256, depth=2, out_dim=3),
                     ),  # fine
                 ],
                 **kwargs,  # will feed these into the lower level networks
                 ):
        super().__init__()
        kwargs = dotdict(kwargs)  # for recursive update
        network_cfgs = [copy.deepcopy(kwargs).update(network_cfg) for network_cfg in network_cfgs]  # for recursive update
        self.networks = nn.ModuleList([
            NETWORKS.build(network_cfg)  # by default, use kwargs, if modified by cfg, use cfg
            for network_cfg in network_cfgs
        ])
        self.execute = self.compute

    # The actual function that does the forwarding for you

    def compute_level(
        self, forward_function=VolumetricVideoNetwork.compute.__name__, level: int = -1,
        *args, **kwargs,
    ) -> Tuple[jt.Var, jt.Var]:
        return getattr(self.networks[level], forward_function)(*args, **kwargs)

    # Convenience functions
    def compute_coarse(
        self, forward_function=VolumetricVideoNetwork.compute.__name__, level: int = 0,
        *args, **kwargs,
    ):
        # make sure `level` does not exceed the number of networks
        level = min(level, len(self.networks) - 1)
        return self.compute_level(forward_function, level, *args, **kwargs)

    def compute_fine(
        self, forward_function=VolumetricVideoNetwork.compute.__name__, level: int = -1,
        *args, **kwargs,
    ):
        return self.compute_level(forward_function, level, *args, **kwargs)

    # Actual forwarding considering only the last layer
    def compute_geometry(self, xyz, t, dist, batch):
        return self.compute_fine(VolumetricVideoNetwork.compute_geometry.__name__, -1, xyz, t, dist, batch)

    def compute(self, xyz, dir, t, dist, batch):
        return self.compute_fine(VolumetricVideoNetwork.compute.__name__, -1, xyz, dir, t, dist, batch)
