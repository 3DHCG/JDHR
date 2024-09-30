# This file contains optimizable camera parameters
# Implemented in SO3xR3, exponential map of rotation and translation from screw rt motion


import jittor as jt
from os.path import join

from jdhr.engine import cfg
from jdhr.engine import CAMERAS
from jdhr.utils.base_utils import dotdict
from jdhr.utils.net_utils import NoopModule


@CAMERAS.register_module()
class NoopCamera(NoopModule):  # TODO: Implement intrinsics optimization
    def __init__(self, **kwargs):
        super().__init__()
        pass

    def forward_srcs(self, batch: dotdict):
        return batch

    def forward_cams(self, batch: dotdict):
        return batch

    def forward_rays(self, ray_o: jt.Var, ray_d: jt.Var, batch, use_z_depth: bool = False, correct_pix: bool = True):
        return ray_o, ray_d

    def execute(self, ray_o: jt.Var, ray_d: jt.Var, batch, use_z_depth: bool = False, correct_pix: bool = True):
        return ray_o, ray_d, batch
