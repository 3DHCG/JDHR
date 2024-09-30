import jittor as jt
from jittor import nn
from jdhr.engine import SUPERVISORS
from jdhr.engine.registry import call_from_cfg
from jdhr.utils.console_utils import *
from jdhr.utils.console_utils import dotdict
from jdhr.utils.loss_utils import ImgLossType, mse, mIoU_loss
from jdhr.models.supervisors.volumetric_video_supervisor import VolumetricVideoSupervisor

@SUPERVISORS.register_module()
class OpacitySupervisor(VolumetricVideoSupervisor):
    def __init__(self,
                 network: nn.Module,
                 ent_loss_weight: float = 0.0,
                 **kwargs):
        call_from_cfg(super().__init__, kwargs, network=network)

        self.ent_loss_weight = ent_loss_weight

    def compute_loss(self, output: dotdict, batch: dotdict, loss: jt.Var, scalar_stats: dotdict, image_stats: dotdict):
        if 'occ' in output and self.ent_loss_weight > 0:
            ent_loss = -jt.mean(output.occ * jt.log(output.occ))
            scalar_stats.ent_loss = ent_loss
            loss += self.ent_loss_weight * ent_loss

        return loss
