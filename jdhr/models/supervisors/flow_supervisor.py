import jittor as jt
from jittor import nn

from jdhr.engine import SUPERVISORS
from jdhr.engine.registry import call_from_cfg
from jdhr.utils.console_utils import *
from jdhr.utils.console_utils import dotdict
from jdhr.utils.loss_utils import ImgLossType, mse, mIoU_loss
from jdhr.models.supervisors.volumetric_video_supervisor import VolumetricVideoSupervisor

@SUPERVISORS.register_module()
class FlowSupervisor(VolumetricVideoSupervisor):
    def __init__(self,
                 network: nn.Module,
                 flow_loss_weight: float = 0.0,
                 normalize: bool = True,
                 **kwargs,
                 ):
        call_from_cfg(super().__init__, kwargs, network=network)

        self.flow_loss_weight = flow_loss_weight
        self.normalize = normalize

    def compute_loss(self, output: dotdict, batch: dotdict, loss: jt.Var, scalar_stats: dotdict, image_stats: dotdict):
        if 'flo_map' in output and 'flow' in batch and 'flow_weight' in batch and \
            self.flow_loss_weight > 0:
            sum_loss = nn.l1_loss(output.flo_map, batch.flow, reduction='none').mean(dim=-1, keepdim=True)
            if self.normalize:
                flow_loss = jt.sum((sum_loss * batch.flow_weight) / (jt.sum(batch.flow_weight) + 1e-8))
            else:
                flow_loss = jt.mean((sum_loss * batch.flow_weight))
            scalar_stats.flow_loss = flow_loss
            loss += self.flow_loss_weight * flow_loss

        return loss
