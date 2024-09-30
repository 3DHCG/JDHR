# The user might specify multiple supervisors
# Call the supervise function, aggregate losses and stats
# Default loss module (called supervisor)
import copy
import numpy as np
from typing import Union, List
import jittor as jt
from jittor import nn
from jdhr.engine import SUPERVISORS
from jdhr.utils.console_utils import *
from jdhr.engine.registry import call_from_cfg
from jdhr.models.supervisors.mask_supervisor import MaskSupervisor
from jdhr.models.supervisors.flow_supervisor import FlowSupervisor
from jdhr.models.supervisors.depth_supervisor import DepthSupervisor
from jdhr.models.supervisors.normal_supervisor import NormalSupervisor
from jdhr.models.supervisors.opacity_supervisor import OpacitySupervisor
from jdhr.models.supervisors.proposal_supervisor import ProposalSupervisor
from jdhr.models.supervisors.geometry_supervisor import GeometrySupervisor
from jdhr.models.supervisors.temporal_supervisor import TemporalSupervisor
from jdhr.models.supervisors.displacement_supervisor import DisplacementSupervisor
from jdhr.models.supervisors.volumetric_video_supervisor import VolumetricVideoSupervisor


@SUPERVISORS.register_module()
class SequentialSupervisor(VolumetricVideoSupervisor):
    def __init__(self,
                 network: nn.Module,
                 supervisor_cfgs: List[dotdict] = [
                     dotdict(type=MaskSupervisor.__name__),
                     dotdict(type=FlowSupervisor.__name__),
                     dotdict(type=DepthSupervisor.__name__),
                     dotdict(type=NormalSupervisor.__name__),
                     dotdict(type=OpacitySupervisor.__name__),
                     dotdict(type=ProposalSupervisor.__name__),
                     dotdict(type=GeometrySupervisor.__name__),
                     dotdict(type=TemporalSupervisor.__name__),
                     dotdict(type=DisplacementSupervisor.__name__),
                     dotdict(type=VolumetricVideoSupervisor.__name__), # NOTE: Put this last for PSNR to be displayed last along with full loss
                 ],
                 **kwargs,
                 ):
        kwargs = dotdict(kwargs)  # for recursive update
        call_from_cfg(super().__init__, kwargs, network=network)
        supervisor_cfgs = [copy.deepcopy(kwargs).update(supervisor_cfg) for supervisor_cfg in supervisor_cfgs]  # for recursive update
        self.supervisors: nn.ModuleList[VolumetricVideoSupervisor] = nn.ModuleList([SUPERVISORS.build(supervisor_cfg, network=network) for supervisor_cfg in supervisor_cfgs])

    def compute_loss(self, output: dotdict, batch: dotdict, loss: jt.Var, scalar_stats: dotdict, image_stats: dotdict):
        for supervisor in self.supervisors:
            loss = supervisor.compute_loss(output, batch, loss, scalar_stats, image_stats)  # loss will be added
        return loss
