import jittor as jt
from jittor import nn
from jdhr.engine import MODELS
from jdhr.utils.net_utils import NoopModule, make_buffer


@MODELS.register_module()
class NoopModel(NoopModule):
    def __init__(self,
                 **kwargs,  # suppress warnings
                 ):
        super().__init__()

        # For iteration based device tracking
        self.device_tracker = make_buffer(jt.empty(0))

        # For APIs
        self.camera = nn.Module()
        self.sampler = nn.Module()
        self.network = nn.Module()
        self.renderer = nn.Module()
        self.supervisor = nn.Module()
