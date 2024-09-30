# Literally return the input as is
#import torch
#from torch import nn
import jittor as jt
from jdhr.engine import EMBEDDERS
from jdhr.utils.base_utils import dotdict
from jittor import nn
from jdhr.utils.net_utils import NoopModule

@EMBEDDERS.register_module()
class EmptyEmbedder(NoopModule):
    def __init__(self, out_dim=0, **kwargs):
        super().__init__()
        self.out_dim = 0  # no embedding, no output

    def execute(self, inputs: jt.Var, batch: dotdict = None):
        return jt.zeros(*inputs.shape[:-1], 0, dtype=inputs.dtype)  # empty tensor
