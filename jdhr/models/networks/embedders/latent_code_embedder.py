import numpy as np
from typing import Union
from jittor import nn
import jittor as jt
from jdhr.engine import EMBEDDERS
from jdhr.utils.console_utils import *
from jdhr.utils.base_utils import dotdict
from jdhr.utils.net_utils import make_params
from jdhr.utils.grid_utils import grid_sample_1d


@EMBEDDERS.register_module()
class LatentCodeEmbedder(nn.Module):
    def __init__(self,
                 # we want to make this length agnostic
                 n_codes: int = 1000,  # if you've got more than these number of frames, it will be interpolated
                 out_dim: int = 128,  # shape of latent code
                 dtype: Union[str, jt.dtype] = jt.float,
                 std: float = 0.01,
                 ):
        super().__init__()

        self.n_codes = n_codes
        self.out_dim = out_dim
        dtype = getattr(jt, dtype) if isinstance(dtype, str) else dtype

        # The optimizable latent codes, will be interpolated or sampled from
        latent_codes = jt.zeros(n_codes, out_dim, dtype=dtype)
        jt.init.gauss_(latent_codes, mean=0, std=std)
        self.latent_codes = make_params(latent_codes)  # N, C

    # All in a variable & batch format
    def execute(self, t: jt.Var, batch: dotdict = None):
        # B, P, 1

        B, L_out = t.shape[0], t.shape[1]
        L_in, C = self.latent_codes.shape

        grid = t * 2 - 1  # -1, 1
        grid = grid.reshape(B, L_out,1)  # B, L_out, 1
        codes = self.latent_codes[None].expand(B, L_in, C).permute(0, 2, 1)  # B, C, L_in
        feat = grid_sample_1d(codes, grid, align_corners=False, padding_mode='border')
        feat = feat.permute(0, 2, 1)  # B, C, L_out -> B, L_out, C
        feat = feat.view(t.shape[:-1] + (-1,))  # B, L_out, C -> whatever

        return feat
