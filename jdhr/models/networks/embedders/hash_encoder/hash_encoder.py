import jittor as jt
from jittor import nn
from jdhr.engine import EMBEDDERS
from .grid_encode import GridEncode
from typing import List
import numpy as np

@EMBEDDERS.register_module()
class TcnnHashEmbedderjt(nn.Module):
    from jdhr.models.cameras.optimizable_camera import OptimizableCamera
    def __init__(self, n_pos_dims=3,
                 n_features_per_level=2,
                 n_levels=16,
                 base_resolution=16,
                 log2_hashmap_size=19,
                 per_level_scale=1.49,
                 *args,**kwargs,):

        self.using_fp16 = True#False#config.model.object.sdf.encoding.fp16
        self.hash_func = "p0 ^ p1 * 2654435761u ^ p2 * 805459861u"
        self.hash_func_header = f"""
#define get_index(p0,p1,p2) {self.hash_func}
        """
        self.encoder = GridEncode(self.hash_func_header,
                                  per_level_scale=per_level_scale,
                                  n_pos_dims=n_pos_dims,
                                  n_features_per_level=n_features_per_level,
                                  n_levels=n_levels, base_resolution=base_resolution,
                                  log2_hashmap_size=log2_hashmap_size,
                                  using_fp16=self.using_fp16,
                                  loss_scale=128.0)
        self.grad_type = 'float32'
        if self.using_fp16:
            self.grad_type = 'float16'
        self.m_grid = jt.init.uniform(
            [self.encoder.m_n_params], low=-1e-4, high=1e-4)
        self.out_dim=n_features_per_level*n_levels

    def execute(self, x):
        m_grid = self.m_grid.float16() if self.using_fp16 else self.m_grid

        output = self.encoder(x, m_grid)
        assert(output.dtype == self.grad_type)
        return output
