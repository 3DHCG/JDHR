import os
import numpy as np
from os.path import join
from typing import List, Dict, Literal
from jittor import nn
import jittor as jt
from jdhr.engine import EMBEDDERS, cfg
from jdhr.utils.console_utils import *
from jdhr.utils.base_utils import dotdict
from jdhr.utils.net_utils import make_buffer, make_params

from jdhr.models.cameras.optimizable_camera import OptimizableCamera
from jdhr.models.networks.embedders.encoding import MultiResHashGrid
from jdhr.models.networks.embedders.hash_embedder import HashEmbedder


@EMBEDDERS.register_module()
class KPlanesEmbedder(nn.Module):
    n_frames = OptimizableCamera.n_views if OptimizableCamera.closest_using_t else OptimizableCamera.n_frames

    def __init__(self,
                 # Tcnn multi-resolution hash encoding texture related configuration
                 n_levels: int = 4,
                 n_features_per_level: int = 8,#8
                 b: float = 2.0,
                 base_resolution: int = 64,

                 # Manual initialization control
                 std: float = 1e-1,  # only used when backbone == grid
                 log2_hashmap_size: int = 16,  # only used when backbone == hash or tcnn (last level will use hash)
                 agg_method: Literal['cat', 'sum', 'prod'] = 'sum',
                 backbone: Literal['hash', 'tcnn', 'torch'] = 'hash',  # be smarter, and be careful on storage

                 # Normalization of coordinate sampling
                 bounds: List[List[int]] = OptimizableCamera.bounds,
                 dim_time: int = n_frames,

                 # Repeat
                 n_repeat: int = 1,  # number of duplicate feature grids for each of the spacial dimension, analogous to tensor decomposition
                 **kwargs,
                 ):
        super().__init__()
        self.n_repeat = n_repeat
        self._bounds = jt.array(bounds, dtype=jt.float)

        self.backbone = backbone
        if backbone == 'torch':
            self.spatial_embedding = nn.ParameterList([
                nn.Parameter(jt.zeros(3, n_features_per_level, int(base_resolution * b**i), int(base_resolution * b**i)))  # [xy, xz, yz], C, H, H
                for i in range(n_levels)])
            self.temporal_embedding = nn.ParameterList([
                nn.Parameter(jt.zeros(3, n_features_per_level, int(base_resolution * b**i), dim_time))  # [xy, xz, yz], C, H, T
                for i in range(n_levels)])
            for data in self.spatial_embedding:
                data.data.uniform_(-std, std)
            for data in self.temporal_embedding:
                data.data.uniform_(-std, std)

        elif backbone == 'hash' or backbone == 'tcnn':
            config = dotdict(n_levels=n_levels,
                             n_features_per_level=n_features_per_level,
                             b=b,
                             base_resolution=base_resolution,
                             log2_hashmap_size=log2_hashmap_size,
                             sum=False,  # cat features from different levels,
                             bounds=[[-1, -1], [1, 1]],  # already normalized in outer level, no need for secondary normalization
                             in_dim=2,  # image
                             include_input=False,
                             **kwargs,
                             )
            if backbone == 'hash':
                # overwrites (almost always smaller)
                self.spatial_embedding = nn.ModuleList([HashEmbedder(**config) for i in range(3 * n_repeat)])  # xy, xz, yz  MultiResHashGrid HashEmbedder
                self.temporal_embedding = nn.ModuleList([HashEmbedder(**config, predefined_sizes=[dim_time, -1]) for i in range(3 * n_repeat)])  # tx, ty, tz  MultiResHashGrid
            elif backbone == 'tcnn':
                self.spatial_embedding = nn.ModuleList([TcnnHashEmbedder(**config) for i in range(3 * n_repeat)])  # xy, xz, yz
                self.temporal_embedding = nn.ModuleList([TcnnHashEmbedder(**config, predefined_sizes=[dim_time, -1]) for i in range(3 * n_repeat)])  # tx, ty, tz
        else:
            raise NotImplementedError

        self.agg_method = agg_method
        if backbone == 'torch':  # different out dims, # ?: Is this getting too messy?
            if agg_method == 'cat':
                self.out_dim = n_features_per_level * n_levels * 6
            elif agg_method == 'prod' or agg_method == 'sum':
                self.out_dim = n_features_per_level * n_levels
            else:
                raise NotImplementedError
        else:
            out_dims = [s.out_dim for s in self.spatial_embedding] + [t.out_dim for t in self.temporal_embedding]
            if agg_method == 'cat':
                self.out_dim = sum(out_dims) // n_repeat
            elif agg_method == 'prod' or agg_method == 'sum':
                self.out_dim = out_dims[0]
            else:
                raise NotImplementedError

        if self.n_repeat != 1 and self.backbone == 'torch':
            raise NotImplementedError('We do not support repeatation of torch defined textures yet.')

        self._xy = jt.array([0, 1], dtype=jt.int32)  # to avoid synchronization
        self._xz = jt.array([0, 2], dtype=jt.int32)  # to avoid synchronization
        self._yz = jt.array([1, 2], dtype=jt.int32)  # to avoid synchronization
        #self.pre_handle = self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    #def _load_state_dict_pre_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Historical reasons
    #    keys = list(state_dict.keys())
    #    xy_key = f'{prefix}xy'
    #    if xy_key not in keys:
    #        state_dict[xy_key] = self.xy
    #    xz_key = f'{prefix}xz'
    #    if xz_key not in keys:
    #        state_dict[xz_key] = self.xz
    #    yz_key = f'{prefix}yz'
    #    if yz_key not in keys:
    #        state_dict[yz_key] = self.yz

    def execute(self, xyz: jt.Var, t: jt.Var, batch: dotdict = None):
        bash = xyz.shape  # batch shape
        xyz = xyz.view(-1, xyz.shape[-1])
        t = t.reshape(-1, t.shape[-1])
        xyz = (xyz - self._bounds[0]) / (self._bounds[1] - self._bounds[0])  # normalized, N, 3
        # get, xy, xz, yz, tx, ty, tz
        spatial_coords = jt.stack([xyz[..., self._xy.long()], xyz[..., self._xz.long()], xyz[..., self._yz.long()]], dim=0) * 2. - 1.  # [xy, xz, yz] from [0, 1] -> [-1, 1] -> 3, N, 2
        temporal_coords = jt.stack([jt.cat([t, xyz[..., :1]], dim=-1),  # tx
                                       jt.cat([t, xyz[..., 1:2]], dim=-1),  # ty
                                       jt.cat([t, xyz[..., 2:3]], dim=-1)], dim=0) * 2. - 1.  # tz -> [tx, ty, tz] from [0, 1] -> [-1, 1] -> 3, N, 2
        if self.backbone == 'torch':
            spatial_feats = []
            temporal_feats = []
            # NOTE: About the arrangement of temporal index
            # NOTE: The dataset provides a t that maps 0 -> 1 to 0 -> 99 (0 correspondes to the first frame, 99 to the last frame)
            # NOTE: Thus this mimicks a sampling of align corners == True
            for data in self.spatial_embedding:
                # 3, 1, N, 2 -> 3, C, 1, N -> 3, C, N -> 3, N, C
                feat = nn.grid_sample(data, spatial_coords[:, None], mode='bilinear', padding_mode='border', align_corners=False)[:, :, 0].permute((0, 2, 1))  # xy need to be reverted
                spatial_feats.append(feat)
            for data in self.temporal_embedding:
                # 3, 1, N, 2 -> 3, C, 1, N -> 3, C, N -> 3, N, C
                feat = nn.grid_sample(data, temporal_coords[:, None], mode='bilinear', padding_mode='border', align_corners=True)[:, :, 0].permute((0, 2, 1))  # xy need to be reverted
                temporal_feats.append(feat)
            spatial_feat = jt.cat(spatial_feats, dim=-1)  # 3, N, 4C
            temporal_feat = jt.cat(temporal_feats, dim=-1)  # 3, N, 4C
            feat = jt.cat([spatial_feat, temporal_feat], dim=0)  # 6, N, 4C
        else:
            spatial_feats = []
            temporal_feats = []
            for i in range(self.n_repeat):
                for j in range(3):
                    spatial_feats.append(self.spatial_embedding[i * 3 + j](spatial_coords[j]))
                    temporal_feats.append(self.temporal_embedding[i * 3 + j](temporal_coords[j]))
            spatial_feat = jt.stack(spatial_feats)  # 3, N, 4C
            temporal_feat = jt.stack(temporal_feats)  # 3, N, 4C
            feat = jt.cat([spatial_feat, temporal_feat], dim=0)  # 6, N, 4C

        if self.agg_method == 'cat':
            val = jt.cat([item for item in feat], dim=-1)  # N, 24C
        elif self.agg_method == 'prod':
            val = jt.prod(feat, dim=0)  # N, 4C
        elif self.agg_method == 'sum':
            val = jt.sum(feat, dim=0)  # N, 4C
        else:
            raise NotImplementedError
        return val.view(*bash[:-1], val.shape[-1])  # B, N, 24C (or 4c)
