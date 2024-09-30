import numpy as np
import jittor as jt
from jittor import nn
from jdhr.models.cameras.optimizable_camera import OptimizableCamera
from jdhr.models.networks.embedders.hash_encoder.hash_encoder import TcnnHashEmbedderjt
from jdhr.engine import EMBEDDERS, cfg
from typing import List, Dict, Literal

@EMBEDDERS.register_module()
class Decomposition4D(nn.Module):
    def __init__(
        self,
        n_levels: int = 16,
        n_features_per_level: int = 8,
        log2_hashmap_size: int = 19,
        base_resolution: int = 64,
        finest_resolution: int = 2048,
        vectors_finest_resolution: int = 2048,
        bounds: List[List[int]] = OptimizableCamera.bounds,
    ):
        """HumanRF's feature grid representation that uses 4D decomposition via four 3D multi-scale hash grids,
        and four 1D dense grids.

        Args:
            ngp_n_levels (int, optional):
                The number of levels in the 3D multi-scale feature grids. Defaults to 16.
            ngp_n_features_per_level (int, optional):
                Can be 1,2,4 or 8. The final feature dimension will be equal to [ngp_n_features_per_level] * [ngp_n_levels].
                Defaults to 2.
            ngp_log2_hashmap_size (int, optional):
                Each 3D hash grid in the decomposition will have a hash map size of 2^[log2_hashmap_size].
                Defaults to 19.
            ngp_base_resolution (int, optional):
               Resolution of the coarsest level in the 3D multi-scale feature grids. Defaults to 32.
            ngp_finest_resolution (int, optional):
                Resolution of the finest level in the 3D multi-scale feature grids. Defaults to 2048.
            vectors_finest_resolution (int, optional):
                Resolution of the 1D dense grids. Defaults to 2048.
        """
        super().__init__()

        self.bounds = np.array(bounds, dtype="float32")
        per_level_scale = np.exp(np.log(finest_resolution / base_resolution) / (n_levels - 1))

        feature_size = n_levels * n_features_per_level
        self.feature_size=feature_size
        self.out_dim=feature_size
        #self.vectors = jt.array(jt.randn((4, vectors_finest_resolution, feature_size), dtype=jt.float) * 0.1)
        self.x_encoding=nn.Embedding(num_embeddings=vectors_finest_resolution, embedding_dim=feature_size)
        self.y_encoding=nn.Embedding(num_embeddings=vectors_finest_resolution, embedding_dim=feature_size)
        self.z_encoding=nn.Embedding(num_embeddings=vectors_finest_resolution, embedding_dim=feature_size)
        self.t_encoding=nn.Embedding(num_embeddings=vectors_finest_resolution, embedding_dim=feature_size)

        self.xyz_encoding=TcnnHashEmbedderjt(
            n_pos_dims=3,
            n_features_per_level=n_features_per_level,
            n_levels=n_levels,
            base_resolution=base_resolution,
            log2_hashmap_size=log2_hashmap_size,
            per_level_scale=per_level_scale,
        )
        self.xyt_encoding=TcnnHashEmbedderjt(
            n_pos_dims=3,
            n_features_per_level=n_features_per_level,
            n_levels=n_levels,
            base_resolution=base_resolution,
            log2_hashmap_size=log2_hashmap_size,
            per_level_scale=per_level_scale,
        )
        self.yzt_encoding=TcnnHashEmbedderjt(
            n_pos_dims=3,
            n_features_per_level=n_features_per_level,
            n_levels=n_levels,
            base_resolution=base_resolution,
            log2_hashmap_size=log2_hashmap_size,
            per_level_scale=per_level_scale,
        )
        self.xzt_encoding=TcnnHashEmbedderjt(
            n_pos_dims=3,
            n_features_per_level=n_features_per_level,
            n_levels=n_levels,
            base_resolution=base_resolution,
            log2_hashmap_size=log2_hashmap_size,
            per_level_scale=per_level_scale,
        )

    def execute(self, xyz, times):
        B,P,_=xyz.shape
        xyz=xyz.view(-1,3)
        times=times.view(-1,1)
        xyz = (xyz - self.bounds[0]) / (self.bounds[1] - self.bounds[0]) #normalize
        xyzt = jt.cat((xyz, times), dim=-1) #(num_points, 4)
        xyz_features = self.xyz_encoding(xyz) #(num_points, feature_dim)
        xyt_features = self.xyt_encoding(xyzt[..., [0, 1, 3]]) #(num_points, feature_dim)
        yzt_features = self.yzt_encoding(xyzt[..., [1, 2, 3]]) #(num_points, feature_dim)
        xzt_features = self.xzt_encoding(xyzt[..., [0, 2, 3]]) #(num_points, feature_dim)

        x_features = self.x_encoding(xyzt[..., 0])
        y_features = self.y_encoding(xyzt[..., 1])
        z_features = self.z_encoding(xyzt[..., 2])
        t_features = self.t_encoding(xyzt[..., 3])

        result=xyz_features*t_features+xyt_features*z_features+yzt_features*x_features+xzt_features*y_features

        result=result.view(B,P,-1)
        return result
