# This embedder will provide special embedding for different view_sample and latent_sample
import numpy as np
from jittor import nn
import jittor as jt
from typing import Union
from jdhr.engine import EMBEDDERS
from jdhr.utils.console_utils import *
from jdhr.utils.base_utils import dotdict
from jdhr.utils.net_utils import make_params
from jdhr.utils.grid_utils import grid_sample_1d
from jdhr.models.networks.embedders.latent_code_embedder import LatentCodeEmbedder
from jdhr.models.cameras.optimizable_camera import OptimizableCamera


@EMBEDDERS.register_module()
class SpacetimeEmbedder(nn.Module):
    def __init__(self,
                 # We want to make this length agnostic
                 #  n_spaces: int = OptimizableCamera.n_views * 2,  # if you've got more than these number of frames, it will be interpolated
                 #  n_times: int = OptimizableCamera.n_frames * 2,  # if you've got more than these number of frames, it will be interpolated
                 #  out_dim: int = 8,  # shape of latent code
                 space_embedder_cfg: dotdict = dotdict(type=LatentCodeEmbedder.__name__, n_codes=OptimizableCamera.n_views * 2, out_dim=8),
                 time_embedder_cfg: dotdict = dotdict(type=LatentCodeEmbedder.__name__, n_codes=OptimizableCamera.n_frames * 2, out_dim=8),
                 dtype: Union[str, jt.dtype] = jt.float,
                 ):
        super().__init__()
        # self.out_dim = 2 * out_dim
        # self.space_embedding = LatentCodeEmbedder(n_spaces, out_dim, dtype)  # should sample with view index
        # self.time_embedding = LatentCodeEmbedder(n_times, out_dim, dtype)  # should sample with frame index
        self.space_embedding: LatentCodeEmbedder = EMBEDDERS.build(space_embedder_cfg, dtype=dtype)
        self.time_embedding: LatentCodeEmbedder = EMBEDDERS.build(time_embedder_cfg, dtype=dtype)
        self.out_dim = self.space_embedding.out_dim + self.time_embedding.out_dim
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Supports loading points and features with different shapes
        self.space_embedding.latent_codes.data = self.space_embedding.latent_codes.new_empty(state_dict[f'{prefix}space_embedding.latent_codes'].shape)
        self.time_embedding.latent_codes.data = self.time_embedding.latent_codes.new_empty(state_dict[f'{prefix}time_embedding.latent_codes'].shape)

    # All in a variable & batch format
    def execute(self, t: jt.Var, batch: dotdict):
        # B, P, 1
        v = batch.v[:, None, None].expand(t.shape).to(t.dtype)
        space_feat = self.space_embedding(v, batch)
        time_feat = self.time_embedding(v, batch)
        feat = jt.cat([space_feat, time_feat], dim=-1)
        return feat
