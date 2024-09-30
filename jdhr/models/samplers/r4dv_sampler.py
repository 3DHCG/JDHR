# Point planes IBR sampler
# Use the point as geometry
# Use K-planes as feature bases
# Use IBR for rendering the final rgb color -> hoping for a sharper result

import jittor as jt
from jittor import nn
from types import MethodType

from jdhr.utils.console_utils import *
from jdhr.utils.sh_utils import eval_sh
from jdhr.utils.timer_utils import timer
from jdhr.utils.math_utils import normalize
from jdhr.utils.net_utils import make_params, make_buffer

from jdhr.engine import cfg
from jdhr.engine.registry import call_from_cfg
from jdhr.engine import SAMPLERS, EMBEDDERS, REGRESSORS

from jdhr.models.samplers.gaussiant_sampler import GaussianTSampler
from jdhr.models.networks.regressors.mlp_regressor import MlpRegressor
from jdhr.models.samplers.point_planes_sampler import PointPlanesSampler
from jdhr.models.networks.embedders.kplanes_embedder import KPlanesEmbedder
from jdhr.models.networks.volumetric_video_network import VolumetricVideoNetwork
from jdhr.models.networks.regressors.spherical_harmonics import SphericalHarmonics
from jdhr.models.networks.embedders.positional_encoding_embedder import PositionalEncodingEmbedder
from jdhr.models.networks.embedders.geometry_image_based_embedder import GeometryImageBasedEmbedder
from jdhr.models.networks.regressors.image_based_spherical_harmonics import ImageBasedSphericalHarmonics


@SAMPLERS.register_module()
class R4DVSampler(PointPlanesSampler):
    def __init__(self,
                 network: VolumetricVideoNetwork,  # always as the first argument of sampler
                 use_diffgl: bool = True,

                 ibr_embedder_cfg: dotdict = dotdict(type=GeometryImageBasedEmbedder.__name__),  # easily returns nan
                 ibr_regressor_cfg: dotdict = dotdict(type=ImageBasedSphericalHarmonics.__name__),  # easily returns nan

                 opt_cnn_warmup: int = 1000,  # optimize for 1000 iterations
                 opt_cnn_every: int = 100,  # optimize every 100 iterations after
                 render_gs: bool = False,

                 **kwargs,
                 ):
        kwargs = dotdict(kwargs)
        self.kwargs = kwargs

        call_from_cfg(super().__init__, kwargs, network=network, use_diffgl=use_diffgl)  # later arguments will overwrite former ones
        del self.dir_embedder  # no need for this
        del self.rgb_regressor
        self.ibr_embedder: GeometryImageBasedEmbedder = EMBEDDERS.build(ibr_embedder_cfg)  # forwarding the images
        self.ibr_regressor: ImageBasedSphericalHarmonics = REGRESSORS.build(ibr_regressor_cfg, in_dim=self.xyz_embedder.out_dim + 3, src_dim=self.ibr_embedder.src_dim)
        #self.pre_handle = self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)
        #self.type(self.dtype)

        self.opt_cnn_warmup = opt_cnn_warmup
        self.opt_cnn_every = opt_cnn_every

        self.render_radius = MethodType(GaussianTSampler.render_radius, self)  # override the method
        self.sh_deg = 0  # only input colors
        self.scale_mod = 1.0
        self.render_gs = render_gs

    def render_points(self, xyz: jt.Var, rgb: jt.Var, rad: jt.Var, occ: jt.Var, batch: dotdict):
        if self.render_gs:
            #print("render_gsrender_gsrender_gs")
            sh0 = (rgb[..., None] - 0.5) / 0.28209479177387814
            rgb, acc, dpt = self.render_radius(xyz, sh0, rad, occ, batch)  # B, HW, C
        else:
            rgb, acc, dpt = super().render_points(xyz, rgb, rad, occ, batch)  # almost always use render_cudagl
        return rgb, acc, dpt

    #def type(self, dtype: jt.dtype):
        #super().type(dtype)
    #    if hasattr(self, 'pcd_embedder'):
            #if self.pcd_embedder.spatial_embedding[0].tcnn_encoding.dtype != dtype:
            #    prev_pcd_embedder = self.pcd_embedder
            #    self.pcd_embedder: KPlanesEmbedder = EMBEDDERS.build(self.kwargs.pcd_embedder_cfg, dtype=dtype)  # unchanged and loaded as is
            #    self.pcd_embedder.load_state_dict(prev_pcd_embedder.state_dict())
            #    self.pcd_embedder
            #else:
    #        self.pcd_embedder._xy.data = self.pcd_embedder._xy.to(jt.int32)
    #        self.pcd_embedder._xz.data = self.pcd_embedder._xz.to(jt.int32)
    #        self.pcd_embedder._yz.data = self.pcd_embedder._yz.to(jt.int32)

    #    if hasattr(self, 'xyz_embedder'):
            #if self.xyz_embedder.spatial_embedding[0].tcnn_encoding.dtype != dtype:
            #    prev_xyz_embedder = self.xyz_embedder
            #    self.xyz_embedder: KPlanesEmbedder = EMBEDDERS.build(self.kwargs.xyz_embedder_cfg, dtype=dtype)  # unchanged and loaded as is
            #   self.xyz_embedder.load_state_dict(prev_xyz_embedder.state_dict())
            #    self.xyz_embedder
            #else:
    #        self.xyz_embedder._xy.data = self.xyz_embedder._xy.to(jt.int32)
    #        self.xyz_embedder._xz.data = self.xyz_embedder._xz.to(jt.int32)
    #        self.xyz_embedder._yz.data = self.xyz_embedder._yz.to(jt.int32)

    #def _load_state_dict_pre_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    #    super()._load_state_dict_pre_hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        # Historical reasons
    #    keys = list(state_dict.keys())
    #    for key in keys:
    #        if f'{prefix}ibr_regressor.feat_agg' in key:
    #            del state_dict[key]

    #    keys = list(state_dict.keys())
    #    for key in keys:
    #        if f'{prefix}ibr_regressor.rgb_mlp.linears' in key:
     #           state_dict[key.replace(f'{prefix}ibr_regressor.rgb_mlp.linears', f'{prefix}ibr_regressor.rgb_mlp.mlp.linears')] = state_dict[key]
    #            del state_dict[key]

    #    keys = list(state_dict.keys())
    #    for key in keys:
    #        if f'{prefix}ibr_regressor.sh_mlp.linears' in key:
    #            state_dict[key.replace(f'{prefix}ibr_regressor.sh_mlp.linears', f'{prefix}ibr_regressor.sh_mlp.mlp.linears')] = state_dict[key]
    #            del state_dict[key]

    def execute(self, batch: dotdict, return_frags: bool = False):
        timer.record('post processing')

        self.init_points(batch)
        self.update_points(batch)
        pcd, pcd_t = self.sample_pcd_pcd_t(batch)  # B, P, 3, B, P, 1

        pcd_feat = self.pcd_embedder(pcd, pcd_t)  # B, N, C

        resd = self.resd_regressor(pcd_feat)  # B, N, 3

        xyz = pcd + resd  # B, N, 3

        xyz_feat = self.xyz_embedder(xyz, pcd_t)  # same time

        rad, occ = self.geo_regressor(xyz_feat)  # B, N, 1

        timer.record('geometry')

        # These could be cached on points
        optimize_cnn =  not (batch.meta.iter % self.opt_cnn_every) or (batch.meta.iter <= self.opt_cnn_warmup)
        src_feat = self.ibr_embedder(xyz, batch, optimize_cnn=optimize_cnn)  # MARK: implicit update of batch.output

        dir = normalize(xyz.detach() - (-batch.R.transpose(-2,-1) @ batch.T).transpose(-2,-1)).repeat(xyz_feat.shape[0],1,1) # B, N, 3

        rgb = self.ibr_regressor(jt.cat([xyz_feat, dir], dim=-1), batch)  # B,  N, 3
        timer.record('appearance')

        if return_frags:
            return pcd, xyz, rgb, rad, occ

        # Perform points rendering
        rgb, acc, dpt = self.render_points(xyz, rgb, rad, occ, batch)  # B, HW, C

        self.store_output(pcd, xyz, rgb, acc, dpt, batch)
        timer.record('rendering')
