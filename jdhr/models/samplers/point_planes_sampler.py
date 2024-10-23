# Similar to point radiance, but uses a tensor4d for features and location updates
# How do we pass features to locations?
# There's a rendered location (different from the explicit pcd)
# Every pcd has a updated residual that could be optimized and shared
# This point decoder should fill in the gaps
# There are two kplanes? or just a shared one that resamples
# During rendering, we quantize the planes and store their features to a vq? (as a post processing? maybe not neccessary)
# TODO: Extract this implementation as itself
# Now it's a very dirty sampler inheritance

import jittor as jt
import numpy as np
from jittor import nn

from os.path import join
from functools import lru_cache, partial

from jdhr.utils.console_utils import *
from jdhr.utils.base_utils import dotdict
from jdhr.utils.data_utils import export_pts, to_x, load_pts
from jdhr.utils.chunk_utils import multi_gather, multi_scatter
from jdhr.utils.net_utils import make_params, make_buffer, VolumetricVideoModule
from jdhr.utils.math_utils import normalize, affine_inverse, affine_padding, point_padding,affine_padding_np,point_padding_np,affine_inverse_np
from jdhr.utils.fcds_utils import voxel_down_sample, farthest_down_sample, remove_outlier, get_pytorch3d_camera_params, surface_points, sample_filter_random_points, get_pulsar_camera_params, voxel_surface_down_sample, duplicate, farthest, random, filter_bounds, SamplingType

from jdhr.engine import cfg, args
from jdhr.engine import EMBEDDERS, REGRESSORS, SAMPLERS
from jdhr.models.cameras.optimizable_camera import OptimizableCamera
from jdhr.models.networks.volumetric_video_network import VolumetricVideoNetwork

from jdhr.models.networks.embedders.noop_embedder import NoopEmbedder
from jdhr.models.networks.regressors.mlp_regressor import MlpRegressor
from jdhr.models.networks.embedders.kplanes_embedder import KPlanesEmbedder
from jdhr.models.networks.embedders.decomposition4d import Decomposition4D
from jdhr.models.networks.regressors.spherical_harmonics import SphericalHarmonics
from jdhr.models.networks.regressors.displacement_regressor import DisplacementRegressor
from jdhr.models.networks.embedders.positional_encoding_embedder import PositionalEncodingEmbedder
from jdhr.models.networks.embedders.freq_encoder import FrequencyEncoder

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from jdhr.models.networks.volumetric_video_network import VolumetricVideoNetwork
    from jdhr.runners.volumetric_video_viewer import VolumetricVideoViewer
    from jdhr.models.networks.multilevel_network import MultilevelNetwork


@SAMPLERS.register_module()
class PointPlanesSampler(VolumetricVideoModule):

    n_frames = OptimizableCamera.n_views if OptimizableCamera.closest_using_t else OptimizableCamera.n_frames
    frame_sample = OptimizableCamera.view_sample if OptimizableCamera.closest_using_t else OptimizableCamera.frame_sample

    def __init__(self,
                 network: VolumetricVideoNetwork,
                 # PCD related configs
                 n_frames: int = n_frames,
                 frame_sample: List[int] = frame_sample,
                 n_points: int = 262144,  # initial number of points (65536 points per frame) # this is an importance hyperparameter

                 # PCD rendering & aggregationg related configs
                 opt_pcd: bool = True,
                 bin_size: int = None,
                 pts_per_pix: int = 15,
                 max_pts_per_bin: int = None,
                 volume_rendering: bool = True,
                 bg_brightness: float = 0.0,

                 radius_min: float = 0.001,  # maybe, smaller radius works better when we have a large number of points
                 radius_max: float = 0.015,  # instability during training, it's much easiert o expand the radius
                 radius_shift: float = -5.0,  # applied before sigmoid
                 alpha_shift: float = 5.0,
                 geo_init: float = 0.0,

                 pcd_embedder_cfg: dotdict = dotdict(type=KPlanesEmbedder.__name__, agg_method='cat', n_levels=2),  # smaller deformation holds a large number of features
                 resd_regressor_cfg: dotdict = dotdict(type=DisplacementRegressor.__name__, width=64, depth=2, scale=0.1, out_dim=3),  # 3d residual, 10cm range
                 geo_regressor_cfg: dotdict = dotdict(type=MlpRegressor.__name__, width=64, depth=2, out_dim=2),

                 xyz_embedder_cfg: dotdict = dotdict(type=KPlanesEmbedder.__name__, agg_method='cat', n_levels=4),  # holds a large number of features
                 dir_embedder_cfg: dotdict = dotdict(type=NoopEmbedder, in_dim=3),  # HACK: hack to make network size match
                 rgb_regressor_cfg: dotdict = dotdict(type=SphericalHarmonics, sh_deg=3, width=64, depth=2),

                 use_pulsar: bool = False,
                 use_diffgl: bool = False,
                 use_cudagl: bool = False,
                 gamma: float = 1e-3,  # i don't understand...

                 # Visual hull initilization related
                 points_dir: str = 'surfs',
                 points_aligned: bool = False,  # are files in points_dir aligned?
                 points_expanded: bool = True,  # are files in points_dir expanded?
                 points_only: bool = False,  # only expand surfs and exit
                 reload_points: bool = False,  # force reloading of surface
                 skip_loading_points: bool = args.type != 'train',  # only when you're loading from a checkpoint # MARK: GLOBAL

                 should_preprocess: bool = False,  # use a complex random process to prepare the point cloud for later
                 n_duplicate: int = 2,  # duplicate 2 times before finding best surfaces

                 sampling_type: SamplingType = SamplingType.MARCHING_CUBES_RECONSTRUCTION.name,
                 voxel_size: float = 0.015,  # small, is it possible to automatically determine voxel size?
                 surface_radius: float = 0.065,
                 surface_K: int = 100,  # need to avoid oom here

                 # Initialization to avoid speed bumps
                 init_H: int = 0,  # if not zero, will call prepare_opengl on this
                 init_W: int = 0,  # if not zero, will call prepare_opengl on this
                 dtype: jt.dtype = jt.float,

                 **kwargs,
                 ):
        super().__init__(network)

        # Point cloud configs
        self.opt_pcd = opt_pcd
        self.n_points = n_points
        self.n_frames = n_frames
        self.frame_sample = frame_sample

        # Initialization related
        self.surface_K = surface_K
        self.voxel_size = voxel_size  # controls the initial updates
        self.n_duplicate = n_duplicate
        self.sampling_type = SamplingType[sampling_type]
        self.surface_radius = surface_radius
        self.should_preprocess = should_preprocess

        # Point cloud rendering configs
        self.bin_size = bin_size
        self.pts_per_pix = pts_per_pix
        self.max_pts_per_bin = max_pts_per_bin
        self.volume_rendering = volume_rendering
        self.bg_brightness = bg_brightness

        # Renderer related
        self.use_pulsar = use_pulsar
        self.use_diffgl = use_diffgl
        self.use_cudagl = use_cudagl

        # Prepare the pulsar renderer
        # Since pulsar requirest the size of the image to be known, we need to find a way to change this
        # TODO: Move pulsar preparation to its own implementation without disturbance to the whole class
        self.gamma = gamma
        self.max_H, self.max_W = 0, 0

        # Load initialized or uninitialized vhulls
        self.points_dir = points_dir
        self.points_only = points_only
        self.points_aligned = True#False  # align only once, but not when loading checkpoints
        self.points_expanded = True#False  # initialize only once, but not when loading checkpoints or loading from expanded dir
        skip_loading_points = skip_loading_points and not points_only  # will always load init vhulls if points_only

        make_params_or_buffer = make_params if opt_pcd else make_buffer

        # Some initialization possibility
        self.pcds = jt.array(jt.zeros([n_frames,n_points,3]))
        self.pcds.requires_grad=True
        self.rgbs: List[jt.Var] = []
        self.nors: List[jt.Var] = []
        self.rads: List[jt.Var] = []
        self.occs: List[jt.Var] = []

        if skip_loading_points:
            pass
        else:
            data_root = OptimizableCamera.data_root
            points_dir = OptimizableCamera.vhulls_dir
            if exists(join(data_root, self.points_dir)) and \
                    len(os.listdir(join(data_root, self.points_dir))) >= n_frames and \
                    not reload_points:
                points_dir = self.points_dir
                self.points_aligned = points_aligned
                self.points_expanded = points_expanded

            # Sometimes, e does not contain the full count
            b, e, s = PointPlanesSampler.frame_sample
            points_path = join(data_root, points_dir)

            # Sometimes, the points folder does not contain every frames, how do we know that it's here?
            # Most methods use file sorting
            # points_files = np.asarray(sorted(os.listdir(points_path)))[b:e:s]
            points_files = os.listdir(points_path)
            points_int = [int(f.split('.')[0]) for f in points_files]

            for f in tqdm(range(n_frames), desc=f'Loading init pcds from {blue(points_path)}'):
                idx = b + f * s
                idx = points_int.index(idx)
                point_file = points_files[idx]

                # Some initialization possibility
                pcd, rgb, norm, scalars = load_pts(join(points_path, point_file))
                self.pcds.append(jt.array(pcd))
                # Sometimes these special variables do not exist
                if rgb is not None: self.rgbs.append(jt.array(rgb))
                if norm is not None: self.nors.append(jt.array(norm))
                if 'radius' in scalars: self.rads.append(jt.array(scalars.radius))
                if 'alpha' in scalars: self.occs.append(jt.array(scalars.alpha))


        #self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)
        #self._register_state_dict_hook(self._state_dict_hook)

        # Could be removed after optimization
        self.pcd_embedder: KPlanesEmbedder = EMBEDDERS.build(pcd_embedder_cfg)
        #self.pcdt_embedder=FrequencyEncoder()###add
        self.resd_regressor: DisplacementRegressor = REGRESSORS.build(resd_regressor_cfg, in_dim=self.pcd_embedder.out_dim)

        # Underlying NeRF
        self.xyz_embedder: KPlanesEmbedder = EMBEDDERS.build(xyz_embedder_cfg)
        #self.xyzt_embedder=FrequencyEncoder()###add
        self.dir_embedder: PositionalEncodingEmbedder = EMBEDDERS.build(dir_embedder_cfg)

        self.radius_min = radius_min
        self.radius_max = radius_max
        self.radius_shift = radius_shift
        self.alpha_shift = alpha_shift
        self.geo_init = geo_init

        def geo_actvn(x: jt.Var,
                      radius_min: float = self.radius_min,
                      radius_max: float = self.radius_max,
                      radius_shift: float = self.radius_shift,
                      alpha_shift: float = self.alpha_shift):
            r, a = x.split([1, 1], dim=-1)
            r = (r + radius_shift).sigmoid() * (radius_max - radius_min) + radius_min
            a = (a + alpha_shift).sigmoid()
            return r, a
        self.geo_actvn = geo_actvn

        self.geo_regressor: MlpRegressor = REGRESSORS.build(geo_regressor_cfg, in_dim=self.xyz_embedder.out_dim, out_actvn=geo_actvn)
        self.rgb_regressor: MlpRegressor = REGRESSORS.build(rgb_regressor_cfg, in_dim=self.xyz_embedder.out_dim + self.dir_embedder.out_dim)  # MARK: Lighting fast!
        jt.init.gauss_(self.geo_regressor.mlp.linears[-1].bias,self.geo_init, std=1e-4)  # small displacement by default

        self.dtype = getattr(jt, dtype) if isinstance(dtype, str) else dtype
        #self.type(self.dtype)

        self.skip_loading_points = skip_loading_points
        self.init_points()

    @staticmethod  # i'm literally...
    def _state_dict_hook(self, state_dict, prefix, local_metadata):
        if 'sampler.pulsar.device_tracker' in state_dict:
            del state_dict['sampler.pulsar.device_tracker']

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Supports loading points and features with different shapes
        if hasattr(self, 'pcds'):
            pcd_keys = []
            for f, pcd in enumerate(self.pcds):
                pcd.data = pcd.data.new_empty(state_dict[f'{prefix}pcds.{f}'].shape)
                pcd_keys.append(f'{prefix}pcds.{f}')
            self.points_expanded = True  # no expansion when loading checkpoints
            self.points_aligned = True  # need aligned when new check points are loaded

            keys = list(state_dict.keys())
            for key in keys:
                if key.startswith(f'{prefix}pcds.') and key not in pcd_keys:
                    del state_dict[key]

        # Supports lazy initialization of pulsar renderer
        if f'{prefix}pulsar.device_tracker' in state_dict:
            del state_dict[f'{prefix}pulsar.device_tracker']

    def render_imgui(self, viewer: 'VolumetricVideoViewer', batch: dotdict):
        if not self.use_cudagl and not self.use_diffgl:
            from imgui_bundle import ImVec2
            from jdhr.utils.viewer_utils import add_debug_text_2d
            slow_pytorch3d_render_msg = 'Using slow PyTorch3D rendering backend. Please see the errors in the terminal.'
            color = 0xff5533ff
            viewer.add_debug_text_2d(slow_pytorch3d_render_msg, color)

    @jt.no_grad()
    def init_points(self, batch: dotdict = None):
        if self.skip_loading_points:
            return  # do not call this
        # Hacky initialization logic
        if not self.points_expanded:
            self.expand_points()  # make them expand to the desired size (with proper sampling)
            self.points_expanded = True

        if self.points_only:
            exit(0)

        # Maybe align with camera transformations
        if not self.points_aligned:
            if 'runner' in cfg and hasattr(cfg.runner.val_dataset, 'c2w_avg'):
                self.align_points()  # make them align with input camera parameters
                self.points_aligned = True

    def align_points_np(self,pcd):
        dataset = cfg.runner.val_dataset  # MARK: GLOBAL
        c2w_avg = np.array(dataset.c2w_avg)
        pcd_new = point_padding_np(pcd) @ affine_inverse_np(affine_padding_np(c2w_avg)).swapaxes(-2,-1)  # this might be affected by amp
        pcd_new = pcd_new.astype(pcd.dtype)
        pcd_new = pcd_new[..., :3] / pcd_new[..., 3:]
        return pcd_new

    @jt.no_grad()
    def align_points(self, batch: dotdict = None):
        dataset = cfg.runner.val_dataset  # MARK: GLOBAL
        _c2w_avg = jt.array(dataset.c2w_avg)

        def align_points(pcd: jt.Var, pcd_t: jt.Var = None):  # B, N, 3
            pcd_new = point_padding(pcd) @ affine_inverse(affine_padding(_c2w_avg)).transpose(-2,-1)  # this might be affected by amp
            pcd_new = pcd_new.type_as(pcd)
            pcd_new = pcd_new[..., :3] / pcd_new[..., 3:]
            return pcd_new
        self.apply_to_pcds(align_points, quite=False)

    @jt.no_grad()
    def expand_points(self, batch: dotdict = None):
        # Save processing results to disk
        data_root = OptimizableCamera.data_root
        # bounds = jt.Var(OptimizableCamera.bounds, device='cuda')  # use CPU tensor for now?
        # bounds = (point_padding(bounds) @ affine_padding(dataset.c2w_avg).mT)[..., :3]  # homo
        names = sorted(os.listdir(join(data_root, OptimizableCamera.vhulls_dir)))
        vhulls_dir = self.points_dir
        for i, pcd in enumerate(tqdm(self.pcds, desc='Expanding pcds')):
            #self.pcds[i] = self.pcds[i].to(cuda)  # move this point cloud to cuda

            if self.should_preprocess:
                # Full treatment for adding more points
                # self.apply_to_pcds(partial(filter_bounds, bounds=bounds), range=[i, i + 1])  # duplication (n_points * 2)
                self.apply_to_pcds(partial(sample_filter_random_points, K=self.n_points * 10, update_radius=0.05, filter_K=10), range=[i, i + 1])  # growing (duplication)
                self.apply_to_pcds(partial(voxel_down_sample, voxel_size=self.voxel_size), range=[i, i + 1])  # expand to the desired size
                self.apply_to_pcds(partial(remove_outlier, K=20), range=[i, i + 1])  # expand to the desired size

                while len(self.pcds[i]) < self.n_points:
                    self.apply_to_pcds(duplicate, range=[i, i + 1])  # duplication (n_points * 2)
                self.apply_to_pcds(partial(random, n_points=self.n_points), range=[i, i + 1])  # n_points number of points
                while len(self.pcds[i]) < self.n_points:
                    self.apply_to_pcds(duplicate, range=[i, i + 1])  # duplication (n_points * 2)
                for _ in range(self.n_duplicate):
                    self.apply_to_pcds(duplicate, range=[i, i + 1])  # duplication (n_points * 2)

            if self.sampling_type == SamplingType.SURFACE_DISTRIBUTION:
                self.apply_to_pcds(partial(surface_points, radius=self.surface_radius, K=self.surface_K, n_points=int(self.n_points * 3)), range=[i, i + 1])  # filter the points to the surface
                self.apply_to_pcds(partial(random, n_points=self.n_points))  # n_points number of points
            elif self.sampling_type == SamplingType.FARTHEST_DOWN_SAMPLE:
                # TODO: Retest this
                self.apply_to_pcds(partial(farthest, length=None, n_points=self.n_points), range=[i, i + 1])  # n_points number of points
            elif self.sampling_type == SamplingType.POISSON_RECONSTRUCTION:
                pass
            elif self.sampling_type == SamplingType.RANDOM_DOWN_SAMPLE:
                self.apply_to_pcds(partial(random, n_points=self.n_points))  # n_points number of points
            elif self.sampling_type == SamplingType.MARCHING_CUBES_RECONSTRUCTION:
                # breakpoint()
                # self.apply_to_pcds(partial(filter_bounds, bounds=bounds), range=[i, i + 1])  # duplication (n_points * 2)
                assert len(self.pcds[i])
                while len(self.pcds[i]) < self.n_points:
                    self.apply_to_pcds(duplicate, range=[i, i + 1])  # duplication (n_points * 2)
                self.apply_to_pcds(partial(voxel_surface_down_sample, voxel_size=self.voxel_size, dist_th=self.surface_radius, n_points=self.n_points), range=[i, i + 1])
            else:
                raise NotImplementedError

            # Everything stored on disk should be in world coordinates
            export_pts(self.pcds[i], filename=join(data_root,
                                                   vhulls_dir,
                                                   names[i],
                                                   ))

        jt.gc()  # For unknown reasons, the memory won't get released after extracting visual hull? # FIXME: Requires further debugging on this

    @jt.no_grad()
    def apply_to_pcds(self,
                      function: Callable,
                      with_t: bool = True,
                      with_inds: bool = False,
                      with_batch: bool = False,  # whether to invoke the `function` in a heterogeneous batch
                      quite: bool = True,
                      range: List[int] = [0, None],
                      ):
        # Maybe apply action on batched pcds
        if with_batch:
            from structures import Pointclouds
            pcds: Pointclouds = Pointclouds([p.data for p in self.pcds])  # batched point clouds
            pcd_news = time_function()(function)(pcds.points_padded(), pcds.num_points_per_cloud())

        b, e, s = self.frame_sample
        times = jt.arange(b, e, s)
        times = (times - b) / np.clip(e - 1 - b, 1, None)  # MARK: same as frame_to_t in VolumetricVideoDatase

        name = function.func.__name__ if isinstance(function, partial) else function.__name__
        indices = jt.arange(0, len(self.pcds))
        times = times

        pcds = self.pcds

        pcds_new: nn.ParameterList[jt.Var] = nn.ParameterList()
        for i, (t, pcd) in enumerate(tqdm(zip(times, pcds), total=len(pcds), desc=name, disable=quite)):
            i = indices[i]  # control range

            pcd_old = pcd[None]  # will this still contain the gradients?

            pcd_t = t[None, None].expand(*pcd_old.shape[:-1], 1)  # B, N, 1

            # Apply action on pcds
            if with_batch: pcd_new = pcd_news[i][None]
            else:
                kwargs = dotdict()
                kwargs.pcd = pcd_old
                if with_t: kwargs.pcd_t = pcd_t
                if with_inds: kwargs.i = i
                pcd_new = function(**kwargs)  # larger number of reg samples

            # Restore parameters

            pcds_new.append(pcd_new[0].detach())


        self.pcds=pcds_new


    def render_points(self, *args, **kwargs):

        if self.use_cudagl: return self.render_cudagl(*args, **kwargs)
        elif self.use_diffgl: return self.render_diffgl(*args, **kwargs)

    def prepare_opengl(self,
                       module_attribute: str = 'cudagl',
                       renderer_class=None,
                       dtype: jt.dtype = jt.half,
                       tex_dtype: jt.dtype = jt.half,
                       H: int = 1024,
                       W: int = 1024,
                       size: int = 262144,
                       ):
        # Lazy initialization of EGL context

        H=int(H)
        W=int(W)
        if 'eglctx' not in cfg and 'window' not in cfg:
            log(f'Init eglctx with h, w: {H}, {W}')
            from jdhr.utils.egl_utils import eglContextManager
            from jdhr.utils.gl_utils import common_opengl_options
            cfg.eglctx = eglContextManager(W, H)  # !: BATCH
            common_opengl_options()

        # Lazy initialization of cuda renderer and opengl buffers, this is only a placeholder
        if not hasattr(self, module_attribute):
            log(f'Init {module_attribute} with size: {size}')
            rand1 = jt.rand(size, 1, dtype=dtype)
            rand3 = jt.rand(size, 3, dtype=dtype)
            opengl = renderer_class(verts=rand3,  # !: BATCH
                                    colors=rand3,
                                    scalars=dotdict(radius=rand1, alpha=rand1),
                                    pts_per_pix=self.pts_per_pix,
                                    render_type=renderer_class.RenderType.POINTS,
                                    dtype=dtype,
                                    tex_dtype=tex_dtype,
                                    H=H,
                                    W=W)  # this will preallocate sizes
            setattr(self, module_attribute, opengl)


    def render_diffgl(self,
                      xyz: jt.Var, feat: jt.Var, rad: jt.Var, occ: jt.Var,
                      batch: dotdict,
                      return_frags: bool = False,
                      return_full: bool = False,
                      ):
        from jdhr.utils.gl_utils import HardwarePeeling
        self.prepare_opengl('diffgl', HardwarePeeling, self.dtype, self.dtype, batch.meta.H[0], batch.meta.W[0], xyz.shape[1])
        self.diffgl: HardwarePeeling
        self.diffgl.pts_per_pix = self.pts_per_pix
        return self.diffgl.execute(xyz, feat, rad, occ, batch, return_frags, return_full)

    @jt.no_grad()
    def update_points(self, batch: dotdict):
        """
        This function implements the Gaussian densification techniques mentioned in the 3D Gaussian Splatting
        It is called before a forward pass where the gradient of the previous update is still visible and valid
        We only implement the splitting and cloning techniques mentioned in their paper, along with the pruning based on alpha
        We do not implement the view space and world space removal of large points, along with the manual update of alpha values
        They perform this update every 100 iterations, they compute the magnitude of the gradient on a particular point's position and deem that point needing update if it exceeds 2e-4.
        For large points with magnitude of scale larger than [NOT STATED IN PAPER], they split the guassian in half and fill it with the corresponding scale and position [AGAIN NOT EXPLICITLY STATED IN THE PAPER]
        For small points with magnitude of scale smaller than [NOT STATED IN PAPER], they clone it and move the point in the direction of the positional gradient by [AGAIN NOT EXPLICITLY STATED IN THE PAPER]
        """

    def sample_index_time(self, batch):
        if OptimizableCamera.closest_using_t:
            i = jt.zeros_like(batch.meta.latent_index)
            t = jt.zeros_like(batch.t)
            return i, t

        i = batch.meta.latent_index  # always 0 for static scenes
        g = batch.latent_index  # always 0 for static scenes
        t = batch.t

        if not self.training:  # Will this be enough? How do we differentiate between val and train dataset input?
            tb, te, ts = self.frame_sample  # dataloader, processed to be OK
            b, e, s = cfg.val_dataset_cfg.frame_sample
            t = (((b + g * s) - tb) / np.clip(te - 1 - tb, 1, None))  # directly on gpu
            i = ((b + i * s) // ts - tb)#.int()  # if everything works out, this should be of type int
        return i, t

    def sample_pcd_pcd_t(self, batch):
        index, time = self.sample_index_time(batch)
        pcd = jt.stack([self.pcds[l] for l in index])  # B, N, 3 # avoid explicit syncing
        pcd_t = time[..., None, None].expand(-1, *pcd.shape[1:-1], 1)  # B, N, 1
        return pcd, pcd_t

    def store_output(self, pcd: jt.Var, xyz: jt.Var, rgb: jt.Var, acc: jt.Var, dpt: jt.Var, batch: dotdict):
        if pcd is not None: batch.output.resd = xyz - pcd  # for residual loss here
        batch.output.rgb_map = rgb.view(rgb.shape[0], -1, 3)  # B, H * W, 3
        batch.output.acc_map = acc.view(acc.shape[0], -1, 1)  # B, H * W, 1
        batch.output.dpt_map = dpt.view(dpt.shape[0], -1, 1)  # B, H * W, 1

        # Maybe use random background color
        if self.bg_brightness >= 0:
            batch.output.bg_color = jt.full_like(batch.output.rgb_map, self.bg_brightness)  # only for training and comparing with gt
        else:
            batch.output.bg_color = jt.rand_like(batch.output.rgb_map)

        # Add a background color
        if self.bg_brightness != 0:
            batch.output.rgb_map = batch.output.rgb_map + batch.output.bg_color * (1 - batch.output.acc_map)

    def execute(self, batch: dotdict):
        self.init_points(batch)
        self.update_points(batch)  # will never work because of alpha
        pcd, pcd_t = self.sample_pcd_pcd_t(batch)  # B, P, 3, B, P, 1

        # These could be discarded
        pcd_feat = self.pcd_embedder(pcd, pcd_t)  # B, N, C
        resd = self.resd_regressor(pcd_feat)  # B, N, 3
        xyz = pcd + resd  # B, N, 3

        # These could be cached -> or could it be expanded?
        xyz_feat = self.xyz_embedder(xyz, pcd_t)  # same time

        # These could be stored
        rad, occ = self.geo_regressor(xyz_feat)  # B, N, 2

        # These are unavoidable
        dir = normalize(xyz.detach() - (-batch.R.tranpose(-2,-1) @ batch.T).tranpose(-2,-1))  # B, N, 3
        dir_feat = self.dir_embedder(dir)  # B, N, ...
        rgb = self.rgb_regressor(jt.cat([xyz_feat, dir_feat], dim=-1))  # B, N, 3

        # Perform points rendering
        rgb, acc, depth = self.render_points(xyz, rgb, rad, occ, batch)  # B, HW, C
        self.store_output(pcd, xyz, rgb, acc, depth, batch)
