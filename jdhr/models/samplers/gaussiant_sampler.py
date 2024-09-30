"""
This is the cleaned up version of the original GaussianModel and GaussianSampler
Including it's core functionalities:
- Differentiable gaussian splatting
- Explicitly defined parameters?
- Image-based rendering?
- Cloning? Pruning? Splitting?
"""

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from jdhr.runners.volumetric_video_viewer import VolumetricVideoViewer

import jittor as jt
import numpy as np
from jittor import nn
from typing import Literal
from jittor.optim import Adam

from jdhr.engine import cfg, args
from jdhr.engine import SAMPLERS
from jdhr.engine.registry import call_from_cfg
from jdhr.utils.base_utils import dotdict

from jdhr.utils.console_utils import *
from jdhr.utils.console_utils import dotdict
from jdhr.utils.bound_utils import get_bounds
from jdhr.utils.chunk_utils import multi_gather, multi_scatter
from jdhr.utils.gaussian_utils import GaussianModel, in_frustrum
from jdhr.utils.net_utils import normalize, typed, update_optimizer_state
from jdhr.utils.data_utils import load_pts, export_pts, to_x, to_cuda, to_cpu, to_tensor, remove_batch

from jdhr.models.cameras.optimizable_camera import OptimizableCamera
from jdhr.models.samplers.point_planes_sampler import PointPlanesSampler
from jdhr.models.networks.volumetric_video_network import VolumetricVideoNetwork
from jdhr.dataloaders.datasets.volumetric_video_dataset import VolumetricVideoDataset


@SAMPLERS.register_module()
class GaussianTSampler(PointPlanesSampler):
    def __init__(self,
                 # Legacy APIs
                 network: VolumetricVideoNetwork = None,  # ignore this

                 # Initializations
                 sh_deg: int = 3,
                 init_occ: float = 0.75,
                 scale_min: float = 1e-4,  # 0.0001m should not be too small
                 scale_max: float = 1e1,  # 100m should not be too large
                 scale_mod: float = 1.0,

                 # Densify & pruning configs
                 densify_until_iter: int = 15000,
                 densify_from_iter: int = 500,
                 densification_interval: int = 100,
                 opacity_reset_interval: int = 3000e9,  # UNUSED:
                 sh_update_iter: int = 1000,
                 densify_grad_threshold: float = 0.0002,
                 percent_dense: float = 0.01,
                 size_threshold: float = None,  # UNUSED:
                 min_opacity: float = 0.005,
                 preload_gs: str = '',

                 # Housekeepings
                 **kwargs,
                 ):

        # Initialize parents and remove unwanted modules
        self.kwargs = dotdict(kwargs)
        call_from_cfg(super().__init__, kwargs, network=network)

        del self.pcd_embedder
        del self.xyz_embedder
        del self.resd_regressor
        del self.geo_regressor
        del self.dir_embedder
        del self.rgb_regressor

        # Gaussian models for every frame
        self.sh_deg = sh_deg
        self.scale_mod = scale_mod
        self.pcds: nn.ParameterList[GaussianModel] = nn.ParameterList([
            GaussianModel(
                pcd,
                self.rgbs[i] if len(self.rgbs) else None,
                self.occs[i] if len(self.occs) else init_occ,
                self.rads[i].expand(self.rads[i].shape[0], 3) if len(self.rads) else None,
                sh_deg,
                scale_min,
                scale_max
            )
            for i, pcd in enumerate(self.pcds)
        ])

        # Update parameters (densification & pruning)
        self.densify_until_iter = densify_until_iter
        self.densify_from_iter = densify_from_iter
        self.densification_interval = densification_interval
        self.opacity_reset_interval = opacity_reset_interval
        self.densify_grad_threshold = densify_grad_threshold
        self.sh_update_iter = sh_update_iter
        self.size_threshold = size_threshold
        self.percent_dense = percent_dense
        self.min_opacity = min_opacity
        self.last_output = None  # will only store the updates for one of the points

        # Debug options
        self.scale_mult = 1.0
        self.alpha_mult = 1.0

        # Test time controls
        self.post_handle = self.register_load_state_dict_post_hook(self._load_state_dict_post_hook)

        if preload_gs:
            self.load_from_file(preload_gs)

    def render_imgui(self, viewer: 'VolumetricVideoViewer', batch: dotdict):
        from imgui_bundle import imgui
        self.scale_mult = imgui.slider_float(f'Scale multiplier', self.scale_mult, 0.1, 5.0)[1]  # 0.1mm
        self.alpha_mult = imgui.slider_float(f'Alpha multiplier', self.alpha_mult, 0.1, 5.0)[1]  # 0.1mm

        for i, pcd in enumerate(self.pcds):
            imgui.text(f'Number of points: {len(pcd._xyz)}')

    def render_gaussians(self, xyz: jt.Var, sh: jt.Var, scale3: jt.Var, rot4: jt.Var, occ1: jt.Var, batch: dotdict):#????
        # Lazy imports
        from diff_gauss import rasterize_gaussians, GaussianRasterizationSettings, GaussianRasterizer
        from jdhr.utils.gaussian_utils import prepare_gaussian_camera

        # Remove batch dimension
        xyz, sh, scale3, rot4, occ1 = remove_batch([xyz, sh, scale3, rot4, occ1])

        # Prepare the camera transformation for Gaussian
        gaussian_camera = to_x(prepare_gaussian_camera(batch), jt.float)

        # is_in_frustrum = in_frustrum(xyz, gaussian_camera.full_proj_transform)
        # print('Number of points to render:', is_in_frustrum.sum().item())

        # Prepare rasterization settings for gaussian
        raster_settings = GaussianRasterizationSettings(
            image_height=gaussian_camera.image_height,
            image_width=gaussian_camera.image_width,
            tanfovx=gaussian_camera.tanfovx,
            tanfovy=gaussian_camera.tanfovy,
            bg=jt.full([3], self.bg_brightness if hasattr(self, 'bg_brightness') else 0.0, device=xyz.device),  # GPU
            scale_modifier=self.scale_mod if hasattr(self, 'bg_brightness') else 1.0,
            viewmatrix=gaussian_camera.world_view_transform,
            projmatrix=gaussian_camera.full_proj_transform,
            sh_degree=self.sh_deg if hasattr(self, 'sh_deg') else 0,
            campos=gaussian_camera.camera_center,
            prefiltered=False,
            debug=self.debug if hasattr(self, 'debug') else False,
        )

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        scr = jt.zeros_like(xyz, requires_grad=True) + 0  # gradient magic
        if scr.requires_grad: scr.retain_grad()
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        rendered_image, rendered_depth, rendered_alpha, radii = typed(jt.float, jt.float)(rasterizer)(
            means3D=xyz,
            means2D=scr,
            shs=sh.mT,
            colors_precomp=None,
            opacities=occ1,
            scales=scale3,
            rotations=rot4,
            cov3D_precomp=None,
        )

        rgb = rendered_image[None].permute(0, 2, 3, 1)
        acc = rendered_alpha[None].permute(0, 2, 3, 1)
        dpt = rendered_depth[None].permute(0, 2, 3, 1)
        batch.output.rad = radii[None]  # Store radii for later use
        batch.output.scr = scr  # Store screen space points for later use, # !: BATCH
        return rgb, acc, dpt

    def render_mips(self, xyz: jt.Var, sh: jt.Var, scale3: jt.Var, rot4: jt.Var, occ1: jt.Var, batch: dotdict):
        # Lazy imports
        from diff_mip_rasterization import rasterize_mips, MipRasterizationSettings, MipRasterizer
        from jdhr.utils.gaussian_utils import prepare_gaussian_camera

        # Remove batch dimension
        xyz, sh, scale3, rot4, occ1 = remove_batch([xyz, sh, scale3, rot4, occ1])

        # Prepare the camera transformation for Gaussian
        gaussian_camera = to_x(prepare_gaussian_camera(batch), jt.float)

        if self.training:
            subpixel_offset = jt.rand((int(gaussian_camera.image_height), int(gaussian_camera.image_width), 2), dtype=xyz.dtype)
        else:
            subpixel_offset = xyz.new_zeros((int(gaussian_camera.image_height), int(gaussian_camera.image_width), 2))

        # Prepare rasterization settings for gaussian
        raster_settings = MipRasterizationSettings(
            image_height=gaussian_camera.image_height,
            image_width=gaussian_camera.image_width,
            tanfovx=gaussian_camera.tanfovx,
            tanfovy=gaussian_camera.tanfovy,
            bg=jt.full([3], self.bg_brightness if hasattr(self, 'bg_brightness') else 0.0),  # GPU
            scale_modifier=self.scale_mod if hasattr(self, 'scale_mod') else 1.0,
            viewmatrix=gaussian_camera.world_view_transform,
            projmatrix=gaussian_camera.full_proj_transform,
            sh_degree=self.sh_deg if hasattr(self, 'sh_deg') else 0,
            campos=gaussian_camera.camera_center,
            prefiltered=False,
            kernel_size=self.kernel_size if hasattr(self, 'kernel_size') else 0.1,
            subpixel_offset=subpixel_offset,
            debug=self.debug if hasattr(self, 'debug') else False,
        )

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        scr = jt.zeros_like(xyz, requires_grad=True) + 0  # gradient magic
        if scr.requires_grad: scr.retain_grad()
        rasterizer = MipRasterizer(raster_settings=raster_settings)
        rendered_image, radii = typed(jt.float, jt.float)(rasterizer)(
            means3D=xyz,
            means2D=scr,
            shs=sh.mT,
            colors_precomp=None,
            opacities=occ1,
            scales=scale3,
            rotations=rot4,
            cov3D_precomp=None,
        )

        rgb = rendered_image[None].permute(0, 2, 3, 1)
        batch.output.rad = radii[None]  # Store radii for later use
        batch.output.scr = scr  # Store screen space points for later use, # !: BATCH
        return rgb, rgb[..., :1], rgb[..., :1]

    def render_radius(self, xyz: jt.Var, sh: jt.Var, radius: jt.Var, occ1: jt.Var, batch: dotdict):
        # Lazy imports
        from jdhr.diff_point_rasterizater import PointRasterizationSettings, PointRasterizer
        from jdhr.utils.gaussian_utils import prepare_gaussian_camera
        assert sh.ndim == 4, 'Should input 4 dim SH: B, N, C, SH'

        # Remove batch dimension
        xyz, sh, radius, occ1 = remove_batch([xyz, sh, radius, occ1])

        # Prepare the camera transformation for Gaussian
        gaussian_camera = to_x(prepare_gaussian_camera(batch), jt.float)

        # Prepare rasterization settings for gaussian
        raster_settings = PointRasterizationSettings(
            image_height=gaussian_camera.image_height,
            image_width=gaussian_camera.image_width,
            tanfovx=gaussian_camera.tanfovx,
            tanfovy=gaussian_camera.tanfovy,
            bg=jt.full([3], self.bg_brightness if hasattr(self, 'bg_brightness') else 0.0),  # GPU
            scale_modifier=self.scale_mod if hasattr(self, 'scale_mod') else 1.0,
            viewmatrix=gaussian_camera.world_view_transform,
            projmatrix=gaussian_camera.full_proj_transform,
            sh_degree=self.sh_deg if hasattr(self, 'sh_deg') else 0,
            campos=gaussian_camera.camera_center,
            prefiltered=False,
            debug=self.debug if hasattr(self, 'debug') else False,
        )

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        scr = jt.array(jt.zeros_like(xyz))+ 0  # gradient magic
        #if scr.requires_grad: scr.retain_grad()
        rasterizer = PointRasterizer(raster_settings=raster_settings)
        #print("gaussian",xyz.shape,scr.shape,sh.shape, radius.shape, occ1.shape)
        rendered_image, rendered_depth, rendered_alpha, radii = typed(jt.float, jt.float)(rasterizer)(
            means3D=xyz,
            means2D=scr,
            shs=sh.transpose(-2,-1),
            colors_precomp=None,
            opacities=occ1,
            radius=radius,
        )

        rgb = rendered_image[None].permute(0, 2, 3, 1)
        acc = rendered_alpha[None].permute(0, 2, 3, 1)
        dpt = rendered_depth[None].permute(0, 2, 3, 1)
        batch.output.rad = radii[None]  # Store radii for later use
        batch.output.scr = scr  # Store screen space points for later use, # !: BATCH
        return rgb, acc, dpt

    @jt.no_grad()
    def update_gaussians(self, batch: dotdict):
        if not self.training: return

        # Prepare global variables
        iter: int = batch.meta.iter  # controls whether we're to update in this iteration
        output = self.last_output  # contains necessary information for updating gaussians
        optimizer: Adam = cfg.runner.optimizer

        # Update for all frames
        if iter > 0 and iter < self.densify_until_iter and iter % self.sh_update_iter == 0:
            for pcd in self.pcds:
                pcd.oneupSHdegree()

        # Update only the rendered frame
        if iter > 0 and iter < self.densify_until_iter and output is not None:

            # The dictionary for controlling the optimizer
            optimizer_state = dotdict()

            # Update all rendered gaussians in the batch
            for i, pcd in enumerate(output.pcd):  # removing batch dim
                pcd: GaussianModel

                # Preparing optimizer states for update
                for name, params in pcd.named_parameters():
                    if params.requires_grad:
                        optimizer_state[params] = dotdict(
                            name=name,
                            old_keep=jt.ones_like(params).bool(),
                            new_keep=jt.ones_like(params).bool(),
                            new_params=None,
                        )

                # Preparing gaussian stats for update
                radii = output.rad[i]
                visibility_filter = radii > 0
                viewspace_point_tensor = output.scr  # no indexing, otherwise no grad # !: BATCH
                if output.scr.grad is None: continue  # previous rendering was an evaluation
                pcd.max_radii2D[visibility_filter] = jt.max(pcd.max_radii2D[visibility_filter], radii[visibility_filter])
                pcd.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # Perform densification and pruning
                if iter > self.densify_from_iter and iter % self.densification_interval == 0:
                    pcd.densify_and_prune(self.densify_grad_threshold, self.min_opacity, self.scale_mod, self.size_threshold, self.percent_dense, optimizer_state)
                    log(yellow_slim('Densification and pruning done! ' +
                                    f'min opacity: {pcd.get_opacity.min().item():.4f} ' +
                                    f'max opacity: {pcd.get_opacity.max().item():.4f} ' +
                                    f'number of points: {pcd.get_xyz.shape[0]}'))

                # Perform opacity reset
                if iter > self.densify_from_iter and iter % self.opacity_reset_interval == 0:
                    pcd.reset_opacity(optimizer_state)
                    log(yellow_slim('Resetting opacity done! ' +
                                    f'min opacity: {pcd.get_opacity.min().item():.4f} ' +
                                    f'max opacity: {pcd.get_opacity.max().item():.4f}'))

            # Update the actual optimizer states
            update_optimizer_state(optimizer, optimizer_state)

    @jt.no_grad()
    def _load_state_dict_pre_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Supports loading points and features with different shapes
        if hasattr(self, 'pcds'):
            pcd_keys = []
            for f, pcd in enumerate(self.pcds):
                for name, params in pcd.named_parameters():
                    params.data = params.data.new_empty(state_dict[f'{prefix}pcds.{f}.{name}'].shape)
                    pcd_keys.append(f'{prefix}pcds.{f}.{name}')
            self.points_aligned = True  # need aligned when new check points are loaded
            self.points_expanded = True  # need aligned when new check points are loaded

            keys = list(state_dict.keys())
            for key in keys:
                if key.startswith(f'{prefix}pcds.') and key not in pcd_keys:
                    del state_dict[key]

        # Historical reason
        if f'{prefix}occs.0' in state_dict:
            del state_dict[f'{prefix}occs.0']

        # Historical reason
        if f'{prefix}rgbs.0' in state_dict:
            del state_dict[f'{prefix}rgbs.0']

        # Historical reason
        if f'{prefix}rads.0' in state_dict:
            del state_dict[f'{prefix}rads.0']

    @jt.no_grad()
    def _load_state_dict_post_hook(self, module, incompatible_keys):
        # Load tighter bounds from the trained models
        dataset: VolumetricVideoDataset = cfg.runner.val_dataloader.dataset
        dataset.vhull_bounds = [dataset.bounds for _ in range(len(self.pcds))]
        for i in range(len(self.pcds)):
            if self.pcds[i] is not None:
                dataset.vhull_bounds[i] = get_bounds(self.pcds[i].get_xyz[None], padding=0.01)[0].cpu()  # MARK: SYNC

    def execute(self, batch: dotdict):
        # Initialization & densification & pruning
        self.init_points(batch)
        self.update_gaussians(batch)

        # Construct renderable parameters
        index, time = self.sample_index_time(batch)
        xyz = jt.stack([self.pcds[l].get_xyz for l in index])  # B, N, 3
        scale3 = jt.stack([self.pcds[l].get_scaling for l in index])  # B, N, 3
        rot4 = jt.stack([self.pcds[l].get_rotation for l in index])  # B, N, 4
        alpha = jt.stack([self.pcds[l].get_opacity for l in index])  # B, N, 1
        sh = jt.stack([self.pcds[l].get_features for l in index]).mT  # B, N, C, SH

        # Perform points rendering
        rgb, acc, dpt = self.render_gaussians(xyz, sh, scale3 * self.scale_mult, rot4, alpha * self.alpha_mult, batch)  # B, HW, C

        # Prepare output
        batch.output.pcd = [self.pcds[l] for l in index]
        self.store_output(None, xyz, rgb, acc, dpt, batch)
        self.last_output = batch.output  # retain gradients after updates
