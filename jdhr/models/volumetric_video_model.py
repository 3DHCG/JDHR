# Default pipeline for volumetric videos
# This corresponds to the tranditional implementation's renderer
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from jdhr.runners.volumetric_video_viewer import VolumetricVideoViewer
    from jdhr.runners.volumetric_video_runner import VolumetricVideoRunner

import time
import jittor as jt
from jittor import nn
from typing import Union
from jdhr.utils.console_utils import *
from jdhr.utils.timer_utils import timer  # global timer
from jdhr.utils.base_utils import dotdict
from jdhr.utils.data_utils import to_x
from jdhr.utils.ray_utils import get_rays
from jdhr.utils.chunk_utils import chunkify
from jdhr.utils.bound_utils import get_near_far_aabb, monotonic_near_far

from jdhr.engine import cfg, args
from jdhr.engine import MODELS, CAMERAS, SAMPLERS, NETWORKS, RENDERERS, SUPERVISORS, REGRESSORS, EMBEDDERS
from jdhr.models.cameras.noop_camera import NoopCamera
#from jdhr.models.cameras.optimizable_camera import OptimizableCamera
from jdhr.models.networks.multilevel_network import MultilevelNetwork
from jdhr.models.samplers.importance_sampler import ImportanceSampler
from jdhr.models.samplers.r4dv_sampler import R4DVSampler
from jdhr.models.renderers.volume_renderer import VolumeRenderer
from jdhr.models.supervisors.volumetric_video_supervisor import VolumetricVideoSupervisor

# sampler (o, d, t -> z -> xyzt vd) ->
# network (xyzt vd -> rgb, occ) ->
# renderer (rgb, occ -> output) ->
# supervisor (output, batch -> loss)


@MODELS.register_module()
class VolumetricVideoModel(nn.Module):
    def __init__(self,
                 camera_cfg: dotdict = dotdict(type=NoopCamera.__name__),
                 sampler_cfg: dotdict = dotdict(type=ImportanceSampler.__name__),
                 network_cfg: dotdict = dotdict(type=MultilevelNetwork.__name__),
                 renderer_cfg: dotdict = dotdict(type=VolumeRenderer.__name__),
                 supervisor_cfg: dotdict = dotdict(type=VolumetricVideoSupervisor.__name__),

                 apply_optcam: bool = True,  # apply optimized cameras even if not training
                 use_z_depth: bool = False,  # use ray direciton depth or z axis depth
                 correct_pix: bool = True,  # move pixel coordinates to the middle of the pixel
                 move_to_cpu: bool = False,  # move chunkify data to cpu during rendering
                 chunkify_rays: bool = True,  # whether to split input rays during forward evaluation
                 train_chunk_size: int = 4096,  # B * P = 8192
                 render_chunk_size: int = 4096,  # B * P = 8192 (B ~= 1)
                 print_render_progress: bool = False,
                 let_user_handle_input: bool = False,

                 dtype: Union[str, jt.dtype] = jt.float,
                 ):
        super().__init__()

        self.camera: NoopCamera = CAMERAS.build(camera_cfg)
        self.network: MultilevelNetwork = NETWORKS.build(network_cfg)
        self.sampler: ImportanceSampler = SAMPLERS.build(sampler_cfg, network=self.network)
        self.renderer: VolumeRenderer = RENDERERS.build(renderer_cfg, network=self.network)
        self.supervisor: VolumetricVideoSupervisor = SUPERVISORS.build(supervisor_cfg, network=self.network)

        self.dtype = getattr(jt, dtype) if isinstance(dtype, str) else dtype
        self.apply_optcam = apply_optcam
        self.use_z_depth = use_z_depth
        self.correct_pix = correct_pix
        self.move_to_cpu = move_to_cpu
        self.chunkify_rays = chunkify_rays
        self.train_chunk_size = train_chunk_size
        self.render_chunk_size = render_chunk_size
        self.print_render_progress = print_render_progress
        self.let_user_handle_input = let_user_handle_input

    def render_imgui(self, viewer: 'VolumetricVideoViewer', batch: dotdict):
        if hasattr(self.camera, 'render_imgui'): self.camera.render_imgui(viewer, batch)
        if hasattr(self.sampler, 'render_imgui'): self.sampler.render_imgui(viewer, batch)
        if hasattr(self.network, 'render_imgui'): self.network.render_imgui(viewer, batch)
        if hasattr(self.renderer, 'render_imgui'): self.renderer.render_imgui(viewer, batch)
        if hasattr(self.supervisor, 'render_imgui'): self.supervisor.render_imgui(viewer, batch)

    def decorate_grad(self, runner: 'VolumetricVideoRunner', batch: dotdict):
        if hasattr(self.camera, 'decorate_grad'): self.camera.decorate_grad(runner, batch)
        if hasattr(self.sampler, 'decorate_grad'): self.sampler.decorate_grad(runner, batch)
        if hasattr(self.network, 'decorate_grad'): self.network.decorate_grad(runner, batch)
        if hasattr(self.renderer, 'decorate_grad'): self.renderer.decorate_grad(runner, batch)
        if hasattr(self.supervisor, 'decorate_grad'): self.supervisor.decorate_grad(runner, batch)

    def render_volume(self, xyz: jt.Var, dir: jt.Var, t: jt.Var, dist: jt.Var, batch: dotdict):
        # Network
        rgb, occ = self.network.compute(xyz, dir, t, dist, batch)  # how to get annotation on forward params?

        # Prepare special data passing object
        output = batch.output  # will be integrated
        # del batch.output
        return output

    def render_rays(self,  # used for vanialla NeRF training
                    ray_o: jt.Var, ray_d: jt.Var, near: jt.Var, far: jt.Var, t: jt.Var,
                    batch: dotdict):
        # This is the main code path for training
        # Sampler
        xyz, dir, t, dist = self.sampler.sample(ray_o, ray_d, near, far, t, batch)  # B, P, S

        # Network
        rgb, occ = self.network.compute(xyz, dir, t, dist, batch)  # how to get annotation on forward params?

        # Renderer
        rgb_map = self.renderer.render(rgb, occ, batch)  # unused rgb map

        # Prepare special data passing object
        output = batch.output  # will be integrated
        del batch.output
        return output

    def prepare_camera(self, batch: dotdict):
        batch.K = to_x(batch.K, self.dtype)
        batch.R = to_x(batch.R, self.dtype)
        batch.T = to_x(batch.T, self.dtype)

        # Maybe forward input camera parameters
        if self.training or (self.apply_optcam and args.type != 'gui'):
            batch = self.camera.forward_cams(batch)  # MARK: SYNC

        # Always forward IBR required camera parameters
        if 'src_exts' in batch:
            batch.src_exts = to_x(batch.src_exts, self.dtype)
            batch.src_ixts = to_x(batch.src_ixts, self.dtype)
            batch = self.camera.forward_srcs(batch)  # MARK: NO SYNC FOR NOW

        # Maybe forward input ray directions
        if 'ray_o' in batch:
            batch.ray_o = to_x(batch.ray_o, self.dtype)
            batch.ray_d = to_x(batch.ray_d, self.dtype)
            if self.training or (self.apply_optcam and args.type != 'gui'):
                batch.ray_o, batch.ray_d = self.camera.forward_rays(batch.ray_o, batch.ray_d, batch, self.use_z_depth, self.correct_pix)

        if 't' in batch:
            batch.t = to_x(batch.t, self.dtype)

        if 'n' in batch:
            batch.n = to_x(batch.n, self.dtype)
            batch.f = to_x(batch.f, self.dtype)

        if 'near' in batch:
            batch.near = to_x(batch.near, self.dtype)
            batch.far = to_x(batch.far, self.dtype)

        if 'bounds' in batch:
            batch.bounds = to_x(batch.bounds, self.dtype)

        if 'xyz' in batch:
            batch.xyz = to_x(batch.xyz, self.dtype)
            batch.dir = to_x(batch.dir, self.dtype)
            batch.dist = to_x(batch.dist, self.dtype)

        return batch

    def extract_input(self, batch: dotdict):
        # No matter the input type, we forward K, R, T with view_index and latent_index for camera optimization

        if self.let_user_handle_input:
            return (None, None, None, None, None), self.render_rays  # let the user handle theirselves

        if 'xyz' in batch:  # forward volume # NOTE: no camera optimization in this path
            # B, P, C
            xyz: jt.Var = batch.xyz
            dir: jt.Var = batch.dir
            t: jt.Var = batch.t
            dist: jt.Var = batch.dist
            t = t[..., None, None].expand(-1, *xyz.shape[1:-1], 1)  # B, P, 1

            return (xyz, dir, t, dist), self.render_volume

        elif 'near' in batch:  # forward rays # NOTE: no camera optimization in this path
            # Most of the NeRF-based method goes through this path

            # B, P, C
            ray_o: jt.Var = batch.ray_o
            ray_d: jt.Var = batch.ray_d
            near: jt.Var = batch.near  # expected to be in the correct shape
            far: jt.Var = batch.far  # expected to be in the correct shape
            t: jt.Var = batch.t  # expected to be in the correct shape

            return (ray_o, ray_d, near, far, t), self.render_rays

        elif 'ray_o' in batch:  # NOTE: no camera optimization in this path
            # B, P, C
            ray_o: jt.Var = batch.ray_o
            ray_d: jt.Var = batch.ray_d
            bounds: jt.Var = batch.bounds  # ? plural
            t: jt.Var = batch.t
            n, f = batch.n, batch.f

            near, far = get_near_far_aabb(bounds, ray_o, ray_d)
            near, far = monotonic_near_far(near, far, n, f)
            t = t[..., None, None].expand(-1, *ray_o.shape[1:-1], 1)  # B, P, 1

            batch.near = near
            batch.far = far
            return (ray_o, ray_d, near, far, t), self.render_rays

        elif 'bounds' in batch:  # forward whole image # this is the most widely used form of model input # MARK: handles camera optimization
            # B, P, C
            bounds: jt.Var = batch.bounds  # ? plural
            n: jt.Var = batch.n  # B,
            f: jt.Var = batch.f  # B,
            t: jt.Var = batch.t  # B,

            H, W, K, R, T = batch.meta.H[0].item(), batch.meta.W[0].item(), batch.K, batch.R, batch.T  # !: BATCH
            ray_o, ray_d, coords = get_rays(H, W, K, R, T, z_depth=self.use_z_depth, correct_pix=self.correct_pix, ret_coord=True)  # maybe without normalization
            ray_o, ray_d, coords = ray_o.view(-1, H * W, 3), ray_d.view(-1, H * W, 3), coords.view(-1, H * W, 2)

            near, far = get_near_far_aabb(bounds, ray_o, ray_d)
            near, far = monotonic_near_far(near, far, n, f)
            t = t[..., None, None].expand(-1, *ray_o.shape[1:-1], 1)  # B, P, 1

            batch.ray_o = ray_o
            batch.ray_d = ray_d
            batch.near = near
            batch.far = far
            batch.coords = coords
            return (ray_o, ray_d, near, far, t), self.render_rays

        else:
            raise NotImplementedError

    def execute(self, batch: dotdict):
        # B, P, C
        batch = self.prepare_camera(batch)
        input, function = self.extract_input(batch)

        # Rendering part of model (manually chunkify, makes it configurable)
        chunk_size = self.train_chunk_size if self.training else self.render_chunk_size
        should_chunk = self.chunkify_rays and input[0].shape[1] > chunk_size
        if should_chunk:
            move_to_cpu = self.move_to_cpu and not self.training  # always store things on gpu when training
            print_progress = self.print_render_progress and not self.training
            rendering_function = chunkify(chunk_size, print_progress=print_progress, move_to_cpu=move_to_cpu)(function)
        else:
            rendering_function = function  # when not chunking, do not move things around

        timer.record()
        output = rendering_function(*input, batch=batch)
        output.time = timer.record('model')
        # Loss computing part of the network
        if self.training:
            # Supervisor
            loss, scalar_stats, image_stats = self.supervisor.supervise(output, batch)
            output.loss = loss
            output.scalar_stats = scalar_stats
            output.image_stats = image_stats

        return output
