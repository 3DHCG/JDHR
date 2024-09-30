# This file contains optimizable camera parameters
# Implemented in SO3xR3, exponential map of rotation and translation from screw rt motion

import numpy as np
import jittor as jt
from jittor import nn
from copy import copy
from os.path import join
from jdhr.engine import cfg
from jdhr.engine import CAMERAS
from jdhr.utils.console_utils import *
from jdhr.utils.base_utils import dotdict
from jdhr.utils.lie_utils import exp_map_SO3xR3
from jdhr.utils.ray_utils import get_rays_from_ij
from jdhr.utils.net_utils import make_params, freeze_module, load_network
from jdhr.utils.math_utils import affine_padding, affine_inverse, vector_padding, point_padding


@CAMERAS.register_module()
class OptimizableCamera(nn.Module):
    # TODO: Implement intrinsics optimization
    # MARK: EVIL GLOBAL CONFIG
    bounds = cfg.dataset_cfg.bounds if 'bounds' in cfg.dataset_cfg else [[-1, -1, -1], [1, 1, 1]]  # only used for initialization
    bounds = np.array(bounds, dtype=float)

    center = bounds.sum(-2) / 2
    radius = (bounds[1] - bounds[0]).max() / 2
    square_bounds = np.stack([center - radius, center + radius])
    scene_scale = cfg.dataset_cfg.scene_scale if 'scene_scale' in cfg.dataset_cfg else 1.0  # is the scene of real world scale?
    duration = cfg.dataset_cfg.duration if 'duration' in cfg.dataset_cfg else 1.0  # length in real world time
    # world_up = cfg.viewer_cfg.camera_cfg.world_up if 'world_up' in cfg.dataset_cfg else [0, 0, 1]  # only used for initialization

    data_root = cfg.dataset_cfg.data_root if 'data_root' in cfg.dataset_cfg else ''
    vhulls_dir = cfg.dataset_cfg.vhulls_dir if 'vhulls_dir' in cfg.dataset_cfg else 'vhulls'
    images_dir = cfg.dataset_cfg.images_dir if 'images_dir' in cfg.dataset_cfg else 'images'

    view_sample = cfg.dataset_cfg.view_sample if 'view_sample' in cfg.dataset_cfg else [0, None, 1]
    frame_sample = cfg.dataset_cfg.frame_sample if 'frame_sample' in cfg.dataset_cfg else [0, None, 1]

    cams = os.listdir(join(data_root, images_dir)) if exists(join(data_root, images_dir)) else []
    view_sample, frame_sample = copy(view_sample), copy(frame_sample)
    if len(frame_sample) == 3:
        frame_sample[1] = frame_sample[1] or (len(os.listdir(join(data_root, images_dir, cams[0]))) if len(cams) else 1)  # will error out if using this module
        n_frames = (frame_sample[1] - frame_sample[0]) // frame_sample[2]
    else:
        n_frames = len(frame_sample)
    if len(view_sample) == 3:
        view_sample[1] = view_sample[1] or len(cams)  # will error out if using this module
        n_views = (view_sample[1] - view_sample[0]) // view_sample[2]  # FIXME: DIFFERENT MEANING
    else:
        n_views = len(view_sample)
    intri_file = cfg.dataset_cfg.intri_file if 'intri_file' in cfg.dataset_cfg else 'intri.yml'
    extri_file = cfg.dataset_cfg.extri_file if 'extri_file' in cfg.dataset_cfg else 'extri.yml'

    # TODO: Remove the closest using t setting
    closest_using_t = cfg.dataset_cfg.closest_using_t if 'closest_using_t' in cfg.dataset_cfg else False
    moves_through_time = not exists(join(data_root, intri_file)) or not exists(join(data_root, extri_file))

    def __init__(self,
                 n_views: int = n_views,
                 n_frames: int = n_frames,
                 moves_through_time: bool = moves_through_time,
                 pretrained_camera: str = '',
                 freeze_camera: bool = False,
                 freeze_extri: bool = False,
                 freeze_intri: bool = False,
                 focal_limit: float = 10.0,
                 shift_limit: float = 0.05,  # extremely unstable
                 dtype: str = 'float',
                 **kwargs,
                 ):
        super().__init__()
        self.n_views = n_views
        self.n_frames = n_frames if moves_through_time else 1
        self.dtype = getattr(jt, dtype) if isinstance(dtype, str) else dtype

        self.extri_resd = make_params(jt.zeros(self.n_frames, self.n_views, 6, dtype=self.dtype))  # F, V, 6
        self.intri_resd = make_params(jt.zeros(self.n_frames, self.n_views, 4, dtype=self.dtype))  # F, V, 4
        self.focal_limit = focal_limit
        self.shift_limit = shift_limit

        if exists(pretrained_camera):
            load_network(self, pretrained_camera, prefix='camera.')  # will load some of the parameters from this model

        if freeze_camera:
            freeze_module(self)

        if freeze_extri:
            self.extri_resd.requires_grad_(False)

        if freeze_intri:
            self.intri_resd.requires_grad_(False)

        self.pre_handle = self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):

        # Historical reasons
        if prefix + 'pose_resd' in state_dict:
            state_dict[prefix + 'extri_resd'] = state_dict[prefix + 'pose_resd']
            del state_dict[prefix + 'pose_resd']

        if prefix + 'extri_resd' in state_dict:
            if state_dict[prefix + 'extri_resd'].shape[0] == self.n_views and state_dict[prefix + 'extri_resd'].shape[1] == self.n_frames:
                state_dict[prefix + 'extri_resd'] = state_dict[prefix + 'extri_resd'].transpose(0, 1)

        if prefix + 'intri_resd' in state_dict:
            if state_dict[prefix + 'intri_resd'].shape[0] == self.n_views and state_dict[prefix + 'intri_resd'].shape[1] == self.n_frames:
                state_dict[prefix + 'intri_resd'] = state_dict[prefix + 'intri_resd'].transpose(0, 1)

        if prefix + 'extri_resd' not in state_dict:
            state_dict[prefix + 'extri_resd'] = self.extri_resd

        if prefix + 'intri_resd' not in state_dict:
            state_dict[prefix + 'intri_resd'] = self.intri_resd

    def forward_srcs(self, batch: dotdict):
        s_inds = batch.src_inds  # B, S, selected source views
        t_inds = batch.t_inds
        if OptimizableCamera.closest_using_t:
            f_inds = s_inds.clamp(0, self.n_frames - 1)
            v_inds = t_inds.clamp(0, self.n_views - 1)
        else:
            f_inds = t_inds.clamp(0, self.n_frames - 1)
            v_inds = s_inds.clamp(0, self.n_views - 1)

        extri_resd = self.extri_resd[f_inds, v_inds].to(batch.src_exts)  # B, S, 6
        extri_resd = exp_map_SO3xR3(extri_resd.detach())  # do not optimize through sampling, unstable
        w2c_resd = affine_padding(extri_resd)  # B, S, 3, 4
        w2c_opt = w2c_resd @ batch.src_exts
        batch.src_exts = w2c_opt
        # UNUSED: This is not used for now
        # batch.meta.src_exts = batch.src_exts.to('cpu', non_blocking=True)

        intri_resd = self.intri_resd[f_inds, v_inds].to(batch.src_ixts)  # B, S, 4
        src_ixts = jt.zeros_like(batch.src_ixts)
        src_ixts[..., 0, 0] = intri_resd[..., 0, 0].sign() * intri_resd[..., 0, 0].abs().clamp(self.focal_limit) + batch.src_ixts[..., 0, 0]  # fx
        src_ixts[..., 1, 1] = intri_resd[..., 1, 1].sign() * intri_resd[..., 1, 1].abs().clamp(self.focal_limit) + batch.src_ixts[..., 1, 1]  # fy
        src_ixts[..., 0, 2] = intri_resd[..., 0, 2].sign() * intri_resd[..., 0, 2].abs().clamp(self.shift_limit) + batch.src_ixts[..., 0, 2]  # cx
        src_ixts[..., 1, 2] = intri_resd[..., 1, 2].sign() * intri_resd[..., 1, 2].abs().clamp(self.shift_limit) + batch.src_ixts[..., 1, 2]  # cy
        src_ixts[..., 2, 2] = 1.0
        return batch

    def forward_pose(self, batch: dotdict):
        if 'w2c_ori' in batch:
            return batch.w2c_ori, batch.w2c_opt, batch.w2c_resd, batch.int_ori, batch.int_opt, batch.int_resd

        view_index = batch.view_index  # B, # avoid synchronization
        latent_index = batch.latent_index  # B,

        view_index = view_index.clamp(0, self.n_views - 1)
        latent_index = latent_index.clamp(0, self.n_frames - 1)  # TODO: FIX VIEW AND CAMERA AND FRAME AND LATEN INDICES

        extri_resd = self.extri_resd[latent_index, view_index].to(batch.R)  # fancy indexing? -> B, 6
        extri_resd = exp_map_SO3xR3(extri_resd)  # B, 3, 4

        # Use left multiplication
        w2c_resd = affine_padding(extri_resd)  # B, 3, 4
        w2c_ori = affine_padding(jt.cat([batch.R, batch.T], dim=-1))  # B, 3, 4
        w2c_opt = w2c_resd @ w2c_ori

        batch.w2c_ori = w2c_ori
        batch.w2c_opt = w2c_opt
        batch.w2c_resd = w2c_resd

        # Optimize intrinsic parameters
        intri_resd = self.intri_resd[latent_index, view_index].to(batch.R)  # fancy indexing? -> B, 4
        int_ori = batch.K
        int_opt = jt.zeros_like(int_ori)
        int_opt[..., 0, 0] = intri_resd[..., 0].sign() * intri_resd[..., 0].abs().clamp(self.focal_limit) + int_ori[..., 0, 0]  # fx
        int_opt[..., 1, 1] = intri_resd[..., 1].sign() * intri_resd[..., 1].abs().clamp(self.focal_limit) + int_ori[..., 1, 1]  # fy
        int_opt[..., 0, 2] = intri_resd[..., 2].sign() * intri_resd[..., 2].abs().clamp(self.shift_limit) + int_ori[..., 0, 2]  # cx
        int_opt[..., 1, 2] = intri_resd[..., 3].sign() * intri_resd[..., 3].abs().clamp(self.shift_limit) + int_ori[..., 1, 2]  # cy
        int_opt[..., 2, 2] = 1.0
        int_resd = int_opt - int_ori

        batch.int_ori = int_ori
        batch.int_opt = int_opt
        batch.int_resd = int_resd

        return w2c_ori, w2c_opt, w2c_resd, int_ori, int_opt, int_resd

    def forward_cams(self, batch: dotdict):
        w2c_ori, w2c_opt, w2c_resd, int_ori, int_opt, int_resd = self.forward_pose(batch)
        batch.orig_R = batch.R
        batch.orig_T = batch.T
        batch.orig_K = batch.K
        batch.R = w2c_opt[..., :3, :3]
        batch.T = w2c_opt[..., :3, 3:]
        batch.K = int_opt

        #batch.meta.meta_stream = torch.cuda.Stream()
        #batch.meta.meta_stream.wait_stream(torch.cuda.current_stream())  # stream synchronization matters
        #with torch.cuda.stream(batch.meta.meta_stream):
        #    batch.meta.R = batch.R.to('cpu', non_blocking=True)
        #    batch.meta.T = batch.T.to('cpu', non_blocking=True)
        #    batch.meta.K = batch.K.to('cpu', non_blocking=True)
        return batch

    def forward_rays(self, ray_o: jt.Var, ray_d: jt.Var, batch: dotdict, use_z_depth: bool = False, correct_pix: bool = True):
        w2c_ori, w2c_opt, w2c_resd, int_ori, int_opt, int_resd = self.forward_pose(batch)
        inv_w2c_opt = affine_inverse(w2c_opt)

        if 'coords' in batch:
            i, j = batch.coords.unbind(-1)
            ray_o, ray_d = get_rays_from_ij(i, j, int_opt, w2c_opt[..., :3, :3], w2c_opt[..., :3, 3:], use_z_depth=use_z_depth, correct_pix=correct_pix)

        else:
            # FIXME: Add optimizing K in ray forwarding (modify ray_d in some way?)

            # The transformed points should be left multiplied with w2c_opt, thus premult the inverse of the resd
            ray_o = point_padding(ray_o) @ w2c_ori.tranpose(-2,-1) @ inv_w2c_opt.tranpose(-2,-1) # B, N, 4 @ B, 4, 4
            ray_d = vector_padding(ray_d) @ w2c_ori.tranpose(-2,-1) @ inv_w2c_opt.tranpose(-2,-1)  # B, N, 4 @ B, 4, 4

        return ray_o[..., :3], ray_d[..., :3]

    def execute(self, ray_o: jt.Var, ray_d: jt.Var, batch, use_z_depth: bool = False, correct_pix: bool = True):
        batch = self.forward_cams(batch)
        batch = self.forward_srcs(batch)
        ray_o, ray_d = self.forward_rays(ray_o, ray_d, batch, use_z_depth, correct_pix)
        return ray_o, ray_d, batch
