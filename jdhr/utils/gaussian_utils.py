import os
import math
import numpy as np
#import torch
import jittor as jt
from jittor import nn
#from jittor.nn import functional as F

from jdhr.utils.console_utils import *
from jdhr.utils.sh_utils import eval_sh
from jdhr.utils.blend_utils import batch_rodrigues
from jdhr.utils.data_utils import to_x, add_batch, load_pts
from jdhr.utils.net_utils import make_buffer, make_params, typed
from jdhr.utils.math_utils import torch_inverse_2x2, point_padding


def render_diff_gauss(xyz3: jt.Var, rgb3: jt.Var, cov: jt.Var, occ1: jt.Var, camera: dotdict):
    from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
    # Prepare rasterization settings for gaussian
    raster_settings = GaussianRasterizationSettings(
        image_height=camera.image_height,
        image_width=camera.image_width,
        tanfovx=camera.tanfovx,
        tanfovy=camera.tanfovy,
        bg=jt.full([3], 0.0),  # GPU
        scale_modifier=1.0,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=0,
        campos=camera.camera_center,
        prefiltered=True,
        debug=False,
    )

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    scr = jt.zeros_like(xyz3) + 0  # gradient magic
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    rendered_image, rendered_depth, rendered_alpha, radii = typed(jt.float, jt.float)(rasterizer)(
        means3D=xyz3,
        means2D=scr,
        shs=None,
        colors_precomp=rgb3,
        opacities=occ1,
        scales=None,
        rotations=None,
        cov3D_precomp=cov,
    )

    rgb = rendered_image[None].permute(0, 2, 3, 1)
    acc = rendered_alpha[None].permute(0, 2, 3, 1)
    dpt = rendered_depth[None].permute(0, 2, 3, 1)
    H = camera.image_height
    W = camera.image_width
    meta = dotdict({'radii': radii / float(max(H, W)), 'scr': scr, 'H': H, 'W': W})
    return rgb, acc, dpt, meta


def render_fdgs(xyz3: jt.Var, rgb3: jt.Var, cov: jt.Var, occ1: jt.Var, camera: dotdict):
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    # Prepare rasterization settings for gaussian
    raster_settings = GaussianRasterizationSettings(
        image_height=camera.image_height,
        image_width=camera.image_width,
        tanfovx=camera.tanfovx,
        tanfovy=camera.tanfovy,
        bg=jt.full([3], 0.0),  # GPU
        scale_modifier=1.0,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=0,
        campos=camera.camera_center,
        prefiltered=True,
        debug=False
    )

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    scr = jt.zeros_like(xyz3) + 0  # gradient magic
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    rendered_image, radii = typed(jt.float, jt.float)(rasterizer)(
        means3D=xyz3,
        means2D=scr,
        shs=None,
        colors_precomp=rgb3,
        opacities=occ1,
        scales=None,
        rotations=None,
        cov3D_precomp=cov,
    )

    rgb = rendered_image[None].permute(0, 2, 3, 1)
    acc = jt.ones_like(rgb[..., :1])
    dpt = jt.zeros_like(rgb[..., :1])
    H = camera.image_height
    W = camera.image_width
    meta = dotdict({'radii': radii / float(max(H, W)), 'scr': scr, 'H': H, 'W': W})
    return rgb, acc, dpt, meta



def in_frustrum(xyz: jt.Var, full_proj_matrix: jt.Var, xy_padding: float = 0.5, padding: float = 0.01):
    ndc = point_padding(xyz) @ full_proj_matrix # this is now in clip space
    ndc = ndc[..., :3] / ndc[..., 3:]
    return (ndc[..., 2] > -1 - padding) & (ndc[..., 2] < 1 + padding) & (ndc[..., 0] > -1 - xy_padding) & (ndc[..., 0] < 1. + xy_padding) & (ndc[..., 1] > -1 - xy_padding) & (ndc[..., 1] < 1. + xy_padding)  # N,



def rgb2sh0(rgb: jt.Var):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0



def sh02rgb(sh: jt.Var):
    C0 = 0.28209479177387814
    return sh * C0 + 0.5



def get_jacobian(pix_xyz: jt.Var,  # B, P, 3, point in screen space
                 ):
    J = pix_xyz.new_zeros(pix_xyz.shape + (3, ))  # B, P, 3, 3
    J[..., 0, 0] = 1 / pix_xyz[..., 2]
    J[..., 1, 1] = 1 / pix_xyz[..., 2]
    J[..., 0, 2] = -pix_xyz[..., 0] / pix_xyz[..., 2]**2
    J[..., 1, 2] = -pix_xyz[..., 1] / pix_xyz[..., 2]**2
    J[..., 2, 2] = 1
    return J



def gaussian_2d(xy: jt.Var,  # B, H, W, 2, screen pixel locations for evaluation
                mean_xy: jt.Var,  # B, H, W, K, 2, center of the gaussian in screen space
                cov_xy: jt.Var,  # B, H, W, 2, 2, covariance of the gaussian in screen space
                # pow: float = 1,  # when pow != 1, not a real gaussian, but easier to control fall off
                # we want to the values at 3 sigma to zeros -> easier to control volume rendering?
                ):
    inv_cov_xy = torch_inverse_2x2(cov_xy)  # B, P, 2, 2
    minus_mean = xy[..., None, :] - mean_xy  # B, P, K, 2
    # weight = torch.exp(-0.5 * torch.einsum('...d,...de,...e->...', x_minus_mean, inv_cov_xy, x_minus_mean))  # B, H, W, K
    xTsigma_new = (minus_mean[..., None] * inv_cov_xy[..., None, :, :]).sum(dim=-2)  # B, P, K, 2
    xTsigma_x = (xTsigma_new * minus_mean).sum(dim=-1)  # B, P, K
    return xTsigma_x



def gaussian_3d(scale3: jt.Var,  # B, P, 3, the scale of the 3d gaussian in 3 dimensions
                rot3: jt.Var,  # B, P, 3, the rotation of the 3D gaussian (angle-axis)
                R: jt.Var,  # B, 3, 3, camera rotation
                ):
    sigma0 = jt.diag(scale3)  # B, P, 3, 3
    rotmat = batch_rodrigues(rot3)  # B, P, 3, 3
    R_sigma = rotmat @ sigma0
    covariance = R @ R_sigma @ R_sigma.transpose(-2,-1) @ R.transpose(-2,-1)
    return covariance  # B, P, 3, 3



def inverse_sigmoid(x):
    return jt.log(x / (1 - x))



def strip_lowerdiag(L: jt.Var):
    # uncertainty = jt.zeros((L.shape[0], 6), dtype=L.dtype, device=L.device)

    # uncertainty[:, 0] = L[:, 0, 0].clip(0.0)  # sanitize covariance matrix
    # uncertainty[:, 1] = L[:, 0, 1]
    # uncertainty[:, 2] = L[:, 0, 2]
    # uncertainty[:, 3] = L[:, 1, 1].clip(0.0)  # sanitize covariance matrix
    # uncertainty[:, 4] = L[:, 1, 2]
    # uncertainty[:, 5] = L[:, 2, 2].clip(0.0)  # sanitize covariance matrix
    # return uncertainty

    inds = jt.Var(np.triu_indices(3, 3)) # 2, 6
    return L[:, inds[0], inds[1]]


def strip_symmetric(sym):
    return strip_lowerdiag(sym)



def build_rotation(q: jt.Var):
    assert q.shape[-1] == 4
    # norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])

    # q = r / norm[:, None]
    q = nn.normalize(q, dim=-1)

    R = jt.zeros((q.size(0), 3, 3), dtype=q.dtype)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R



def build_scaling_rotation(s: jt.Var, q: jt.Var):
    L = jt.zeros((s.shape[0], 3, 3), dtype=s.dtype)
    R = build_rotation(q)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def fov2focal(fov, pixels):
    return pixels / (2 * np.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * np.arctan(pixels / (2 * focal))



def getWorld2View(R: jt.Var, t: jt.Var):
    """
    R: ..., 3, 3
    T: ..., 3, 1
    """
    sh = R.shape[:-2]
    T = jt.init.eye(4, dtype=R.dtype)  # 4, 4
    for i in range(len(sh)):
        T = T.unsqueeze(0)
    T = T.expand(sh + (4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3:] = t
    return T



def getProjectionMatrix(K: jt.Var, H: jt.Var, W: jt.Var, znear: jt.Var, zfar: jt.Var):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    s = K[0, 1]
    one = K[2, 2]

    P = jt.zeros(4, 4, dtype=K.dtype)

    P[0, 0] = 2 * fx / W
    P[0, 1] = 2 * s / W
    P[0, 2] = -1 + 2 * (cx / W)

    P[1, 1] = 2 * fy / H
    P[1, 2] = -1 + 2 * (cy / H)

    P[2, 2] = -(zfar + znear) / (znear - zfar)
    P[2, 3] = 2 * zfar * znear / (znear - zfar)

    P[3, 2] = one

    return P


def prepare_gaussian_camera(batch):
    output = dotdict()
    H, W, K, R, T, n, f = batch.H[0], batch.W[0], batch.K[0], batch.R[0], batch.T[0], batch.n[0], batch.f[0]
    cpu_H, cpu_W, cpu_K, cpu_R, cpu_T, cpu_n, cpu_f = batch.meta.H[0], batch.meta.W[0], batch.meta.K[0], batch.meta.R[0], batch.meta.T[0], batch.meta.n[0], batch.meta.f[0]

    output.image_height = cpu_H
    output.image_width = cpu_W

    output.K = K
    output.R = R
    output.T = T

    fl_x = batch.meta.K[0][0, 0]  # use cpu K
    fl_y = batch.meta.K[0][1, 1]  # use cpu K

    output.FoVx = focal2fov(fl_x, cpu_W)
    output.FoVy = focal2fov(fl_y, cpu_H)

    output.world_view_transform = getWorld2View(R, T).transpose(0, 1)
    output.projection_matrix = getProjectionMatrix(K, H, W, n, f).transpose(0, 1)
    output.full_proj_transform = jt.matmul(output.world_view_transform, output.projection_matrix)
    output.camera_center = (-R.transpose(-2,-1) @ T)[..., 0]  # B, 3, 1 -> 3,

    # Set up rasterization configuration
    output.tanfovx = math.tan(output.FoVx * 0.5)
    output.tanfovy = math.tan(output.FoVy * 0.5)

    return output


def convert_to_gaussian_camera(K: jt.Var,
                               R: jt.Var,
                               T: jt.Var,
                               H: jt.Var,
                               W: jt.Var,
                               n: jt.Var,
                               f: jt.Var,
                               cpu_K: jt.Var,
                               cpu_R: jt.Var,
                               cpu_T: jt.Var,
                               cpu_H: int,
                               cpu_W: int,
                               cpu_n: float = 0.01,
                               cpu_f: float = 100.,
                               ):
    output = dotdict()

    output.image_height = cpu_H
    output.image_width = cpu_W

    output.K = K
    output.R = R
    output.T = T

    output.znear = cpu_n
    output.zfar = cpu_f

    output.FoVx = focal2fov(cpu_K[0, 0].cpu(), cpu_W.cpu())  # MARK: MIGHT SYNC IN DIST TRAINING, WHY?
    output.FoVy = focal2fov(cpu_K[1, 1].cpu(), cpu_H.cpu())  # MARK: MIGHT SYNC IN DIST TRAINING, WHY?

    # Use .float() to avoid AMP issues
    output.world_view_transform = getWorld2View(R, T).transpose(0, 1).float()  # this is now to be right multiplied
    output.projection_matrix = getProjectionMatrix(K, H, W, n, f).transpose(0, 1).float()  # this is now to be right multiplied
    output.full_proj_transform = jt.matmul(output.world_view_transform, output.projection_matrix).float()   # 4, 4
    output.camera_center = (-R.transpose(-2,-1) @ T)[..., 0].float()  # B, 3, 1 -> 3,

    # Set up rasterization configuration
    output.tanfovx = np.tan(output.FoVx * 0.5)
    output.tanfovy = np.tan(output.FoVy * 0.5)

    return output


class GaussianModel(nn.Module):
    def __init__(self,
                 xyz: jt.Var = None,
                 colors: jt.Var = None,
                 init_occ: float = 0.1,
                 init_scale: jt.Var = None,
                 sh_deg: int = 3,
                 scale_min: float = 1e-4,
                 scale_max: float = 1e1,
                 ):
        super().__init__()


        def scaling_activation(x, scale_min: float = scale_min, scale_max: float = scale_max):
            return jt.sigmoid(x) * (scale_max - scale_min) + scale_min


        def scaling_inverse_activation(x, scale_min: float = scale_min, scale_max: float = scale_max):
            return jt.logit(((x - scale_min) / (scale_max - scale_min)).clamp(1e-5, 1 - 1e-5))

        self.setup_functions(scaling_activation=scaling_activation, scaling_inverse_activation=scaling_inverse_activation)

        # SH realte configs
        self.active_sh_degree = make_buffer(jt.zeros(1))
        self.max_sh_degree = sh_deg

        # Initalize trainable parameters
        self.create_from_pcd(xyz, colors, init_occ, init_scale)

        # Densification related parameters
        self.max_radii2D = make_buffer(jt.zeros(self.get_xyz.shape[0]))
        self.xyz_gradient_accum = make_buffer(jt.zeros((self.get_xyz.shape[0], 1)))
        self.denom = make_buffer(jt.zeros((self.get_xyz.shape[0], 1)))

        # Perform some model messaging before loading
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def setup_functions(self,
                        scaling_activation=jt.exp,
                        scaling_inverse_activation=jt.log,
                        opacity_activation=jt.sigmoid,
                        inverse_opacity_activation=inverse_sigmoid,
                        rotation_activation=jt.normalize,
                        ):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = getattr(jt, scaling_activation) if isinstance(scaling_activation, str) else scaling_activation
        self.opacity_activation = getattr(tojtrch, opacity_activation) if isinstance(opacity_activation, str) else opacity_activation
        self.rotation_activation = getattr(jt, rotation_activation) if isinstance(rotation_activation, str) else rotation_activation

        self.scaling_inverse_activation = getattr(jt, scaling_inverse_activation) if isinstance(scaling_inverse_activation, str) else scaling_inverse_activation
        self.opacity_inverse_activation = getattr(jt, inverse_opacity_activation) if isinstance(inverse_opacity_activation, str) else inverse_opacity_activation
        self.covariance_activation = build_covariance_from_scaling_rotation

    @property
    def device(self):
        return self.get_xyz.device

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return jt.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, xyz: jt.Var, colors: jt.Var = None, opacities: float = 0.1, scales: jt.Var = None):
        from simple_knn._C import distCUDA2
        if xyz is None:
            xyz = jt.empty(0, 3, device='cuda')  # by default, init empty gaussian model on CUDA

        features = jt.zeros((xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2))
        if colors is not None:
            SH = rgb2sh0(colors)
            features[:, :3, 0] = SH
        features[:, 3: 1:] = 0

        if scales is None:
            dist2 = jt.clamp(distCUDA2(xyz.float().cuda()), min=0.0000001)
            scales = self.scaling_inverse_activation(jt.sqrt(dist2))[..., None].repeat(1, 3)
        else:
            scales = self.scaling_inverse_activation(scales)

        rots = jt.rand((xyz.shape[0], 4))
        rots[:, 0] = 1

        if not isinstance(opacities, jt.Var) or len(opacities) != len(xyz):
            opacities = opacities * jt.ones((xyz.shape[0], 1), dtype=jt.float)
        opacities = self.opacity_inverse_activation(opacities)

        self._xyz = make_params(xyz)
        self._features_dc = make_params(features[:, :, :1].transpose(1, 2).contiguous())
        self._features_rest = make_params(features[:, :, 1:].transpose(1, 2).contiguous())
        self._scaling = make_params(scales)
        self._rotation = make_params(rots)
        self._opacity = make_params(opacities)

    @jt.no_grad()
    def _load_state_dict_pre_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Supports loading points and features with different shapes
        if prefix is not '' and not prefix.endswith('.'): prefix = prefix + '.'  # special care for when we're loading the model directly
        for name, params in self.named_parameters():
            if f'{prefix}{name}' in state_dict:
                params.data = params.data.new_empty(state_dict[f'{prefix}{name}'].shape)

    def reset_opacity(self, optimizer_state):
        for _, val in optimizer_state.items():
            if val.name == '_opacity':
                break
        opacities_new = inverse_sigmoid(jt.min(self.get_opacity, jt.ones_like(self.get_opacity) * 0.01))
        self._opacity.set_(opacities_new.detach())
        self._opacity.grad = None
        val.old_keep = jt.zeros_like(val.old_keep, dtype=jt.bool)
        val.new_keep = jt.zeros_like(val.new_keep, dtype=jt.bool)
        val.new_params = self._opacity
        # optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        # self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = jt.zeros_like(tensor)
                stored_state["exp_avg_sq"] = jt.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask: jt.Var):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = mask.logical_not()
        # optimizable_tensors = self._prune_optimizer(valid_points_mask)

        # self._xyz = optimizable_tensors["xyz"]
        # self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        # self._opacity = optimizable_tensors["opacity"]
        # self._scaling = optimizable_tensors["scaling"]
        # self._rotation = optimizable_tensors["rotation"]

        self._xyz.set_(self._xyz[valid_points_mask].detach())
        self._xyz.grad = None
        self._features_dc.set_(self._features_dc[valid_points_mask].detach())
        self._features_dc.grad = None
        self._features_rest.set_(self._features_rest[valid_points_mask].detach())
        self._features_rest.grad = None
        self._opacity.set_(self._opacity[valid_points_mask].detach())
        self._opacity.grad = None
        self._scaling.set_(self._scaling[valid_points_mask].detach())
        self._scaling.grad = None
        self._rotation.set_(self._rotation[valid_points_mask].detach())
        self._rotation.grad = None

        self.xyz_gradient_accum.set_(self.xyz_gradient_accum[valid_points_mask])
        self.xyz_gradient_accum.grad = None
        self.denom.set_(self.denom[valid_points_mask])
        self.denom.grad = None
        self.max_radii2D.set_(self.max_radii2D[valid_points_mask])
        self.max_radii2D.grad = None

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = jt.cat((stored_state["exp_avg"], jt.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = jt.cat((stored_state["exp_avg_sq"], jt.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(jt.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(jt.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, optimizer_state):
        d = dotdict({
            "_xyz": new_xyz,
            "_features_dc": new_features_dc,
            "_features_rest": new_features_rest,
            "_opacity": new_opacities,
            "_scaling": new_scaling,
            "_rotation": new_rotation,
        })

        # optimizable_tensors = self.cat_tensors_to_optimizer(d)
        # self._xyz = optimizable_tensors["xyz"]
        # self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        # self._opacity = optimizable_tensors["opacity"]
        # self._scaling = optimizable_tensors["scaling"]
        # self._rotation = optimizable_tensors["rotation"]

        for name, new_params in d.items():
            params: nn.Parameter = getattr(self, name)
            params.set_(jt.cat((params.data, new_params), dim=0).detach())
            params.grad = None

        device = self.get_xyz.device
        self.xyz_gradient_accum.set_(jt.zeros((self.get_xyz.shape[0], 1), device=device))
        self.xyz_gradient_accum.grad = None
        self.denom.set_(jt.zeros((self.get_xyz.shape[0], 1), device=device))
        self.denom.grad = None
        self.max_radii2D.set_(jt.zeros((self.get_xyz.shape[0]), device=device))
        self.max_radii2D.grad = None

        for val in optimizer_state.values():
            name = val.name
            val.new_keep = jt.cat((val.new_keep, jt.zeros_like(d[name], dtype=jt.bool, requires_grad=False)), dim=0)
            val.new_params = getattr(self, name)
            assert val.new_keep.shape == val.new_params.shape

    def densify_and_split(self, grads, grad_threshold, scene_extent, percent_dense, min_opacity, max_screen_size, optimizer_state, N=2):
        n_init_points = self.get_xyz.shape[0]
        device = self.get_xyz.device
        # Extract points that satisfy the gradient condition
        padded_grad = jt.zeros((n_init_points), device=device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = padded_grad >= grad_threshold
        selected_pts_mask = jt.logical_and(selected_pts_mask,
                                              jt.max(self.get_scaling, dim=1).values > percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = jt.zeros((stds.size(0), 3), device=device)
        samples = jt.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = jt.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, optimizer_state)

        prune_filter = jt.cat((selected_pts_mask, jt.zeros(N * selected_pts_mask.sum(), device=device, dtype=bool)))
        self.prune_points(prune_filter)
        old_keep_mask = prune_filter[:grads.shape[0]].logical_not()
        for val in optimizer_state.values():
            name = val.name
            val.old_keep[old_keep_mask.logical_not()] = False
            val.new_keep = val.new_keep[prune_filter.logical_not()]
            val.params = getattr(self, name)
            assert val.old_keep.sum() == val.new_keep.sum()
            assert val.new_keep.shape == val.new_params.shape

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * scene_extent
            prune_mask = jt.logical_or(jt.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        _old_keep_mask = old_keep_mask.clone()
        mask_mask = old_keep_mask[old_keep_mask]
        _mask = prune_mask[:mask_mask.shape[0]]
        mask_mask[_mask] = False
        old_keep_mask[_old_keep_mask] = mask_mask
        for val in optimizer_state.values():
            name = val.name
            val.old_keep[old_keep_mask.logical_not()] = False
            val.new_keep = val.new_keep[prune_mask.logical_not()]
            val.params = getattr(self, name)
            assert val.old_keep.sum() == val.new_keep.sum()
            assert val.new_keep.shape == val.new_params.shape

    def densify_and_clone(self, grads, grad_threshold, scene_extent, percent_dense, optimizer_state):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = jt.norm(grads, dim=-1) >= grad_threshold
        selected_pts_mask = jt.logical_and(selected_pts_mask,
                                              jt.max(self.get_scaling, dim=1).values <= percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, optimizer_state)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, percent_dense, optimizer_state):

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent, percent_dense, optimizer_state)
        self.densify_and_split(grads, max_grad, extent, percent_dense, min_opacity, max_screen_size, optimizer_state)

        #torch.cuda.empty_cache()#?????

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += jt.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        from plyfile import PlyData, PlyElement
        os.makedirs(dirname(path), exist_ok=True)

        # The original gaussian model uses a different activation
        # Normalization for rotation, so no conversion
        # Exp on scaling, need to -> world space -> log

        # Doing inverse_sigmoid here will lead to NaNs
        _opacity = self._opacity
        if self.opacity_activation != F.sigmoid and \
                self.opacity_activation != jt.sigmoid and \
                not isinstance(self.opacity_activation, nn.Sigmoid):
            opacity = self.opacity_activation(opacity)
            _opacity = inverse_sigmoid(opacity)

        scale = self._scaling
        scale = self.scaling_activation(scale)
        _scale = jt.log(scale)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = _opacity.detach().cpu().numpy()
        scale = _scale.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path: str):
        xyz, _, _, scalars = load_pts(path)

        # The original gaussian model uses a different activation
        xyz = jt.array(xyz)
        rotation = jt.array(np.concatenate([scalars['rot_{}'.format(i)] for i in range(4)], axis=-1))
        scaling = jt.array(np.concatenate([scalars['scale_{}'.format(i)] for i in range(3)], axis=-1))
        scaling = jt.exp(scaling)
        scaling = self.scaling_inverse_activation(scaling)
        opacity = jt.array(scalars['opacity'])

        # Doing inverse_sigmoid here will lead to NaNs
        if self.opacity_activation != F.sigmoid and \
                self.opacity_activation != jt.sigmoid and \
                not isinstance(self.opacity_activation, nn.Sigmoid):
            opacity = inverse_sigmoid(opacity)
            opacity = self.opacity_inverse_activation(opacity)

        # Load the SH colors
        features_dc = jt.empty((xyz.shape[0], 3, 1))
        features_dc[:, 0] = jt.array(np.asarray(scalars["f_dc_0"]))
        features_dc[:, 1] = jt.array(np.asarray(scalars["f_dc_1"]))
        features_dc[:, 2] = jt.array(np.asarray(scalars["f_dc_2"]))

        extra_f_names = [k for k in scalars.keys() if k.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))

        # Load max_sh_degree from file
        for i in range(4):
            if len(extra_f_names) == 3 * (i + 1) ** 2 - 3:
                self.max_sh_degree = i
                # assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
                features_rest = jt.zeros((xyz.shape[0], len(extra_f_names), 1))

                for idx, attr_name in enumerate(extra_f_names):
                    features_rest[:, idx] = jt.array(np.asarray(scalars[attr_name]))

                # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
                features_rest = features_rest.view(features_rest.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)

        state_dict = dotdict()
        state_dict._xyz = xyz
        state_dict._features_dc = features_dc.transpose(-2,-1)
        state_dict._features_rest = features_rest.transpose(-2,-1)
        state_dict._opacity = opacity
        state_dict._scaling = scaling
        state_dict._rotation = rotation

        self.load_state_dict(state_dict, strict=False)
        self.active_sh_degree.data.fill_(self.max_sh_degree)

    def render(self, batch: dotdict, scale_mult: float = 1.0, alpha_mult: float = 1.0):
        # TODO: Make rendering function easier to read, now there're at least 3 types of gaussian rendering function
        from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer

        # Prepare renderable parameters, without batch
        xyz = self.get_xyz
        scale3 = self.get_scaling
        rot4 = self.get_rotation
        occ = self.get_opacity
        sh = self.get_features

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
            bg=jt.full([3], 0.0, device=xyz.device),  # GPU # TODO: make these configurable
            scale_modifier=1.0,  # TODO: make these configurable
            viewmatrix=gaussian_camera.world_view_transform,
            projmatrix=gaussian_camera.full_proj_transform,
            sh_degree=self.active_sh_degree,
            campos=gaussian_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        scr = jt.zeros_like(xyz) + 0  # gradient magic
        if scr.requires_grad: scr.retain_grad()
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        rendered_image, rendered_depth, rendered_alpha, radii = typed(jt.float, jt.float)(rasterizer)(
            means3D=xyz,
            means2D=scr,
            shs=sh,
            colors_precomp=None,
            opacities=occ * alpha_mult,
            scales=scale3 * scale_mult,
            rotations=rot4,
            cov3D_precomp=None,
        )

        # No batch dimension
        rgb = rendered_image.permute(1, 2, 0)
        acc = rendered_alpha.permute(1, 2, 0)
        dpt = rendered_depth.permute(1, 2, 0)

        return rgb, acc, dpt  # H, W, C


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: jt.Var, scaling_modifier=1.0, override_color=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = jt.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device=pc.get_xyz.device) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = jt.clamp(sh2rgb + 0.5, min=0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, rendered_depth, rendered_alpha, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return dotdict({
        "render": rendered_image,
        "alpha": rendered_alpha,
        "depth": rendered_depth,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii
    })


def naive_render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: jt.Var, scaling_modifier=1.0, override_color=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = jt.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = jt.clamp(sh2rgb + 0.5, min=0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # rendered_image, radii, rendered_depth = rasterizer(
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=jt.zeros_like(bg_color),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )
    rasterizer.raster_settings = raster_settings

    colors_precomp = jt.ones_like(means3D, requires_grad=False).contiguous()
    rendered_alpha, _ = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    colors_precomp = nn.pad(means3D, (0, 1), value=1.0) @ viewpoint_camera.world_view_transform
    colors_precomp = jt.norm(colors_precomp[..., :3] - viewpoint_camera.camera_center, dim=-1, keepdim=True)
    colors_precomp = jt.repeat_interleave(colors_precomp, 3, dim=-1).contiguous()
    rendered_depth, _ = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return dotdict({
        "render": rendered_image[:3],
        "alpha": rendered_alpha[:1],
        "depth": rendered_depth[:1],
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii
    })


def construct_list_of_attributes(self: dotdict):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
        l.append('f_dc_{}'.format(i))
    for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(self._scaling.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(self._rotation.shape[1]):
        l.append('rot_{}'.format(i))
    return l


def save_gs(self: dotdict, path):
    from plyfile import PlyData, PlyElement
    os.makedirs(dirname(path), exist_ok=True)

    # The original gaussian model uses a different activation
    # Normalization for rotation, so no conversion
    # Exp on scaling, need to -> world space -> log

    # Doing inverse_sigmoid here will lead to NaNs
    _opacity = self._opacity

    _scale = self._scaling

    xyz = self._xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = _opacity.detach().cpu().numpy()
    scale = _scale.detach().cpu().numpy()
    rotation = self._rotation.detach().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(self)]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
