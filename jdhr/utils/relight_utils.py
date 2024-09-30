import cv2
#import torch
import jittor as jt
import numpy as np
from jittor import misc
from jittor import nn
from jdhr.utils.base_utils import dotdict
from jdhr.utils.ray_utils import get_rays
from jdhr.utils.math_utils import normalize
from jdhr.utils.image_utils import resize_image
import math

def gen_light_dir(H: int, W: int, R: jt.Var):
    # R, T in w2c
    # front direction is z of R
    R = R.clone()  # avoid modifying the original value
    R = R[0]  # !: BATCH
    R = R.transpose(-2,-1)  # w2c to c2w, stores camera axis

    # elimiate rotation on other axes, only retain horizontal rotation
    front = R[:, 2]  # 3,
    down = jt.zeros_like(R[:, 1])  # 3,
    down[2] = nn.sign(R[:, 1][2])  # 3, z indicates world up or down
    right = normalize(jt.cross(down, front))  # 3,
    front = normalize(jt.cross(right, down))
    R[:, 0] = right  # right
    R[:, 1] = down  # down, the probe has a different orientation, addressed in sample_envmap_image
    R[:, 2] = front  # front

    # flip the z, y axis, the probe has a different orientation
    R[:, 1], R[:, 2] = -R[:, 2].clone(), -R[:, 1].clone()

    # the ray directions are in world space (always)
    # gen_light_xyz will generate the ray directions with center at torch.eye as the transformation matrix
    # camera rotates the lookat axis, so we need to rotate the ray directions to align with the lookat direction
    # in the original probe ray_d, lookat should be 0, 0, 1, in the camera space, it is R[:, 2]
    ray_d = normalize(gen_light_xyz(H, W)[0])  # H, W, 3 (in camera space)
    ray_d = ray_d @ R.transpose(-2,-1)   # convert to world space
    return ray_d


def add_light_probe(rgb: jt.Var, probe: jt.Var, batch: dotdict, cfg):
    # preparing shape
    sh = rgb.shape
    B = rgb.shape[0]
    H, W = batch.meta.H.item(), batch.meta.W.item()
    eH, eW = cfg.env_h, cfg.env_w
    uW = int(W * cfg.probe_size_ratio)
    uH = int(uW * eH / eW)

    # the actual sampling
    ray_d = gen_light_dir(uH, uW, batch.cam_R)[None]  # !: BATCH
    rgb = rgb.view(B, H, W, 3)
    rgb[:, :uH, :uW, :] = sample_envmap_image(probe, ray_d)
    rgb = rgb.view(sh)
    return rgb


def rotate_envmap(novel_light: dotdict[str, dotdict], index, repeat, probe_width, image_width):
    keys = list(novel_light.keys())
    if repeat <= 0:
        return keys[index], novel_light[keys[index]]

    n_rotation = probe_width * repeat  # downscale the probe
    n_light = len(novel_light)

    i = index // n_rotation
    j = index % n_rotation
    name = f'{keys[i]}-{j:04d}'

    envmap = novel_light[keys[i]]
    probe: jt.Var = envmap.probe  # eH, eW, 3
    image: jt.Var = envmap.image  # iH, iW, 3

    eH, eW = probe.shape[-3:-1]
    iH, iW = image.shape[-3:-1]
    uH, uW = eH * repeat, eW * repeat

    probe_shift = eW / uW * j
    image_shift = iW / uW * j

    def shift_image(image: jt.Var, shift: float):
        # use bilinear interpolation to shift image in a float fasion
        # for now, this function only supports image with a batch size
        B, H, W = image.shape[:3]
        i, j = jt.meshgrid(jt.arange(0, H), jt.arange(0, W))
        grid = jt.stack([j, i], dim=-1)  # H, W, 2
        grid = grid[None].expand(B, H, W, 2)  # B, H, W, 2
        grid = grid.float()  # grid sample expects float input
        grid = grid + 0.5  # pixel center
        grid[..., 0] = grid[..., 0] + shift  # shift is defined in pixel space, can be a float
        grid[..., 0] = grid[..., 0] % W  # wrap around
        grid[..., 0] = grid[..., 0] / W * 2 - 1  # avoid using squeeze
        grid[..., 1] = grid[..., 1] / H * 2 - 1  # MARK: gradient
        # strange resampling issues....
        sampled = nn.grid_sample(image.permute(0, 3, 1, 2), grid, align_corners=False, mode='bilinear', padding_mode='border').permute(0, 2, 3, 1)  # B, H, W, 3
        return sampled

    # upscaled_probe = resize_image(probe, uH, uW, 'bilinear')  # uH, uW, 3, for olat, use the correct rotate
    # # upscaled_probe = resize_image(probe, uH, uW, 'area')  # uH, uW, 3, for olat, use the correct rotate
    # # upscaled_probe = F.interpolate(probe.permute(0, 3, 1, 2), size=(uH, uW), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)  # uH, uW, 3
    # rotated_probe = torch.roll(upscaled_probe, shifts=j, dims=-2)  # uH, uW, 3
    # rotated_probe = resize_image(rotated_probe, eH, eW, 'area')  # uH, uW, 3
    # # rotated_probe = F.interpolate(rotated_probe.permute(0, 3, 1, 2), size=(eH, eW), mode='area').permute(0, 2, 3, 1)  # eH, eW, 3
    # rotated_image = torch.roll(image, shifts=int(iW / uW * j), dims=-2)  # iH, iW, 3

    return name, dotdict(probe=shift_image(probe, probe_shift), image=shift_image(image, image_shift))


def sample_envmap_image(image: jt.Var, ray_d: jt.Var):
    sh = ray_d.shape
    if image.ndim == 4:
        image = image[0]
    ray_d = ray_d.view(-1, 3)
    # envmap: H, W, C
    # viewdirs: N, 3

    # https://github.com/zju3dv/InvRender/blob/45e6cdc5e3c9f092b5d10e2904bbf3302152bb2f/code/model/sg_render.py
    image = image.permute(2, 0, 1).unsqueeze(0)

    theta = jt.arccos(ray_d[:, 2]).reshape(-1) - 1e-6
    phi = jt.arctan2(ray_d[:, 1], ray_d[:, 0]).reshape(-1)  # 0 - pi

    # normalize to [-1, 1]
    query_y = (theta / math.pi) * 2 - 1
    query_x = - phi / math.pi
    grid = jt.stack((query_x, query_y)).permute(1, 0).unsqueeze(0).unsqueeze(0)

    rgb = nn.grid_sample(image, grid, align_corners=False, padding_mode='border')
    rgb = rgb.squeeze().permute(1, 0)
    return rgb.view(sh)


def reflect(ray_d: jt.Var, norm: jt.Var):
    dot = (ray_d * norm).sum(dim=-1, keepdim=True)
    return 2 * (norm * dot) - ray_d


def read_hdr(path):
    # TODO: will this support openexr? could not find valid openexr python binding
    # TODO: implement saving in hdr format
    """Reads an HDR map from disk.

    Args:
        path (str): Path to the .hdr file.

    Returns:
        numpy.ndarray: Loaded (float) HDR map with RGB channels in order.
    """
    with open(path, 'rb') as h:
        buffer_ = np.fromstring(h.read(), np.uint8)
    bgr = cv2.imdecode(buffer_, cv2.IMREAD_UNCHANGED)
    # bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32)


def area_hot_img(h, w, c, i, j):
    """Makes a float32 HxWxC tensor with 1s at (i, j, *) and 0s everywhere else.
    """
    one_hot = np.zeros((h, w, c), dtype=np.float32)
    # one_hot[i - 1:i + 2, j - 1:j + 2, :] = 1 # MARK: assumption
    one_hot[i, j, :] = 1
    return one_hot


def expand_envmap_probe(envmap: jt.Var, x: jt.Var):  # B, eH, eW, P, 3
    B, P, _ = x.shape
    if envmap.ndim == 3:
        envmap = envmap[None]  # add batch dimension if needed
    return envmap[:, :, :, None, :].relu().expand(B, *envmap.shape[1:3], P, 3)  # should be in linear, no negative light

    # relu will make expand contiguous, consumes lots of memory


def linear2srgb(linear: jt.Var):
    srgb_linear_thres = 0.0031308
    srgb_linear_coeff = 12.92
    srgb_exponential_coeff = 1.055
    srgb_exponent = 2.4

    linear = linear.clamp(0., 1.)
    tensor_linear = linear * srgb_linear_coeff
    tensor_nonlinear = srgb_exponential_coeff * (jt.pow(linear + 1e-7, 1 / srgb_exponent)) - (srgb_exponential_coeff - 1)

    is_linear = linear <= srgb_linear_thres
    tensor_srgb = jt.ternary(is_linear, tensor_linear, tensor_nonlinear)

    return tensor_srgb


def srgb2linear(srgb: jt.Var):
    srgb_linear_thres = 0.0031308
    linear_srgb_thres = 0.04045
    srgb_linear_coeff = 12.92
    srgb_exponential_coeff = 1.055
    srgb_exponent = 2.4

    srgb = srgb.clamp(0, 1)
    tensor_linear = srgb / srgb_linear_coeff
    tensor_nonlinear = ((srgb * srgb_exponential_coeff) / srgb_exponential_coeff) ** srgb_exponent

    is_linear = srgb <= linear_srgb_thres
    tensor_linear = jt.ternary(is_linear, tensor_linear, tensor_nonlinear)
    return tensor_linear


def uniform_sample_sph(n, r=1, convention='lat-lng'):
    r"""Uniformly samples points on the sphere
    [`source <https://mathworld.wolfram.com/SpherePointPicking.html>`_].

    Args:
        n (int): Total number of points to sample. Must be a square number.
        r (float, optional): Radius of the sphere. Defaults to :math:`1`.
        convention (str, optional): Convention for spherical coordinates.
            See :func:`cart2sph` for conventions.

    Returns:
        numpy.ndarray: Spherical coordinates :math:`(r, \theta_1, \theta_2)`
        in radians. The points are ordered such that all azimuths are looped
        through first at each elevation.
    """
    n_ = jt.sqrt(n)
    if n_ != int(n_):
        raise ValueError("%d is not perfect square" % n)
    n_ = int(n_)

    pts_r_theta_phi = []
    for u in jt.linspace(0, 1, n_):
        for v in jt.linspace(0, 1, n_):
            theta = jt.arccos(2 * u - 1)  # [0, pi]
            phi = 2 * math.pi * v  # [0, 2pi]
            pts_r_theta_phi.append((r, theta, phi))
    pts_r_theta_phi = jt.Var(np.vstack(pts_r_theta_phi.numpy()))

    # Select output convention
    if convention == 'lat-lng':
        pts_sph = _convert_sph_conventions(
            pts_r_theta_phi, 'theta-phi_to_lat-lng')
    elif convention == 'theta-phi':
        pts_sph = pts_r_theta_phi
    else:
        raise NotImplementedError(convention)

    return pts_sph


def cart2sph(pts_cart, convention='lat-lng'):
    r"""Converts 3D Cartesian coordinates to spherical coordinates.

    Args:
        pts_cart (array_like): Cartesian :math:`x`, :math:`y` and
            :math:`z`. Of shape N-by-3 or length 3 if just one point.
        convention (str, optional): Convention for spherical coordinates:
            ``'lat-lng'`` or ``'theta-phi'``:

            .. code-block:: none

                   lat-lng
                                            ^ z (lat = 90)
                                            |
                                            |
                       (lng = -90) ---------+---------> y (lng = 90)
                                          ,'|
                                        ,'  |
                   (lat = 0, lng = 0) x     | (lat = -90)

            .. code-block:: none

                theta-phi
                                            ^ z (theta = 0)
                                            |
                                            |
                       (phi = 270) ---------+---------> y (phi = 90)
                                          ,'|
                                        ,'  |
                (theta = 90, phi = 0) x     | (theta = 180)

    Returns:
        numpy.ndarray: Spherical coordinates :math:`(r, \theta_1, \theta_2)`
        in radians.
    """
    pts_cart = jt.array(pts_cart)

    # Validate itorch.ts
    is_one_point = False
    if pts_cart.shape == (3,):
        is_one_point = True
        pts_cart = pts_cart.reshape(1, 3)
    elif pts_cart.ndim != 2 or pts_cart.shape[1] != 3:
        raise ValueError("Shape of itorch.t must be either (3,) or (n, 3)")

    # Compute r
    r = jt.sqrt(jt.sum(jt.Var(np.square(pts_cart)), dim=1))

    # Compute latitude
    z = pts_cart[:, 2]
    lat = jt.arcsin(z / r)

    # Compute longitude
    x = pts_cart[:, 0]
    y = pts_cart[:, 1]
    lng = jt.arctan2(y, x)  # choosing the quadrant correctly

    # Assemble
    pts_r_lat_lng = jt.stack((r, lat, lng), dim=-1)

    # Select output convention
    if convention == 'lat-lng':
        pts_sph = pts_r_lat_lng
    elif convention == 'theta-phi':
        pts_sph = _convert_sph_conventions(
            pts_r_lat_lng, 'lat-lng_to_theta-phi')
    else:
        raise NotImplementedError(convention)

    if is_one_point:
        pts_sph = pts_sph.reshape(3)

    return pts_sph


def _convert_sph_conventions(pts_r_angle1_angle2, what2what):
    """Internal function converting between different conventions for
    spherical coordinates. See :func:`cart2sph` for conventions.
    """
    if what2what == 'lat-lng_to_theta-phi':
        pts_r_theta_phi = jt.zeros(pts_r_angle1_angle2.shape)
        # Radius is the same
        pts_r_theta_phi[:, 0] = pts_r_angle1_angle2[:, 0]
        # Angle 1
        pts_r_theta_phi[:, 1] = math.pi / 2 - pts_r_angle1_angle2[:, 1]
        # Angle 2
        ind = pts_r_angle1_angle2[:, 2] < 0
        pts_r_theta_phi[ind, 2] = 2 * math.pi + pts_r_angle1_angle2[ind, 2]
        pts_r_theta_phi[jt.logical_not(ind), 2] = \
            pts_r_angle1_angle2[jt.logical_not(ind), 2]
        return pts_r_theta_phi

    if what2what == 'theta-phi_to_lat-lng':
        pts_r_lat_lng = jt.zeros(pts_r_angle1_angle2.shape)
        # Radius is the same
        pts_r_lat_lng[:, 0] = pts_r_angle1_angle2[:, 0]
        # Angle 1
        pts_r_lat_lng[:, 1] = math.pi / 2 - pts_r_angle1_angle2[:, 1]
        # Angle 2
        ind = pts_r_angle1_angle2[:, 2] > math.pi
        pts_r_lat_lng[ind, 2] = pts_r_angle1_angle2[ind, 2] - 2 * math.pi
        pts_r_lat_lng[jt.logical_not(ind), 2] = \
            pts_r_angle1_angle2[jt.logical_not(ind), 2]
        return pts_r_lat_lng

    raise NotImplementedError(what2what)


def sph2cart(pts_sph: jt.Var, convention='lat-lng') -> jt.Var:
    """Inverse of :func:`cart2sph`.

    See :func:`cart2sph`.
    """
    # Validate itorch.ts
    is_one_point = False
    if pts_sph.shape == (3,):
        is_one_point = True
        pts_sph = pts_sph.reshape(1, 3)
    elif pts_sph.ndim != 2 or pts_sph.shape[1] != 3:
        raise ValueError("Shape of itorch.t must be either (3,) or (n, 3)")

    # Convert to latitude-longitude convention, if necessary
    if convention == 'lat-lng':
        pts_r_lat_lng = pts_sph
    elif convention == 'theta-phi':
        pts_r_lat_lng = _convert_sph_conventions(
            pts_sph, 'theta-phi_to_lat-lng')
    else:
        raise NotImplementedError(convention)

    # Compute x, y and z
    r = pts_r_lat_lng[:, 0]
    lat = pts_r_lat_lng[:, 1]
    lng = pts_r_lat_lng[:, 2]
    z = r * jt.sin(lat)
    x = r * jt.cos(lat) * jt.cos(lng)
    y = r * jt.cos(lat) * jt.sin(lng)

    # Assemble and return
    pts_cart = jt.stack((x, y, z), dim=-1)

    if is_one_point:
        pts_cart = pts_cart.reshape(3)

    return pts_cart


def spher2cart(theta, phi):
    """Convert spherical coordinates into Cartesian coordinates (radius 1)."""
    r = jt.sin(theta)
    x = r * jt.cos(phi)
    y = r * jt.sin(phi)
    z = jt.cos(theta)
    return jt.stack([x, y, z], dim=-1)


def gen_uniform_light_xyz(envmap_h, envmap_w, envmap_r=1e2, device="cuda"):
    # See: https://www.bogotobogo.com/Algorithms/uniform_distribution_sphere.php
    theta_half = 1 / envmap_h / 2
    phi_half = 1 / envmap_w / 2
    theta = jt.acos(2.0 * jt.linspace(0 + theta_half, 1 - theta_half, envmap_h) - 1.0)
    phi = 2.0 * math.pi * jt.linspace(0 + phi_half, 1 - phi_half, envmap_w)
    theta, phi = jt.meshgrid(theta, phi)

    ray_d = spher2cart(theta, phi)  # T, 3, z always bigger than zero
    xyz = ray_d * envmap_r
    area = 4 * math.pi / (envmap_h * envmap_w)
    area = jt.full(xyz.shape[:2], area)  # should be uniform

    return xyz, area


def gen_light_xyz(envmap_h, envmap_w, envmap_r=1e2, device='cuda'):
    """Additionally returns the associated solid angles, for integration.
    """
    # OpenEXR "latlong" format
    # lat = pi/2
    # lng = pi
    #     +--------------------+
    #     |                    |
    #     |                    |
    #     +--------------------+
    #                      lat = -pi/2
    #                      lng = -pi
    # theta = 0
    # phi = pi
    #     +--------------------+
    #     |                    |
    #     |                    |
    #     +--------------------+
    #                      theta = pi
    #                      phi = -pi
    # lat_step_size = math.pi / (envmap_h + 2) # 0.5 in pixel coordinates
    # lng_step_size = 2 * math.pi / (envmap_w + 2) # 0.5 in pixel coordinates
    lat_half = math.pi / envmap_h / 2  # 0.5 in pixel coordinates
    lng_half = 2 * math.pi / envmap_w / 2  # 0.5 in pixel coordinates
    # Try to exclude the problematic polar points (linspace includes endpoints)
    lats = jt.linspace(math.pi / 2 - lat_half, -math.pi / 2 + lat_half, envmap_h)
    lngs = jt.linspace(math.pi - lng_half, -math.pi + lng_half, envmap_w)
    lngs, lats = jt.meshgrid(lngs, lats)

    # To Cartesian
    rlatlngs = jt.Var(np.dstack((envmap_r * jt.ones_like(lats), lats, lngs).numpy()))
    rlatlngs = rlatlngs.reshape(-1, 3)
    xyz = sph2cart(rlatlngs)
    xyz = xyz.reshape(envmap_h, envmap_w, 3)

    # Calculate the area of each pixel on the unit sphere (useful for
    # integration over the sphere)
    sin_colat = jt.sin(math.pi / 2 - lats)
    areas = 4 * math.pi * sin_colat / jt.sum(sin_colat)

    assert 0 not in areas, "There shouldn't be light pixel that doesn't contribute"

    return xyz, areas


class Microfacet:
    """As described in:
        Microfacet Models for Refraction through Rough Surfaces [EGSR '07]
    """

    def __init__(self,
                 default_rough=0.1, default_albedo=0.8, f0=0.04,
                 lambert_only=False, glossy_only=False):
        self.default_albedo = default_albedo
        self.default_rough = default_rough
        self.f0 = f0

        self.lambert_only = lambert_only
        self.glossy_only = glossy_only

    def __call__(self,
                 pts2l: jt.Var,
                 pts2c: jt.Var,
                 normal: jt.Var,
                 albedo: jt.Var = None,
                 rough: jt.Var = None,
                 fresnel: jt.Var = None,
                 ):
        """All in the world coordinates.

        Too low roughness is OK in the forward pass, but may be numerically
        unstable in the backward pass

        pts2l: NxLx3
        pts2c: Nx3
        normal: Nx3
        albedo: Nx3
        rough: Nx1
        """
        if not isinstance(albedo, jt.Var):  # scalar input
            albedo = jt.full((*pts2c.shape[:-1], 3), albedo if albedo is not None else self.default_albedo, device=normal.device)
        if not isinstance(rough, jt.Var):  # scalar input
            rough = jt.full((*pts2c.shape[:-1], 1), rough if rough is not None else self.default_rough, device=normal.device)

        sh = pts2l.shape
        if len(sh) == 5:
            B, eH, eW, P, C = sh
            pts2l = pts2l.reshape(B, eH * eW, P, 3).permute(0, 2, 1, 3).reshape(B * P, eH * eW, 3)
            pts2c = pts2c.reshape(B * P, 3)
            normal = normal.reshape(B * P, 3)
            albedo = albedo.reshape(B * P, 3)
            rough = rough.reshape(B * P, 1)
        elif len(sh) == 3:
            B, P, C = sh
            pts2l = pts2l.reshape(B * P, 1, 3)  # assume only one light
            pts2c = pts2c.reshape(B * P, 3)
            normal = normal.reshape(B * P, 3)
            albedo = albedo.reshape(B * P, 3)
            rough = rough.reshape(B * P, 1)
        elif len(sh) == 2:
            P, C = sh
            pts2l = pts2l.reshape(P, 1, 3)  # assume only one light
            pts2c = pts2c.reshape(P, 3)
            normal = normal.reshape(P, 3)
            albedo = albedo.reshape(P, 3)
            rough = rough.reshape(P, 1)
        else:
            pass  # whatever, might just error out

        # Normalize directions and normals
        pts2l = misc.normalize(pts2l, p=2, dim=-1, eps=1e-7)
        pts2c = misc.normalize(pts2c, p=2, dim=-1, eps=1e-7)
        normal = misc.normalize(normal, p=2, dim=-1, eps=1e-7)

        # Glossy
        h = pts2l + pts2c[:, None, :]  # NxLx3
        h = misc.normalize(h, p=2, dim=-1, eps=1e-7)
        f = self._get_f(pts2l, h)  # NxL, fresnel term (Schlick's approx)
        alpha = rough ** 2
        d = self._get_d(h, normal, alpha=alpha)  # NxL, normal distribution
        g = self._get_g(pts2c, h, normal, alpha=alpha)  # NxL, shadow masking term
        # g = torch.ones_like(g)
        # d = torch.ones_like(d)
        # f = torch.ones_like(f)

        l_dot_n = jt.linalg.einsum('ijk,ik->ij', pts2l, normal)
        v_dot_n = jt.linalg.einsum('ij,ij->i', pts2c, normal)
        denom = 4 * jt.abs(l_dot_n) * jt.abs(v_dot_n)[:, None]
        microfacet = safe_divide(f * g * d, denom)  # NxL
        brdf_glossy = microfacet[:, :, None].repeat(1, 1, 3)

        # Diffuse
        # http://www.joshbarczak.com/blog/?p=272
        # Mix two shaders
        brdf_lambert = albedo[:, None, :].repeat(1, pts2l.shape[1], 1) / math.pi
        if self.lambert_only:
            brdf = brdf_lambert  # Nx3
        elif self.glossy_only:
            brdf = brdf_glossy
        else:
            brdf = brdf_glossy + brdf_lambert  # Nx3  # TODO: energy conservation?

        if len(sh) == 5:
            return brdf.reshape(B, P, eH * eW, 3).permute(0, 2, 1, 3).reshape(B, eH, eW, P, 3)
        elif len(sh) == 3:
            return brdf.reshape(B, P, 3)
        elif len(sh) == 2:
            return brdf.reshape(P, 3)
        else:
            return brdf

    @staticmethod
    def _get_g(v, m, n, alpha=0.1):
        """Geometric function (GGX).
        """
        cos_theta_v = jt.linalg.einsum('ij,ij->i', n, v)
        cos_theta = jt.linalg.einsum('ijk,ik->ij', m, v)
        denom = cos_theta_v[:, None]
        div = safe_divide(cos_theta, denom)
        chi = jt.ternary(div > 0, 1., 0.)
        cos_theta_v_sq = jt.Var(np.square(cos_theta_v))
        cos_theta_v_sq = jt.clamp(cos_theta_v_sq, 0., 1.)
        denom = cos_theta_v_sq
        tan_theta_v_sq = safe_divide(1 - cos_theta_v_sq, denom)
        tan_theta_v_sq = jt.Var(np.clamp(tan_theta_v_sq, 0., 1e10))
        denom = 1 + jt.sqrt(1 + alpha ** 2 * tan_theta_v_sq[:, None])
        g = safe_divide(chi * 2, denom)
        return g  # (n_pts, n_lights)

    @staticmethod
    def _get_d(m, n, alpha=0.1):
        """Microfacet distribution (GGX).
        """
        cos_theta_m = jt.linalg.einsum('ijk,ik->ij', m, n)
        chi = jt.ternary(cos_theta_m > 0, 1., 0.)
        cos_theta_m_sq = jt.Var(np.square(cos_theta_m))
        denom = cos_theta_m_sq
        tan_theta_m_sq = safe_divide(1 - cos_theta_m_sq, denom)
        denom = math.pi * jt.Var(np.square(cos_theta_m_sq)) * jt.Var(np.square(
            alpha ** 2 + tan_theta_m_sq))
        d = safe_divide(alpha ** 2 * chi, denom)
        return d  # (n_pts, n_lights)

    def _get_f(self, l, m):
        """Fresnel (Schlick's approximation).
        """
        cos_theta = jt.linalg.einsum('ijk,ijk->ij', l, m)
        f = self.f0 + (1 - self.f0) * (1 - cos_theta) ** 5
        return f  # (n_pts, n_lights)


def safe_divide(x, denom, eps=1e-20):
    # https://stackoverflow.com/questions/18234311/how-close-to-division-by-zero-can-i-get
    # The answer: very close...
    return jt.div(x, denom.clamp(eps))
