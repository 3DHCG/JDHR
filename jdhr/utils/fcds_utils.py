# Feature Cloud Sequence utilities
# This files builds the components for the feature cloud sequence sampler

#import torch
import jittor as jt
from typing import List, Dict, Union
from jittor import nn
from jdhr.utils.console_utils import *
from jdhr.utils.net_utils import MLP
from jdhr.utils.base_utils import dotdict
from jdhr.utils.raster_utils import get_ndc_perspective_matrix
from jdhr.utils.chunk_utils import multi_gather, multi_scatter
from jdhr.utils.math_utils import normalize_sum, affine_inverse, affine_padding
from jdhr.utils.ball_query_utils import ball_query
from jdhr.utils.sample_farthest_points_utils import sample_farthest_points
from enum import Enum, auto


class SamplingType(Enum):
    MARCHING_CUBES_RECONSTRUCTION = auto()  # use surface reconstruction and distance thresholding
    POISSON_RECONSTRUCTION = auto()  # use surface reconstruction and distance thresholding
    FARTHEST_DOWN_SAMPLE = auto()  # use the fartherest down sampling algorithm
    SURFACE_DISTRIBUTION = auto()
    RANDOM_DOWN_SAMPLE = auto()
    VOXEL_DOWN_SAMPLE = auto()


def estimate_occupancy_field(xyz: jt.Var, rad: jt.Var, occ: jt.Var):
    # This method builds a function to evaluate the occupancy field of the point cloud density field
    # We sample the point cloud with a ball query for the largest radius in the set
    # The actual alpha is decreased as the distance to the closest points
    # If multiple points fall into the region of interest, we compute for alpha on all of them and performs a add operation
    #from pytorch3d.ops import ball_query
    max_rad = rad.max()
    # B, N, 3
    # B, N, 1
    # B, N, 1

    def field(pts: jt.Var, K=10):
        # pts: B, P, 3
        sh = pts.shape
        pts = pts.view(pts.shape[0], -1, 3)
        knn = ball_query(pts, xyz, K, radius=max_rad)#?????
        dists,idx = knn  # B, P, K
        msk = idx != -1
        idx = jt.ternary(msk, idx, 0).long()
        pix_rad = multi_gather(rad[..., 0], idx.view(idx.shape[0], -1), dim=-1).view(idx.shape)  # B, P, K
        pix_occ = multi_gather(occ[..., 0], idx.view(idx.shape[0], -1), dim=-1).view(idx.shape)  # B, P, K
        pix_occ = pix_occ * (1 - dists / (pix_rad * pix_rad))  # B, P, K
        pix_occ = jt.ternary(msk, pix_occ, 0)
        pix_occ = pix_occ.clamp(0, 1)
        pix_occ = pix_occ.sum(dim=-1, keepdim=True)  # B, P, 1
        return pix_occ.view(*sh[:-1], 1)

    return field

# @torch.jit.script


def prepare_feedback_transform(H: int, W: int, K: jt.Var, R: jt.Var, T: jt.Var,
                               n: jt.Var,
                               f: jt.Var,
                               xyz: jt.Var,
                               rgb: jt.Var,
                               rad: jt.Var):
    ixt = get_ndc_perspective_matrix(K, H, W, n[..., 0], f[..., 0]).to(xyz.dtype)  # to opengl, remove last dim of n and f
    w2c = affine_padding(jt.cat([R, T], dim=-1)).to(xyz.dtype)
    c2w = affine_inverse(w2c)
    c2w[..., 0] *= 1  # flip x
    c2w[..., 1] *= -1  # flip y
    c2w[..., 2] *= -1  # flip z
    ext = affine_inverse(c2w)
    pix_xyz = jt.cat([xyz, jt.ones_like(xyz[..., :1])], dim=-1) @ ext.transpose(-2,-1) @ ixt.transpose(-2,-1)
    pix_rad = abs(H * ixt[..., 1, 1][..., None, None] * rad / pix_xyz[..., -1:])  # z: B, 1 * B, N, world space radius -> ndc radius B, N, 1

    # Prepare data to be rendered
    data = jt.cat([pix_xyz, rgb, pix_rad], dim=-1).reshape(-1,)  # organize the data inside vbo
    return data


def matrix_to_rotation_6d(matrix: jt.Var) -> jt.Var:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


@run_once
def warn_once_about_pulsar_fxfy():
    log(yellow(
        "Pulsar only supports a single focal lengths. For converting OpenCV "
        "focal lengths, we average them for x and y directions. "
        "The focal lengths for x and y you provided differ by more than 1%, "
        "which means this could introduce a noticeable error."
    ))


def get_pulsar_camera_params(
    R: jt.Var,
    tvec: jt.Var,
    camera_matrix: jt.Var,
    image_size: jt.Var,
    znear: float = 0.1,
) -> jt.Var:
    assert len(camera_matrix.size()) == 3, "This function requires batched inputs!"
    assert len(R.size()) == 3, "This function requires batched inputs!"
    assert len(tvec.size()) in (2, 3), "This function reuqires batched inputs!"

    # Validate parameters.
    image_size_wh = image_size.to(R).flip(dims=(1,))
    assert jt.all(
        image_size_wh > 0
    ), "height and width must be positive but min is: %s" % (
        str(image_size_wh.min().item())
    )
    assert (
        camera_matrix.size(1) == 3 and camera_matrix.size(2) == 3
    ), "Incorrect camera matrix shape: expected 3x3 but got %dx%d" % (
        camera_matrix.size(1),
        camera_matrix.size(2),
    )
    assert (
        R.size(1) == 3 and R.size(2) == 3
    ), "Incorrect R shape: expected 3x3 but got %dx%d" % (
        R.size(1),
        R.size(2),
    )
    if len(tvec.size()) == 2:
        tvec = tvec.unsqueeze(2)
    assert (
        tvec.size(1) == 3 and tvec.size(2) == 1
    ), "Incorrect tvec shape: expected 3x1 but got %dx%d" % (
        tvec.size(1),
        tvec.size(2),
    )
    # Check batch size.
    batch_size = camera_matrix.size(0)
    assert R.size(0) == batch_size, "Expected R to have batch size %d. Has size %d." % (
        batch_size,
        R.size(0),
    )
    assert (
        tvec.size(0) == batch_size
    ), "Expected tvec to have batch size %d. Has size %d." % (
        batch_size,
        tvec.size(0),
    )
    # Check image sizes.
    image_w = image_size_wh[0, 0]
    image_h = image_size_wh[0, 1]
    assert jt.all(
        image_size_wh[:, 0] == image_w
    ), "All images in a batch must have the same width!"
    assert jt.all(
        image_size_wh[:, 1] == image_h
    ), "All images in a batch must have the same height!"
    # Focal length.
    fx = camera_matrix[:, 0, 0].unsqueeze(1)
    fy = camera_matrix[:, 1, 1].unsqueeze(1)
    # Check that we introduce less than 1% error by averaging the focal lengths.
    fx_y = fx / fy
    if jt.any(fx_y > 1.01) or jt.any(fx_y < 0.99):
        warn_once_about_pulsar_fxfy()
    f = (fx + fy) / 2
    # Normalize f into normalized device coordinates.
    focal_length_px = f / image_w
    # Transfer into focal_length and sensor_width.
    # NOTE: Using jt.Var instead of jt.array will cause cpu gpu sync
    focal_length = jt.float32([znear - 1e-5])
    focal_length = focal_length[None, :].repeat(batch_size, 1)
    sensor_width = focal_length / focal_length_px
    # Principal point.
    cx = camera_matrix[:, 0, 2].unsqueeze(1)
    cy = camera_matrix[:, 1, 2].unsqueeze(1)
    # Transfer principal point offset into centered offset.
    cx = -(cx - image_w / 2)
    cy = cy - image_h / 2
    # Concatenate to final vector.
    param = jt.cat([focal_length, sensor_width, cx, cy], dim=1)
    R_trans = R.permute(0, 2, 1)
    cam_pos = -jt.bmm(R_trans, tvec).squeeze(2)
    cam_rot = matrix_to_rotation_6d(R_trans)
    cam_params = jt.cat([cam_pos, cam_rot, param], dim=1)
    return cam_params


def get_opencv_camera_params(batch: dotdict):
    H = batch.meta.H[0].item()  # !: BATCH
    W = batch.meta.W[0].item()  # !: BATCH
    K = batch.K
    R = batch.R
    T = batch.T
    C = -batch.R.transpose(-2,-1) @ batch.T  # B, 3, 1
    return H, W, K, R, T, C


def get_pytorch3d_camera_params(batch: dotdict):
    # Extract pytorc3d camera parameters from batch input
    # R and T are applied on the right (requires a transposed R from OpenCV camera format)
    # Coordinate system is different from that of OpenCV (cv: right down front, 3d: left up front)
    # However, the correction has to be down on both T and R... (instead of just R)
    C = -batch.R.transpose(-2,-1) @ batch.transpose()  # B, 3, 1
    R = batch.R.clone()
    R[..., 0, :] *= -1  # flip x row
    R[..., 1, :] *= -1  # flip y row
    T = (-R @ C)[..., 0]  # c2w back to w2c
    R = R.transpose(-2,-1)  # applied left (left multiply to right multiply, god knows why...)

    H = batch.meta.H[0].item()  # !: BATCH
    W = batch.meta.W[0].item()  # !: BATCH
    K = get_pytorch3d_ndc_K(batch.K, H, W)

    return H, W, K, R, T, C

# TODO: Remove pcd_t and with_t semantics, this is a legacy API


def voxel_surface_down_sample(pcd: jt.Var, pcd_t: jt.Var = None, voxel_size: float = 0.01, dist_th: float = 0.025, n_points: int = 65536):
    # !: BATCH
    # TODO: Use number of vertices for good estimation
    import open3d as o3d
    import numpy as np
    import mcubes
    from jdhr.utils.sample_utils import point_mesh_distance
    from jdhr.utils.fusion_utils import voxel_reconstruction
    #from pytorch3d.ops import knn_points, ball_query, sample_farthest_points

    # Performing voxel surface reconstruction
    vertices, triangles = voxel_reconstruction(pcd, voxel_size)

    # Convert mesh data to torch tensors
    triangles_torch = jt.float32(vertices[triangles])

    # Calculate distances using point_mesh_distance
    dists, _ = point_mesh_distance(pcd[0], triangles_torch)

    # Select points based on distances
    valid = (dists < dist_th).nonzero()[..., 0]
    while (len(valid) - n_points) / n_points > 0.005:
        # There are too many valid points, should control its number
        ratio = len(valid) / len(pcd[0])  # the ratio of valid points
        n_expected = int(n_points / ratio)  # the expected number of points before surface sampling
        pcd = random(pcd, n_points=n_expected)

        # Calculate distances using point_mesh_distance
        dists, _ = point_mesh_distance(pcd[0], triangles_torch)

        # Select points based on distances
        valid = (dists < dist_th).nonzero()[..., 0]

    _, valid = dists.topk(n_points, dim=-1, sorted=False, largest=False)
    pcd_new = jt.index_select(pcd[0], 0, valid)[None]

    return pcd_new


def filter_bounds(pcd: jt.Var, pcd_t: jt.Var = None, bounds: jt.Var = None):
    valid = ((pcd - bounds[..., 0, :]) > 0).all(dim=-1) & ((pcd - bounds[..., 1, :]) < 0).all(dim=-1)  # mask: B, N
    valid = valid[0].nonzero()[None]  # B, S -> B, V # MARK: SYNC
    pcd = multi_gather(pcd, valid, dim=-2)
    return pcd


def duplicate(pcd: jt.Var, pcd_t: jt.Var = None, std: float = 0.005 * 0.1):
    # return pcd.repeat_interleave(2, dim=-2), ind.repeat_interleave(2, dim=-2)
    pcd_new = jt.normal(pcd, std=std)
    return jt.cat([pcd, pcd_new], dim=-2)


def farthest(pcd: jt.Var, pcd_t: jt.Var = None, lengths: jt.Var = None, n_points: int = 65536):
    #from pytorch3d.ops import knn_points, ball_query, sample_farthest_points
    idx = sample_farthest_points(pcd, n_points)  # N, K (padded)#?????
    return multi_gather(pcd, idx)


def random(pcd: jt.Var, pcd_t: jt.Var = None, n_points: int = 65536, std: float = 0.001):
    inds = jt.stack([jt.randperm(pcd.shape[-2])[:n_points] for b in range(len(pcd))])  # B, S,
    return multi_gather(pcd, inds)


def voxel_down_sample(pcd: jt.Var, pcd_t: jt.Var = None, voxel_size=0.005):
    import open3d as o3d
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd.view(-1, 3).detach().cpu().numpy())
    o3d_pcd = o3d_pcd.voxel_down_sample(voxel_size)
    return jt.array(np.array(o3d_pcd.points)).to(pcd.dtype, non_blocking=True).view(pcd.shape[0], -1, 3)


def remove_outlier(pcd: jt.Var, pcd_t: jt.Var = None, K: int = 20, std_ratio=2.0, return_inds=False):  # !: BATCH
    import open3d as o3d
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd.view(-1, 3).detach().cpu().numpy())
    cl, ind = o3d_pcd.remove_statistical_outlier(nb_neighbors=K, std_ratio=std_ratio)
    if return_inds:
        return jt.array(np.array(ind))[None]  # N,
    return jt.array(np.array(o3d_pcd.points)[np.array(ind)]).to( pcd.dtype, non_blocking=True).view(pcd.shape[0], -1, 3)


def farthest_down_sample(pcd: jt.Var, pcd_t: jt.Var = None, K: int = 65536):
    import open3d as o3d
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd.view(-1, 3).detach().cpu().numpy())
    o3d_pcd = o3d_pcd.farthest_point_down_sample(K)
    return jt.array(np.array(o3d_pcd.points)).to( pcd.dtype, non_blocking=True).view(pcd.shape[0], -1, 3)


def sample_random_points(pcd: jt.Var, pcd_t: jt.Var = None, K: int = 500):
    bounds = jt.stack([pcd.min(dim=-2) - 0.033, pcd.max(dim=-2) + 0.033], dim=-2)  # B, 2, 3
    pts = jt.rand(*pcd.shape[:-2], K, 3) * (bounds[..., 1:, :] - bounds[..., :1, :]) + bounds[..., :1, :]
    return pts


def sample_filter_random_points(pcd: jt.Var, pcd_t: jt.Var = None, K: int = 500, update_radius=0.05, filter_K=10):
    pts = sample_random_points(pcd, pcd_t, K)  # ugly interface
    pts = filter_points(pts, pcd, update_radius, filter_K)
    return pts


def get_pytorch3d_ndc_K(K: jt.Var, H: int, W: int):
    M = min(H, W)
    K = jt.cat([K, jt.zeros_like(K[..., -1:, :])], dim=-2)
    K = jt.cat([K, jt.zeros_like(K[..., :, -1:])], dim=-1)
    K[..., 3, 2] = 1  # ...? # HACK: pytorch3d magic
    K[..., 2, 2] = 0  # ...? # HACK: pytorch3d magic
    K[..., 2, 3] = 1  # ...? # HACK: pytorch3d magic

    K[..., 0, 1] = 0
    K[..., 1, 0] = 0
    K[..., 2, 0] = 0
    K[..., 2, 1] = 0
    # return K

    K[..., 0, 0] = K[..., 0, 0] * 2.0 / M  # fx
    K[..., 1, 1] = K[..., 1, 1] * 2.0 / M  # fy
    K[..., 0, 2] = -(K[..., 0, 2] - W / 2.0) * 2.0 / M  # px
    K[..., 1, 2] = -(K[..., 1, 2] - H / 2.0) * 2.0 / M  # py
    return K


def expand_points_features(render_scale: Union[float, int], pcd_old: jt.Var, ind_old: jt.Var, radius: float):
    # FIXME: Duplicated code for these
    n_points = pcd_old.shape[-2]
    if isinstance(render_scale, int):
        target_n_points = render_scale
        n_points = pcd_old.shape[-2]
        render_scale = target_n_points / n_points
    target_n_points = int(render_scale * n_points)
    return generate_points_features(target_n_points, pcd_old, ind_old, radius)


def expand_points(render_scale: Union[float, int], pcd_old: jt.Var, radius: float):
    n_points = pcd_old.shape[-2]
    if isinstance(render_scale, int):
        target_n_points = render_scale
        n_points = pcd_old.shape[-2]
        render_scale = target_n_points / n_points
    target_n_points = int(render_scale * n_points)
    return generate_points(target_n_points, pcd_old, radius)


def generate_points_features(n_points: int, pcd_old: jt.Var, ind_old: jt.Var, radius: float):
    pcd_new = sample_random_points(pcd_old, K=n_points)
    pcd_new, ind_new = update_points_features(pcd_new, pcd_old, ind_old, radius)
    return pcd_new, ind_new


def generate_points(n_points: int, pcd_old: jt.Var, radius: float):
    pcd_new = sample_random_points(pcd_old, K=n_points)
    pcd_new = update_points(pcd_new, pcd_old, radius)
    return pcd_new


def surface_points(pcd: jt.Var, pcd_t: jt.Var = None, radius: float = 0.05, K: int = 500, n_points: float = 16384):
    # Try to retain the surface points
    #from pytorch3d.ops import knn_points, ball_query

    # 1. Perform a ball query (with a large upper limit number of points)
    # 2. Sort all points based on the number of neighbors
    close = ball_query(pcd, pcd, K,radius=radius)  # B, S, K
    dists, idx = close

    dists = jt.ternary(idx == -1, float('inf'), 0.1)  # B, S, K, equal weight, just for filtering
    idx = jt.ternary(idx == -1, 0, idx)  # B, S, K

    # Find mean points
    B, S, C = pcd.shape
    weights = weight_function(dists, radius)[..., None]  # B, S, K, 1
    pcd_new = multi_gather(pcd, idx.view(B, S * K)).view(B, S, K, -1)
    pcd_new = (pcd_new * weights).sum(dim=-2)  # B, S, 3

    # Find mean deviation
    dists = (pcd_new - pcd).norm(dim=-1)  # B, S,
    valid = (dists).topk(n_points, dim=-1, sorted=False)[1]  # B, K
    pcd_new = multi_gather(pcd, valid, dim=-2)

    return pcd_new


def surface_points_features(pcd_old: jt.Var, ind_old: jt.Var, radius: float = 0.05, K: int = 500, n_points: float = 16384):
    # Try to retain the surface points
    #from pytorch3d.ops import knn_points, ball_query

    # 1. Perform a ball query (with a large upper limit number of points)
    # 2. Sort all points based on the number of neighbors
    close = ball_query(pcd_old, pcd_old, K,radius=radius)  # B, S, K
    dists, idx = close

    dists = jt.ternary(idx == -1, float('inf'), 0.1)  # B, S, K, equal weight, just for filtering
    idx = jt.ternary(idx == -1, 0, idx)  # B, S, K

    # Find mean points
    B, S, C = pcd_old.shape
    weights = weight_function(dists, radius)[..., None]  # B, S, K, 1
    pcd_new = multi_gather(pcd_old, idx.view(B, S * K)).view(B, S, K, -1)
    pcd_new = (pcd_new * weights).sum(dim=-2)  # B, S, 3

    # Find mean deviation
    dists = (pcd_new - pcd_old).norm(dim=-1)  # B, S,
    valid = (dists).topk(n_points, dim=-1, sorted=False)[1]  # B, K
    pcd_new = multi_gather(pcd_old, valid, dim=-2)
    ind_new = multi_gather(ind_old, valid, dim=-2)

    return pcd_new, ind_new


def filter_points(pcd_new: jt.Var, pcd_old: jt.Var, radius: float = 0.05, K: int = 10, fill_ratio: float = 0.1):
    # This will lead to shrinking
    #from pytorch3d.ops import knn_points, ball_query

    close = ball_query(pcd_new, pcd_old, K)  # B, S, K
    dists, idx = close.dists, close.idx
    # !: BATCH
    good = (idx != -1).sum(dim=-1) / K > fill_ratio
    valid = good[0].nonzero()[None]  # B, S -> B, V # MARK: SYNC

    idx = multi_gather(idx, valid, dim=-2)
    dists = multi_gather(dists, valid, dim=-2)
    pcd_new = multi_gather(pcd_new, valid, dim=-2)
    dists = jt.ternary(idx == -1, float('inf'), dists)  # B, S, K
    idx = jt.ternary(idx == -1, 0, idx)  # B, S, K

    B, S, C = pcd_new.shape
    B, N, C = pcd_old.shape
    pcd_new = multi_gather(pcd_old, idx.view(B, S * K)).view(B, S, K, -1)  # B, S, K, 3
    weights = weight_function(dists, radius)[..., None]  # B, S, K, 1
    pcd_new = (pcd_new * weights).sum(dim=-2)
    return pcd_new


def filter_points_features(pcd_new: jt.Var, pcd_old: jt.Var, ind_old: jt.Var, radius: float = 0.05, K: int = 10, fill_ratio: float = 0.1):
    # This will lead to shrinking
    #from pytorch3d.ops import knn_points, ball_query

    close = ball_query(pcd_new, pcd_old, K,radius=radius)  # B, S, K
    dists, idx = close
    # !: BATCH
    good = (idx != -1).sum(dim=-1) / K > fill_ratio
    valid = good[0].nonzero()[None]  # B, S -> B, V # MARK: SYNC

    idx = multi_gather(idx, valid, dim=-2)
    dists = multi_gather(dists, valid, dim=-2)
    pcd_new = multi_gather(pcd_new, valid, dim=-2)
    dists = jt.ternary(idx == -1, float('inf'), dists)  # B, S, K
    idx = jt.ternary(idx == -1, 0, idx)  # B, S, K

    B, S, C = pcd_new.shape
    B, N, C = pcd_old.shape
    pcd_new = multi_gather(pcd_old, idx.view(B, S * K)).view(B, S, K, -1)  # B, S, K, 3
    ind_new = multi_gather(ind_old, idx.view(B, S * K)).view(B, S, K, -1)  # B, S, K, C
    weights = weight_function(dists, radius)[..., None]  # B, S, K, 1
    pcd_new = (pcd_new * weights).sum(dim=-2)
    ind_new = (ind_new * weights).sum(dim=-2)
    # pcd_new = pcd_new.mean(dim=-2)
    # ind_new = ind_new.mean(dim=-2)
    return pcd_new, ind_new


def update_points_features(pcd_new: jt.Var, pcd_old: jt.Var, ind_old: jt.Var, radius: float = 0.05, K: int = 5):
    # This will lead to shrinking
    #from pytorch3d.ops import knn_points, ball_query

    # close = ball_query(pcd_new, pcd_old, radius=radius, return_nn=False, K=K)  # B, S, K
    close = jt.misc.knn(pcd_new, pcd_old,K)  # B, S, K
    dists, idx = close

    B, S, C = pcd_new.shape
    B, N, C = pcd_old.shape
    pcd_new = multi_gather(pcd_old, idx.view(B, S * K)).view(B, S, K, -1)  # B, S, K, 3
    ind_new = multi_gather(ind_old, idx.view(B, S * K)).view(B, S, K, -1)  # B, S, K, C
    weights = weight_function(dists, radius)[..., None]  # B, S, K, 1
    pcd_new = (pcd_new * weights).sum(dim=-2)
    ind_new = (ind_new * weights).sum(dim=-2)
    # pcd_new = pcd_new.mean(dim=-2)
    # ind_new = ind_new.mean(dim=-2)
    return pcd_new, ind_new


def update_points(pcd_new: jt.Var, pcd_old: jt.Var, radius: float = 0.05, K: int = 5):
    # This will lead to shrinking
    #from pytorch3d.ops import knn_points, ball_query

    # close = ball_query(pcd_new, pcd_old, radius=radius, return_nn=False, K=K)  # B, S, K
    close = jt.misc.knn(pcd_new, pcd_old,K)  # B, S, K
    dists, idx = close

    B, S, C = pcd_new.shape
    B, N, C = pcd_old.shape
    pcd_new = multi_gather(pcd_old, idx.view(B, S * K)).view(B, S, K, -1)  # B, S, K, 3
    weights = weight_function(dists, radius)[..., None]  # B, S, K, 1
    pcd_new = (pcd_new * weights).sum(dim=-2)
    # pcd_new = pcd_new.mean(dim=-2)
    return pcd_new


def update_features(pcd_new: jt.Var, pcd_old: jt.Var, ind_old: jt.Var, radius: float = 0.05, K: int = 5):
    # This will lead to shrinking
    #from pytorch3d.ops import knn_points, ball_query

    # close = ball_query(pcd_new, pcd_old, radius=radius, return_nn=False, K=K)  # B, S, K
    close = jt.misc.knn(pcd_new, pcd_old,K)  # B, S, K
    dists, idx = close

    B, S, C = pcd_new.shape
    B, N, C = pcd_old.shape
    ind_new = multi_gather(ind_old, idx.view(B, S * K)).view(B, S, K, -1)  # B, S, K, C
    weights = weight_function(dists, radius)[..., None]  # B, S, K, 1
    ind_new = (ind_new * weights).sum(dim=-2)
    # ind_new = ind_new.mean(dim=-2)
    return ind_new


def weight_function(d2: jt.Var, radius: float = 0.05, delta: float = 0.001):
    # Radius weighted function from structured local radiance field
    weights = (-d2 / (2 * radius ** 2)).exp().clip(0)  # B, S, K
    weights = normalize_sum(weights)
    return weights
