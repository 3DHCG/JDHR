#import torch
#import torch.nn.functional as F

import jittor
import jittor as jt
from jittor import misc
from jittor import nn
from typing import List
from functools import lru_cache
from jdhr.utils.math_utils import torch_inverse_3x3
from jdhr.utils.SE3_exp_map import se3_exp_map
from jittor import init
from scipy.spatial.transform import Rotation as R

def batch_rodrigues(
    rot_vecs: jt.Var,  # B, N, 3
    eps: float = misc.finfo("float32").eps
) -> jt.Var:
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: jt.Var BxNx3
            array of N axis-angle vectors
        Returns
        -------
        R: jt.Var BxNx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[:-1]
    dtype = rot_vecs.dtype

    angle = (rot_vecs + eps).norm(p=2, dim=-1, keepdim=True)  # B, N, 3
    rot_dir = rot_vecs / angle

    cos = angle.cos()[..., None, :]
    sin = angle.sin()[..., None, :]

    # Bx1 arrays
    rx, ry, rz = rot_dir.split(1, dim=-1)
    zeros = jt.zeros(batch_size + (1,), dtype=dtype)
    K = jt.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=-1).view(batch_size + (3, 3))

    ident = init.eye(3, dtype=dtype)
    for i in range(len(batch_size)): ident = ident[None]
    rot_mat = ident + sin * K + (1 - cos) * K @ K
    return rot_mat


def transform_mat(R: jt.Var, t:jt.Var) -> jt.Var:
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return jt.cat([nn.pad(R, [0, 0, 0, 1]),
                      nn.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(
    rot_mats: jt.Var,
    joints: jt.Var,
    parents: jt.Var,
    dtype=jt.float32
) -> jt.Var:
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : jt.Var BxNx3x3
        Tensor of rotation matrices
    joints : jt.Var BxNx3
        Locations of joints
    parents : jt.Var BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : jt.Var BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : jt.Var BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    joints = jt.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = jt.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = jt.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = nn.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - nn.pad(
        jt.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms


def apply_r(vds, se3):
    # se3: (B, N, 6), pts: (B, N, 3)
    B, N, _ = se3.shape
    se3 = se3.view(-1, se3.shape[-1])
    vds = vds.view(-1, vds.shape[-1])
    Rs = batch_rodrigues(se3[:, :3])  # get rotation matrix: (N, 3, 3)
    vds = nn.bmm(Rs, vds[:, :, None])[:, :, 0]  # batch matmul to apply rotation, and get (N, 3) back
    vds = vds.view(B, N, -1)
    return vds


def apply_rt(pts, se3):
    # se3: (B, N, 6), pts: (B, N, 3)
    B, N, _ = se3.shape
    se3 = se3.view(-1, se3.shape[-1])
    pts = pts.view(-1, pts.shape[-1])
    Rs = batch_rodrigues(se3[:, :3])  # get rotation matrix: (N, 3, 3)
    pts = nn.bmm(Rs, pts[:, :, None])[:, :, 0]  # batch matmul to apply rotation, and get (N, 3) back
    # TODO: retrain these...
    pts += se3[:, 3:]  # apply transformation
    pts = pts.view(B, N, -1)
    return pts


def get_aspect_bounds(bounds) -> jt.Var:
    # bounds: B, 2, 3
    half_edge = (bounds[:, 1:] - bounds[:, :1]) / 2  # 1, 1, 3
    half_long_edge = half_edge.max(dim=-1, keepdim=True).expand(-1, -1, 3)
    middle_point = half_edge + bounds[:, :1]  # 1, 1, 3
    return jt.cat([middle_point - half_long_edge, middle_point + half_long_edge], dim=-2)


@lru_cache
def get_ndc_transform(bounds: jt.Var, preserve_aspect_ratio: bool = False) -> jt.Var:
    if preserve_aspect_ratio:
        bounds = get_aspect_bounds(bounds)
    n_batch = bounds.shape[0]

    # move to -1
    # scale to 1
    # scale * 2
    # move - 1

    move0 = init.eye(4)[None].expand(n_batch, -1, -1)
    move0[:, :3, -1] = -bounds[:, :1]

    scale0 = init.eye(4)[None].expand(n_batch, -1, -1)
    scale0[:, jt.arange(3), jt.arange(3)] = 1 / (bounds[:, 1:] - bounds[:, :1])

    scale1 = init.eye(4)[None].expand(n_batch, -1, -1)
    scale1[:, jt.arange(3), jt.arange(3)] = 2

    move1 = init.eye(4)[None].expand(n_batch, -1, -1)
    move1[:, :3, -1] = -1

    M = move1.matmul(scale1.matmul(scale0.matmul(move0)))

    return M  # only scale and translation has value


@lru_cache
def get_inv_ndc_transform(bounds: jt.Var, preserve_aspect_ratio: bool = False) -> jt.Var:
    M = get_ndc_transform(bounds, preserve_aspect_ratio)
    invM = scale_trans_inverse(M)
    return invM


@lru_cache
def get_dir_ndc_transform(bounds: jt.Var, preserve_aspect_ratio: bool = False) -> jt.Var:
    # https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/geometry/transforming-normals
    invM = get_inv_ndc_transform(bounds, preserve_aspect_ratio)
    return invM.transpose(-2, -1)



def scale_trans_inverse(M: jt.Var) -> jt.Var:
    n_batch = M.shape[0]
    invS = 1 / M[:, jt.arange(3), jt.arange(3)]
    invT = -M[:, :3, 3:] * invS[..., None]
    invM = init.eye(4)[None].expand(n_batch, -1, -1)
    invM[:, jt.arange(3), jt.arange(3)] = invS
    invM[:, :3, 3:] = invT

    return invM


def ndc(pts, bounds, preserve_aspect_ratio=False) -> jt.Var:
    # both with batch dimension
    # pts has no last dimension
    M = get_ndc_transform(bounds, preserve_aspect_ratio)
    R = M[:, :3, :3]
    T = M[:, :3, 3:]
    pts = pts.matmul(R.transpose(-2, -1)) + T.transpose(-2, -1)
    return pts


def inv_ndc(pts, bounds, preserve_aspect_ratio=False) -> jt.Var:
    M = get_inv_ndc_transform(bounds, preserve_aspect_ratio)
    R = M[:, :3, :3]
    T = M[:, :3, 3:]
    pts = pts.matmul(R.transpose(-2, -1)) + T.transpose(-2, -1)
    return pts


def dir_ndc(dir, bounds, preserve_aspect_ratio=False) -> jt.Var:
    M = get_dir_ndc_transform(bounds, preserve_aspect_ratio)
    R = M[:, :3, :3]
    dir = dir.matmul(R.transpose(-2, -1))
    return dir


@lru_cache
def get_rigid_transform(poses: jt.Var, joints: jt.Var, parents: jt.Var):
    # pose: B, N, 3
    # joints: B, N, 3
    # parents: B, N
    # B, N, _ = poses.shape
    R = batch_rodrigues(poses.view(-1, 3))  # N, 3, 3
    J, A = batch_rigid_transform(R[None], joints, parents.view(-1))  # MARK: doc of this is wrong about parent
    return J, A


def get_rigid_transform_nobatch(poses: jt.Var, joints: jt.Var, parents: jt.Var):
    # pose: N, 3
    # joints: N, 3
    # parents: N
    R = batch_rodrigues(poses)  # N, 3, 3
    J, A = batch_rigid_transform(R[None], joints[None], parents)  # MARK: doc of this is wrong about parent
    J, A = J[0], A[0]  # remove batch dimension
    return J, A


# def apply_rt(xyz: jt.Var, rt: jt.Var):
#     # xyz: B, P, 3
#     # rt: B, P, 6
#     R = batch_rodrigues(rt[..., :3].view(-1, 3)).view(rt.shape[:-1] + (3, 3))  # B, P, 3, 3
#     T = rt[..., 3:]  # B, P, 3
#     return (R @ xyz[..., None])[..., 0] + T


def mat2rt(A: jt.Var) -> jt.Var:
    """calculate 6D rt representation of blend weights and bones
    bw: n_batch, n_points, n_bones
    A: n_batch, n_bones, 4, 4
    """
    # bw
    # 1. get blended transformation from bw and bones
    # 2. get quaternion from matrix
    # 3. get axis-angle from quaternion
    # 4. slice out the translation
    # 5. concatenation
    # A = blend_transform(input, batch.A)
    B,N,_,_=A.shape
    a=jt.flatten(A,start_dim=0, end_dim=1)
    a=a.numpy()
    r=R.from_matrix(a[..., :3, :3])
    r=r.as_rotvec()
    print(r)
    t = a[..., :3, 3]  # n_batch, n_points, 3, drops last dimension
    rt = jt.cat([r, t], dim=-1)
    return rt.reshape((B,N,6))


def screw2rt(screw: jt.Var) -> jt.Var:
    from pytorch3d import transforms
    from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_axis_angle, axis_angle_to_matrix
    return mat2rt(se3_exp_map(screw.view(-1, screw.shape[-1])).permute(0, 2, 1)).view(*screw.shape)#????


def blend_transform(bw: jt.Var, A: jt.Var):
    """blend the transformation
    bw: n_batch, n_points, n_bones
    A: n_batch, n_bones, 4, 4
    """
    A = (bw.unsqueeze(-1).unsqueeze(-1) * A.unsqueeze(-4)).sum(dim=-3)
    return A


def tpose_points_to_ndc_points(pts: jt.Var, M: jt.Var) -> jt.Var:
    R = M[:, :3, :3]
    T = M[:, :3, 3:]
    pts = pts.matmul(R.transpose(-2, -1)) + T.transpose(-2, -1)
    return pts


def tpose_dirs_to_ndc_dirs(dirs: jt.Var, M: jt.Var) -> jt.Var:
    R = M[:, :3, :3]
    dirs = dirs.matmul(R.transpose(-2, -1))
    return dirs


def world_dirs_to_pose_dirs(wdirs, R):
    """
    wpts: n_batch, n_points, 3
    R: n_batch, 3, 3
    """
    pts = jt.matmul(wdirs, R)
    return pts


def pose_dirs_to_world_dirs(pdirs, R):
    """
    wpts: n_batch, n_points, 3
    R: n_batch, 3, 3
    """
    pts = jt.matmul(pdirs, R.transpose(1, 2))
    return pts


def world_points_to_pose_points(wpts, R, Th):
    """
    wpts: n_batch, n_points, 3
    R: n_batch, 3, 3
    Th: n_batch, 1, 3
    """
    if Th.ndim == 2:
        Th = Th[..., None, :]  # add fake point dimension
    pts = jt.matmul(wpts - Th, R)
    return pts


def pose_points_to_world_points(ppts, R, Th):
    """
    ppts: n_batch, n_points, 3
    R: n_batch, 3, 3
    Th: n_batch, 1, 3
    """
    if Th.ndim == 2:
        Th = Th[..., None, :]  # add fake point dimension
    pts = jt.matmul(ppts, R.transpose(1, 2)) + Th
    return pts


def pose_dirs_to_tpose_dirs(ddirs, bw=None, A=None, A_bw=None, R_inv=None):
    """transform directions from the pose space to the T pose
    ddirs: n_batch, n_points, 3
    bw: n_batch, n_points, n_bones
    A: n_batch, n_bones, 4, 4
    """
    A_bw = blend_transform(bw, A) if A_bw is None else A_bw
    # since the blended rotation matrix is not a pure rotation anymore, we transform with inverse transpose
    R = A_bw[..., :3, :3]  # never None
    R_transpose = R.transpose(-2, -1)  # inverse transpose of inverse(R)
    pts = jt.sum(R_transpose * ddirs.unsqueeze(-2), dim=-1)
    return pts


def pose_points_to_tpose_points(ppts: jt.Var, bw=None, A=None, A_bw=None, R_inv=None):
    """transform points from the pose space to the T pose
    ppts: n_batch, n_points, 3
    bw: n_batch, n_points, n_bones
    A: n_batch, n_bones, 4, 4
    """
    A_bw = blend_transform(bw, A) if A_bw is None else A_bw
    pts = ppts - A_bw[..., :3, 3]
    R_inv = torch_inverse_3x3(A_bw[..., :3, :3]) if R_inv is None else R_inv
    pts = jt.sum(R_inv * pts.unsqueeze(-2), dim=-1)
    return pts


def tpose_points_to_pose_points(pts, bw=None, A=None, A_bw=None, R_inv=None):
    """transform points from the T pose to the pose space
    ppts: n_batch, n_points, 3
    bw: n_batch, n_points, n_bones
    A: n_batch, n_bones, 4, 4
    """
    A_bw = blend_transform(bw, A) if A_bw is None else A_bw
    R = A_bw[..., :3, :3]
    pts = jt.sum(R * pts.unsqueeze(-2), dim=-1)
    pts = pts + A_bw[..., :3, 3]
    return pts


def tpose_dirs_to_pose_dirs(ddirs, bw=None, A=None, A_bw=None, R_inv=None):
    """transform directions from the T pose to the pose space
    ddirs: n_batch, n_points, 3
    bw: n_batch, n_points, n_bones
    A: n_batch, n_bones, 4, 4
    """
    A_bw = blend_transform(bw, A) if A_bw is None else A_bw

    # since the blended rotation matrix is not a pure rotation anymore, we transform with inverse transpose
    R_inv = torch_inverse_3x3(A_bw[..., :3, :3]) if R_inv is None else R_inv
    R_inv_trans = R_inv.transpose(-2, -1)  # inverse transpose of the rotation

    pts = jt.sum(R_inv_trans * ddirs.unsqueeze(-2), dim=-1)
    return pts


world_points_to_view_points = world_points_to_pose_points  # input w2c, apply w2c
view_points_to_world_points = pose_points_to_world_points  # input w2c, inversely apply w2c


# def grid_sample_blend_weights(grid_coords, bw):
#     # the blend weight is indexed by xyz
#     grid_coords = grid_coords[:, None, None]
#     bw = nn.grid_sample(bw,
#                        grid_coords,
#                        padding_mode='border',
#                        align_corners=True)
#     bw = bw[:, :, 0, 0]
#     return bw


# def pts_sample_blend_weights_surf(pts, verts, faces, values) -> jt.Var:
#     # surf samp 126988 pts: 127.36531300470233
#     # b, n, D
#     bw, dists = sample_closest_points_on_surface(pts, verts, faces, values)
#     bw = torch.cat([bw, dists], dim=-1)  # b, n, D+1
#     return bw.permute(0, 2, 1)  # b, D+1, n


# def pts_sample_blend_weights_vert(pts, verts, values) -> jt.Var:
#     # b, n, D
#     bw, dists = sample_closest_points(pts, verts, values)
#     bw = torch.cat([bw, dists], dim=-1)  # b, n, D+1
#     return bw.permute(0, 2, 1)  # b, D+1, n


# def pts_sample_blend_weights_vert_blend(pts, verts, values, K=5) -> jt.Var:
#     # vert samp K=5 126988 pts: 6.205926998518407
#     # b, n, D
#     bw, dists = sample_blend_K_closest_points(pts, verts, values, K)
#     bw = torch.cat([bw, dists], dim=-1)  # b, n, D+1
#     return bw.permute(0, 2, 1)  # b, D+1, n
# BLENDING


def pts_sample_blend_weights(pts, bw, bounds):
    """sample blend weights for points
    pts: n_batch, n_points, 3
    bw: n_batch, d, h, w, 25
    bounds: n_batch, 2, 3
    """
    pts = pts.clone()

    # interpolate blend weights
    min_xyz = bounds[:, 0]
    max_xyz = bounds[:, 1]
    bounds = max_xyz[:, None] - min_xyz[:, None]
    grid_coords = (pts - min_xyz[:, None]) / bounds
    grid_coords = grid_coords * 2 - 1
    # convert xyz to zyx, since the blend weight is indexed by xyz
    grid_coords = grid_coords[..., [2, 1, 0]]

    # the blend weight is indexed by xyz
    bw = bw.permute(0, 4, 1, 2, 3)
    grid_coords = grid_coords[:, None, None]
    bw = nn.grid_sample(bw,
                       grid_coords,
                       padding_mode='border',
                       align_corners=True)
    bw = bw[:, :, 0, 0]

    return bw


def grid_sample_A_blend_weights(nf_grid_coords, bw):
    """
    nf_grid_coords: batch_size x n_samples x 24 x 3
    bw: batch_size x 24 x 64 x 64 x 64
    """
    bws = []
    for i in range(24):
        nf_grid_coords_ = nf_grid_coords[:, :, i]
        nf_grid_coords_ = nf_grid_coords_[:, None, None]
        bw_ = nn.grid_sample(bw[:, i:i + 1],
                            nf_grid_coords_,
                            padding_mode='border',
                            align_corners=True)
        bw_ = bw_[:, :, 0, 0]
        bws.append(bw_)
    bw = jt.cat(bws, dim=1)
    return bw


def get_sampling_points(bounds, n_samples):
    sh = bounds.shape
    min_xyz = bounds[:, 0]
    max_xyz = bounds[:, 1]
    x_vals = jt.rand([sh[0], n_samples])
    y_vals = jt.rand([sh[0], n_samples])
    z_vals = jt.rand([sh[0], n_samples])
    vals = jt.stack([x_vals, y_vals, z_vals], dim=2)
    #vals = vals.to(bounds.device)
    pts = (max_xyz - min_xyz)[:, None] * vals + min_xyz[:, None]
    return pts


def forward_node_graph(verts: jt.Var, graph_rt: jt.Var, graph_nodes: jt.Var, graph_bones: jt.Var, graph_weights: jt.Var) -> jt.Var:
    n_batch = graph_rt.shape[0]
    verts = verts.expand(n_batch, *verts.shape[1:])
    graph_nodes = graph_nodes.expand(n_batch, *graph_nodes.shape[1:])
    graph_bones = graph_bones.expand(n_batch, *graph_bones.shape[1:])
    graph_weights = graph_weights.expand(n_batch, *graph_weights.shape[1:])

    # graph_bones: B, V, 4
    r, t = graph_rt.split([3, 3], dim=-1)
    R = batch_rodrigues(r.view(-1, 3)).view(n_batch, -1, 3, 3)
    vi = verts[..., None, :].expand(n_batch, -1, graph_bones.shape[-1], -1)  # B, V, 4, 3

    pj = graph_nodes[jt.arange(n_batch)[..., None, None], graph_bones]  # B, V, 4, 3
    tj = t[jt.arange(n_batch)[..., None, None], graph_bones]  # translation B, V, 4, 3
    Rj = R[jt.arange(n_batch)[..., None, None], graph_bones]  # rotation B, V, 4, 3, 3

    wj = graph_weights[..., None].expand(-1, -1, -1, 3)  # B, V, 4, 3
    vj = Rj.matmul((vi - pj)[..., None])[..., 0] + pj + tj  # B, V, 4, 3
    vi = (vj * wj).sum(dim=-2)
    return vi


def forward_deform_lbs(cverts: jt.Var, deform: jt.Var, weights: jt.Var, A: jt.Var, R: jt.Var = None, T: jt.Var = None, big_A=None) -> jt.Var:
    n_batch = A.shape[0]
    weights = weights.expand(n_batch, *weights.shape[1:])
    if deform is not None:
        tverts = cverts + deform
    else:
        tverts = cverts
    if big_A is not None:
        tverts = pose_points_to_tpose_points(tverts, weights, big_A)
    pverts = tpose_points_to_pose_points(tverts, weights, A)
    if R is not None and T is not None:
        wverts = pose_points_to_world_points(pverts, R, T)
    else:
        wverts = pverts
    return wverts


def inverse_deform_lbs(wverts: jt.Var, deform: jt.Var, weights: jt.Var, A: jt.Var, R: jt.Var, T: jt.Var, big_A=None) -> jt.Var:
    n_batch = deform.shape[0]
    weights = weights.expand(n_batch, *weights.shape[1:])
    pverts = world_points_to_pose_points(wverts, R, T)
    tverts = pose_points_to_tpose_points(pverts, weights, A)
    if big_A is not None:
        tverts = tpose_points_to_pose_points(tverts, weights, big_A)
    cverts = tverts - deform
    return cverts


def bilinear_interpolation(input: jt.Var, shape: List[int]) -> jt.Var:
    # input: B, H, W, C
    # shape: [target_height, target_width]
    return nn.interpolate(input.permute(0, 3, 1, 2), shape, mode='bilinear', align_corners=False).permute(0, 2, 3, 1)


def rand_sample_sum_to_one(dim, samples, negative_one=False):
    # negative_one: allow sampling to negative one?
    exp_sum = (0.5 * (dim - 1))
    bbweights = jt.rand(samples, dim - 1)  # 1024, 5
    bbweights_sum = bbweights.sum(dim=-1)
    extra_mask = bbweights_sum > exp_sum
    bbweights[extra_mask] = 1 - bbweights[extra_mask]
    last_row = (bbweights_sum - exp_sum).abs()
    bbweights = jt.cat([bbweights, last_row[..., None]], dim=-1)
    bbweights = bbweights / exp_sum

    if negative_one:
        bbweights = bbweights * 2 - 1 / dim
    return bbweights
    # bbweights = bbweights / (bbweights.sum(dim=-1, keepdim=True) + eps) # MARK: wrong normalization
    # __import__('ipdb').set_trace()


def linear_sample_sum_to_one(dim, samples, multiplier=5.0):
    interval = dim - 1
    samples_per_iter = samples // interval
    samples_last_iter = samples - (interval - 1) * samples_per_iter

    # except last dimension
    weights = jt.zeros(samples, dim)
    for i in range(interval - 1):
        active = jt.linspace(1, 0, samples_per_iter)
        active = active - 0.5
        active = active * multiplier
        active = active.sigmoid()
        active = active - 0.5
        active = active / active.max() / 2
        active = active + 0.5
        next = 1 - active
        weights[i * samples_per_iter:i * samples_per_iter + samples_per_iter, i] = active
        weights[i * samples_per_iter:i * samples_per_iter + samples_per_iter, i + 1] = next

    active = jt.linspace(1, 0, samples_last_iter)
    next = 1 - active
    weights[(interval - 1) * samples_per_iter:, interval - 1] = active
    weights[(interval - 1) * samples_per_iter:, interval] = next

    return weights


def interpolate_poses(poses, bbweights):

    return


def interpolate_shapes(shapes, bbweights):
    # bbposes: jt.Var = torch.einsum('sn,nvd->svd', bbweights, poses)
    bbshapes: jt.Var = jt.linalg.einsum('sn,nvd->svd', bbweights, shapes)
    # bbdeformed: jt.Var = bbshapes + optim_tpose.verts[None]  # resd to shape
    return bbshapes
