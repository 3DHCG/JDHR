#import torch
import jittor as jt
import numpy as np

def monotonic_near_far(near:jt.Var, far:jt.Var, n:jt.Var, f:jt.Var):
    #print("near, far",near.shape, far.shape,n.shape,f.shape)
    #n = n[..., None, None]
    #f = f[..., None, None]
    #print("near, far",near.shape, far.shape,n.shape,f.shape)
    #near, far = near.clamp(n, f), far.clamp(n, f)
    for i in range(near.shape[0]):
        near[i], far[i] = near[i].clamp(n[i], f[i]), far[i].clamp(n[i], f[i])
    valid_mask = near < far
    valid_near_plane = jt.ternary(valid_mask, near, f).min()
    valid_far_plane = jt.ternary(valid_mask, far, n).max()
    near, far = jt.ternary(valid_mask, near, valid_near_plane), jt.ternary(valid_mask, far, valid_far_plane)  # what ever for these points
    #near, far = near.clamp(n, f), far.clamp(n, f)
    #print("near, far",near.shape, far.shape,n.shape,f.shape)
    for i in range(near.shape[0]):
        near[i], far[i] = near[i].clamp(n[i], f[i]), far[i].clamp(n[i], f[i])
    return near, far
    
def monotonic_near_far_np(near, far, n, f):
    n = n[..., None, None]
    f = f[..., None, None]
    near, far = near.clip(n, f), far.clip(n, f)
    valid_mask = near < far
    valid_near_plane = np.where(valid_mask, near, f).min()
    valid_far_plane = np.where(valid_mask, far, n).max()
    near, far = np.where(valid_mask, near, valid_near_plane), np.where(valid_mask, far, valid_far_plane)  # what ever for these points
    near, far = near.clip(n, f), far.clip(n, f)
    return near, far

def get_bound_corners(bounds:jt.Var):
    #bounds=bounds.data#print("bounds",type(bounds),bounds.shape)
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    #min_x, min_y, min_z = min_x.data.item(), min_y.data.item(), min_z.data.item()
    #max_x, max_y, max_z = max_x.data.item(), max_y.data.item(), max_z.data.item()
    a=jt.cat([min_x, min_y, min_z])
    b=jt.cat([min_x, min_y, max_z])
    c=jt.cat([min_x, max_y, min_z])
    d=jt.cat([min_x, max_y, max_z])
    e=jt.cat([max_x, min_y, min_z])
    f=jt.cat([max_x, min_y, max_z])
    g=jt.cat([max_x, max_y, min_z])
    h=jt.cat([max_x, max_y, max_z])
    corners_3d=jt.cat((a,b,c,d,e,f,g,h),dim=0)
    corners_3d=corners_3d.reshape(-1,3)
    #print("bounds",type(corners_3d),corners_3d.shape)
    return corners_3d 

def get_bound_corners_np(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d

def project(xyz:jt.Var, K:jt.Var, R:jt.Var, T:jt.Var):
    """
    xyz: [...N, 3], ... means some batch dim
    K: [3, 3]
    R: [3, 3]
    T: [3, 1]
    """
    RT = jt.cat([R, T], dim=-1)
    xyz = xyz @ RT[..., :3].transpose(-2, -1) + RT[..., 3:].transpose(-2, -1)
    xyz = xyz @ K.transpose(-2, -1)
    xy = xyz[..., :2] / xyz[..., 2:]
    return xy

def project_np(xyz:jt.Var, K:jt.Var, R:jt.Var, T:jt.Var):
    """
    xyz: [...N, 3], ... means some batch dim
    K: [3, 3]
    R: [3, 3]
    T: [3, 1]
    """
    RT = np.concatenate([R, T], axis=-1)
    xyz = xyz @ RT[..., :3].swapaxes(-2,-1) + RT[..., 3:].swapaxes(-2,-1)
    xyz = xyz @ K.swapaxes(-2,-1)
    xy = xyz[..., :2] / xyz[..., 2:]
    return xy

 
def transform(xyz, RT):
    """
    xyz: [...N, 3], ... means some batch dim
    RT: [3, 4]
    """
    xyz = xyz @ RT[:, :3].transpose() + RT[:, 3:].transpose()
    return xyz


def get_bound_2d_bound(bounds:jt.Var, K:jt.Var, R:jt.Var, T:jt.Var, H, W, pad=25):  # pad more, be safe
    #print(bounds)
    if bounds.ndim == 3: corners_3d = jt.stack([get_bound_corners(b) for b in bounds])
    else: corners_3d = get_bound_corners(bounds)
    if isinstance(H, jt.Var): H = H#.numpy()#.detach().clone().data.item()
    if isinstance(W, jt.Var): W = W#.numpy()#.detach().clone().data.item() #.clone()#.detach()#.data.item()
    corners_2d = project(corners_3d, K, R, T)
    corners_2d = corners_2d.round().int()
    #print("WH",W,H)
    x_min = (corners_2d[..., 0].min() - pad).clamp(0,W)
    x_max = (corners_2d[..., 0].max() + pad).clamp(0,W)
    y_min = (corners_2d[..., 1].min() - pad).clamp(0,H)
    y_max = (corners_2d[..., 1].max() + pad).clamp(0,H)
    #print("asda", x_min, y_min, x_max, y_max)
    x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min

    return x, y, w, h

def get_bound_2d_bound_np(bounds, K, R, T, H, W, pad=25):  # pad more, be safe
    #print(bounds)
    if bounds.ndim == 3: corners_3d = np.stack([get_bound_corners_np(b) for b in bounds])
    else: corners_3d = get_bound_corners_np(bounds)
    #if isinstance(H, jt.Var): H = H#.numpy()#.detach().clone().data.item()
    #if isinstance(W, jt.Var): W = W#.numpy()#.detach().clone().data.item() #.clone()#.detach()#.data.item()
    corners_2d = project_np(corners_3d, K, R, T)
    corners_2d = corners_2d.round()#.int()
    #print("WH",W,H)
    x_min = (corners_2d[..., 0].min() - pad).clip(0,W)
    x_max = (corners_2d[..., 0].max() + pad).clip(0,W)
    y_min = (corners_2d[..., 1].min() - pad).clip(0,H)
    y_max = (corners_2d[..., 1].max() + pad).clip(0,H)
    #print("asda", x_min, y_min, x_max, y_max)
    x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min

    return x, y, w, h

def get_bound_3d_near_far(bounds:jt.Var, R:jt.Var, T:jt.Var):
    corners_3d_worlds = get_bound_corners(bounds)
    corners_3d_camera = transform(corners_3d_worlds, jt.cat([R, T], dim=-1))
    near = corners_3d_camera[..., -1].min()
    far = corners_3d_camera[..., -1].max()
    return near, far


# MipNeRF360 space contraction



def contract(x:jt.Var, r: float = 1.0, p: float= float('inf')):
    l = x.norm(dim=-1, keepdim=True, p=1) + 1e-13
    m = l <= r

    # For smaller than radius points: x = x
    # For larger than radius points: (2 - r/|x|) * r * x / |x|
    x = x * m + m.logical_not() * (2 - r / l) * r * x / l
    return x


def get_bounds(xyz, padding=0.05):  # 5mm padding? really?
    # xyz: n_batch, n_points, 3

    min_xyz = jt.min(xyz, dim=1)  # torch min with dim is ...
    max_xyz = jt.max(xyz, dim=1)
    min_xyz -= padding
    max_xyz += padding
    bounds = jt.stack([min_xyz, max_xyz], dim=1)
    return bounds
    diagonal = bounds[..., 1:] - bounds[..., :1]  # n_batch, 1, 3
    bounds[..., 1:] = bounds[..., :1] + jt.ceil(diagonal / voxel_size) * voxel_size  # n_batch, 1, 3
    return bounds

def get_bounds_np(xyz, padding=0.05):  # 5mm padding? really?
    # xyz: n_batch, n_points, 3

    min_xyz = np.min(xyz, axis=1)  # torch min with dim is ...
    max_xyz = np.max(xyz, axis=1)
    min_xyz -= padding
    max_xyz += padding
    bounds = np.stack([min_xyz, max_xyz], axis=1)
    return bounds
    #diagonal = bounds[..., 1:] - bounds[..., :1]  # n_batch, 1, 3
    #bounds[..., 1:] = bounds[..., :1] + jt.ceil(diagonal / voxel_size) * voxel_size  # n_batch, 1, 3
    #return bounds


def get_near_far_aabb(bounds:jt.Var, ray_o:jt.Var, ray_d:jt.Var, epsilon: float = 1e-8):
    """
    calculate intersections with 3d bounding box
    bounds: n_batch, 2, 3, min corner and max corner
    ray_o: n_batch, n_points, 3
    ray_d: n_batch, n_points, 3, assume already normalized
    return: near, far: B, P, 1

    NOTE: This function might produce inf or -inf, need a clipping
    """
    if ray_o.ndim >= bounds.ndim:
        diff = ray_o.ndim - bounds.ndim
        for i in range(diff):
            bounds = bounds.unsqueeze(-3)  # match the batch dimensions, starting from second

    # NOTE: here, min in tmin means the intersection with point bound_min, not minimum
    tmin = (bounds[..., :1, :] - ray_o) / (ray_d + epsilon)  # (b, 1, 3) - (b, 1, 3) / (b, n, 3) -> (b, n, 3)
    tmax = (bounds[..., 1:, :] - ray_o) / (ray_d + epsilon)  # (b, n, 3)
    # near plane is where the intersection has a smaller value on corresponding dimension than the other point
    t1 = jt.minimum(tmin, tmax)  # (b, n, 3)
    t2 = jt.maximum(tmin, tmax)
    # near plane is the maximum of x, y, z intersection point, entering AABB: enter every dimension
    near = t1.max(dim=-1, keepdim=True)  # (b, n)
    far = t2.min(dim=-1, keepdim=True)
    return near, far


def sample_depth_near_far(near:jt.Var, far:jt.Var, n_samples: int, perturb: bool = False):
    # ray_o: n_batch, n_rays, 3
    # ray_d: n_batch, n_rays, 3
    # near: n_batch, n_rays
    # far: n_batch, n_rays
    # sample points and depth values for every ray between near-far
    # and possibly do some pertubation to introduce randomness in training
    # dists: n_batch, n_rays, n_steps
    # pts: n_batch, n_rays, n_steps, 3

    # calculate the steps for each ray
    s_vals = jt.linspace(0., 1., steps=n_samples, dtype=near.dtype)
    z_vals = near * (1. - s_vals) + far * s_vals

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = jt.cat([mids, z_vals[..., -1:]], -1)
        lower = jt.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        z_rand = jt.rand(*z_vals.shape, dtype=upper.dtype)
        z_vals = lower + (upper - lower) * z_rand

    return z_vals


def sample_points_near_far(ray_o, ray_d, near, far, n_samples: int, perturb: bool):
    # ray_o: n_batch, n_rays, 3
    # ray_d: n_batch, n_rays, 3
    # near: n_batch, n_rays
    # far: n_batch, n_rays
    # sample points and depth values for every ray between near-far
    # and possibly do some pertubation to introduce randomness in training
    # dists: n_batch, n_rays, n_steps
    # pts: n_batch, n_rays, n_steps, 3

    # calculate the steps for each ray
    z_vals = sample_depth_near_far(near, far, n_samples, perturb)

    # (n_batch, n_rays, n_samples, 3)
    xyz = ray_o[..., None, :] + ray_d[..., None, :] * z_vals[..., None]

    return xyz, z_vals
