#import torch
#from torch import nn
from jdhr.utils.math_utils import torch_inverse_3x3, normalize,torch_inverse_3x3_np,normalize_np
import jittor as jt
from jittor import nn
import numpy as np

def create_meshgrid(H: int, W: int, indexing: str = 'ij', ndc: bool = False,
                    correct_pix: bool = True, dtype: jt.dtype = jt.float32):
    # kornia has meshgrid, but not the best
    i = jt.arange(H, dtype=dtype)
    j = jt.arange(W, dtype=dtype)
    if correct_pix:
        i = i + 0.5
        j = j + 0.5
    if ndc:
        i = i / H * 2 - 1
        j = j / W * 2 - 1
    ij = jt.meshgrid(i, j)  # defaults to ij
    ij = jt.stack(ij, dim=-1)  # Ht, Wt, 2

    return ij


def get_rays(H: int, W: int, K: jt.Var, R: jt.Var, T: jt.Var, is_inv_K: bool = False,
             z_depth: bool = False, correct_pix: bool = True, ret_coord: bool = False):
    # calculate the world coodinates of pixels
    i, j = np.meshgrid(np.arange(H, dtype=R.dtype),
                          np.arange(W, dtype=R.dtype))
    bss = K.shape[:-2]
    for _ in range(len(bss)): i, j = i[None], j[None]
    #print("III",i.shape,j.shape,bss + i.shape[len(bss):],bss + j.shape[len(bss):])
    i, j = i.tile(bss + i.shape[len(bss):]), j.tile(bss + j.shape[len(bss):])
    # 0->H, 0->W
    return get_rays_from_ij(i, j, K, R, T, is_inv_K, z_depth, correct_pix, ret_coord)


def get_rays_from_ij(i: jt.Var, j: jt.Var,
                     K: jt.Var, R: jt.Var, T: jt.Var,
                     is_inv_K: bool = False, use_z_depth: bool = False,
                     correct_pix: bool = True, ret_coord: bool = False):
    # i: B, P or B, H, W or P or H, W
    # j: B, P or B, H, W or P or H, W
    # K: B, 3, 3
    # R: B, 3, 3
    # T: B, 3, 1
    nb_dim = len(K.shape[:-2])  # number of batch dimensions
    np_dim = len(i.shape[nb_dim:])  # number of points dimensions
    #print("K.float()",K.dtype)
    if not is_inv_K: invK = torch_inverse_3x3_np(K)#.type_as(K)
    else: invK = K
    ray_o = - R.swapaxes(-2,-1) @ T  # B, 3, 1

    # Prepare the shapes
    for _ in range(np_dim): invK = np.expand_dims(invK,axis=-3)
    #print("i.shape + (3, 3)",i.shape,invK.shape)
    invK = np.tile(invK,(i.shape + (1, 1)))
    for _ in range(np_dim): R = np.expand_dims(R,axis=-3)#R.unsqueeze(-3)
    R = np.tile(R,(i.shape + (1, 1)))
    for _ in range(np_dim): T = np.expand_dims(T,axis=-3)#T.unsqueeze(-3)
    T = np.tile(T,(i.shape + (1, 1)))
    for _ in range(np_dim): ray_o = np.expand_dims(ray_o,axis=-3)#ray_o.unsqueeze(-3)
    ray_o = np.tile(ray_o,(i.shape + (1, 1)))

    # Pixel center correction
    if correct_pix: i, j = i + 0.5, j + 0.5
    else: i, j = i.float(), j.float()

    # 0->H, 0->W
    # int -> float; # B, H, W, 3, 1 or B, P, 3, 1 or P, 3, 1 or H, W, 3, 1
    xy1 = np.stack([j, i, np.ones_like(i)], axis=-1)[..., None]
    pixel_camera = invK @ xy1  # B, H, W, 3, 1 or B, P, 3, 1
    #print("pixel_camera",R.shape)
    #print(",(pixel_camera - T).shape",pixel_camera.shape,T.shape)
    pixel_world = R.swapaxes(-2,-1) @ (pixel_camera - T)  # B, P, 3, 1

    # Calculate the ray direction
    pixel_world = pixel_world[..., 0]
    ray_o = ray_o[..., 0]
    ray_d = pixel_world - ray_o  # use pixel_world depth as is (no curving)
    if not use_z_depth: ray_d = normalize_np(ray_d)  # B, P, 3, 1

    if not ret_coord: return ray_o, ray_d
    elif correct_pix: return ray_o, ray_d, (np.stack([i, j], axis=-1) - 0.5).int32()  # B, P, 2
    else: return ray_o, ray_d, np.stack([i, j], axis=-1).int32()  # B, P, 2


def weighted_sample_rays(wet: jt.Var,  # weight of every pixel (high weights -> sample more)
                         K: jt.Var,  # intrinsic
                         R: jt.Var,  # extrinsic
                         T: jt.Var,  # extrinsic
                         n_rays: int = -1,  # -1 means use all pixels (non-zero mask)
                         use_z_depth: bool = False,
                         correct_pix: bool = True,
                         ):
    # When random is set, will ignore n_rays and sample the whole image
    # 1. Need to consider incorporating masks, bounds (only some of the pixels are sampled)
    # 2. Exactly how slow can the vanilla implementation be?
    # with timeit(weighted_sample_coords.__name__):
    #print("wet",wet.shape,K.shape,R.shape,T.shape)
    n_rays=int(n_rays)
    coords = weighted_sample_coords(wet, n_rays)  # B, P, 2
    #print("coords",coords.shape)
    #if(not isinstance(coords,tuple)):
    i, j = coords.T[0],coords.T[1]#unbind(-1)
    #else:
    #    i, j = coords

    # Maybe refactor these? Built rays directly
    # with timeit(get_rays_from_ij.__name__):
    ray_o, ray_d = get_rays_from_ij(i, j, K, R, T, use_z_depth=use_z_depth, correct_pix=correct_pix)  # MARK: 0.xms
    return ray_o, ray_d, coords  # gt, input, input, others


def indices_to_coords(idx: jt.Var, H: int, W: int):
    i = idx // W
    j = idx % W
    return np.stack([i, j], axis=-1)  # ..., 2


def coords_to_indices(ij: jt.Var, H, W):
    i, j = ij.unbind(-1)
    return i * W + j  # ... (no last dim)


def weighted_sample_coords(wet: jt.Var, n_rays: int = -1) -> jt.Var:
    # Here, msk is a sampling weight controller, and only those with >0 values will be sampled

    # Non training sampling
    if n_rays == -1:
        #print("222222222",np.nonzero(np.ones_like(wet[..., 0])).shape)
        return np.transpose(np.nonzero(np.ones_like(wet[..., 0])))  # MARK: assume sorted

    sh = wet.shape
    H, W = wet.shape[-3:-1]
    weight = wet.reshape(*sh[:-3], -1)  # B, H, W, 1 ->  B, -1 (H * W)

    if weight.sum() == weight.size:  # ?: Will this be too slow?
        indices = np.random.randint(0, H * W, (n_rays,))  # MARK: 0.1ms
    else:
        indices = np.random.multinomial(weight, n_rays, replacement=True)  # B, P, # MARK: 3-5ms
    coords = indices_to_coords(indices, H, W)  # B, P, 2 # MARK: 0.xms
    return coords


def parameterize(x: jt.Var, v: jt.Var):
    s = (-x * v).sum(dim=-1, keepdim=True)  # closest distance along ray to origin
    o = s * v + x  # closest point along ray to origin
    return o, s
