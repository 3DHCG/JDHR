#import torch
#import torch.nn as nn
#import torch.nn.functional as F
import jittor as jt
from jittor import nn

def grid_sample_1d(input:jt.Var , grid:jt.Var , *args, **kwargs):
    # input: B, C, L_in
    # grid: B, 1, L_out

    # Add a dimension to pass through torch's implementation
    # Though https://github.com/luo3300612/grid_sample1d/blob/main/grid_sample1d/grid_sample1d_cuda_kernel.cu
    # and https://github.com/AliaksandrSiarohin/cuda-gridsample-grad2
    # might be better than this makeshift
    # Will temporarily use torch implementation
    # Maybe switch to cuda one in future impls

    # This snippet is copied from: https://github.com/luo3300612/grid_sample1d
    input = input[..., None]  # B, C, L_in, 1, use H
    grid = grid[..., None, :]  # B, L_out, 1, 1, use W
    grid = jt.cat([jt.ones_like(grid) / 2, grid], dim=-1)
    z = nn.grid_sample(input, grid, *args, **kwargs)  # B, C, L_out, 1
    z = z[..., 0]  # B, C, L_out, 1 -> B, C, L_out
    return z


def grid_sample(input:jt.Var , grid:jt.Var , *args, **kwargs)->jt.Var:
    # https://github.com/pytorch/pytorch/issues/34704
    # RuntimeError: derivative for grid_sampler_2d_backward is not implemented
    # this implementation might be slower than the cuda one
    if args or kwargs:
        # warnings.warn(message=f'unused arguments for grid_sample: {args}, {kwargs}')
        return nn.grid_sample(input, grid, *args, **kwargs)
    if input.ndim == 3:
        # invoke 1d grid_sampling that falls through
        assert grid.ndim == 3, '3d input needs a 3d grid'
        return grid_sample_1d(input, grid, args, kwargs)
    elif input.ndim == 4:
        # invoke 2d custom grid_sampling
        assert grid.ndim == 4, '4d input needs a 4d grid'
        return grid_sample_2d(input, grid)
    elif input.ndim == 5:
        # invoke 3d custom grid_sampling
        assert grid.ndim == 5, '5d input needs a 5d grid'
        return grid_sample_3d(input, grid)
    else:
        raise NotImplementedError(f'grid_sample not implemented for {input.ndim}d input')


def grid_sample_2d(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    with jt.no_grad():
        ix_nw = jt.floor(ix)
        iy_nw = jt.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    with jt.no_grad():
        jt.clamp(ix_nw, 0, IW - 1, out=ix_nw)
        jt.clamp(iy_nw, 0, IH - 1, out=iy_nw)

        jt.clamp(ix_ne, 0, IW - 1, out=ix_ne)
        jt.clamp(iy_ne, 0, IH - 1, out=iy_ne)

        jt.clamp(ix_sw, 0, IW - 1, out=ix_sw)
        jt.clamp(iy_sw, 0, IH - 1, out=iy_sw)

        jt.clamp(ix_se, 0, IW - 1, out=ix_se)
        jt.clamp(iy_se, 0, IH - 1, out=iy_se)

    image = image.view(N, C, IH * IW)

    nw_val = jt.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = jt.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = jt.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = jt.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) +
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val


def grid_sample_3d(image, optical):
    N, C, ID, IH, IW = image.shape
    _, D, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]
    iz = optical[..., 2]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    iz = ((iz + 1) / 2) * (ID - 1)
    with jt.no_grad():

        ix_tnw = jt.floor(ix)
        iy_tnw = jt.floor(iy)
        iz_tnw = jt.floor(iz)

        ix_tne = ix_tnw + 1
        iy_tne = iy_tnw
        iz_tne = iz_tnw

        ix_tsw = ix_tnw
        iy_tsw = iy_tnw + 1
        iz_tsw = iz_tnw

        ix_tse = ix_tnw + 1
        iy_tse = iy_tnw + 1
        iz_tse = iz_tnw

        ix_bnw = ix_tnw
        iy_bnw = iy_tnw
        iz_bnw = iz_tnw + 1

        ix_bne = ix_tnw + 1
        iy_bne = iy_tnw
        iz_bne = iz_tnw + 1

        ix_bsw = ix_tnw
        iy_bsw = iy_tnw + 1
        iz_bsw = iz_tnw + 1

        ix_bse = ix_tnw + 1
        iy_bse = iy_tnw + 1
        iz_bse = iz_tnw + 1

    tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz)
    tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz)
    tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz)
    tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz)
    bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse)
    bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw)
    bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne)
    bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw)

    with jt.no_grad():

        jt.clamp(ix_tnw, 0, IW - 1, out=ix_tnw)
        jt.clamp(iy_tnw, 0, IH - 1, out=iy_tnw)
        jt.clamp(iz_tnw, 0, ID - 1, out=iz_tnw)

        jt.clamp(ix_tne, 0, IW - 1, out=ix_tne)
        jt.clamp(iy_tne, 0, IH - 1, out=iy_tne)
        jt.clamp(iz_tne, 0, ID - 1, out=iz_tne)

        jt.clamp(ix_tsw, 0, IW - 1, out=ix_tsw)
        jt.clamp(iy_tsw, 0, IH - 1, out=iy_tsw)
        jt.clamp(iz_tsw, 0, ID - 1, out=iz_tsw)

        jt.clamp(ix_tse, 0, IW - 1, out=ix_tse)
        jt.clamp(iy_tse, 0, IH - 1, out=iy_tse)
        jt.clamp(iz_tse, 0, ID - 1, out=iz_tse)

        jt.clamp(ix_bnw, 0, IW - 1, out=ix_bnw)
        jt.clamp(iy_bnw, 0, IH - 1, out=iy_bnw)
        jt.clamp(iz_bnw, 0, ID - 1, out=iz_bnw)

        jt.clamp(ix_bne, 0, IW - 1, out=ix_bne)
        jt.clamp(iy_bne, 0, IH - 1, out=iy_bne)
        jt.clamp(iz_bne, 0, ID - 1, out=iz_bne)

        jt.clamp(ix_bsw, 0, IW - 1, out=ix_bsw)
        jt.clamp(iy_bsw, 0, IH - 1, out=iy_bsw)
        jt.clamp(iz_bsw, 0, ID - 1, out=iz_bsw)

        jt.clamp(ix_bse, 0, IW - 1, out=ix_bse)
        jt.clamp(iy_bse, 0, IH - 1, out=iy_bse)
        jt.clamp(iz_bse, 0, ID - 1, out=iz_bse)

    image = image.view(N, C, ID * IH * IW)

    tnw_val = jt.gather(image, 2, (iz_tnw * IW * IH + iy_tnw * IW + ix_tnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tne_val = jt.gather(image, 2, (iz_tne * IW * IH + iy_tne * IW + ix_tne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tsw_val = jt.gather(image, 2, (iz_tsw * IW * IH + iy_tsw * IW + ix_tsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tse_val = jt.gather(image, 2, (iz_tse * IW * IH + iy_tse * IW + ix_tse).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bnw_val = jt.gather(image, 2, (iz_bnw * IW * IH + iy_bnw * IW + ix_bnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bne_val = jt.gather(image, 2, (iz_bne * IW * IH + iy_bne * IW + ix_bne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bsw_val = jt.gather(image, 2, (iz_bsw * IW * IH + iy_bsw * IW + ix_bsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bse_val = jt.gather(image, 2, (iz_bse * IW * IH + iy_bse * IW + ix_bse).long().view(N, 1, D * H * W).repeat(1, C, 1))

    out_val = (tnw_val.view(N, C, D, H, W) * tnw.view(N, 1, D, H, W) +
               tne_val.view(N, C, D, H, W) * tne.view(N, 1, D, H, W) +
               tsw_val.view(N, C, D, H, W) * tsw.view(N, 1, D, H, W) +
               tse_val.view(N, C, D, H, W) * tse.view(N, 1, D, H, W) +
               bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W) +
               bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W) +
               bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W) +
               bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W))

    return out_val
