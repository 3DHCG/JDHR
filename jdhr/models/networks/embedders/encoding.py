# --- bulit in ---
import math
# --- 3rd party ---
import numpy as np
import jittor as  jt
from jittor import nn
# --- my module ---
from jdhr.engine import EMBEDDERS
"""
The MIT License (MIT)
Copyright (c) 2022 Joe Hsiao (Ending2015a)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.
"""

# --- constants ---
PRIMES = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]


@jt.no_grad()
def fast_hash(ind: jt.Var, primes: jt.Var, hashmap_size: int):
    """Hashing function from:
    https://github.com/NVlabs/tiny-cuda-nn/blob/master/include/tiny-cuda-nn/encodings/grid.h#L76-L92
    """
    d = ind.shape[-1]
    ind = (ind * primes[:d]) & 0xffffffff  # uint32
    for i in range(1, d):
      ind[..., 0] ^= ind[..., i]
    return ind[..., 0] % hashmap_size

class _HashGrid(nn.Module):
    def __init__(
        self,
        dim: int,
        n_features: int,
        hashmap_size: int,
        resolution: float
        ):
        super().__init__()
        self.dim = dim
        self.n_features = n_features
        self.hashmap_size = hashmap_size
        self.resolution = resolution

        # you can add more primes for supporting more dimensions
        assert self.dim <= len(PRIMES), \
          f"HashGrid only supports < {len(PRIMES)}-D inputs"

        # create look-up table
        self.embedding = nn.Embedding(hashmap_size, n_features)


        self._primes = jt.array(PRIMES, dtype=jt.int32)


        # create interpolation binary mask
        n_neigs = 1 << self.dim
        neigs = np.arange(n_neigs, dtype=np.int32).reshape((-1, 1))
        dims = np.arange(self.dim, dtype=np.int32).reshape((1, -1))
        self._bin_mask = jt.array(neigs & (1 << dims) == 0, dtype=bool) # (neig, dim)


    def execute(self, x: jt.Var):
        # x: (b..., dim), torch.float32, range: [0, 1]
        bdims = len(x.shape[:-1])
        x = x * self.resolution
        xi = x.long()
        xf = x - xi.float().detach()
        xi = xi.unsqueeze(dim=-2) # (b..., 1, dim)
        xf = xf.unsqueeze(dim=-2) # (b..., 1, dim)
        # to match the input batch shape
        _bin_mask = self._bin_mask.reshape((1,)*bdims + self._bin_mask.shape) # (1..., neig, dim)
        # get neighbors' indices and weights on each dim
        inds = jt.ternary(_bin_mask, xi, xi+1) # (b..., neig, dim)
        ws = jt.ternary(_bin_mask, 1-xf, xf) # (b...., neig, dim)
        # aggregate nehgibors' interp weights
        w = ws.prod(dim=-1, keepdim=True) # (b..., neig, 1)
        # hash neighbors' id and look up table
        hash_ids = fast_hash(inds, self._primes, self.hashmap_size) # (b..., neig)
        neig_data = self.embedding(hash_ids) # (b..., neig, feat)
        return jt.sum(neig_data * w, dim=-2) # (b..., feat)

@EMBEDDERS.register_module()
class MultiResHashGrid(nn.Module):
    #from jdhr.models.cameras.optimizable_camera import OptimizableCamera
    def __init__(
        self,
        in_dim: int,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 15,
        base_resolution: int = 16,
        finest_resolution: int = 2048,
        *args,**kwargs,

        ):
        """NVidia's hash grid encoding
        https://nvlabs.github.io/instant-ngp/

        The output dimensions is `n_levels` * `n_features_per_level`,
        or your can simply access `model.output_dim` to get the output dimensions

        Args:
          dim (int): input dimensions, supports at most 7D data.
          n_levels (int, optional): number of grid levels. Defaults to 16.
          n_features_per_level (int, optional): number of features per grid level.
            Defaults to 2.
          log2_hashmap_size (int, optional): maximum size of the hashmap of each
            level in log2 scale. According to the paper, this value can be set to
            14 ~ 24 depending on your problem size. Defaults to 15.
          base_resolution (int, optional): coarsest grid resolution. Defaults to 16.
          finest_resolution (int, optional): finest grid resolution. According to
            the paper, this value can be set to 512 ~ 524288. Defaults to 512.
        """
        super().__init__()
        self.dim = in_dim
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution

        # from paper eq (3)
        if self.n_levels==1:
            b=1
        else:
            b = math.exp((math.log(self.finest_resolution) - math.log(self.base_resolution))/(self.n_levels-1))

        levels = []
        for level_idx in range(self.n_levels):
          resolution = math.floor(self.base_resolution * (b ** level_idx))
          hashmap_size = min(resolution ** self.dim, 2 ** self.log2_hashmap_size)
          levels.append(_HashGrid(
            dim = self.dim,
            n_features = self.n_features_per_level,
            hashmap_size = hashmap_size,
            resolution = resolution
          ))
        self.levels = nn.ModuleList(levels)

        self.input_dim = self.dim
        self.out_dim = self.n_levels * self.n_features_per_level

    def execute(self, x: jt.Var):

        return jt.cat([level(x) for level in self.levels], dim=-1)
