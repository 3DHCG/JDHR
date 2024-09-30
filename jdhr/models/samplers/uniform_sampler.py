import jittor as jt
from jittor import nn
from jdhr.engine import cfg
from jdhr.engine import SAMPLERS
from jdhr.utils.base_utils import dotdict
from jdhr.utils.prop_utils import s_vals_to_z_vals
from jdhr.utils.net_utils import VolumetricVideoModule
from jdhr.utils.nerf_utils import linear_sampling, ray2xyz


@SAMPLERS.register_module()
class UniformSampler(VolumetricVideoModule):
    # Could be uniform in anything, uniform in disparity or weighted uniform
    def __init__(self,
                 network: nn.Module,
                 uniform_disparity: bool = False,  # uniform sampling in disparity space or not
                 n_samples: int = 64,  # number of samples

                 **kwargs,
                 ):
        super().__init__(network, **kwargs)
        # I feared this would lead to extra disk usage (saving multiple copies), it didn't
        # Only a saving and loading overhead (loading the sampler would also overwrite previous datasets)
        self.uniform_disparity = uniform_disparity
        self.n_samples = n_samples

        self.g = (lambda x: 1 / (x + 1e-10)) if uniform_disparity else (lambda x: x)
        self.ig = (lambda x: 1 / (x + 1e-10)) if uniform_disparity else (lambda x: x)

        self.execute = self.sample

    def sample_depth(self,
                     ray_o: jt.Var, ray_d: jt.Var,  # not used, but needed for api,
                     near: jt.Var, far: jt.Var,
                     t: jt.Var,  # not used, but needed for api,
                     batch: dotdict,  # not used, but needed for api,
                     ):  # some sampler do not need batch input, give them the chance
        # ray_o: B, P, 3
        # ray_d: B, P, 3
        # t: B, P, 1
        # return: B, P, S, 3

        # Get shapes
        B, P, _ = near.shape
        S = self.n_samples

        # Actual sampling
        s_vals = linear_sampling(B, P, S, dtype=ray_o.dtype, perturb=self.training)  # 0 -> B, P, S

        # Uniform disparity or not
        z_vals = s_vals_to_z_vals(s_vals, near, far, g=self.g, ig=self.ig)

        output = dotdict()
        output.s_vals = s_vals  # for rendering depth and stuff?
        output.z_vals = z_vals  # for rendering depth and stuff?
        batch.output.update(output)
        return z_vals  # returns depth (sampler specific)

    def sample(self, ray_o: jt.Var, ray_d: jt.Var, near: jt.Var, far: jt.Var, t: jt.Var, batch: dotdict):
        z_vals = self.sample_depth(ray_o, ray_d, near, far, t, batch)
        xyz, dir, t, dist = ray2xyz(ray_o, ray_d, t, z_vals)  # B, P * S, 3 (for easy chunking)
        return xyz, dir, t, dist
