import jittor as jt
from jittor import nn
from jdhr.engine import REGRESSORS
from jdhr.utils.net_utils import MLP
from jdhr.utils.base_utils import dotdict
from jdhr.utils.chunk_utils import chunkify


@REGRESSORS.register_module()
class IbrRegressor(nn.Module):
    def __init__(self,
                 in_dim: int,  # feature channel dim (0, no `geo_feat`, `rgb_feat` or `dir_feat`)
                 src_dim: int = 32 + 3 + 4,  #  (feat, rgb, dir)

                 width: int = 64,
                 depth: int = 1,  # use small regressor network

                 manual_chunking: bool = False,
                 dtype: str = 'float',

                 chunk_size: int = 1e20,
                 ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.manual_chunking = manual_chunking

        self.mlp = MLP(in_dim + src_dim, width, depth, 1, out_actvn=nn.Identity(), dtype=dtype)  # vox(8) + img(16) + geo(64) + img_feat_rgb_dir (32 + 3 + 4)
        self.forward_mlp = chunkify(chunk_size=chunk_size)(self.mlp)

    def execute(self, geo_feat: jt.Var, batch: dotdict):
        # Prepare feature for image based rendering blending (in a narrow sense)
        src_msks: jt.Var = batch.output.src_msks if 'src_msks' in batch.output else None  # (B, S, P, 1)
        src_rgbs: jt.Var = batch.output.src_rgbs  # (B, S, P, 3)
        app_feat: jt.Var = batch.output.app_feat  # (B, S, P, C)

        # Remember shapes
        B, S, P, _ = app_feat.shape

        # Manual chunking, hopefully this will ease the memory usage
        if self.manual_chunking:
            rgb_bws = []
            for i in range(B):  # TODO: PERF
                for j in range(S):
                    feat = jt.cat([geo_feat[i], app_feat[i, j]], dim=-1)  # (P, C)
                    rgb_bws.append(self.forward_mlp(feat))  # (P, 1)
            rgb_bws = jt.stack(rgb_bws)  # (B * S, P, 1)
            rgb_bws = rgb_bws.view(B, S, P, -1)  # (B, S, P, 1)
        else:
            geo_feat = geo_feat[:, None].expand(app_feat.shape[:-1] + (geo_feat.shape[-1], ))  # (B, S, P, C)
            geo_feat = jt.cat([geo_feat, app_feat], dim=-1)  # +7, append the actual image feature
            del app_feat  # release some memory
            rgb_bws = self.forward_mlp(geo_feat)  # (B, S, P, 1)

        # Perform image based rendering， namely source images weighted average.
        if src_msks: rgb_bws = rgb_bws.masked_fill(src_msks == 0, 1e-9)  # (B, S, P, 1)
        rgb_bws = rgb_bws.softmax(-3)  # (B, S, P, 1)
        rgb = (src_rgbs * rgb_bws).sum(-3)  # (B, P, 3)
        return rgb  # this regressor only produces rgb of revised enerf for now, need more care for reusing
