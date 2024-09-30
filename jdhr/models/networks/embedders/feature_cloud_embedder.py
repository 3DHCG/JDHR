import jittor as jt
from jittor import nn
from jdhr.engine import EMBEDDERS, REGRESSORS
from jdhr.utils.base_utils import dotdict
from jdhr.utils.fcds_utils import update_features
from jdhr.models.networks.embedders.positional_encoding_embedder import PositionalEncodingEmbedder


@EMBEDDERS.register_module()
class FeatureCloudEmbedder(nn.Module):
    def __init__(self,
                 in_dim: int = 64,  # smaller input dim to fit inside the memory
                 radius: float = 0.10,  # larger radius for aggregation
                 K: int = 10,  # otherwise oom

                 xyz_embedder_cfg: dotdict = dotdict(type=PositionalEncodingEmbedder.__name__),
                 ) -> None:
        super().__init__()

        self.in_dim = in_dim

        self.radius = radius
        self.K = K

        self.xyz_embedder = EMBEDDERS.build(xyz_embedder_cfg)
        self.out_dim = in_dim + self.xyz_embedder.out_dim

    def execute(self, xyz: jt.Var, batch: dotdict):
        # xyz: B, P * S, 3

        # Find features inside batch
        # Return sampled features

        # This feature is position agnostic
        fcd_feat = update_features(xyz, batch.output.pcd, batch.output.feat, self.radius, self.K)
        xyz_feat = self.xyz_embedder(xyz)
        return jt.cat([fcd_feat, xyz_feat], dim=-1)  # gives more positional information

