from jittor import nn
import jittor as jt
from jdhr.engine import EMBEDDERS, REGRESSORS
from jdhr.utils.base_utils import dotdict
from jdhr.utils.fcds_utils import update_features
from jdhr.utils.pointnet2_utils import PointNeRFAggregator
from jdhr.models.networks.embedders.positional_encoding_embedder import PositionalEncodingEmbedder


@EMBEDDERS.register_module()
class PointNeRFEmbedder(nn.Module):
    def __init__(self,
                 in_dim: int = 64,  # smaller input dim to fit inside the memory
                 radius: float = 0.05,  # aggregation
                 width: int = 64,
                 depth: int = 2,
                 K: int = 5,  # otherwise oom
                 ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim
        self.aggregator = PointNeRFAggregator(in_dim=in_dim + 3, out_dim=in_dim, K=K, W=width, D=depth, radius=radius)  # will be used in aggregation

    def execute(self, xyz: jt.Var, batch: dotdict):
        # xyz: B, P * S, 3

        # Find features inside batch
        # Return sampled features
        density, confidence, weights, feat = self.aggregator(xyz, batch.output.pcd, batch.output.feat)
        batch.output.confidence = confidence # B, S, K, 1

        density = (density * confidence * weights).sum(dim=-2)
        feat = (feat * confidence * weights).sum(dim=-2)
        batch.output.density = density
        return feat
