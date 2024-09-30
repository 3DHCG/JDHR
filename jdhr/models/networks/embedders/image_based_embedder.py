from jittor import nn
import jittor as jt
from jdhr.engine import EMBEDDERS, REGRESSORS
from jdhr.utils.base_utils import dotdict
from jdhr.utils.data_utils import to_x
from jdhr.utils.chunk_utils import chunkify
from jdhr.utils.ibr_utils import sample_feature_volume, sample_feature_image
from jdhr.utils.enerf_utils import FeatureAgg


@EMBEDDERS.register_module()
class ImageBasedEmbedder(nn.Module):
    def __init__(self,
                 agg_cfg: dotdict = dotdict(type=FeatureAgg.__name__),
                 use_vox_feat: bool = True,
                 use_img_feat: bool = True,
                 vox_dim: int = 8,  #
                 src_dim: int = 32 + 3,  #
                 chunk_size: int = 1e20,
                 dtype: str = jt.float,
                 **kwargs,
                 ) -> None:
        super().__init__()
        self.dtype = getattr(jt, dtype) if isinstance(dtype, str) else dtype
        if use_img_feat: self.agg = REGRESSORS.build(agg_cfg, feat_ch=src_dim).to(self.dtype)

        # assert use_vox_feat or use_img_feat, 'At least use some feature'
        self.use_vox_feat = use_vox_feat
        self.use_img_feat = use_img_feat
        self.chunk_size = chunk_size
        self.out_dim = 0
        if use_vox_feat: self.out_dim += vox_dim  # 8
        if use_img_feat: self.out_dim += self.agg.out_dim  # 16

    def execute(self, xyz: jt.Var, batch: dotdict):
        # xyz: B, P * S, 3

        # Find features inside batch
        # Return sampled features

        # If the xyz features are not present, perform sampling
        output = dotdict()

        # Extract things from batch
        src_feat_rgb = batch.output.src_feat_rgb  # last level of source image feature, B, S, C, Hs, Ws
        src_scale = batch.output.src_scale

        del batch.output.src_feat_rgb
        tar_ext, src_exts, src_ixts = to_x([batch.tar_ext, batch.src_exts, batch.src_ixts], xyz.dtype)

        # Sample volume feature (after 3D CNN regression on cost volume)
        if self.use_vox_feat:
            feat_vol = batch.output.feat_vol  # last level of feature volume, B, C, D, Ht, Wt
            s_vals = batch.output.s_vals  # bad naming? B, Ht, Wt, N
            tar_scale_cpu = batch.output.meta.tar_scale
            ren_scale_cpu = batch.output.meta.ren_scale
            del batch.output.feat_vol
            if 'layer_idx' in batch and batch.layer_idx >= 0:
                x, y, w, h = batch.meta.objects_xywh[0, batch.layer_idx].cpu().tolist()
                vox_feat = sample_feature_volume(s_vals, feat_vol, ren_scale_cpu, tar_scale_cpu, x=x, y=y, w=w, h=h)  # B, P, C, # !: CHUNK
            else:
                vox_feat = sample_feature_volume(s_vals, feat_vol, ren_scale_cpu, tar_scale_cpu)  # B, P, C, # !: CHUNK

        # Supports chunkify
        def chunked_sample_feature_image(xyz: jt.Var):
            @chunkify(self.chunk_size, ignore_mismatch=True)
            def reshaped_sample_feature_image(xyz: jt.Var):
                ret = sample_feature_image(xyz, src_feat_rgb, tar_ext, src_exts, src_scale, src_ixts)
                ret = ret.view(-1, *ret.shape[2:])
                return ret
            ret = reshaped_sample_feature_image(xyz)
            ret = ret.view(xyz.shape[0], -1, *ret.shape[1:])
            return ret

        # Sample image feature
        src_feat_rgb_dir = chunked_sample_feature_image(xyz)  # B, S, P, C

        # Store output
        output.src_feat_rgb_dir = src_feat_rgb_dir  # B, S, P, C
        batch.output.update(output)

        # Aggregate image feature
        if not self.use_vox_feat and not self.use_img_feat: return xyz.new_zeros(*xyz.shape[:-1], 0)
        ret = []
        if self.use_vox_feat: ret.append(vox_feat)
        if self.use_img_feat: ret.append(self.agg(src_feat_rgb_dir))  # B, S, P, C -> B, P, C
        ret = jt.cat(ret, dim=-1)
        return ret
