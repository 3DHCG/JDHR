import jittor as jt
from jittor import nn
from jdhr.engine import SUPERVISORS
from jdhr.engine.registry import call_from_cfg

from jdhr.utils.console_utils import *
from jdhr.utils.image_utils import resize_image
from jdhr.utils.loss_utils import lossfun_outer, lossfun_distortion

from jdhr.models.supervisors.volumetric_video_supervisor import VolumetricVideoSupervisor


@SUPERVISORS.register_module()
class ProposalSupervisor(VolumetricVideoSupervisor):
    def __init__(self,
                 network: nn.Module,

                 prop_loss_weight: float = 1.0,
                 dist_loss_weight: float = 0.0,  # mipnerf360 distorsion loss

                 **kwargs,
                 ):
        call_from_cfg(super().__init__, kwargs, network=network)

        self.prop_loss_weight = prop_loss_weight
        self.dist_loss_weight = dist_loss_weight

    def compute_loss(self, output: dotdict, batch: dotdict, loss: jt.Var, scalar_stats: dotdict, image_stats: dotdict):

        def compute_image_loss(rgb_map: jt.Var, rgb_gt: jt.Var,
                               bg_color: jt.Var, msk_gt: jt.Var,
                               H: int = batch.meta.H[0], W: int = batch.meta.W[0],
                               type=self.img_loss_type):
            return self.compute_image_loss(rgb_map, rgb_gt, bg_color, msk_gt, H, W, type)

        # Compute the actual loss here
        if 's_vals' in output and 'weights' in output and \
           self.dist_loss_weight > 0:
            dist_loss = 0.0
            dist_loss += lossfun_distortion(output.s_vals, output.weights).mean()

            if 's_vals_prop' in output and 'weights_prop' in output:
                for i in range(len(output.s_vals_prop)):
                    dist_loss += lossfun_distortion(output.s_vals_prop[i], output.weights_prop[i]).mean()

            scalar_stats.dist_loss = dist_loss
            loss += self.dist_loss_weight * dist_loss

        if 'rgb_maps_prop' in output and 'bg_colors_prop' in output and \
           len(output.rgb_maps_prop) and len(output.bg_colors_prop) and \
           self.prop_loss_weight > 0:
            prop_loss = 0
            for i in range(len(output.rgb_maps_prop)):
                rgb_gt = batch.rgb
                msk_gt = batch.msk
                rgb_map = output.rgb_maps_prop[i]
                bg_color = output.bg_colors_prop[i]
                H, W = batch.meta.H[0].item(), batch.meta.W[0].item()
                if 'Hrs_prop' in output and 'Wrs_prop' in output:
                    B = rgb_map.shape[0]
                    Hr, Wr = output.Hrs_prop[i], output.Wrs_prop[i]
                    rgb_gt = resize_image(rgb_gt.view(B, H, W, -1), size=(Hr, Wr)).view(B, Hr * Wr, -1)  # B, P, 3
                    msk_gt = resize_image(msk_gt.view(B, H, W, -1), size=(Hr, Wr)).view(B, Hr * Wr, -1)  # B, P, 3
                    H, W = Hr, Wr
                _, prop_loss_i = compute_image_loss(rgb_map, rgb_gt, bg_color, msk_gt, H, W)
                prop_loss += prop_loss_i

                if self.perc_loss_weight != 0:
                    _, prop_perc_loss_i = compute_image_loss(rgb_map, rgb_gt, bg_color, msk_gt, H, W, type=ImgLossType.PERC)
                    prop_loss += prop_perc_loss_i * self.perc_loss_weight

            prop_loss = prop_loss / len(output.rgb_maps_prop)
            scalar_stats.prop_loss = prop_loss
            loss += self.prop_loss_weight * prop_loss

        if 's_vals_prop' in output and 'weights_prop' in output and \
           len(output.s_vals_prop) and len(output.weights_prop) and \
           's_vals' in output and 'weights' in output and \
           self.prop_loss_weight > 0:
            prop_loss = 0
            for i in range(len(output.s_vals_prop)):
                prop_loss += lossfun_outer(
                    output.s_vals.detach(), output.weights.detach(),
                    output.s_vals_prop[i], output.weights_prop[i])
            prop_loss = prop_loss.mean()
            scalar_stats.prop_loss = prop_loss
            loss += self.prop_loss_weight * prop_loss

        return loss
