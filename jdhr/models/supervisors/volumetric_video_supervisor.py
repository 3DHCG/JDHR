# Default loss module (called supervisor)
import numpy as np
from typing import Union
import jittor as jt
from jittor import nn
from jdhr.engine import SUPERVISORS
from jdhr.utils.console_utils import *
from jdhr.utils.net_utils import VolumetricVideoModule
from jdhr.utils.loss_utils import l1, wl1, huber, ssim, msssim, lpips, ImgLossType


@SUPERVISORS.register_module()
class VolumetricVideoSupervisor(VolumetricVideoModule):
    def __init__(self,
                 network: nn.Module,
                 img_loss_weight: float = 1.0,  # main reconstruction loss
                 img_loss_type: ImgLossType = ImgLossType.HUBER.name,  # chabonier loss for img_loss
                 perc_loss_weight: float = 0.0,  # smaller loss on perc
                 ssim_loss_weight: float = 0.0,  # 3dgs ssim loss
                 msssim_loss_weight: float = 0.0,  # 3dgs msssim loss
                 dtype: Union[str, jt.dtype] = jt.float,

                 **kwargs,
                 ):
        super().__init__(network)
        self.execute = self.supervise
        self.dtype = getattr(jt, dtype) if isinstance(dtype, str) else dtype

        # Image reconstruction loss
        self.img_loss_weight = img_loss_weight
        self.img_loss_type = ImgLossType[img_loss_type]
        self.perc_loss_weight = perc_loss_weight
        self.ssim_loss_weight = ssim_loss_weight
        self.msssim_loss_weight = msssim_loss_weight

    def compute_image_loss(self, rgb_map: jt.Var, rgb_gt: jt.Var,
                           bg_color: jt.Var, msk_gt: jt.Var,
                           H: int, W: int,
                           type=ImgLossType.HUBER,
                           **kwargs,):

        rgb_gt = rgb_gt + bg_color * (1 - msk_gt)  # MARK: modifying gt for supervision
        H=int(H)
        W=int(W)
        rgb_gt, rgb_map = rgb_gt[:, :H * W], rgb_map[:, :H * W]

        # https://stackoverflow.com/questions/181530/styling-multi-line-conditions-in-if-statements
        resd_sq = (rgb_map - rgb_gt)**2
        mse = resd_sq.mean()
        psnr = (1 / mse.clamp(1e-10)).log() * 10 / np.log(10)

        if type == ImgLossType.PERC:
            rgb_gt = rgb_gt.view(-1, H, W, 3).permute(0, 3, 1, 2)  # B, C, H, W
            rgb_map = rgb_map.view(-1, H, W, 3).permute(0, 3, 1, 2)  # B, C, H, W
            img_loss = lpips(rgb_map, rgb_gt)
        elif type == ImgLossType.CHARB: img_loss = (resd_sq + 0.001 ** 2).sqrt().mean()
        elif type == ImgLossType.HUBER: img_loss = huber(rgb_map, rgb_gt)
        elif type == ImgLossType.L2: img_loss = mse
        elif type == ImgLossType.L1: img_loss = l1(rgb_map, rgb_gt)
        elif type == ImgLossType.SSIM:
            rgb_gt = rgb_gt.view(-1, H, W, 3).permute(0, 3, 1, 2)  # B, C, H, W
            rgb_map = rgb_map.view(-1, H, W, 3).permute(0, 3, 1, 2)  # B, C, H, W
            img_loss = 1. - ssim(rgb_map, rgb_gt)
        elif type == ImgLossType.MSSSIM:
            rgb_gt = rgb_gt.view(-1, H, W, 3).permute(0, 3, 1, 2)  # B, C, H, W
            rgb_map = rgb_map.view(-1, H, W, 3).permute(0, 3, 1, 2)  # B, C, H, W
            img_loss = 1. - msssim(rgb_map, rgb_gt)
        elif type == ImgLossType.WL1:
            rgb_gt = rgb_gt.view(-1, H, W, 3).permute(0, 3, 1, 2)  # B, C, H, W
            img_loss_wet = kwargs['img_loss_wet'].view(-1, H, W, 1).permute(0, 3, 1, 2)
            rgb_map = rgb_map.view(-1, H, W, 3).permute(0, 3, 1, 2)  # B, C, H, W
            img_loss = wl1(rgb_map, rgb_gt, img_loss_wet)
        else: raise NotImplementedError

        return psnr, img_loss

    def compute_loss(self, output: dotdict, batch: dotdict, loss: jt.Var, scalar_stats: dotdict, image_stats: dotdict):

        # NOTE: a loss will be computed and logged if
        # 1. the corresponding loss weight is bigger than zero
        # 2. the corresponding components exist in the output
        def compute_image_loss(rgb_map: jt.Var, rgb_gt: jt.Var,
                               bg_color: jt.Var, msk_gt: jt.Var,
                               H: int = batch.meta.H[0], W: int = batch.meta.W[0],
                               type=self.img_loss_type,
                               **kwargs):
            return self.compute_image_loss(rgb_map, rgb_gt, bg_color, msk_gt, H, W, type, **kwargs)

        if 'rgb_map' in output and self.perc_loss_weight > 0 and batch.meta.n_rays[0].item() == -1:
            if 'patch_h' in batch.meta and 'patch_w' in batch.meta:
                H, W = batch.meta.patch_h[0], batch.meta.patch_w[0]
            else:
                H, W = batch.meta.H[0], batch.meta.W[0]
            H=int(H)
            W=int(W)
            _, perc_loss = compute_image_loss(output.rgb_map, batch.rgb, output.bg_color, batch.msk, H, W, type=ImgLossType.PERC)
            scalar_stats.perc_loss = perc_loss
            loss += self.perc_loss_weight * perc_loss

        if 'rgb_map' in output and \
           self.ssim_loss_weight > 0 and \
           batch.meta.n_rays[0].item() == -1:
            if 'patch_h' in batch.meta and 'patch_w' in batch.meta:
                H, W = batch.meta.patch_h[0], batch.meta.patch_w[0]
            else:
                H, W = batch.meta.H[0].item(), batch.meta.W[0].item()
            H=int(H)
            W=int(W)
            _, ssim_loss = compute_image_loss(output.rgb_map, batch.rgb, output.bg_color, batch.msk, H, W, type=ImgLossType.SSIM)
            scalar_stats.ssim_loss = ssim_loss
            loss += self.ssim_loss_weight * ssim_loss

        if 'rgb_map' in output and \
                self.msssim_loss_weight > 0 and \
                batch.meta.n_rays[0].item() == -1:
            if 'patch_h' in batch.meta and 'patch_w' in batch.meta:
                H, W = batch.meta.patch_h[0], batch.meta.patch_w[0]
            else:
                H, W = batch.meta.H[0], batch.meta.W[0]
            H=int(H)
            W=int(W)
            _, msssim_loss = compute_image_loss(output.rgb_map, batch.rgb, output.bg_color, batch.msk, H, W, type=ImgLossType.MSSSIM)
            scalar_stats.msssim_loss = msssim_loss
            loss += self.msssim_loss_weight * msssim_loss

        if 'rgb_map' in output and \
           self.img_loss_weight > 0:
            if 'img_loss_wet' in batch:
                psnr, img_loss = compute_image_loss(output.rgb_map, batch.rgb, output.bg_color, batch.msk, img_loss_wet=batch.img_loss_wet)
            else:
                psnr, img_loss = compute_image_loss(output.rgb_map, batch.rgb, output.bg_color, batch.msk)
            scalar_stats.psnr = psnr
            scalar_stats.img_loss = img_loss
            loss += self.img_loss_weight * img_loss

        return loss

    # If we want to split the supervisor to include more sets of losses?
    def supervise(self, output: dotdict, batch: dotdict):
        loss = output.get('loss', 0)   # accumulated final loss
        loss_stats = output.get('loss_stats', dotdict())  # give modules ability to record something
        image_stats = output.get('image_stats', dotdict())  # give modules ability to record something
        scalar_stats = output.get('scalar_stats', dotdict())  # give modules ability to record something

        loss = self.compute_loss(output, batch, loss, scalar_stats, image_stats)

        for k, v in loss_stats.items():
            loss += v.mean()  # network computed loss

        output.loss = loss
        loss_stats.loss = loss  # these are the things to accumulated for loss
        scalar_stats.loss = loss  # these are the things to record and log
        return loss, scalar_stats, image_stats
