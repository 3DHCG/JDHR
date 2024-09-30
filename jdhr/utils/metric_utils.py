"""
Given images, output scalar metrics on CPU
Used for evaluation. For training, please check out loss_utils
"""

#import torch
import jittor as jt
import numpy as np

from jdhr.utils.console_utils import *
from jdhr.utils.loss_utils import mse as compute_mse
from jdhr.utils.loss_utils import lpips as compute_lpips
from skimage.metrics import structural_similarity as compare_ssim

from enum import Enum, auto


@jt.no_grad()
def psnr(x: jt.Var, y: jt.Var):
    mse = compute_mse(x, y).mean()
    psnr = (1 / mse.clamp(1e-10)).log() * 10 / np.log(10)
    return psnr.item()  # tensor to scalar


@jt.no_grad()
def ssim(x: jt.Var, y: jt.Var):
    return np.mean([
        compare_ssim(
            _x.detach().cpu().numpy(),
            _y.detach().cpu().numpy(),
            channel_axis=-1,
            data_range=2.0
        )
        for _x, _y in zip(x, y)
    ]).astype(float).item()


@jt.no_grad()
def lpips(x: jt.Var, y: jt.Var):
    if x.ndim == 3: x = x.unsqueeze(0)
    if y.ndim == 3: y = y.unsqueeze(0)
    x = x.permute(0, 3, 1, 2)
    y = y.permute(0, 3, 1, 2)
    return compute_lpips(x, y).item()


class Metrics(Enum):
    PSNR = psnr
    SSIM = ssim
    LPIPS = lpips
