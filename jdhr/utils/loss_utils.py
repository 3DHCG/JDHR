#import torch
import jittor as jt
import numpy as np
#import torch.nn as nn
#import torch.nn.functional as F
import jittor.nn as nn
import jittor.models.vgg as vgg
from collections import namedtuple
from typing import Callable
from jittor import misc
from jittor import nn
from math import exp
#from torch.autograd import Variable

from jdhr.utils.prop_utils import searchsorted, matchup_channels
from jdhr.utils.console_utils import *

from enum import Enum, auto


class ElasticLossReduceType(Enum):
    WEIGHT = auto()
    MEDIAN = auto()


class ImgLossType(Enum):
    PERC = auto()  # lpips
    CHARB = auto()
    HUBER = auto()
    L1 = auto()
    L2 = auto()
    SSIM = auto()
    MSSSIM = auto()
    WL1 = auto()


class DptLossType(Enum):
    SMOOTHL1 = auto()
    L1 = auto()
    L2 = auto()
    SSIMSE = auto()
    SSIMAE = auto()
    SILOG = auto()
    CONTINUITY = auto()
    RANKING = auto()


def compute_val_pair_around_range(pts: jt.Var, decoder: Callable[[jt.Var], jt.Var], diff_range: float):
    # sample around input point and compute values
    # pts and its random neighbor are concatenated in second dimension
    # if needed, decoder should return multiple values together to save computation
    neighbor = pts + (jt.rand_like(pts) - 0.5) * diff_range
    full_pts = jt.cat([pts, neighbor], dim=-2)  # cat in n_masked dim
    raw: jt.Var = decoder(full_pts)  # (n_batch, n_masked, 3)
    return raw

# from mipnerf360


def inner_outer(t0, t1, y1):
    """Construct inner and outer measures on (t1, y1) for t0."""
    cy1 = jt.cat([jt.zeros_like(y1[..., :1]), jt.cumsum(y1, dim=-1)], dim=-1)  # 129
    idx_lo, idx_hi = searchsorted(t1, t0)

    cy1_lo = jt.gather(cy1, -1, idx_lo)  # 128
    cy1_hi = jt.gather(cy1, -1, idx_hi)

    y0_outer = cy1_hi[..., 1:] - cy1_lo[..., :-1]  # 127
    y0_inner = jt.ternary(idx_hi[..., :-1] <= idx_lo[..., 1:], cy1_lo[..., 1:] - cy1_hi[..., :-1], 0)
    return y0_inner, y0_outer

# from mipnerf360


def lossfun_outer(t: jt.Var, w: jt.Var, t_env: jt.Var, w_env: jt.Var, eps=jt.finfo('float32').eps):
    # accepts t.shape[-1] = w.shape[-1] + 1
    t, w = matchup_channels(t, w)
    t_env, w_env = matchup_channels(t_env, w_env)
    """The proposal weight should be an upper envelope on the nerf weight."""
    _, w_outer = inner_outer(t, t_env, w_env)
    # We assume w_inner <= w <= w_outer. We don't penalize w_inner because it's
    # more effective to pull w_outer up than it is to push w_inner down.
    # Scaled half-quadratic loss that gives a constant gradient at w_outer = 0.
    return (w - w_outer).clamp(0.).pow(2) / (w + eps)


def blur_stepfun(x, y, r):
    xr, xr_idx = jt.sort(jt.cat([x - r, x + r], dim=-1))
    y1 = (jt.cat([y, jt.zeros_like(y[..., :1])], dim=-1) -
          jt.cat([jt.zeros_like(y[..., :1]), y], dim=-1)) / (2 * r)
    y2 = jt.gather(jt.cat([y1, -y1], dim=-1),-1,xr_idx[..., :-1])
    yr = jt.cumsum((xr[..., 1:] - xr[..., :-1]) *
                      jt.cumsum(y2, dim=-1), dim=-1).clamp(min=0)
    yr = jt.cat([jt.zeros_like(yr[..., :1]), yr], dim=-1)
    return xr, yr


def sorted_interp_quad(x, xp, fpdf, fcdf):
    """interp in quadratic"""

    # Identify the location in `xp` that corresponds to each `x`.
    # The final `True` index in `mask` is the start of the matching interval.
    mask = x[..., None, :] >= xp[..., :, None]

    def find_interval(x, return_idx=False):
        # Grab the value where `mask` switches from True to False, and vice versa.
        # This approach takes advantage of the fact that `x` is sorted.
        x0_idx, x0 = jt.argmax(jt.ternary(mask, x[..., None], x[..., :1, None]), -2)
        x1_idx, x1 = jt.argmin(jt.ternary(mask.logical_not(), x[..., None], x[..., -1:, None]), -2)
        if return_idx:
            return x0, x1, x0_idx, x1_idx
        return x0, x1

    fcdf0, fcdf1, fcdf0_idx, fcdf1_idx = find_interval(fcdf, return_idx=True)
    fpdf0 = jt.gather(fpdf, -1,fcdf0_idx)
    fpdf1 = jt.gather(fpdf, -1,fcdf1_idx)
    xp0, xp1 = find_interval(xp)

    offset = jt.clamp((0 if jt.isnan((x - xp0) / (xp1 - xp0)) else (x - xp0) / (xp1 - xp0)), 0, 1)#torch.nan_to_num((x - xp0) / (xp1 - xp0), 0)
    ret = fcdf0 + (x - xp0) * (fpdf0 + fpdf1 * offset + fpdf0 * (1 - offset)) / 2
    return ret


def lossfun_zip_outer(t, w, t_env, w_env, pulse_width, eps=1e-6):
    t, w = matchup_channels(t, w)
    t_env, w_env = matchup_channels(t_env, w_env)

    w_normalize = w / jt.clamp(t[..., 1:] - t[..., :-1], min=eps)

    t_, w_ = blur_stepfun(t, w_normalize, pulse_width)
    w_ = jt.clamp(w_, min=0.)
    assert (w_ >= 0.0).all()

    # piecewise linear pdf to piecewise quadratic cdf
    area = 0.5 * (w_[..., 1:] + w_[..., :-1]) * (t_[..., 1:] - t_[..., :-1])

    cdf = jt.cat([jt.zeros_like(area[..., :1]), jt.cumsum(area, dim=-1)], dim=-1)

    # query piecewise quadratic interpolation
    cdf_interp = sorted_interp_quad(t_env, t_, w_, cdf)
    # difference between adjacent interpolated values
    w_s = jt.Var(np.diff(cdf_interp, dim=-1))#????

    return ((w_s - w_env).clamp(0.).pow(2) / (w_env + eps)).mean()


def lossfun_distortion(t: jt.Var, w: jt.Var):
    # accepts t.shape[-1] = w.shape[-1] + 1
    t, w = matchup_channels(t, w)
    """Compute iint w[i] w[j] |t[i] - t[j]| di dj."""
    # The loss incurred between all pairs of intervals.
    ut = (t[..., 1:] + t[..., :-1]) / 2  # 64
    dut = jt.abs(ut[..., :, None] - ut[..., None, :])  # 64
    loss_inter = jt.sum(w * jt.sum(w[..., None, :] * dut, dim=-1), dim=-1)

    # The loss incurred within each individual interval with itself.
    loss_intra = jt.sum(w**2 * (t[..., 1:] - t[..., :-1]), dim=-1) / 3

    return loss_inter + loss_intra


def interval_distortion(t0_lo, t0_hi, t1_lo, t1_hi):
    """Compute mean(abs(x-y); x in [t0_lo, t0_hi], y in [t1_lo, t1_hi])."""
    # Distortion when the intervals do not overlap.
    d_disjoint = jt.abs((t1_lo + t1_hi) / 2 - (t0_lo + t0_hi) / 2)

    # Distortion when the intervals overlap.
    d_overlap = (2 *
                 (jt.minimum(t0_hi, t1_hi)**3 - jt.maximum(t0_lo, t1_lo)**3) +
                 3 * (t1_hi * t0_hi * jt.abs(t1_hi - t0_hi) +
                      t1_lo * t0_lo * jt.abs(t1_lo - t0_lo) + t1_hi * t0_lo *
                      (t0_lo - t1_hi) + t1_lo * t0_hi *
                      (t1_lo - t0_hi))) / (6 * (t0_hi - t0_lo) * (t1_hi - t1_lo))

    # Are the two intervals not overlapping?
    are_disjoint = (t0_lo > t1_hi) | (t1_lo > t0_hi)

    return jt.ternary(are_disjoint, d_disjoint, d_overlap)


def anneal_loss_weight(weight: float, gamma: float, iter: int, mile: int):
    # exponentially anneal the loss weight
    return weight * gamma ** min(iter / mile, 1)


def gaussian_entropy_relighting4d(albedo_pred):
    albedo_entropy = 0
    for i in range(3):
        channel = albedo_pred[..., i]
        hist = GaussianHistogram(15, 0., 1., sigma=jt.var(channel))
        h = hist(channel)
        if h.sum() > 1e-6:
            h = h.div(h.sum()) + 1e-6
        else:
            h = jt.ones_like(h)
        albedo_entropy += jt.sum(-h * jt.log(h))
    return albedo_entropy


class GaussianHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(GaussianHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (jt.arange(bins).float() + 0.5)

    def execute(self, x):
        x = jt.unsqueeze(x, 0) - jt.unsqueeze(self.centers, 1)
        x = jt.exp(-0.5 * (x / self.sigma)**2) / (self.sigma * np.sqrt(np.pi * 2)) * self.delta
        x = x.sum(dim=1)
        return x


def gaussian_entropy(x: jt.Var, *args, **kwargs):
    eps = 1e-6
    hps = 1e-9
    h = gaussian_histogram(x, *args, **kwargs)
    # h = (h / (h.sum(dim=0) + hps)).clamp(eps)  # 3,
    # entropy = (-h * h.log()).sum(dim=0).sum(dim=0)  # per channel entropy summed
    entropy = 0
    for i in range(3):
        hi = h[..., i]
        if hi.sum() > eps:
            hi = hi / hi.sum() + eps
        else:
            hi = jt.ones_like(hi)
        entropy += jt.sum(-hi * jt.log(hi))
    return entropy


def gaussian_histogram(x: jt.Var, bins: int = 15, min: float = 0.0, max: float = 1.0):
    x = x.view(-1, x.shape[-1])  # N, 3
    sigma = x.var(dim=0)  # 3,
    delta = (max - min) / bins
    centers = min + delta * (jt.arange(bins, dtype=x.dtype) + 0.5)  # BIN
    x = x[None] - centers[:, None, None]  # BIN, N, 3
    x = (-0.5 * (x / sigma).pow(2)).exp() / (sigma * np.sqrt(np.pi * 2)) * delta  # BIN, N, 3
    x = x.sum(dim=1)
    return x  # BIN, 3


def reg_diff_crit(x: jt.Var, iter_step: int, max_weight: float = 1e-4, ann_iter: int = 100 * 500):
    weight = min(iter_step, ann_iter) * max_weight / ann_iter
    return reg(x), weight


def reg_raw_crit(x: jt.Var, iter_step: int, max_weight: float = 1e-4, ann_iter: int = 100 * 500):
    weight = min(iter_step, ann_iter) * max_weight / ann_iter
    n_batch, n_pts_x2, D = x.shape
    n_pts = n_pts_x2 // 2
    length = x.norm(dim=-1, keepdim=True)  # length
    vector = x / (length + 1e-8)  # vector direction (normalized to unit sphere)
    # loss_length = mse(length[:, n_pts:, :], length[:, :n_pts, :])
    loss_vector = reg((vector[:, n_pts:, :] - vector[:, :n_pts, :]))
    # loss = loss_length + loss_vector
    loss = loss_vector
    return loss, weight


def lpips(x: jt.Var, y: jt.Var):
    # B, 3, H, W
    # B, 3, H, W

    if not hasattr(lpips, 'compute_lpips'):
        from jdhr.utils.lpips.lpips import LPIPS
        log('Initializing LPIPS network')
        #print("shape",x.shape,y.shape,type(x),type(y))
        lpips.compute_lpips = LPIPS(net='vgg', verbose=True)

    return lpips.compute_lpips.execute(x , y ).mean()#?????


def eikonal(x: jt.Var, th=1.0) -> jt.Var:
    return ((x.norm(dim=-1) - th)**2).mean()


def sdf_mask_crit(ret, batch):
    msk_sdf = ret['msk_sdf']
    msk_label = ret['msk_label']

    alpha = 50
    alpha_factor = 2
    alpha_milestones = [10000, 20000, 30000, 40000, 50000]
    for milestone in alpha_milestones:
        if batch['iter_step'] > milestone:
            alpha = alpha * alpha_factor

    msk_sdf = -alpha * msk_sdf
    mask_loss = nn.binary_cross_entropy_with_logits(msk_sdf, msk_label) / alpha

    return mask_loss


def cross_entropy(x: jt.Var, y: jt.Var):
    # x: unormalized input logits
    # channel last cross entropy loss
    x = x.view(-1, x.shape[-1])  # N, C
    y = y.view(-1, y.shape[-1])  # N, C
    return nn.cross_entropy_loss(x, y)


def huber(x: jt.Var, y: jt.Var):
    return nn.smooth_l1_loss(x, y, reduction='mean')


def smoothl1(x: jt.Var, y: jt.Var):
    return nn.smooth_l1_loss(x, y)


def mse(x: jt.Var, y: jt.Var):
    return ((x.float() - y.float())**2).mean()


def dot(x: jt.Var, y: jt.Var):
    return (x * y).sum(dim=-1)


def l1(x: jt.Var, y: jt.Var):
    return l1_reg(x - y)


def wl1(x: jt.Var, y: jt.Var, w: jt.Var):
    return l1_reg(w * (x - y))


def l2(x: jt.Var, y: jt.Var):
    return l2_reg(x - y)


def l1_reg(x: jt.Var):
    # return x.abs().sum(dim=-1).mean()
    return x.abs().mean()


def l2_reg(x: jt.Var) -> jt.Var:
    # return (x**2).sum(dim=-1).mean()
    return (x**2).mean()


def bce_loss(x: jt.Var, y: jt.Var):
    return nn.bce_loss(x, y)

def cosine_similarity(x1, x2, dim=1, eps=1e-8):

    dot_product = jt.sum(x1 * x2, dim=dim, keepdim=True)

    norm_x1 = jt.norm(x1, p=2, dim=dim, keepdim=True)
    norm_x2 = jt.norm(x2, p=2, dim=dim, keepdim=True)

    product = norm_x1 * norm_x2
    product = jt.ternary(product == 0, jt.ones_like(product) * eps, product)

    cosine_sim = dot_product / product

    return cosine_sim.squeeze(-1)

def cos(x: jt.Var, y: jt.Var):
    return (1 - cosine_similarity(x, y, dim=-1)).mean()


def mIoU_loss(x: jt.Var, y: jt.Var):
    """
    Compute the mean intersection of union loss over masked regions
    x, y: B, N, 1
    """
    I = (x * y).sum(-1).sum(-1)
    U = (x + y).sum(-1).sum(-1) - I
    mIoU = (I / U.detach()).mean()
    return 1 - mIoU


def reg(x: jt.Var) -> jt.Var:
    return x.norm(dim=-1).mean()


def thresh(x: jt.Var, a: jt.Var, eps: float = 1e-8):
    return 1 / (l2(x, a) + eps)


def elastic_crit(jac: jt.Var) -> jt.Var:
    """Compute the raw 'log_svals' type elastic energy, and
    remap it using the Geman-McClure type of robust loss.
    Args:
        jac (jt.Var): (B, N, 3, 3), the gradient of warpped xyz with respect to the original xyz
    Return:
        elastic_loss (jt.Var): (B, N),
    """
    # !: CUDA IMPLEMENTATION OF SVD IS EXTREMELY SLOW
    # old_device = jac.device
    # jac = jac.cpu()
    # svd_backward: Setting compute_uv to false in torch.svd doesn't compute singular matrices, and hence we cannot compute backward. Please use torch.svd(compute_uv=True)
    _, S, _ = jt.linalg.svd(jac)           # (B, N, 3)
    # S = S.to(old_device)
    log_svals = jt.log(jt.clamp(S, min=1e-6))     # (B, N, 3)
    sq_residual = jt.sum(log_svals**2, dim=-1)       # (B, N)
    # TODO: determine whether it is a good choice to compute the robust loss here
    elastic_loss = general_loss_with_squared_residual(sq_residual, alpha=-2.0, scale=0.03)
    return elastic_loss


def general_loss_with_squared_residual(squared_x, alpha, scale):
    r"""The general loss that takes a squared residual.
    This fuses the sqrt operation done to compute many residuals while preserving
    the square in the loss formulation.
    This implements the rho(x, \alpha, c) function described in "A General and
    Adaptive Robust Loss Function", Jonathan T. Barron,
    https://arxiv.org/abs/1701.03077.
    Args:
        squared_x: The residual for which the loss is being computed. x can have
        any shape, and alpha and scale will be broadcasted to match x's shape if
        necessary.
        alpha: The shape parameter of the loss (\alpha in the paper), where more
        negative values produce a loss with more robust behavior (outliers "cost"
        less), and more positive values produce a loss with less robust behavior
        (outliers are penalized more heavily). Alpha can be any value in
        [-infinity, infinity], but the gradient of the loss with respect to alpha
        is 0 at -infinity, infinity, 0, and 2. Varying alpha allows for smooth
        interpolation between several discrete robust losses:
            alpha=-Infinity: Welsch/Leclerc Loss.
            alpha=-2: Geman-McClure loss.
            alpha=0: Cauchy/Lortentzian loss.
            alpha=1: Charbonnier/pseudo-Huber loss.
            alpha=2: L2 loss.
        scale: The scale parameter of the loss. When |x| < scale, the loss is an
        L2-like quadratic bowl, and when |x| > scale the loss function takes on a
        different shape according to alpha.
    Returns:
        The losses for each element of x, in the same shape as x.
    """
    # https://pytorch.org/docs/stable/type_info.html
    eps = jt.Var(misc.finfo(jt.float32).eps)

    # convert the float to jt.Var
    alpha = jt.Var(alpha).to(squared_x.device)
    scale = jt.Var(scale).to(squared_x.device)

    # This will be used repeatedly.
    squared_scaled_x = squared_x / (scale ** 2)

    # The loss when alpha == 2.
    loss_two = 0.5 * squared_scaled_x
    # The loss when alpha == 0.
    loss_zero = log1p_safe(0.5 * squared_scaled_x)
    # The loss when alpha == -infinity.
    loss_neginf = -jt.exp(-0.5 * squared_scaled_x)+1#-torch.expm1(-0.5 * squared_scaled_x)
    # The loss when alpha == +infinity.
    loss_posinf = expm1_safe(0.5 * squared_scaled_x)

    # The loss when not in one of the above special cases.
    # Clamp |2-alpha| to be >= machine epsilon so that it's safe to divide by.
    beta_safe = jt.maximum(eps, jt.abs(alpha - 2.))
    # Clamp |alpha| to be >= machine epsilon so that it's safe to divide by.
    alpha_safe = jt.ternary(
        jt.greater_equal(alpha, jt.Var(0.)), jt.ones_like(alpha),
        -jt.ones_like(alpha)) * jt.maximum(eps, jt.abs(alpha))
    loss_otherwise = (beta_safe / alpha_safe) * (
        jt.pow(squared_scaled_x / beta_safe + 1., 0.5 * alpha) - 1.)

    # Select which of the cases of the loss to return.
    loss = jt.ternary(
        alpha == -jt.Var(float('inf')), loss_neginf,
        jt.ternary(
            alpha == 0, loss_zero,
            jt.ternary(
                alpha == 2, loss_two,
                jt.ternary(alpha == jt.Var(float('inf')), loss_posinf, loss_otherwise))))

    return scale * loss


def log1p_safe(x):
    """The same as torch.log1p(x), but clamps the input to prevent NaNs."""
    return jt.log(jt.minimum(x, jt.Var(3e37)) + 1)#(jt.minimum(x, jt.Var(3e37)))


def expm1_safe(x):
    """The same as torch.expm1(x), but clamps the input to prevent NaNs."""
    return jt.exp(jt.minimum(x, jt.Var(87.5)))-1#torch.expm1(jt.minimum(x, jt.Var(87.5)))


def compute_plane_tv(t):
    batch_size, c, h, w = t.shape
    count_h = batch_size * c * (h - 1) * w
    count_w = batch_size * c * h * (w - 1)
    h_tv = ((t[..., 1:, :] - t[..., :h - 1, :])*(t[..., 1:, :] - t[..., :h - 1, :])).sum()
    w_tv = ((t[..., :, 1:] - t[..., :, :w - 1])*(t[..., :, 1:] - t[..., :, :w - 1])).sum()
    return 2 * (h_tv / count_h + w_tv / count_w)  # This is summing over batch and c instead of avg


def compute_planes_tv(embedding):
    tv_loss = 0
    for emb in embedding:
        tv_loss += compute_plane_tv(emb)
    return tv_loss


def compute_plane_smoothness(t):
    batch_size, c, h, w = t.shape
    # Convolve with a second derivative filter, in the time dimension which is dimension 2
    first_difference = t[..., 1:] - t[..., :w - 1]  # [batch, c, h-1, w]
    second_difference = first_difference[..., 1:] - first_difference[..., :w - 2]  # [batch, c, h-2, w]
    # Take the L2 norm of the result
    return ((second_difference)*(second_difference)).mean()


def compute_time_planes_smooth(embedding):
    loss = 0.
    for emb in embedding:
        loss += compute_plane_smoothness(emb)
    return loss


def gaussian(window_size, sigma):
    gauss = jt.Var([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = nn.Parameter(_2D_window.expand(channel, 1, window_size, window_size), requires_grad=True)
    return window


def gsssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window#.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = nn.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = nn.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = nn.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = nn.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = nn.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(x: jt.Var, y: jt.Var):
    #from pytorch_msssim import ssim as compute_ssim#????
    from jdhr.utils.msssim_utils import _SSIMForMultiScale
    x_np=x.numpy()
    y_np=y.numpy()
    ssim,_=_SSIMForMultiScale(x_np, y_np, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
    return jt.array(ssim)


def msssim(x: jt.Var, y: jt.Var):
    #from pytorch_msssim import ms_ssim as compute_msssim#????
    from jdhr.utils.msssim_utils import MultiScaleSSIM
    x_np=x.numpy()
    y_np=y.numpy()
    msssim=MultiScaleSSIM(x_np, y_np, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
    return jt.array(msssim)


# from MonoSDF
def compute_scale_and_shift(prediction, target, mask):
    # System matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = jt.sum(mask * prediction * prediction, (1, 2))
    a_01 = jt.sum(mask * prediction, (1, 2))
    a_11 = jt.sum(mask, (1, 2))

    # Right hand side: b = [b_0, b_1]
    b_0 = jt.sum(mask * prediction * target, (1, 2))
    b_1 = jt.sum(mask * target, (1, 2))

    # Solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = jt.zeros_like(b_0)
    x_1 = jt.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # Average of all valid pixels of the batch
    # Avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = jt.sum(M)

    if divisor == 0: return 0
    else: return jt.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # Mean of average of valid pixels of an image
    # Avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()
    image_loss[valid] = image_loss[valid] / M[valid]

    return jt.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):
    # Number of valid pixels
    M = jt.sum(mask, (1, 2))  # (B,)

    # L2 loss
    res = prediction - target  # (B, H, W)
    image_loss = jt.sum(mask * res * res, (1, 2))  # (B,)

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = jt.sum(mask, (1, 2))

    diff = prediction - target
    diff = jt.mul(mask, diff)

    grad_x = jt.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = jt.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = jt.mul(mask_x, grad_x)

    grad_y = jt.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = jt.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = jt.mul(mask_y, grad_y)

    image_loss = jt.sum(grad_x, (1, 2)) + jt.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def execute(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=1, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def execute(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


class ScaleAndShiftInvariantMSELoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def execute(self, prediction, target, mask):
        # Deal with the channel dimension, the input dimension may have (B, C, H, W) or (B, H, W)
        if prediction.ndim == 4: prediction = prediction[:, 0]  # (B, H, W)
        if target.ndim == 4: target = target[:, 0]  # (B, H, W)
        if mask.ndim == 4: mask = mask[:, 0]  # (B, H, W)

        # Compute scale and shift
        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)
        total = self.__data_loss(self.__prediction_ssi, target, mask)

        # Add regularization if needed
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)
# from MonoSDF


def median_normalize(x, mask):
    """ Median normalize a tensor for all valid pixels.
        This operation is performed without batch dimension.
    Args:
        x (jt.Var): (H, W), original tensor
        mask (jt.Var): (H, W), mask tensor
    Return:
        y (jt.Var): (H, W), median normalized tensor
    """
    M = jt.sum(mask)

    # Return original tensor if there is no valid pixel
    if M == 0:
        return x

    # Compute median and scale
    t = jt.Var(np.median(x[mask == 1]))#torch.quantile(x[mask == 1], q=0.5)  # scalar
    s = jt.sum(x[mask == 1] - t) / M  # scalar

    # Return median normalized tensor
    return (x - t) / s


def mae_loss(prediction, target, mask, reduction=reduction_batch_based):
    # Number of valid pixels
    M = jt.sum(mask, (1, 2))  # (B,)

    # L1 loss
    res = (prediction - target).abs()  # (B, H, W)
    image_loss = jt.sum(mask * res, (1, 2))  # (B,)

    return reduction(image_loss, 2 * M)


class MAELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def execute(self, prediction, target, mask):
        return mae_loss(prediction, target, mask, reduction=self.__reduction)


class ScaleAndShiftInvariantMAELoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MAELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

    def execute(self, prediction, target, mask):
        # Deal with the channel dimension, the input dimension may have (B, C, H, W) or (B, H, W)
        if prediction.ndim == 4: prediction = prediction[:, 0]  # (B, H, W)
        if target.ndim == 4: target = target[:, 0]  # (B, H, W)
        if mask.ndim == 4: mask = mask[:, 0]  # (B, H, W)

        # TODO: Maybe there is a better way to do the batching
        # But `torch.quantile` does not support multiple `dim` argument for now
        for i in range(prediction.shape[0]):
            prediction[i] = median_normalize(prediction[i], mask[i])  # (H, W)
            target[i] = median_normalize(target[i], mask[i])  # (H, W)

        # Compute the scale-and-shift invariant MAE loss
        total = self.__data_loss(prediction, target, mask)

       # Add regularization if needed
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.prediction, target, mask)

        return total


# Modified version of Adabins repository
# https://github.com/shariqfarooq123/AdaBins/blob/0952d91e9e762be310bb4cd055cbfe2448c0ce20/loss.py#L7
class ScaleInvariantLogLoss(nn.Module):
    def __init__(self, alpha=10.0, beta=0.15, eps=0.0):
        super(ScaleInvariantLogLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        # The eps is added to avoid log(0) and division by zero
        # But it should be gauranteed that the network output is always non-negative
        self.eps = eps

    def execute(self, prediction, target, mask):
        # Deal with the channel dimension, the input dimension may have (B, C, H, W) or (B, H, W)
        if prediction.ndim == 4: prediction = prediction[:, 0]  # (B, H, W)
        if target.ndim == 4: target = target[:, 0]  # (B, H, W)
        if mask.ndim == 4: mask = mask[:, 0]  # (B, H, W)

        total = 0
        # Maybe there is a better way to do the batching
        for i in range(prediction.shape[0]):
            g = jt.log(prediction[i][mask[i]] + self.eps) - jt.log(target[i][mask[i]] + self.eps)  # (N,)
            Dg = jt.var(g) + self.beta * jt.pow(jt.mean(g), 2)  # scalar
            total += self.alpha * jt.sqrt(Dg)

        return total
