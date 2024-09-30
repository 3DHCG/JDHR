#import torch
from typing import Tuple, Callable, List
import jittor as jt
from jittor import misc
from jittor import nn

def matchup_channels(t: jt.Var, w: jt.Var):
    if t.ndim == w.ndim + 1:
        t = t[..., 0]  # remove last dimension
    if t.shape[-1] != w.shape[-1] + 1:
        t = jt.cat([t, jt.ones_like(t[..., -1:])], dim=-1)  # 65
    return t, w



def interpolate(x: jt.Var, xp: jt.Var, fp: jt.Var) -> jt.Var:
    """One-dimensional linear interpolation for monotonically increasing sample
    points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: the :math:`x`-coordinates of the data points, must be increasing.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`.

    Returns:
        the interpolated values, same size as `x`.
    """
    if x.ndim == xp.ndim - 1:
        x = x[None]

    m = (fp[..., 1:] - fp[..., :-1]) / (xp[..., 1:] - xp[..., :-1] + 1e-8)  # slope
    b = fp[..., :-1] - (m * xp[..., :-1])

    indices = jt.sum(jt.greater_equal(x[..., :, None], xp[..., None, :]), -1) - 1  # torch.ge:  x[i] >= xp[i] ? true: false  torch.ge(x[..., :, None], xp[..., None, :])
    indices = jt.clamp(indices, 0, m.shape[-1] - 1)

    return m.gather(dim=-1, index=indices) * x + b.gather(dim=-1, index=indices)



def integrate_weights(w: jt.Var):
    """Compute the cumulative sum of w, assuming all weight vectors sum to 1.
    The output's size on the last dimension is one greater than that of the input,
    because we're computing the integral corresponding to the endpoints of a step
    function, not the integral of the interior/bin values.
    Args:
      w: Tensor, which will be integrated along the last axis. This is assumed to
        sum to 1 along the last axis, and this function will (silently) break if
        that is not the case.
    Returns:
      cw0: Tensor, the integral of w, where cw0[..., 0] = 0 and cw0[..., -1] = 1
    """
    cw = jt.cumsum(w[..., :-1], dim=-1).clamp(max_v=1.0)
    shape = cw.shape[:-1] + (1,)
    # Ensure that the CDF starts with exactly 0 and ends with exactly 1.
    cw0 = jt.cat([cw.new_zeros(shape), cw, cw.new_ones(shape)], dim=-1)
    return cw0



def weighted_percentile(t: jt.Var, w: jt.Var, ps: List[float]):
    """Compute the weighted percentiles of a step function. w's must sum to 1."""
    t, w = matchup_channels(t, w)
    cw = integrate_weights(w)
    # We want to interpolate into the integrated weights according to `ps`.
    # Vmap fn to an arbitrary number of leading dimensions.
    cw_mat = cw.reshape([-1, cw.shape[-1]])
    t_mat = t.reshape([-1, t.shape[-1]])
    wprctile_mat = interpolate(jt.array(ps).to(t, non_blocking=True),
                               cw_mat,
                               t_mat)
    wprctile = wprctile_mat.reshape(cw.shape[:-1] + (len(ps),))
    return wprctile


def s_vals_to_z_vals(s: jt.Var,
                     tn: jt.Var,
                     tf: jt.Var,
                     g: Callable[[jt.Var], jt.Var] = lambda x: 1 / x,
                     ig: Callable[[jt.Var], jt.Var] = lambda x: 1 / x,
                     ):
    # transfer ray depth from s space to t space (with inverse of g)
    return ig(s * g(tf) + (1 - s) * g(tn))


def z_vals_to_s_vals(t: jt.Var,
                     tn: jt.Var,
                     tf: jt.Var,
                     g: Callable[[jt.Var], jt.Var] = lambda x: 1 / x,
                     ):
    # transfer ray depth from t space back to s space (with function g)
    return (g(t) - g(tn)) / (g(tf) - g(tn) + 1e-8)

# Hierarchical sampling (section 5.2)


def searchsorted(a: jt.Var, v: jt.Var) -> Tuple[jt.Var, jt.Var]:
    """Find indices where v should be inserted into a to maintain order.
    This behaves like jnp.searchsorted (its second output is the same as
    jnp.searchsorted's output if all elements of v are in [a[0], a[-1]]) but is
    faster because it wastes memory to save some compute.
    Args:
      a: tensor, the sorted reference points that we are scanning to see where v
        should lie.
      v: tensor, the query points that we are pretending to insert into a. Does
        not need to be sorted. All but the last dimensions should match or expand
        to those of a, the last dimension can differ.
    Returns:
      (idx_lo, idx_hi), where a[idx_lo] <= v < a[idx_hi], unless v is out of the
      range [a[0], a[-1]] in which case idx_lo and idx_hi are both the first or
      last index of a.
    """
    i = jt.arange(a.shape[-1])  # 128
    v_ge_a = v[..., None, :] >= a[..., :, None]
    idx_lo = jt.max(jt.ternary(v_ge_a, i[..., :, None], i[..., :1, None]), -2)  # 128
    idx_hi = jt.min(jt.ternary(v_ge_a.logical_not(), i[..., :, None], i[..., -1:, None]), -2)
    return idx_lo, idx_hi


def invert_cdf(u, t, w):
    """Invert the CDF defined by (t, w) at the points specified by u in [0, 1)."""
    # Compute the PDF and CDF for each weight vector.
    cw = integrate_weights(w)
    # Interpolate into the inverse CDF.
    t_new = interpolate(u, cw, t)
    return t_new


def importance_sampling(t: jt.Var,
                        w: jt.Var,
                        num_samples: int,
                        perturb=True,
                        single_jitter=False,
                        ):
    """Piecewise-Constant PDF sampling from a step function.

    Args:
        rng: random number generator (or None for `linspace` sampling).
        t: [..., num_bins + 1], bin endpoint coordinates (must be sorted)
        w_logits: [..., num_bins], logits corresponding to bin weights
        num_samples: int, the number of samples.
        single_jitter: bool, if True, jitter every sample along each ray by the same
        amount in the inverse CDF. Otherwise, jitter each sample independently.
        deterministic_center: bool, if False, when `rng` is None return samples that
        linspace the entire PDF. If True, skip the front and back of the linspace
        so that the centers of each PDF interval are returned.
        use_gpu_resampling: bool, If True this resamples the rays based on a
        "gather" instruction, which is fast on GPUs but slow on TPUs. If False,
        this resamples the rays based on brute-force searches, which is fast on
        TPUs, but slow on GPUs.

    Returns:
        t_samples: jnp.ndarray(float32), [batch_size, num_samples].
    """
    if t.ndim == w.ndim + 1:
        t = t[..., 0]  # remove last dim

    # preparing for size change
    sh = *t.shape[:-1], num_samples  # B, P, I
    t = t.reshape(-1, t.shape[-1])
    w = w.reshape(-1, w.shape[-1])

    # assuming sampling in s space
    if t.shape[-1] != w.shape[-1] + 1:
        t = jt.cat([t, jt.ones_like(t[..., -1:])], dim=-1)

    # eps = torch.finfo(torch.float32).eps
    eps = 1e-8

    # Draw uniform samples.

    # `u` is in [0, 1) --- it can be zero, but it can never be 1.
    u_max = eps + (1 - eps) / num_samples
    max_jitter = (1 - u_max) / (num_samples - 1) - eps if perturb else 0
    d = 1 if single_jitter else num_samples
    u = (
        jt.linspace(0, 1 - u_max, num_samples) +
        jt.rand(t.shape[:-1] + (d,), dtype=t.dtype) * max_jitter
    )

    u = invert_cdf(u, t, w)

    # preparing for size change
    u = u.reshape(sh)
    return u


def weight_to_pdf(t: jt.Var, w: jt.Var, eps=misc.finfo('float32').eps**2):
    t, w = matchup_channels(t, w)
    """Turn a vector of weights that sums to 1 into a PDF that integrates to 1."""
    return w / (t[..., 1:] - t[..., :-1]).clip(eps)


def pdf_to_weight(t: jt.Var, p: jt.Var):
    t, p = matchup_channels(t, p)
    """Turn a PDF that integrates to 1 into a vector of weights that sums to 1."""
    return p * (t[..., 1:] - t[..., :-1])


def max_dilate(t, w, dilation, domain=(-float('inf'), float('inf'))):
    t, w = matchup_channels(t, w)
    """Dilate (via max-pooling) a non-negative step function."""
    t0 = t[..., :-1] - dilation
    t1 = t[..., 1:] + dilation
    t_dilate = jt.sort(jt.cat([t, t0, t1], dim=-1), dim=-1)[0]
    t_dilate = t_dilate.clip(*domain)
    w_dilate = jt.max(
        jt.ternary(
            (t0[..., None, :] <= t_dilate[..., None])
            & (t1[..., None, :] > t_dilate[..., None]),
            w[..., None, :],
            0,
        ),
        dim=-1)[..., :-1]
    return t_dilate, w_dilate


def max_dilate_weights(t: jt.Var,
                       w: jt.Var,
                       dilation: float,
                       domain=(-float('inf'), float('inf')),
                       renormalize=False,
                       eps=misc.finfo('float').eps**2):
    """Dilate (via max-pooling) a set of weights."""
    p = weight_to_pdf(t, w)
    t_dilate, p_dilate = max_dilate(t, p, dilation, domain=domain)
    w_dilate = pdf_to_weight(t_dilate, p_dilate)
    if renormalize:
        w_dilate /= jt.sum(w_dilate, dim=-1, keepdim=True).clip(eps)
    return t_dilate, w_dilate


def anneal_weights(t: jt.Var,
                   w: jt.Var,
                   train_frac: float,
                   anneal_slope: float = 10.0,
                   eps=misc.finfo('float').eps ** 2):
    # accepts t.shape[-1] = w.shape[-1] + 1
    t, w = matchup_channels(t, w)

    # Optionally anneal the weights as a function of training iteration.
    if anneal_slope > 0:
        # Schlick's bias function, see https://arxiv.org/abs/2010.09714
        def bias(x, s): return (s * x) / ((s - 1) * x + 1)
        anneal = bias(train_frac, anneal_slope)
    else:
        anneal = 1.

    # A slightly more stable way to compute weights**anneal. If the distance
    # between adjacent intervals is zero then its weight is fixed to 0.
    logits_resample = jt.ternary(
        t[..., 1:] > t[..., :-1],
        anneal * jt.log(w.clip(eps)), -float('inf'))  # MARK: prone to nan

    # If all samples are -inf, softmax will produce a nan (all -torch.inf)
    w = nn.softmax(logits_resample, dim=-1)
    return w


def query(tq, t, y, outside_value=0):
    """Look up the values of the step function (t, y) at locations tq."""
    idx_lo, idx_hi = searchsorted(t, tq)
    yq = jt.ternary(idx_lo == idx_hi, outside_value,jt.gather(jt.cat([y, jt.full_like(y[..., :1], outside_value)], dim=-1),-1,idx_lo ))  # ?
    return yq
