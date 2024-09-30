import jittor as jt
import math

def bucketize(
    a: jt.Var,
    boundaries: jt.Var,
    *,
    out_int32: bool = False,
    right: bool = False,
):
    #torch._check(
    #    boundaries.dim() == 1,
    #    lambda: f"boundaries tensor must be 1 dimension but got dim({boundaries.dim()})",
    #)

    out_dtype = jt.int32 if out_int32 else jt.int64
    n_boundaries = boundaries.shape[-1]
    if n_boundaries == 0:
        return jt.zeros_like(a)
    # We are trying to find the bucket (defined by pairs of consecutive elements of `boundaries`)
    # each element of `a` belongs to. We use binary search to achieve logarithimic complexity,
    # but each step of the search is done "in parallel" over all elements of `a`
    # can't use int32 as indexes, so we have to do all computations with int64 and convert at the end
    start = jt.zeros(a.shape, dtype=jt.int64)
    end = start + n_boundaries
    # Max depth of the binary search
    # Since we can't break out of the loop at different points for different elements of a,
    # we just do the max amount of iterations that binary search requires and add condition
    # tensor (cond_update below) to stop updating once the search terminates

    # For first iteration through loop we can skip some checks, we have separate implementation
    mid = start + (end - start) // 2
    mid_val = boundaries[mid]
    if right:
        cond_mid = mid_val > a
    else:
        cond_mid = mid_val >= a
    start = jt.ternary(cond_mid, start, mid + 1)

    if n_boundaries > 1:
        cond_update = jt.ones_like(a).bool()
        niters = int(math.log2(n_boundaries))
        for _ in range(niters):
            end = jt.ternary(cond_mid & cond_update, mid, end)
            cond_update = start < end
            # start might end up pointing to 1 past the end, we guard against that
            mid = jt.ternary(cond_update, start + (end - start) // 2, 0)
            mid_val = boundaries[mid]
            # If right is true, the buckets are closed on the *left*
            # (i.e., we are doing the equivalent of std::upper_bound in C++)
            # Otherwise they are closed on the right (std::lower_bound)
            if right:
                cond_mid = mid_val > a
            else:
                cond_mid = mid_val >= a
            start = jt.ternary((cond_mid.logical_not()) & cond_update, mid + 1, start)

    return start.to(dtype=out_dtype)