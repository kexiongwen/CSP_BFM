"""Scale-aware reductions without changing the working floating-point dtype.

Binary scaling uses IEEE exponent/significand operations, not exp2/log2 or an
approximate reciprocal. Reduction scales are local to the output element or
norm. These routines avoid avoidable range loss; they are still finite-
precision reductions, not exact arithmetic or a global accuracy guarantee.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def _float_parts(x):
    if x.dtype == jnp.float32:
        uint, width, nmant, bias = jnp.uint32, 32, 23, 127
    elif x.dtype == jnp.float64:
        uint, width, nmant, bias = jnp.uint64, 64, 52, 1023
    else:
        raise TypeError("binary scaling requires float32 or float64")
    one = jnp.asarray(1, dtype=uint)
    bits = jax.lax.bitcast_convert_type(x, uint)
    frac_mask = (one << nmant) - one
    frac = bits & frac_mask
    exp = ((bits >> nmant) & jnp.asarray(2 * bias + 1, uint)).astype(jnp.int32)
    # Normalize subnormal significands with integer operations: CPU/GPU
    # floating point arithmetic may flush a subnormal input to zero.
    shift = jax.lax.clz(frac).astype(jnp.int32) - (width - nmant - 1)
    shift = jnp.clip(shift, 0, nmant)
    sig = jnp.where(exp == 0, frac << shift.astype(uint), frac | (one << nmant))
    exponent = jnp.where(exp == 0, 1 - bias - shift, exp - bias)
    return bits, sig, exponent, exp, uint, width, nmant, bias


def binary_exponent(x):
    """floor(log2(abs(x))) for finite nonzero x; zero maps to zero.

    This extracts the exponent exactly, including near the largest finite
    number, where a floating log2 can round into an overflowing exponent.
    """
    x = jnp.asarray(x)
    _bits, sig, exponent, _exp, *_ = _float_parts(x)
    return jnp.where(sig == 0, 0, exponent)


def scale_pow2(x, exponent):
    """Return x * 2**exponent, with integer exponents broadcast against x.

    Normal results are exact exponent changes. Subnormal results are rounded
    to nearest, ties to even, using integers so they survive flush-to-zero
    arithmetic. Overflow becomes signed infinity; NaNs/infinities and signed
    zero are preserved. No global x64 configuration change is made.
    """
    x = jnp.asarray(x)
    bits, sig, old_e, exp, uint, width, nmant, bias = _float_parts(x)
    shift_by = jnp.clip(jnp.asarray(exponent, dtype=jnp.int32), -16384, 16384)
    new_e = old_e + shift_by
    one = jnp.asarray(1, dtype=uint)
    sign = bits & (one << (width - 1))
    frac_mask = (one << nmant) - one
    normal = sign | ((new_e + bias).astype(uint) << nmant) | (sig & frac_mask)
    # For underflow, retain the significand bits that fit and round once.
    drop = 1 - bias - new_e
    bounded_drop = jnp.clip(drop, 1, nmant + 1).astype(uint)
    kept = sig >> bounded_drop
    discarded = sig & ((one << bounded_drop) - one)
    halfway = one << (bounded_drop - one)
    increment = (discarded > halfway) | ((discarded == halfway) & ((kept & one) != 0))
    subnormal = sign | jnp.where(drop <= nmant + 1, kept + increment.astype(uint), 0)
    infinity = sign | (jnp.asarray(2 * bias + 1, uint) << nmant)
    out = jnp.where(new_e < 1 - bias, subnormal, normal)
    out = jnp.where(new_e > bias, infinity, out)
    out = jnp.where((sig == 0) | (exp == 2 * bias + 1), bits, out)
    return jax.lax.bitcast_convert_type(out, x.dtype)


def power_of_two(exponent, dtype):
    """Construct powers of two through exponent bits, including subnormals."""
    return scale_pow2(jnp.asarray(1, dtype=dtype), exponent)


def max_abs(x, axis=None, keepdims=False):
    """Maximum magnitude without flushing subnormals in a float reduction.

    Positive IEEE encodings have the same order as their magnitudes. NaN
    encodings sort above infinity and remain NaN rather than being hidden.
    """
    x = jnp.asarray(x)
    bits, _sig, _e, _exp, uint, width, _nmant, _bias = _float_parts(x)
    magnitude = bits & ((jnp.asarray(1, uint) << (width - 1)) - jnp.asarray(1, uint))
    largest = jnp.max(magnitude, axis=axis, keepdims=keepdims)
    return jax.lax.bitcast_convert_type(largest, x.dtype)


def pow2_recip(v):
    """Compatibility helper: two reciprocal factors and a finite power scale.

    For positive finite v, scale = 2**floor(log2(v)); zeros use unit factors.
    Prefer scale_pow2 for exponent changes outside a single factor's range.
    """
    v = jnp.asarray(v)
    e = binary_exponent(v)
    e1 = e // 2
    return power_of_two(-e1, v.dtype), power_of_two(-(e - e1), v.dtype), power_of_two(e, v.dtype)


def safe_mean(x, axis=0):
    """Mean with a per-output fallback for nonfinite or subnormal reductions.

    Scaling one output coordinate cannot erase another coordinate's small
    values. The normalized mean is bounded by its input extrema before the
    exponent is restored, preventing rounding just beyond a finite maximum.
    Finite ordinary reductions retain the jnp.mean summation order, including
    cancellation tails that whole-coordinate scaling could erase. This is
    not an exact sum of arbitrarily separated magnitudes within a coordinate.
    """
    x = jnp.asarray(x)
    plain = jnp.mean(x, axis=axis)
    smax = max_abs(x, axis=axis, keepdims=True)
    e = binary_exponent(smax)
    use_scaled = (~jnp.isfinite(plain)) | (jnp.squeeze(e, axis=axis) < np.finfo(x.dtype).minexp)
    # Keep a rounded finite mean inside its convex hull. This also preserves
    # exactly repeated inputs when division by the sample count rounds.
    bounded_plain = jnp.clip(plain, jnp.min(x, axis=axis), jnp.max(x, axis=axis))

    def scaled():
        values = scale_pow2(x, -e)
        average = jnp.mean(values, axis=axis)
        average = jnp.clip(average, jnp.min(values, axis=axis), jnp.max(values, axis=axis))
        restored = scale_pow2(average, jnp.squeeze(e, axis=axis))
        return jnp.where(use_scaled, restored, bounded_plain)

    return jax.lax.cond(jnp.any(use_scaled), scaled, lambda: bounded_plain)


def _axes(x, axis):
    if axis is None:
        return tuple(range(x.ndim))
    if isinstance(axis, int):
        return (axis,)
    return tuple(axis)


def scaled_norm(x, axis=None):
    """Frobenius norm with one scale per reduced slice, including zero slices.

    The output is infinity when the norm itself exceeds the working range.
    Finite input alone does not imply a representable norm.
    """
    x = jnp.asarray(x)
    axes = _axes(x, axis)
    e = binary_exponent(max_abs(x, axis=axes, keepdims=True))
    scaled = scale_pow2(x, -e)
    norm = jnp.sqrt(jnp.sum(scaled * scaled, axis=axes))
    return scale_pow2(norm, jnp.squeeze(e, axis=axes))


def sum_squares(x, axis=None):
    """Original-scale squared norm without intermediate square range loss.

    Inputs may already be residuals. Binary scaling also prevents the compiler
    from expanding a residual square into overflowing operand squares.
    Final values below/above the dtype range round to zero/infinity.
    """
    x = jnp.asarray(x)
    axes = _axes(x, axis)
    e = binary_exponent(max_abs(x, axis=axes, keepdims=True))
    scaled = scale_pow2(x, -e)
    total = jnp.sum(scaled * scaled, axis=axes)
    return scale_pow2(total, 2 * jnp.squeeze(e, axis=axes))
