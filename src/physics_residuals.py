"""Projected 2D diving diagnostics, not physical 3D momentum estimates.

Default mapping is COCO-17. Mass fractions are the male set from de Leva
(1996), J Biomech 29:1223–1230, doi:10.1016/0021-9290(95)00178-6.
Segment positions are explicit midpoint/proxy approximations, NOT de Leva's
anatomical segment COM locations. COCO lacks head, hand and foot endpoints.
Edit SEGMENTS for another format or better anatomical landmarks. Mass totals
are normalized to one. Inertia ignores each segment's intrinsic inertia and
out-of-plane motion. Supply an airborne window for meaningful residuals.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import savgol_filter

# name: (mass fraction, {joint index: position weight}); COCO-17 indices.
SEGMENTS: dict[str, tuple[float, dict[int, float]]] = {
    'head': (.0694, {0: 1.}),
    'trunk': (.4346, {5: .25, 6: .25, 11: .25, 12: .25}),
    'left_upper_arm': (.0271, {5: .5, 7: .5}),
    'right_upper_arm': (.0271, {6: .5, 8: .5}),
    'left_forearm': (.0162, {7: .5, 9: .5}),
    'right_forearm': (.0162, {8: .5, 10: .5}),
    'left_hand': (.0061, {9: 1.}),
    'right_hand': (.0061, {10: 1.}),
    'left_thigh': (.1416, {11: .5, 13: .5}),
    'right_thigh': (.1416, {12: .5, 14: .5}),
    'left_shank': (.0433, {13: .5, 15: .5}),
    'right_shank': (.0433, {14: .5, 16: .5}),
    'left_foot': (.0137, {15: 1.}),
    'right_foot': (.0137, {16: 1.}),
}
TORSO_JOINTS: dict[str, tuple[int, ...]] = {'hips': (11, 12), 'shoulders': (5, 6)}
FloatArray = NDArray[np.float64]


def _pose(pose: ArrayLike) -> FloatArray:
    p = np.asarray(pose, dtype=float)
    if p.ndim != 3 or p.shape[0] == 0 or p.shape[2] != 2:
        raise ValueError('pose must have shape (nonempty frames, joints, 2)')
    if not np.isfinite(p).all():
        raise ValueError('pose must be finite; call interpolate_low_confidence first')
    return p


def _series(value: ArrayLike, name: str) -> FloatArray:
    x = np.asarray(value, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError(f'{name} must be a nonempty finite 1D array')
    return x


def _fps(fps: float) -> None:
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be finite and positive')


def interpolate_low_confidence(
    pose: ArrayLike, confidence: ArrayLike, threshold: float = .5,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Return a copy and (frames,joints) replacement mask.

    Replace confidence below threshold, nonfinite confidence, or either
    nonfinite coordinate. Interpolate linearly in time; hold nearest valid
    sample at endpoints. Raise if any joint has no valid observation.
    """
    p = np.asarray(pose, dtype=float)
    c = np.asarray(confidence, dtype=float)
    if p.ndim != 3 or p.shape[0] == 0 or p.shape[2] != 2 or c.shape != p.shape[:2]:
        raise ValueError('expected pose (frames,joints,2), confidence (frames,joints)')
    if not np.isfinite(threshold):
        raise ValueError('threshold must be finite')
    mask = (c < threshold) | ~np.isfinite(c) | ~np.isfinite(p).all(axis=2)
    result = p.copy()
    t = np.arange(len(p))
    for j in range(p.shape[1]):
        good = ~mask[:, j]
        if not good.any():
            raise ValueError(f'joint {j} has no valid observations')
        for coordinate in range(2):
            result[:, j, coordinate] = np.interp(t, t[good], p[good, j, coordinate])
    return result, mask


def _segments(pose: ArrayLike) -> tuple[FloatArray, FloatArray]:
    p = _pose(pose)
    masses, positions = [], []
    for mass, mapping in SEGMENTS.values():
        if mass < 0 or not np.isclose(sum(mapping.values()), 1.):
            raise ValueError('segment mass must be nonnegative and position weights sum to one')
        masses.append(mass)
        positions.append(sum(p[:, j] * w for j, w in mapping.items()))
    m = np.asarray(masses, dtype=float)
    if not np.isfinite(m).all() or m.sum() <= 0:
        raise ValueError('segment masses must be finite with positive total')
    return np.stack(positions, axis=1), m / m.sum()


def center_of_mass(pose: ArrayLike) -> FloatArray:
    """Return mass-weighted proxy segment centres, shape (frames,2), pixels."""
    positions, masses = _segments(pose)
    return np.einsum('fsd,s->fd', positions, masses)


def _torso(pose: ArrayLike) -> FloatArray:
    p = _pose(pose)
    v = p[:, TORSO_JOINTS['shoulders']].mean(axis=1) - p[:, TORSO_JOINTS['hips']].mean(axis=1)
    if np.any(np.linalg.norm(v, axis=1) <= np.finfo(float).eps):
        raise ValueError('torso length must be positive in every frame')
    return v


def torso_angle(pose: ArrayLike) -> FloatArray:
    """Unwrapped atan2 angle of hip-to-shoulder vector, in image-plane radians.

    Positive direction follows image axes (clockwise if y points down).
    Unwrapping assumes less than pi rotation between consecutive frames.
    """
    v = _torso(pose)
    return np.unwrap(np.arctan2(v[:, 1], v[:, 0]))


def angular_velocity(angle: ArrayLike, fps: float = 25., window_size: int = 9,
                     polyorder: int = 3) -> FloatArray:
    """Savitzky–Golay first derivative, radians/second; polynomial edge fits.

    Window must be odd, fit within the sequence, and exceed polyorder.
    Invalid windows raise rather than silently changing smoothing.
    """
    a = _series(angle, 'angle')
    _fps(fps)
    if (not isinstance(window_size, (int, np.integer)) or window_size % 2 != 1
            or window_size < 3 or window_size > len(a)):
        raise ValueError('window_size must be odd, >=3, and <= frame count')
    if not isinstance(polyorder, (int, np.integer)) or not 1 <= polyorder < window_size:
        raise ValueError('polyorder must be >=1 and below window_size')
    return savgol_filter(a, window_size, polyorder, deriv=1, delta=1 / fps, mode='interp')


def moment_of_inertia(pose: ArrayLike, com: ArrayLike) -> FloatArray:
    """Sum mass*distance² divided by squared torso length; unit body mass."""
    positions, masses = _segments(pose)
    c = np.asarray(com, dtype=float)
    if c.shape != (len(positions), 2) or not np.isfinite(c).all():
        raise ValueError('com must be finite with shape (frames,2)')
    r2 = np.sum((positions - c[:, None]) ** 2, axis=2)
    return (r2 @ masses) / np.sum(_torso(pose) ** 2, axis=1)


def angular_momentum(inertia: ArrayLike, omega: ArrayLike) -> FloatArray:
    """Return normalized projected angular-momentum proxy I*omega per frame."""
    i, w = _series(inertia, 'inertia'), _series(omega, 'omega')
    if i.shape != w.shape or np.any(i < 0):
        raise ValueError('inertia and omega must match; inertia must be nonnegative')
    return i * w


def angular_residual(
    L: ArrayLike, changepoint_window: int = 12,
) -> tuple[FloatArray, float, FloatArray]:
    """Return median deviations, population CV, and absolute sliding t-statistic.

    At t, compare L[t-w:t] with L[t:t+w], where w=changepoint_window.
    Statistic = abs(mean_after-mean_before) / (pooled_std*sqrt(2/w)),
    with pooled sample variance (ddof=1). Edges lacking full windows are
    NaN; a short sequence has all-NaN changepoints. Window must be >=2.
    A scale-relative numerical floor on the standard error makes constant
    plateaus finite; it is not a measurement-noise model. Filtered estimates
    are correlated, so calibrate thresholds empirically, not from a t table.
    This statistic uses future frames and is an offline diagnostic.

    CV is infinity for zero mean with nonzero variation, and zero for an
    identically zero sequence. CV is unstable near zero mean or sign changes.
    """
    values = _series(L, 'L')
    mean, std = float(np.mean(values)), float(np.std(values))
    cv = std / abs(mean) if mean != 0 else (0. if std == 0 else float('inf'))
    w = changepoint_window
    if not isinstance(w, (int, np.integer)) or isinstance(w, bool) or w < 2:
        raise ValueError('changepoint_window must be an integer >=2')
    statistic = np.full(len(values), np.nan)
    floor = 64 * np.finfo(float).eps * max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    for t in range(w, len(values) - w + 1):
        before, after = values[t-w:t], values[t:t+w]
        pooled_variance = (before.var(ddof=1) + after.var(ddof=1)) / 2
        standard_error = max(float(np.sqrt(pooled_variance * 2 / w)), floor)
        statistic[t] = abs(after.mean() - before.mean()) / standard_error
    return np.abs(values - np.median(values)), cv, statistic


def ballistic_residual(com: ArrayLike, fps: float = 25.) -> tuple[FloatArray, float]:
    """Fit y0+v0*t-0.5*g*t² with free signed pixel g; return y-fit and RMSE.

    Image y usually increases downwards, so falling motion has negative g
    in this convention. Linear camera motion and all quadratic y paths are
    absorbed exactly; this diagnostic alone cannot establish ballistic motion.
    """
    c = np.asarray(com, dtype=float)
    if c.ndim != 2 or c.shape[1] != 2 or len(c) < 3 or not np.isfinite(c).all():
        raise ValueError('com must be finite (frames,2) with at least 3 frames')
    _fps(fps)
    t = np.arange(len(c), dtype=float) / fps
    design = np.column_stack((np.ones(len(c)), t, -.5 * t ** 2))
    coefficients = np.linalg.lstsq(design, c[:, 1], rcond=None)[0]
    residual = c[:, 1] - design @ coefficients
    return residual, float(np.sqrt(np.mean(residual ** 2)))
