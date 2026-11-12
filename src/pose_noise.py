"""Ground-truth-free pose noise diagnostics with explicit model assumptions.

Pixel sigma means the standard deviation of ONE coordinate, matching isotropic
Gaussian noise injection, not RMS Euclidean position error (sqrt(2)*sigma).
No estimator can identify general pose error from a single sequence. Bone
variation assumes constant projected lengths and independent endpoint errors;
HF residuals assume locally smooth true motion and temporally white noise.
Confidence requires externally validated model-specific calibration to pixels.
Do not interpret interpolated samples as independent measurements: interpolation
suppresses variability. Prefer original finite observations within a flight window.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import savgol_coeffs, savgol_filter
from physics_residuals import TORSO_JOINTS

# COCO-17 projected rigid-segment proxies. Exclude flexible shoulder/hip widths.
BONES: dict[str, tuple[int, int]] = {
    'left_upper_arm': (5,7), 'right_upper_arm': (6,8),
    'left_forearm': (7,9), 'right_forearm': (8,10),
    'left_thigh': (11,13), 'right_thigh': (12,14),
    'left_shank': (13,15), 'right_shank': (14,16),
}
Calibration = Callable[[NDArray[np.float64]], ArrayLike]


@dataclass(frozen=True)
class NoiseEstimate:
    """Clip coordinate sigma and torso-relative sigma, plus diagnostic details.

    None means pixel sigma is unidentifiable without confidence calibration.
    No combined estimate is returned: assumptions differ across methods.
    """
    sigma_px: float | None
    sigma_torso: float | None
    details: dict[str, object]


def _pose(pose: ArrayLike) -> NDArray[np.float64]:
    p = np.asarray(pose, dtype=float)
    if p.ndim != 3 or p.shape[0] < 3 or p.shape[2] != 2 or not np.isfinite(p).all():
        raise ValueError('pose must be finite (frames>=3, joints, 2)')
    return p


def torso_length(pose: ArrayLike) -> float:
    """Median shoulder-centre to hip-centre distance in pixels."""
    p = _pose(pose)
    shoulders = p[:, TORSO_JOINTS['shoulders']].mean(axis=1)
    hips = p[:, TORSO_JOINTS['hips']].mean(axis=1)
    scale = float(np.median(np.linalg.norm(shoulders-hips, axis=1)))
    if scale <= np.finfo(float).eps:
        raise ValueError('median torso length must be positive')
    return scale


def _estimate(sigma: float | None, scale: float, details: dict[str, object]) -> NoiseEstimate:
    return NoiseEstimate(sigma, None if sigma is None else sigma/scale, details)


def bone_length_variability(
    pose: ArrayLike, bones: Mapping[str, tuple[int, int]] | None = None,
) -> NoiseEstimate:
    """Report each bone's sample length SD; estimate sigma=SD/sqrt(2).

    Clip estimate is RMS of per-bone sigmas. The conversion is first order:
    short bones at high noise have nonlinear distance bias. Shared joints
    correlate bone estimates. Foreshortening, zoom and real length changes
    inflate this diagnostic; coherent joint translation is invisible to it.
    """
    p = _pose(pose)
    mapping = BONES if bones is None else bones
    if not mapping:
        raise ValueError('at least one bone is required')
    lengths = np.stack([np.linalg.norm(p[:,a]-p[:,b], axis=1)
                        for a,b in mapping.values()], axis=1)
    sd = lengths.std(axis=0, ddof=1)
    sigma = float(np.sqrt(np.mean(sd**2)/2))
    return _estimate(sigma, torso_length(p), {
        'bone_length_sd_px': dict(zip(mapping, sd.tolist())),
        'bone_sigma_px': dict(zip(mapping, (sd/np.sqrt(2)).tolist())),
        'frames': len(p),
    })


def high_frequency_residual(
    pose: ArrayLike, window_size: int = 9, polyorder: int = 3,
) -> NoiseEstimate:
    """Estimate coordinate sigma from SG residual RMS, correcting noise gain.

    Discard half-window edges. Divide by sqrt(sum((delta-h)**2)) so filter
    fitting does not systematically underestimate white input noise. Per-joint
    RMS pools x/y. Rapid genuine motion inflates estimates; smooth bias and
    temporally correlated errors can be underestimated. No fps-dependent
    physical cutoff is implied: window_size is in frames.
    """
    p = _pose(pose)
    if (not isinstance(window_size, (int,np.integer)) or window_size < 3
            or window_size % 2 != 1 or window_size > len(p)):
        raise ValueError('window_size must be odd, >=3, and <= frame count')
    if not isinstance(polyorder, (int,np.integer)) or not 0 <= polyorder < window_size:
        raise ValueError('polyorder must be nonnegative and below window_size')
    half = window_size//2
    h = savgol_coeffs(window_size, polyorder)
    kernel = -h.copy()
    kernel[half] += 1
    gain = float(np.linalg.norm(kernel))
    if gain < 1e-10:
        raise ValueError('filter interpolates observations; noise cannot be estimated')
    smooth = savgol_filter(p, window_size, polyorder, axis=0)
    residual = (p-smooth)[half:-half]
    per_joint = np.sqrt(np.mean(residual**2, axis=(0,2)))/gain
    sigma = float(np.sqrt(np.mean(per_joint**2)))
    return _estimate(sigma, torso_length(p), {
        'joint_sigma_px': per_joint.tolist(), 'residual_noise_gain': gain,
        'interior_frames': len(residual), 'window_size': window_size, 'polyorder': polyorder,
    })


def confidence_distribution(
    pose: ArrayLike, confidence: ArrayLike,
    calibration: Calibration | None = None,
) -> NoiseEstimate:
    """Report confidence quantiles; optionally map scores to coordinate sigma.

    Calibration must return nonnegative finite conditional coordinate sigmas
    in pixels with the same shape as confidence. Clip sigma is their RMS.
    A confidence score is not inherently an error magnitude or a probability
    of a particular pixel error. Synthetic calibration tests validate plumbing,
    not transfer to a real pose model, image scale, or viewing condition.
    """
    p = _pose(pose)
    c = np.asarray(confidence, dtype=float)
    if c.shape != p.shape[:2] or not np.isfinite(c).all() or np.any((c<0)|(c>1)):
        raise ValueError('confidence must be finite (frames,joints) in [0,1]')
    details: dict[str, object] = {
        'quantiles': dict(zip(('min','p05','p25','median','p75','p95','max'),
                             np.quantile(c,[0,.05,.25,.5,.75,.95,1]).tolist())),
        'mean': float(c.mean()), 'std': float(c.std()),
        'fraction_below_0_5': float(np.mean(c<.5)), 'count': c.size,
        'calibrated': calibration is not None,
    }
    if calibration is None:
        details['reason'] = 'confidence-to-pixel mapping unavailable'
        return _estimate(None, torso_length(p), details)
    sigmas = np.asarray(calibration(c), dtype=float)
    if sigmas.shape != c.shape or not np.isfinite(sigmas).all() or np.any(sigmas<0):
        raise ValueError('calibration must return finite nonnegative sigmas matching confidence')
    return _estimate(float(np.sqrt(np.mean(sigmas**2))), torso_length(p), details)


def estimate_clip_noise(
    pose: ArrayLike, confidence: ArrayLike, *,
    bones: Mapping[str, tuple[int,int]] | None = None,
    window_size: int = 9, polyorder: int = 3,
    confidence_calibration: Calibration | None = None,
) -> dict[str, NoiseEstimate]:
    """Return separate bone, high-frequency and confidence diagnostics.

    Caller supplies only the flight window. Use sigma_torso to compare clips
    at different image scales; no average across incompatible estimators is
    made. Confidence is reported even when pixel conversion is unavailable.
    """
    return {
        'bone': bone_length_variability(pose,bones),
        'high_frequency': high_frequency_residual(pose,window_size,polyorder),
        'confidence': confidence_distribution(pose,confidence,confidence_calibration),
    }
