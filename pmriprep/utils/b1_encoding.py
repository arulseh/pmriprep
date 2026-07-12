"""Units-aware interpretation of pre-computed B1+ transmit-field maps.

A pre-computed B1+ map can encode its pixel values in several vendor-specific
conventions. The BIDS standard signals the encoding through the file suffix and
the ``Units`` sidecar field (``TB1TFL`` → actual flip in 0.1° units;
``TB1map`` with ``Units="p.u."`` → percent of nominal, 100 = nominal), but
``Units`` is only RECOMMENDED, so it may be absent. This module centralises the
pixel→ratio scaling for every supported encoding and resolves the ``auto`` case
from metadata (upstream) or, as a last resort, from the pixel median.

The B1+ correction ratio is ``pixel / divisor`` where ``divisor`` is the pixel
value corresponding to the nominal flip angle (B1+ = 1.0). Everything downstream
of that division operates on the normalised ratio and is encoding-agnostic.

Notes
-----
Because a mis-identified encoding silently rescales B1+ (and hence T1 and
Ktrans) by up to 10x, the ``auto`` median fallback is deliberately strict: it
resolves only when exactly one encoding maps the median into a physiological
B1+ window, and raises :class:`B1EncodingIndeterminateError` otherwise. This is
the one place the B1 stage gates rather than reports — it is input correctness,
not a QC metric.
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import NamedTuple

import numpy as np

logger = logging.getLogger("nipype.interface")

#: Physiological window for the *median* implied B1+ ratio, used only to
#: disambiguate the ``auto`` encoding. Wider than the QC pass range
#: (``B1_MEAN_RANGE``) because it must admit ±20-30% transmit offsets while
#: still cleanly excluding the 10×/100× mis-scalings of a wrong encoding.
_PHYS_MEDIAN_LOW: float = 0.6
_PHYS_MEDIAN_HIGH: float = 1.6


class B1Encoding(str, Enum):
    """Pixel-value encoding of a pre-computed B1+ map.

    The value string is what appears in config, sidecars, and QC provenance.
    """

    AUTO = "auto"
    ACTUAL_FLIP_DECI_DEG = "actual_flip_deci_deg"  # pixel = actual flip × 10 (TB1TFL)
    ACTUAL_FLIP_DEG = "actual_flip_deg"            # pixel = actual flip in degrees
    PERCENT = "percent"                            # pixel = percent of nominal (p.u.)
    RATIO = "ratio"                                # pixel = B1+ ratio (1.0 = nominal)


class B1EncodingIndeterminateError(ValueError):
    """Raised when ``auto`` cannot uniquely determine a B1+ map's encoding.

    Only raised by the pixel-median fallback, i.e. when no upstream metadata
    (suffix/``Units``) resolved the encoding and the median is ambiguous or
    physiologically implausible under every encoding.
    """


class B1Resolution(NamedTuple):
    """Resolved scaling for a pre-computed B1+ map.

    Attributes
    ----------
    divisor : float
        Value to divide pixels by to obtain the B1+ ratio (pixel at B1+ = 1.0).
    resolved_encoding : B1Encoding
        The concrete encoding used (never :attr:`B1Encoding.AUTO`).
    source : str
        Provenance of the resolution: ``"explicit"`` (from config/metadata,
        resolved upstream) or ``"median_fallback"`` (inferred from pixels).
    detail : str
        Human-readable explanation for QC/logging.
    """

    divisor: float
    resolved_encoding: B1Encoding
    source: str
    detail: str


def _actual_flip_divisor(encoding: B1Encoding, nominal_prep_fa: float | None) -> float:
    """Divisor for an actual-flip encoding; requires the nominal flip angle."""
    if nominal_prep_fa is None or nominal_prep_fa <= 0:
        raise ValueError(
            f"b1_map_encoding={encoding.value!r} requires a positive nominal "
            "flip angle (b1_nominal_prep_fa, from the FlipAngle sidecar), "
            f"got {nominal_prep_fa!r}."
        )
    return (10.0 if encoding is B1Encoding.ACTUAL_FLIP_DECI_DEG else 1.0) * float(
        nominal_prep_fa
    )


def _divisor_for(encoding: B1Encoding, nominal_prep_fa: float | None) -> float:
    """Return the pixel→ratio divisor for a concrete (non-auto) encoding."""
    if encoding in (B1Encoding.ACTUAL_FLIP_DECI_DEG, B1Encoding.ACTUAL_FLIP_DEG):
        return _actual_flip_divisor(encoding, nominal_prep_fa)
    if encoding is B1Encoding.PERCENT:
        return 100.0
    if encoding is B1Encoding.RATIO:
        return 1.0
    raise ValueError(f"Not a concrete B1 encoding: {encoding!r}")


def _resolve_auto(
    nominal_prep_fa: float | None,
    b1_raw: np.ndarray,
    brain_mask: np.ndarray | None,
) -> B1Resolution:
    """Infer the encoding from the pixel median (last-resort fallback).

    Resolves only when exactly one encoding maps the median into the
    physiological window; raises :class:`B1EncodingIndeterminateError` on a
    genuine ambiguity (typically percent vs actual-flip-in-degrees) or when no
    encoding is plausible.
    """
    positive = b1_raw > 0
    if brain_mask is not None:
        positive = positive & (brain_mask > 0)
    if not np.any(positive):
        raise B1EncodingIndeterminateError(
            "Cannot infer B1+ map encoding: no positive voxels. Set Units in "
            "the sidecar or b1_map_encoding explicitly."
        )
    median = float(np.median(b1_raw[positive]))

    # Candidate encodings (drop the actual-flip pair when FA is unavailable).
    candidates = [B1Encoding.RATIO, B1Encoding.PERCENT]
    if nominal_prep_fa is not None and nominal_prep_fa > 0:
        candidates += [B1Encoding.ACTUAL_FLIP_DEG, B1Encoding.ACTUAL_FLIP_DECI_DEG]

    plausible: list[tuple[B1Encoding, float]] = []
    for enc in candidates:
        divisor = _divisor_for(enc, nominal_prep_fa)
        implied = median / divisor
        if _PHYS_MEDIAN_LOW <= implied <= _PHYS_MEDIAN_HIGH:
            plausible.append((enc, implied))

    if len(plausible) == 1:
        enc, implied = plausible[0]
        detail = (
            f"median={median:.3g} → {enc.value} (implied B1+={implied:.3f}); "
            "sole physiologically-plausible encoding"
        )
        logger.warning(
            "B1+ encoding not declared (no Units); inferred %s from pixel "
            "median=%.3g (implied B1+=%.3f). Set Units or b1_map_encoding to "
            "make this explicit.",
            enc.value, median, implied,
        )
        return B1Resolution(
            _divisor_for(enc, nominal_prep_fa), enc, "median_fallback", detail
        )

    if len(plausible) > 1:
        names = ", ".join(f"{e.value}(B1+={v:.2f})" for e, v in plausible)
        raise B1EncodingIndeterminateError(
            f"B1+ map encoding is ambiguous: median={median:.3g} is "
            f"physiologically plausible under multiple encodings [{names}]. "
            "Set Units in the sidecar (e.g. 'p.u.') or b1_map_encoding explicitly."
        )
    raise B1EncodingIndeterminateError(
        f"B1+ map encoding could not be inferred: median={median:.3g} implies no "
        "physiological B1+ under any encoding. Check the input file, or set "
        "b1_map_encoding explicitly."
    )


def resolve_b1_divisor(
    *,
    encoding: B1Encoding,
    nominal_prep_fa: float | None,
    b1_raw: np.ndarray,
    brain_mask: np.ndarray | None = None,
) -> B1Resolution:
    """Resolve the pixel→ratio divisor for a pre-computed B1+ map.

    Parameters
    ----------
    encoding : B1Encoding
        The encoding to apply. When :attr:`B1Encoding.AUTO`, the encoding is
        inferred from the pixel median (config/metadata resolution is expected
        to have happened upstream).
    nominal_prep_fa : float or None
        Nominal preparation flip angle in degrees, required only for the
        actual-flip encodings.
    b1_raw : numpy.ndarray
        Raw B1+ map pixel values (used only for the ``auto`` median fallback).
    brain_mask : numpy.ndarray or None, optional
        Restrict the fallback median to brain voxels when available.

    Returns
    -------
    B1Resolution
        The divisor, the concrete resolved encoding, and provenance.

    Raises
    ------
    B1EncodingIndeterminateError
        When ``encoding`` is ``auto`` and the median is ambiguous or implausible.
    ValueError
        When an actual-flip encoding is requested without a valid flip angle.

    Examples
    --------
    >>> import numpy as np
    >>> r = resolve_b1_divisor(encoding=B1Encoding.PERCENT,
    ...                         nominal_prep_fa=None, b1_raw=np.array([105.0]))
    >>> r.divisor
    100.0
    """
    if encoding is B1Encoding.AUTO:
        return _resolve_auto(nominal_prep_fa, b1_raw, brain_mask)
    divisor = _divisor_for(encoding, nominal_prep_fa)
    return B1Resolution(divisor, encoding, "explicit", f"{encoding.value} (divisor={divisor:g})")


def expected_pixel_max(divisor: float) -> float:
    """Return the nominal full-scale pixel value (pixel at B1+ = 1.0).

    Equal to the divisor; named for use at the pixel-range sanity guard, where
    the reject/warn thresholds are multiples of the nominal full scale.

    Parameters
    ----------
    divisor : float
        The pixel→ratio divisor from :func:`resolve_b1_divisor`.

    Returns
    -------
    float
        The nominal full-scale pixel value.
    """
    return float(divisor)
