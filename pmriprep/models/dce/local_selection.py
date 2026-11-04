"""Physical in-plane references and penalized kinetic-family selection."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.spatial import cKDTree

from pmriprep.models.dce.automatic_admission import (
    ExchangeAdmission,
    exchange_admission,
)
from pmriprep.models.dce.delay_profiles import (
    MAX_LOCAL_DELAY_SECONDS,
    DelayProfile,
    ProfileFamily,
)
from pmriprep.models.dce.native_exchange import NativeInput

_COUNTS = {"patlak": 2, "tofts": 2, "extended_tofts": 3, "2cxm": 4}
_PARAMETERS = {"patlak": 3, "tofts": 2, "extended_tofts": 3, "2cxm": 4}


@dataclass
class FamilyTiming:
    """One centre's timing evidence from complete neighboring family profiles.

    Attributes
    ----------
    index, reference, best : int
        Selected, pooled-reference and minimum-SSE grid indices; -1 is unavailable.
    free : bool
        Whether the penalized free-delay candidate was retained.
    bic, improvement : float
        Selected BIC and reference-minus-free BIC, including one delay penalty.
    support : int
        Complete distinct pool members used at every offset.
    fallback : str or None
        Zero-reference reason: insufficient support, flat profile or fixed zero.
    status : int
        Selected numerical status, zero when no complete centre is available.
    reason : str or None
        Incomplete centre or selected nonconvergence; None means numerical success.
    free_optimum_near_lower, free_optimum_near_upper : bool or None
        Whether the free profile optimum is within one actual grid step of
        zero or the local-delay limit, regardless of the nested test's choice.
    reference_at_bound : bool or None
        Whether the reference is exactly zero or the local-delay limit.
    upper_edge : {'truncated', 'flat_edge'} or None
        Report-only classification when the free optimum is near the upper
        bound. Truncated means upper_edge_evidence >=6.
    upper_edge_evidence : float
        n*log(min SSE at d<=limit-frame_interval / SSE at limit), with the
        same residual floor used for timing. NaN when not applicable/available.
    boundary_unavailable_reason : str or None
        Fixed delay, incomplete profile/range, or no pre-edge comparison sample.
        Unavailable flags are None, never silently treated as false.
    """

    index: int
    reference: int
    best: int
    free: bool
    bic: float
    improvement: float
    support: int
    fallback: str | None
    status: int
    reason: str | None
    free_optimum_near_lower: bool | None = None
    free_optimum_near_upper: bool | None = None
    reference_at_bound: bool | None = None
    upper_edge: Literal["truncated", "flat_edge"] | None = None
    upper_edge_evidence: float = np.nan
    boundary_unavailable_reason: str | None = None


@dataclass
class LocalSelection:
    """Timing and family choice before inference, retention and identification QC.

    Attributes
    ----------
    family : str or None
        Chosen family, or None if automatic mode has no numerical candidate.
        Explicit mode preserves the requested family even if its timing failed.
    timing : dict
        Per-family timing records, including failed requested families.
    scores : dict
        BIC after numerical and automatic-admission eligibility checks.
    admission : ExchangeAdmission or None
        G1/G2 evidence when eligible automatic 2CXM was evaluated.
    exchange_reason : str or None
        Sampling, numerical or admission outcome for automatic 2CXM.
    delta_bic : float
        Raw best-versus-runner-up gap, or NaN for a Patlak-guard downgrade,
        explicit mode or fewer than two finite candidates.
    """

    family: ProfileFamily | None
    timing: dict[ProfileFamily, FamilyTiming]
    scores: dict[ProfileFamily, float]
    admission: ExchangeAdmission | None
    exchange_reason: str | None
    delta_bic: float


def in_plane_neighborhoods(
    coordinates: np.ndarray, affine: np.ndarray
) -> tuple[float, list[np.ndarray]]:
    """Find eligible centres in each physical disc of one complete image slice.

    Parameters
    ----------
    coordinates : ndarray
        Unique integer i/j/k indices of input-fitting-eligible voxels in one slice.
    affine : ndarray
        Finite nonsingular 4x4 voxel-to-mm affine, including obliquity/shear.

    Returns
    -------
    radius : float
        max(2 mm, sqrt(dx²+dy²)) using norms of the in-plane affine columns.
    neighbors : list of ndarray
        Sorted member indices, including each centre, within radius + 1e-6 mm.
        No padding, segmentation, fit-QC mask or through-plane voxel is added.
    """
    xyz, transform = np.asarray(coordinates), np.asarray(affine, dtype=float)
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or not np.issubdtype(xyz.dtype, np.integer)
        or np.any(xyz < 0)
        or len(np.unique(xyz, axis=0)) != len(xyz)
        or (len(xyz) and np.any(xyz[:, 2] != xyz[0, 2]))
    ):
        raise ValueError("Unique integer voxel coordinates in one slice required.")
    if (
        transform.shape != (4, 4)
        or not np.isfinite(transform).all()
        or not np.array_equal(transform[3], [0.0, 0.0, 0.0, 1.0])
        or np.linalg.matrix_rank(transform[:3, :3]) < 3
    ):
        raise ValueError("Finite nonsingular voxel-to-mm affine required.")
    radius = max(2.0, float(np.sqrt(np.sum(transform[:3, :2] ** 2))))
    physical = xyz @ transform[:3, :3].T
    neighbors = cKDTree(physical).query_ball_point(
        physical, radius + 1e-6, return_sorted=True
    )
    return radius, [np.asarray(members, dtype=np.intp) for members in neighbors]


def _validate_profile(profile: DelayProfile) -> None:
    """Require aligned family arrays and unique common observation identity."""
    grid, observed = profile.delays_seconds, profile.observed
    if (
        profile.family not in _COUNTS
        or grid.ndim != 1
        or not len(grid)
        or not np.isfinite(grid).all()
        or grid[0] != 0
        or grid[-1] > MAX_LOCAL_DELAY_SECONDS
        or np.any(np.diff(grid) <= 0)
        or observed.ndim != 1
        or len(observed) < 2
        or not np.issubdtype(observed.dtype, np.integer)
        or np.any(observed < 0)
        or len(np.unique(observed)) != len(observed)
        or profile.sse.ndim != 2
        or profile.sse.shape[0] != len(grid)
        or profile.status.shape != profile.sse.shape
        or profile.params.shape != (*profile.sse.shape, _PARAMETERS[profile.family])
    ):
        raise ValueError(
            "Aligned family profiles, delay grid and observation identity required."
        )


def family_timing(
    profile: DelayProfile, members: np.ndarray, centre: int,
    *, frame_interval_seconds: float,
) -> FamilyTiming:
    """Pool complete family profiles and test the centre's free-delay departure.

    Parameters
    ----------
    profile : DelayProfile
        Finalized conditional fits; no retries occur during selection.
    members : ndarray
        Distinct member indices including the centre.
    centre : int
        Target member index in the shared profile arrays.
    frame_interval_seconds : float
        Original acquisition interval, not the interval between retained frames.
        Used only for report-only upper-bound diagnostics.

    Returns
    -------
    FamilyTiming
        Raw timing, complete support and numerical eligibility. Finite failed
        fits keep their actual residual; incomplete members are excluded at all
        offsets. No next-best-delay substitution is made for selected failure.
    """
    if not np.isfinite(frame_interval_seconds) or frame_interval_seconds <= 0:
        raise ValueError("Positive finite original frame interval required.")
    _validate_profile(profile)
    members = np.asarray(members)
    if (
        members.ndim != 1
        or not np.issubdtype(members.dtype, np.integer)
        or len(np.unique(members)) != len(members)
        or centre not in members
        or np.any(members < 0)
        or np.any(members >= profile.sse.shape[1])
    ):
        raise ValueError("Distinct valid pool members including centre required.")
    params = profile.params[:, members]
    required = params[:, :, [0, 2]] if profile.family == "patlak" else params
    finite = (
        np.isfinite(profile.sse[:, members])
        & (profile.sse[:, members] >= 0)
        & np.isfinite(required).all(axis=2)
        & np.isfinite(profile.status[:, members])
    )
    complete = members[finite.all(axis=0)]
    support = len(complete)
    reference = 0
    fallback: str | None = "insufficient_support"
    n = len(profile.observed)
    if len(profile.delays_seconds) == 1:
        fallback = "fixed_zero"
    elif support >= 2:
        pooled = n * np.log(np.maximum(profile.sse[:, complete], 1e-30)).sum(axis=1)
        fallback = "flat_profile" if np.ptp(pooled) < 6 else None
        reference = 0 if fallback else int(np.argmin(pooled))
    if centre not in complete:
        return FamilyTiming(
            -1,
            reference,
            -1,
            False,
            np.inf,
            np.nan,
            support,
            fallback,
            0,
            "incomplete_family_profile",
            boundary_unavailable_reason="incomplete_family_profile",
        )
    sse = profile.sse[:, centre]
    best = int(np.argmin(sse))
    fixed_bic = float(
        n * np.log(max(sse[reference], 1e-30) / n) + _COUNTS[profile.family] * np.log(n)
    )
    free_bic = float(
        n * np.log(max(sse[best], 1e-30) / n)
        + (_COUNTS[profile.family] + 1) * np.log(n)
    )
    improvement = fixed_bic - free_bic
    free = improvement >= 6
    selected = best if free else reference
    status = int(profile.status[selected, centre])
    timing = FamilyTiming(
        selected,
        reference,
        best,
        free,
        free_bic if free else fixed_bic,
        improvement,
        support,
        fallback,
        status,
        None if status > 0 else "selected_nonconverged",
    )
    _set_boundary_flags(timing, profile.delays_seconds, sse, n, frame_interval_seconds)
    return timing


def _set_boundary_flags(
    timing: FamilyTiming, grid: np.ndarray, sse: np.ndarray, n: int,
    frame_interval_seconds: float,
) -> None:
    """Attach report-only flags while the complete centre profile is available.

    Parameters
    ----------
    timing : FamilyTiming
        Existing choice; selection fields are not modified.
    grid, sse : ndarray
        Full delay grid in seconds and finite nonnegative centre residuals.
    n : int
        Number of common retained observations.
    frame_interval_seconds : float
        Acquisition interval used for the pre-edge comparison range.
    """
    if len(grid) == 1:
        timing.boundary_unavailable_reason = "delay_fixed"
        return
    if grid[-1] != MAX_LOCAL_DELAY_SECONDS:
        timing.boundary_unavailable_reason = "incomplete_delay_range"
        return
    timing.free_optimum_near_lower = timing.best <= 1
    timing.free_optimum_near_upper = timing.best >= len(grid) - 2
    timing.reference_at_bound = timing.reference in (0, len(grid) - 1)
    if not timing.free_optimum_near_upper:
        return
    comparison = grid <= MAX_LOCAL_DELAY_SECONDS - frame_interval_seconds
    if not np.any(comparison):
        timing.boundary_unavailable_reason = "no_pre_edge_samples"
        return
    timing.upper_edge_evidence = float(
        n * np.log(max(float(sse[comparison].min()), 1e-30) / max(float(sse[-1]), 1e-30))
    )
    timing.upper_edge = "truncated" if timing.upper_edge_evidence >= 6 else "flat_edge"


def select_pool(
    profiles: dict[ProfileFamily, DelayProfile],
    aif: np.ndarray,
    times: np.ndarray,
    members: np.ndarray,
    centre: int,
    mode: Literal["auto", "patlak", "tofts", "extended_tofts", "2cxm"] = "auto",
) -> LocalSelection:
    """Select one centre's local timing and requested/automatic kinetic family.

    Parameters
    ----------
    profiles : dict
        Finalized family profiles on identical member/grid/observation support.
    aif, times : ndarray
        Full plasma input in mM and original acquisition times in minutes.
    members, centre : ndarray and int
        Neighborhood member indices and included target index.
    mode : str
        Automatic selection or one explicitly requested family.

    Returns
    -------
    LocalSelection
        Family and timing evidence. Final map retention and identification QC
        are separate; explicit selection never replaces the requested family.
    """
    aif, times = np.asarray(aif), np.asarray(times)
    if (
        times.ndim != 1
        or len(times) < 2
        or times.shape != aif.shape
        or not np.isfinite(times).all()
        or not np.isfinite(aif).all()
        or times[1] <= times[0]
        or not np.allclose(np.diff(times), times[1] - times[0], rtol=1e-12, atol=1e-14)
    ):
        raise ValueError(
            "Full finite AIF and uniform original acquisition times required."
        )
    dt = float((times[1] - times[0]) * 60)
    names: list[ProfileFamily]
    if mode == "auto":
        names = ["patlak", "extended_tofts"]
        if dt <= 2 + 1e-6:
            names.append("2cxm")
    elif mode in _COUNTS:
        names = [mode]
    else:
        raise ValueError(f"Unsupported selection mode: {mode}")
    if any(name not in profiles or profiles[name].family != name for name in names):
        raise ValueError("Profiles for every requested/eligible family are required.")
    first = profiles[names[0]]
    for name in names:
        profile = profiles[name]
        _validate_profile(profile)
        if (
            not np.array_equal(profile.observed, first.observed)
            or not np.array_equal(profile.delays_seconds, first.delays_seconds)
            or profile.sse.shape != first.sse.shape
            or np.any(profile.observed >= len(times))
        ):
            raise ValueError(
                "Families must share member, grid and original observation identity."
            )
    timing = {
        name: family_timing(profiles[name], members, centre, frame_interval_seconds=dt)
        for name in names
    }
    scores = {
        name: choice.bic if choice.reason is None else np.inf
        for name, choice in timing.items()
    }
    admission, exchange_reason = None, None
    if mode != "auto":
        return LocalSelection(names[0], timing, scores, None, None, np.nan)
    if "2cxm" not in names:
        exchange_reason = "sampling_interval_above_2s"
    else:
        choice = timing["2cxm"]
        exchange_reason = choice.reason
        if choice.reason is None:
            cx = profiles["2cxm"].params[choice.index, centre]
            et = profiles["extended_tofts"]
            if (
                np.any(cx < 0)
                or np.any(cx > [500.0, 500.0, 1.0, 1.0])
                or cx[2] + cx[3] > 1 + 1e-12
                or et.status[choice.index, centre] <= 0
            ):
                exchange_reason = "conditional_fit_unavailable"
            else:
                context = NativeInput(
                    aif,
                    times,
                    float(first.delays_seconds[choice.index]),
                    first.observed,
                )
                admission = exchange_admission(
                    context,
                    et.params[choice.index, centre],
                    float(et.sse[choice.index, centre]),
                    float(profiles["2cxm"].sse[choice.index, centre]),
                )
                exchange_reason = admission.reason
        if admission is None or not admission.passed:
            scores["2cxm"] = np.inf
    values = np.array([scores[name] for name in names])
    if not np.isfinite(values).any():
        return LocalSelection(None, timing, scores, admission, exchange_reason, np.nan)
    winner = int(np.argmin(values))
    ordered = np.sort(values)
    delta = float(ordered[1] - ordered[0]) if np.isfinite(ordered[:2]).all() else np.nan
    if winner != 0 and np.isfinite(values[0]) and values[0] - values[winner] < 6:
        winner, delta = 0, np.nan
    return LocalSelection(
        names[winner], timing, scores, admission, exchange_reason, delta
    )
