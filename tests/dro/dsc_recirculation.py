"""Synthetic DSC tissue presets, recirculation AIFs and transit-time residues."""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt
from scipy.special import gammainc


@dataclass(frozen=True)
class TissueParams:
    """Synthetic tissue perfusion parameters.

    CBV is derived via central volume theorem: CBV = CBF * MTT / 60.
    Never independently specified.

    Attributes
    ----------
    name : str
        Human-readable tissue label.
    cbf : float
        Cerebral blood flow (mL/100g/min).
    mtt : float
        Mean transit time (seconds).
    description : str
        Description of the synthetic scenario.
    """

    name: str
    cbf: float
    mtt: float
    description: str

    @property
    def cbv(self) -> float:
        """CBV in mL/100g, derived from central volume theorem."""
        return self.cbf * self.mtt / 60.0

    @property
    def cbv_raw(self) -> float:
        """CBV in raw deconvolution units (CBF * MTT), for assertion."""
        return self.cbf * self.mtt


@dataclass(frozen=True)
class GammaResidueShape:
    """Shape of a gamma transit-time distribution and its survival residue.

    Controls the gamma transit-time distribution and its survival residue.
    At fixed MTT, alpha = 1 gives an exponential residue; increasing alpha
    reduces relative dispersion and approaches a box-shaped residue.

    Attributes
    ----------
    alpha : float
        Gamma shape parameter.
    relative_dispersion : float
        Coefficient of variation, 1/sqrt(alpha) (dimensionless).
    description : str
        Human-readable description.
    """

    alpha: float
    relative_dispersion: float
    description: str


@dataclass(frozen=True)
class NoiseLevel:
    """Noise level for synthetic DSC tests.

    Attributes
    ----------
    fraction : float
        Noise std as fraction of peak tissue concentration.
    label : str
        Short label for test IDs.
    description : str
        Human-readable description.
    """

    fraction: float
    label: str
    description: str


# =============================================================================
# Synthetic parameter presets
# =============================================================================

# CBV derived via central volume theorem: CBV = CBF * MTT / 60
TISSUE_PARAMS: Dict[str, TissueParams] = {
    "normal_gm": TissueParams(
        name="Normal GM",
        cbf=60.0,
        mtt=4.6,
        description="Synthetic gray-matter scenario",
    ),
    "normal_wm": TissueParams(
        name="Normal WM",
        cbf=20.0,
        mtt=4.3,
        description="Synthetic white-matter scenario",
    ),
    "tumor": TissueParams(
        name="Tumor",
        cbf=100.0,
        mtt=6.0,
        description="Synthetic tumor scenario",
    ),
    "tumor_high": TissueParams(
        name="Tumor (hot-spot)",
        cbf=150.0,
        mtt=8.0,
        description="Synthetic high-perfusion tumor scenario",
    ),
    "tumor_low": TissueParams(
        name="Tumor (low-end)",
        cbf=50.0,
        mtt=4.0,
        description="Synthetic low-perfusion tumor scenario",
    ),
    "peritumoral": TissueParams(
        name="Peritumoral",
        cbf=35.0,
        mtt=5.0,
        description="Synthetic tissue scenario",
    ),
    "contralateral_gm": TissueParams(
        name="Contralateral GM",
        cbf=55.0,
        mtt=3.5,
        description="Synthetic tissue scenario",
    ),
    "low_perfusion": TissueParams(
        name="Low Perfusion",
        cbf=12.0,
        mtt=8.0,
        description="Synthetic low-perfusion scenario",
    ),
}

# Gamma transit-time distribution shape presets
GAMMA_RESIDUE_SHAPES: Dict[str, GammaResidueShape] = {
    "low_dispersion": GammaResidueShape(
        alpha=9.0,
        relative_dispersion=0.33,
        description="Low relative dispersion (alpha=9, CV=0.33)",
    ),
    "moderate_dispersion": GammaResidueShape(
        alpha=4.0,
        relative_dispersion=0.5,
        description="Moderate relative dispersion (alpha=4, CV=0.5)",
    ),
    "high_dispersion": GammaResidueShape(
        alpha=2.0,
        relative_dispersion=0.71,
        description="High relative dispersion (alpha=2, CV=0.71)",
    ),
}

NOISE_LEVELS: Dict[str, NoiseLevel] = {
    "zero": NoiseLevel(0.0, "zero", "No noise"),
    "low": NoiseLevel(0.02, "low", "2% of peak"),
    "medium": NoiseLevel(0.05, "medium", "5% of peak"),
    "high": NoiseLevel(0.10, "high", "10% of peak"),
}


# =============================================================================
# AIF generator with recirculation
# =============================================================================


def generate_recirculation_aif(
    n_timepoints: int = 80,
    dt_sec: float = 1.5,
    baseline_frames: int = 10,
    peak_delta_r2: float = 40.0,
    kappa: float = 0.35,
    recirc_delay_sec: float = 15.0,
    recirc_sigma_sec: float = 5.0,
    alpha_fp: float = 3.0,
    beta_fp: float = 2.5,
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Generate a synthetic AIF with a dispersed recirculation component.

    First-pass is a gamma-variate; recirculation is modeled as a
    delayed, broadened copy of the first pass.

    Recirculation: c_recirc = kappa * convolve(c_fp, G(t - delay, sigma))

    Parameters
    ----------
    n_timepoints : int
        Number of time points.
    dt_sec : float
        Temporal resolution in seconds.
    baseline_frames : int
        Number of pre-bolus baseline frames.
    peak_delta_r2 : float
        Peak first-pass delta-R2* (s^-1).
    kappa : float
        Relative recirculation amplitude.
    recirc_delay_sec : float
        Recirculation delay in seconds.
    recirc_sigma_sec : float
        Gaussian dispersion sigma for recirculation (s).
    alpha_fp : float
        First-pass gamma shape parameter.
    beta_fp : float
        First-pass gamma scale parameter (seconds).

    Returns
    -------
    Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
        Tuple of (aif_concentration, time_sec).
    """
    time_sec = np.arange(n_timepoints, dtype=np.float64) * dt_sec
    arrival_time = baseline_frames * dt_sec

    # First pass: gamma-variate (same math as generate_dsc_aif_dro)
    c_fp = np.zeros(n_timepoints, dtype=np.float64)
    mask = time_sec > arrival_time
    if np.any(mask):
        dt_arr = time_sec[mask] - arrival_time
        peak_time = alpha_fp * beta_fp
        peak_factor = (peak_time**alpha_fp) * np.exp(-alpha_fp)
        if peak_factor > 1e-10:
            amplitude = peak_delta_r2 / peak_factor
        else:
            amplitude = peak_delta_r2
        c_fp[mask] = amplitude * (dt_arr**alpha_fp) * np.exp(-dt_arr / beta_fp)

    # Recirculation: Gaussian kernel for dispersion
    if kappa > 0:
        # Build Gaussian kernel centered at recirc_delay
        kernel_length = n_timepoints
        kernel_time = np.arange(kernel_length, dtype=np.float64) * dt_sec
        gaussian_kernel = np.exp(
            -0.5 * ((kernel_time - recirc_delay_sec) / recirc_sigma_sec) ** 2
        )
        # Normalize kernel so convolution preserves area
        kernel_sum = np.sum(gaussian_kernel) * dt_sec
        if kernel_sum > 1e-10:
            gaussian_kernel /= kernel_sum

        # Convolve first pass with gaussian kernel, scale by kappa
        c_recirc_full = np.convolve(c_fp, gaussian_kernel * dt_sec, mode="full")
        c_recirc = kappa * c_recirc_full[:n_timepoints]

        aif = c_fp + c_recirc
    else:
        aif = c_fp.copy()

    return np.asarray(aif, dtype=np.float64), time_sec


# =============================================================================
# Gamma transit-time survival residue
# =============================================================================


def generate_gamma_residue(
    cbf: float,
    alpha: float,
    beta: float,
    n_points: int,
    dt: float,
) -> npt.NDArray[np.floating]:
    """Generate a flow-scaled survival residue for gamma-distributed transit times.

    R(t) = CBF * [1 - gammainc(alpha, t/beta)], with mean transit time
    alpha * beta.

    Parameters
    ----------
    cbf : float
        Cerebral blood flow (mL/100g/min).
    alpha : float
        Gamma shape parameter (controls relative dispersion).
    beta : float
        Gamma scale parameter (seconds). Related to MTT: beta = MTT / alpha.
    n_points : int
        Number of time points.
    dt : float
        Time step in seconds.

    Returns
    -------
    npt.NDArray[np.floating]
        Residue array R(t) = CBF * [1 - gammainc(alpha, t / beta)].
    """
    t = np.arange(n_points, dtype=np.float64) * dt
    # scipy.special.gammainc is the regularized lower incomplete gamma function
    # R(t) = CBF * [1 - P(alpha, t/beta)] where P is regularized lower gamma
    beta_safe = max(beta, 1e-10)
    residue = cbf * (1.0 - gammainc(alpha, t / beta_safe))
    return np.asarray(residue, dtype=np.float64)
