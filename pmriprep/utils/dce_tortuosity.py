"""DCE tortuosity correction utilities.

Functions for DWI-based tortuosity correction to adjust Ktrans values
for restricted diffusion in tumor tissue.

The tortuosity correction accounts for the fact that contrast agent
molecules must navigate around cells and structures in tissue, making
their effective path longer than the straight-line distance.

References:
    - Nicholson C, Syková E. Extracellular space structure revealed by
      diffusion analysis. Trends Neurosci. 1998;21(5):207-215.
    - Tofts PS, et al. Estimating kinetic parameters from dynamic
      contrast-enhanced T1-weighted MRI of a diffusable tracer.
      Magn Reson Med. 1997;38(3):411-421.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt


# Free diffusion coefficient of Gadolinium-DTPA at 37°C in mm²/s
# Literature values range from 2.5-3.5 × 10⁻³ mm²/s
# We use 3.0 × 10⁻³ mm²/s as the standard value
D_FREE_GD: float = 3.0e-3  # mm²/s

# Epsilon guard for ADC calculations to prevent division by zero
# Using 0.1e-3 mm²/s limits max tortuosity to ~30 (= D_free / EPSILON_ADC)
# This prevents extreme corrections from noise/artifacts (Gap 2 fix)
EPSILON_ADC: float = 0.1e-3  # mm²/s

# Typical ADC ranges for validation (mm²/s)
# Normal brain tissue: 0.7-1.0 × 10⁻³
# Tumor tissue: 0.5-1.5 × 10⁻³ (depending on cellularity)
ADC_MIN_VALID: float = 0.1e-3  # mm²/s
ADC_MAX_VALID: float = 3.0e-3  # mm²/s

# Typical tortuosity ranges
# λ = 1.0: unrestricted diffusion (ADC = D_free)
# λ = 1.5-2.5: typical brain tissue
# λ > 3.0: highly restricted (dense tumor)
# Values > 5.0 are physiologically implausible and likely artifacts (Gap 3 fix)
TORTUOSITY_MIN_VALID: float = 1.0
TORTUOSITY_MAX_VALID: float = 5.0


@dataclass
class TortuosityResult:
    """Result of tortuosity calculation.

    Attributes:
        adc: Apparent diffusion coefficient map in mm²/s.
        tortuosity: Tortuosity factor λ = D_free / ADC.
        d_free: Free diffusion coefficient used (mm²/s).
        is_valid: Boolean mask indicating valid voxels.
        n_valid: Number of valid voxels.
        n_total: Total number of voxels.
        qc_metrics: Comprehensive QC metrics dictionary (Gap 6 enhancement).
    """

    adc: npt.NDArray[np.floating]
    tortuosity: npt.NDArray[np.floating]
    d_free: float
    is_valid: npt.NDArray[np.bool_]
    n_valid: int
    n_total: int
    qc_metrics: Optional[dict] = None


@dataclass
class TortuosityCorrectionResult:
    """Result of applying tortuosity correction.

    Attributes:
        ktrans_corrected: Corrected Ktrans map in min⁻¹.
        ktrans_original: Original Ktrans map in min⁻¹.
        tortuosity: Tortuosity factor used for correction.
        correction_factor: Ratio of corrected to original Ktrans.
        was_corrected: Boolean indicating if correction was applied.
    """

    ktrans_corrected: npt.NDArray[np.floating]
    ktrans_original: npt.NDArray[np.floating]
    tortuosity: npt.NDArray[np.floating]
    correction_factor: npt.NDArray[np.floating]
    was_corrected: bool


def calculate_adc(
    dwi_signal: npt.NDArray[np.floating],
    b_values: npt.NDArray[np.floating],
    mask: Optional[npt.NDArray[np.bool_]] = None,
) -> npt.NDArray[np.floating]:
    """Calculate Apparent Diffusion Coefficient from DWI data.

    Uses the mono-exponential diffusion model:
        S(b) = S0 × exp(-b × ADC)

    Therefore:
        ADC = -ln(S/S0) / b

    For multi-b-value data, ADC is estimated using linear regression
    of ln(S) vs b.

    Args:
        dwi_signal: DWI signal array. Can be:
            - 3D (x, y, z) for single b-value pair (S at b)
            - 4D (x, y, z, n_b) for multiple b-values
        b_values: Array of b-values in s/mm². Must include b=0.
            - For 3D input: array([b0, b1]) where b0=0
            - For 4D input: array([b0, b1, ...]) matching last dimension
        mask: Optional brain mask. If provided, ADC is only calculated
            within the mask.

    Returns:
        ADC map in mm²/s. Shape matches spatial dimensions of input.

    Raises:
        ValueError: If b_values don't include b=0.
        ValueError: If number of b-values doesn't match signal dimensions.

    Notes:
        - b-values should be in s/mm² (e.g., 0, 1000)
        - ADC output is in mm²/s (e.g., 0.0007 for normal brain)
        - Negative ADC values (from noise) are clipped to EPSILON_ADC.

    Example:
        >>> # Single b-value pair
        >>> adc = calculate_adc(dwi_b1000, np.array([0, 1000]))
        >>> # Multi-b-value
        >>> adc = calculate_adc(dwi_4d, np.array([0, 500, 1000, 2000]))
    """
    b_values = np.asarray(b_values, dtype=np.float64)

    # Validate b-values
    if b_values.ndim != 1:
        raise ValueError(f"b_values must be 1D array, got {b_values.ndim}D")

    if not np.any(np.isclose(b_values, 0.0)):
        raise ValueError("b_values must include b=0 (reference image)")

    # Find b0 index
    b0_idx = int(np.argmin(np.abs(b_values)))

    # Handle different input dimensions
    if dwi_signal.ndim == 3:
        # Single b-value pair: assume S at high b-value, need S0 separately
        # This case requires exactly 2 b-values
        if len(b_values) != 2:
            raise ValueError(
                f"For 3D input, need exactly 2 b-values, got {len(b_values)}"
            )

        # Assume dwi_signal is S at high b-value
        # User should provide the reference (b0) signal as the first b-value
        raise ValueError(
            "3D input requires 4D data with b0 and b>0 volumes. "
            "Please provide 4D array with shape (x, y, z, n_b)."
        )

    elif dwi_signal.ndim == 4:
        # Multiple b-values: proper multi-b ADC estimation
        n_b = dwi_signal.shape[3]
        if len(b_values) != n_b:
            raise ValueError(
                f"Number of b-values ({len(b_values)}) doesn't match "
                f"4th dimension of signal ({n_b})"
            )

        spatial_shape = dwi_signal.shape[:3]

        # Get S0 (b=0 signal)
        s0 = dwi_signal[..., b0_idx]

        # Prevent log of zero/negative
        s0_safe = np.maximum(s0, EPSILON_ADC)

        if n_b == 2:
            # Two b-values: simple calculation
            # ADC = -ln(S/S0) / b
            other_idx = 1 - b0_idx  # The non-b0 index
            b = b_values[other_idx]
            s = dwi_signal[..., other_idx]

            # Safe signal ratio
            ratio = np.maximum(s, EPSILON_ADC) / s0_safe

            # Calculate ADC
            # ADC = -ln(ratio) / b, but b is in s/mm², ADC in mm²/s
            # So: ADC [mm²/s] = -ln(ratio) / (b [s/mm²])
            adc = -np.log(ratio) / b

        else:
            # Multiple b-values: linear regression of ln(S) vs b
            # ln(S) = ln(S0) - b × ADC
            # y = a + bx where y=ln(S), x=b, slope=-ADC

            # Reshape for vectorized regression
            # Signal: (x, y, z, n_b) -> (n_voxels, n_b)
            n_voxels = int(np.prod(spatial_shape))
            signal_flat = dwi_signal.reshape(n_voxels, n_b)

            # Take log of signal (with safety)
            log_signal = np.log(np.maximum(signal_flat, EPSILON_ADC))

            # Linear regression: ADC = -slope of ln(S) vs b
            # Using numpy polyfit for each voxel
            adc_flat = np.zeros(n_voxels)

            for i in range(n_voxels):
                try:
                    # polyfit returns [slope, intercept] for degree 1
                    coeffs = np.polyfit(b_values, log_signal[i, :], 1)
                    adc_flat[i] = -coeffs[0]  # ADC = -slope
                except (np.linalg.LinAlgError, ValueError):
                    adc_flat[i] = EPSILON_ADC

            adc = adc_flat.reshape(spatial_shape)

    else:
        raise ValueError(
            f"dwi_signal must be 3D or 4D, got {dwi_signal.ndim}D"
        )

    # Apply mask if provided
    if mask is not None:
        adc = np.where(mask, adc, 0.0)

    # Clip negative values (can occur from noise)
    adc = np.maximum(adc, EPSILON_ADC)

    return adc


def calculate_tortuosity(
    adc: npt.NDArray[np.floating],
    d_free: float = D_FREE_GD,
    mask: Optional[npt.NDArray[np.bool_]] = None,
    clip_range: bool = True,
) -> TortuosityResult:
    """Calculate tortuosity factor from ADC map.

    Tortuosity (λ) represents the ratio of free diffusion to restricted
    diffusion in tissue:

        λ = D_free / ADC

    Higher tortuosity indicates more restricted diffusion, requiring
    longer effective paths for molecular transport.

    Args:
        adc: ADC map in mm²/s.
        d_free: Free diffusion coefficient of the tracer in mm²/s.
            Default is 3.0×10⁻³ mm²/s for Gd at 37°C.
        mask: Optional mask for ROI analysis.
        clip_range: If True, clip tortuosity to valid physiological range.

    Returns:
        TortuosityResult with tortuosity map and validity information.

    Notes:
        - λ = 1.0: unrestricted diffusion (ADC equals D_free)
        - λ > 1.0: restricted diffusion (typical tissue)
        - Typical brain λ: 1.5-2.5
        - Dense tumor λ: 2.5-5.0 or higher

    Example:
        >>> result = calculate_tortuosity(adc_map, d_free=D_FREE_GD)
        >>> tortuosity = result.tortuosity
        >>> print(f"Mean tortuosity in mask: {tortuosity[mask].mean():.2f}")
    """
    # Calculate tortuosity: λ = D_free / ADC
    # Prevent division by zero
    adc_safe = np.maximum(adc, EPSILON_ADC)
    tortuosity = d_free / adc_safe

    # Determine valid voxels
    is_valid = np.ones(adc.shape, dtype=bool)

    # Mark as invalid if ADC is outside reasonable range
    is_valid &= adc >= ADC_MIN_VALID
    is_valid &= adc <= ADC_MAX_VALID

    # Apply mask if provided
    if mask is not None:
        is_valid &= mask

    # Clip tortuosity to valid range if requested
    if clip_range:
        tortuosity = np.clip(tortuosity, TORTUOSITY_MIN_VALID, TORTUOSITY_MAX_VALID)

    # Count valid voxels
    n_valid = int(np.sum(is_valid))
    n_total = int(adc.size)

    # Build comprehensive QC metrics dict (Gap 6 enhancement)
    qc_metrics: Optional[dict] = None
    if n_valid > 0:
        adc_valid = adc[is_valid]
        tort_valid = tortuosity[is_valid]
        qc_metrics = {
            "adc": {
                "mean": float(np.mean(adc_valid)),
                "median": float(np.median(adc_valid)),
                "std": float(np.std(adc_valid)),
                "min": float(np.min(adc_valid)),
                "max": float(np.max(adc_valid)),
                "units": "mm²/s",
            },
            "tortuosity": {
                "mean": float(np.mean(tort_valid)),
                "median": float(np.median(tort_valid)),
                "std": float(np.std(tort_valid)),
                "min": float(np.min(tort_valid)),
                "max": float(np.max(tort_valid)),
            },
            "d_free": d_free,
            "d_free_units": "mm²/s",
            "n_valid_voxels": n_valid,
            "n_total_voxels": n_total,
            "valid_fraction": n_valid / n_total if n_total > 0 else 0.0,
        }

    return TortuosityResult(
        adc=adc,
        tortuosity=tortuosity,
        d_free=d_free,
        is_valid=is_valid,
        n_valid=n_valid,
        n_total=n_total,
        qc_metrics=qc_metrics,
    )


def apply_tortuosity_correction(
    ktrans: npt.NDArray[np.floating],
    tortuosity: Union[float, npt.NDArray[np.floating]],
    mask: Optional[npt.NDArray[np.bool_]] = None,
) -> TortuosityCorrectionResult:
    """Apply tortuosity correction to Ktrans map.

    The tortuosity correction adjusts Ktrans to account for restricted
    diffusion in tissue:

        Ktrans_corrected = Ktrans × λ

    This correction is based on the principle that the measured Ktrans
    underestimates the true permeability-surface area product because
    contrast agent molecules must navigate around obstacles in tissue.

    Args:
        ktrans: Ktrans map in min⁻¹.
        tortuosity: Tortuosity factor λ. Can be:
            - Scalar: apply same correction to all voxels
            - Array: voxel-wise tortuosity map
        mask: Optional mask for ROI-specific correction.

    Returns:
        TortuosityCorrectionResult with corrected Ktrans and correction info.

    Notes:
        - Typical correction factor: 1.5-3.0× increase in Ktrans
        - Correction is most significant in highly cellular tumors
        - If tortuosity is not available, set λ=1.0 for no correction

    Example:
        >>> result = apply_tortuosity_correction(ktrans_map, tortuosity=2.0)
        >>> ktrans_corr = result.ktrans_corrected
        >>> # Or with voxel-wise tortuosity:
        >>> result = apply_tortuosity_correction(ktrans_map, tortuosity_map)
    """
    # Ensure tortuosity is array-like for broadcasting
    if np.isscalar(tortuosity):
        tortuosity_arr = np.full_like(ktrans, tortuosity)
    else:
        tortuosity_arr = np.asarray(tortuosity)

    # Apply correction: Ktrans_corrected = Ktrans × λ
    ktrans_corrected = ktrans * tortuosity_arr

    # Calculate correction factor
    # Avoid division by zero where Ktrans is near zero
    ktrans_safe = np.maximum(np.abs(ktrans), 1e-10)
    correction_factor = ktrans_corrected / ktrans_safe

    # Apply mask if provided
    if mask is not None:
        ktrans_corrected = np.where(mask, ktrans_corrected, ktrans)
        correction_factor = np.where(mask, correction_factor, 1.0)

    # Check if any actual correction was applied
    was_corrected = not np.allclose(tortuosity_arr, 1.0)

    return TortuosityCorrectionResult(
        ktrans_corrected=ktrans_corrected,
        ktrans_original=ktrans,
        tortuosity=tortuosity_arr,
        correction_factor=correction_factor,
        was_corrected=was_corrected,
    )


def calculate_and_apply_tortuosity_correction(
    ktrans: npt.NDArray[np.floating],
    dwi_signal: npt.NDArray[np.floating],
    b_values: npt.NDArray[np.floating],
    mask: Optional[npt.NDArray[np.bool_]] = None,
    d_free: float = D_FREE_GD,
) -> Tuple[TortuosityCorrectionResult, TortuosityResult]:
    """Full tortuosity correction pipeline from DWI data.

    Combines ADC calculation, tortuosity estimation, and Ktrans
    correction into a single convenient function.

    Args:
        ktrans: Ktrans map in min⁻¹.
        dwi_signal: 4D DWI data (x, y, z, n_b).
        b_values: Array of b-values in s/mm².
        mask: Optional brain/ROI mask.
        d_free: Free diffusion coefficient in mm²/s.

    Returns:
        Tuple of (TortuosityCorrectionResult, TortuosityResult).

    Example:
        >>> correction, tortuosity = calculate_and_apply_tortuosity_correction(
        ...     ktrans=ktrans_map,
        ...     dwi_signal=dwi_data,
        ...     b_values=np.array([0, 1000]),
        ...     mask=brain_mask,
        ... )
        >>> ktrans_corrected = correction.ktrans_corrected
    """
    # Step 1: Calculate ADC
    adc = calculate_adc(dwi_signal, b_values, mask=mask)

    # Step 2: Calculate tortuosity
    tortuosity_result = calculate_tortuosity(adc, d_free=d_free, mask=mask)

    # Step 3: Apply correction
    correction_result = apply_tortuosity_correction(
        ktrans, tortuosity_result.tortuosity, mask=mask
    )

    return correction_result, tortuosity_result


def load_and_apply_tortuosity_correction(
    ktrans_file: Union[str, Path],
    dwi_file: Union[str, Path],
    b_values: npt.NDArray[np.floating],
    output_file: Union[str, Path],
    mask_file: Optional[Union[str, Path]] = None,
    d_free: float = D_FREE_GD,
) -> Tuple[Path, TortuosityCorrectionResult, TortuosityResult]:
    """File-based tortuosity correction workflow.

    Loads NIfTI files, applies tortuosity correction, and saves result.

    Args:
        ktrans_file: Path to Ktrans NIfTI file.
        dwi_file: Path to DWI NIfTI file (4D with multiple b-values).
        b_values: Array of b-values in s/mm².
        output_file: Path for corrected Ktrans output.
        mask_file: Optional path to brain mask NIfTI.
        d_free: Free diffusion coefficient in mm²/s.

    Returns:
        Tuple of (output_path, TortuosityCorrectionResult, TortuosityResult).

    Raises:
        FileNotFoundError: If input files don't exist.
        ValueError: If image dimensions don't match.
    """
    import nibabel as nib  # type: ignore[import-untyped]

    # Load files
    ktrans_path = Path(ktrans_file)
    dwi_path = Path(dwi_file)

    if not ktrans_path.exists():
        raise FileNotFoundError(f"Ktrans file not found: {ktrans_path}")
    if not dwi_path.exists():
        raise FileNotFoundError(f"DWI file not found: {dwi_path}")

    ktrans_img = nib.load(ktrans_path)
    ktrans_data = ktrans_img.get_fdata()  # type: ignore[attr-defined]

    dwi_img = nib.load(dwi_path)
    dwi_data = dwi_img.get_fdata()  # type: ignore[attr-defined]

    # Validate dimensions match (spatial)
    if ktrans_data.shape[:3] != dwi_data.shape[:3]:
        raise ValueError(
            f"Ktrans shape {ktrans_data.shape[:3]} doesn't match "
            f"DWI shape {dwi_data.shape[:3]}"
        )

    # Load mask if provided
    mask: Optional[npt.NDArray[np.bool_]] = None
    if mask_file is not None:
        mask_path = Path(mask_file)
        if mask_path.exists():
            mask_img = nib.load(mask_path)
            mask = mask_img.get_fdata().astype(bool)  # type: ignore[attr-defined]

    # Apply correction
    correction_result, tortuosity_result = calculate_and_apply_tortuosity_correction(
        ktrans=ktrans_data,
        dwi_signal=dwi_data,
        b_values=b_values,
        mask=mask,
        d_free=d_free,
    )

    # Save output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ktrans_corr_img = nib.Nifti1Image(
        correction_result.ktrans_corrected,
        ktrans_img.affine,  # type: ignore[attr-defined]
        ktrans_img.header,  # type: ignore[attr-defined]
    )
    nib.save(ktrans_corr_img, output_path)

    return output_path, correction_result, tortuosity_result


def generate_tortuosity_qc_plot(
    adc_map: npt.NDArray[np.floating],
    tortuosity_map: npt.NDArray[np.floating],
    ktrans_orig: npt.NDArray[np.floating],
    ktrans_corr: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    out_file: Union[str, Path],
    slice_idx: Optional[int] = None,
) -> str:
    """Generate 6-panel QC figure for tortuosity correction (Gap 5).

    Creates a comprehensive visualization of the tortuosity correction
    process for quality control.

    Panels:
        1. ADC map (×10⁻³ mm²/s, range 0-2)
        2. Tortuosity map (λ, range 1.0-3.0)
        3. Original Ktrans (min⁻¹)
        4. Corrected Ktrans (min⁻¹)
        5. Correction factor heatmap (0.8-2.0, RdBu_r colormap)
        6. Histogram overlay (original vs corrected Ktrans)

    Args:
        adc_map: ADC map in mm²/s.
        tortuosity_map: Tortuosity factor λ = D_free / ADC.
        ktrans_orig: Original Ktrans map in min⁻¹.
        ktrans_corr: Corrected Ktrans map in min⁻¹.
        mask: Brain mask for masking display.
        out_file: Output path for the QC figure.
        slice_idx: Axial slice index to display. If None, auto-selects
            the slice with most brain voxels.

    Returns:
        Path to the saved QC figure.
    """
    import matplotlib.pyplot as plt

    # Auto-select slice with most brain voxels if not specified
    if slice_idx is None:
        slice_idx = int(np.argmax(mask.sum(axis=(0, 1))))

    # Calculate correction factor (avoid div by zero)
    ktrans_safe = np.maximum(np.abs(ktrans_orig), 1e-10)
    correction_factor = ktrans_corr / ktrans_safe

    # Mask the data for display
    adc_display = np.where(mask, adc_map * 1000, np.nan)  # Convert to ×10⁻³
    tort_display = np.where(mask, tortuosity_map, np.nan)
    ktrans_orig_display = np.where(mask, ktrans_orig, np.nan)
    ktrans_corr_display = np.where(mask, ktrans_corr, np.nan)
    cf_display = np.where(mask, correction_factor, np.nan)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Tortuosity Correction QC (slice {slice_idx})", fontsize=14)

    # Panel 1: ADC map
    ax = axes[0, 0]
    im = ax.imshow(
        adc_display[:, :, slice_idx].T,
        cmap="hot",
        vmin=0,
        vmax=2,
        origin="lower",
    )
    ax.set_title("ADC (×10⁻³ mm²/s)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 2: Tortuosity map
    ax = axes[0, 1]
    im = ax.imshow(
        tort_display[:, :, slice_idx].T,
        cmap="viridis",
        vmin=1.0,
        vmax=3.0,
        origin="lower",
    )
    ax.set_title("Tortuosity (λ)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 3: Original Ktrans
    ax = axes[0, 2]
    ktrans_max = np.nanpercentile(ktrans_corr_display, 99)
    im = ax.imshow(
        ktrans_orig_display[:, :, slice_idx].T,
        cmap="hot",
        vmin=0,
        vmax=max(ktrans_max, 0.1),
        origin="lower",
    )
    ax.set_title("Original Ktrans (min⁻¹)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 4: Corrected Ktrans
    ax = axes[1, 0]
    im = ax.imshow(
        ktrans_corr_display[:, :, slice_idx].T,
        cmap="hot",
        vmin=0,
        vmax=max(ktrans_max, 0.1),
        origin="lower",
    )
    ax.set_title("Corrected Ktrans (min⁻¹)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 5: Correction factor heatmap
    ax = axes[1, 1]
    im = ax.imshow(
        cf_display[:, :, slice_idx].T,
        cmap="RdBu_r",
        vmin=0.8,
        vmax=2.0,
        origin="lower",
    )
    ax.set_title("Correction Factor")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 6: Histogram overlay
    ax = axes[1, 2]
    ktrans_orig_masked = ktrans_orig[mask & (ktrans_orig > 0)]
    ktrans_corr_masked = ktrans_corr[mask & (ktrans_corr > 0)]

    if len(ktrans_orig_masked) > 0 and len(ktrans_corr_masked) > 0:
        bins = np.linspace(0, np.percentile(ktrans_corr_masked, 99), 50)
        ax.hist(
            ktrans_orig_masked,
            bins=bins,
            alpha=0.5,
            label="Original",
            color="blue",
        )
        ax.hist(
            ktrans_corr_masked,
            bins=bins,
            alpha=0.5,
            label="Corrected",
            color="red",
        )
        ax.set_xlabel("Ktrans (min⁻¹)")
        ax.set_ylabel("Voxel count")
        ax.set_title("Ktrans Distribution")
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "No valid data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Ktrans Distribution")

    plt.tight_layout()
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return str(out_path)
