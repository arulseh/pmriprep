#!/usr/bin/env python3
"""Run SVB on CPI DRO to compare with MAP/VI results.

Creates a 2D spatial grid of homogeneous tissue patches, then fits with
SVB (FABBER's default) and reports CBF/MTT recovery per tissue type.
Also runs standard VI for comparison.
"""

import sys
import time

import numpy as np

from pmriprep.models.bayesian_utils import build_priors_from_params
from pmriprep.models.dsc.batched_fitting import batched_dsc_vi_fit, batched_dsc_map_fit
from pmriprep.models.dsc.cpi import CPIModel
from pmriprep.models.dsc.spatial_vb import batched_dsc_spatial_vi_fit
from pmriprep.models.dsc.vascular import VascularModel
from pmriprep.utils.dsc_deconvolution import build_volterra_matrix

from tests.dro.conftest import (
    dsc_forward_model,
    generate_dsc_aif_dro,
)


# Tissue types: (cbf, alpha, beta, name)
TISSUES = [
    (0.10, 3.0, 1.5, "normal_gm"),   # MTT=4.5s
    (0.04, 4.0, 1.5, "normal_wm"),   # MTT=6.0s
    (0.16, 2.0, 3.0, "tumor"),       # MTT=6.0s
    (0.13, 3.0, 4.0, "long_mtt"),    # MTT=12.0s
]

N_VOXELS_PER_TISSUE = 16  # 4x4 patch per tissue
N_CP = 10
T_SPAN = 30.0
N_TIMEPOINTS = 60
DT_SEC = 1.5
BASELINE_FRAMES = 10


def make_multi_tissue_grid(noise_frac: float = 0.0, seed: int = 42):
    """Create a 2D grid of tissue patches for SVB testing."""
    rng = np.random.default_rng(seed)

    aif, time_sec = generate_dsc_aif_dro(
        n_timepoints=N_TIMEPOINTS,
        dt_sec=DT_SEC,
        baseline_frames=BASELINE_FRAMES,
    )
    vascular = VascularModel()

    n_tissues = len(TISSUES)
    # 4x4 patch per tissue, laid out as 2 x 2 tissue blocks => 8 x 8 grid
    nx, ny, nz = 8, 8, 1
    total_voxels = 0
    tissue_curves_list = []
    tissue_labels = []
    coords_x, coords_y, coords_z = [], [], []

    for ti, (cbf, alpha, beta, name) in enumerate(TISSUES):
        residue = vascular.residue_function([cbf, alpha, beta, 0.0], time_sec)
        clean_curve = dsc_forward_model(aif, residue, DT_SEC)

        # Place 4x4 patch
        bx = (ti % 2) * 4
        by = (ti // 2) * 4
        for dx in range(4):
            for dy in range(4):
                curve = clean_curve.copy()
                if noise_frac > 0:
                    peak = float(np.max(np.abs(curve)))
                    if peak > 0:
                        curve = curve + rng.normal(0, noise_frac * peak, curve.shape)
                tissue_curves_list.append(curve)
                tissue_labels.append(ti)
                coords_x.append(bx + dx)
                coords_y.append(by + dy)
                coords_z.append(0)
                total_voxels += 1

    tissue_curves = np.stack(tissue_curves_list)
    mask_coords = (
        np.array(coords_x),
        np.array(coords_y),
        np.array(coords_z, dtype=int),
    )
    return tissue_curves, aif, time_sec, mask_coords, (nx, ny, nz), tissue_labels


def build_cpi_fit_kwargs(aif, time_sec):
    """Build common kwargs for CPI fitting."""
    dt = float(time_sec[1] - time_sec[0])
    model = CPIModel(
        n_control_points=N_CP,
        n_acquisition_frames=int(len(time_sec)),
        dt=dt,
    )
    fitting_defs = list(model.get_fitting_parameter_definitions())
    priors = build_priors_from_params(fitting_defs)
    cp_times = np.array(model._control_point_times, dtype=np.float64)
    toeplitz = build_volterra_matrix(aif, dt)

    return dict(
        model_type="cpi",
        aif=aif,
        time_points=time_sec,
        prior_means_nat=np.array([p.mean for p in priors]),
        prior_stds_nat=np.array([p.std for p in priors]),
        transforms=[p.transform for p in priors],
        upper_bounds=[p.upper_bound for p in priors],
        fitting_param_names=[p.name for p in fitting_defs],
        fitting_bounds=[(p.bounds[0], p.bounds[1]) for p in fitting_defs],
        control_point_times=cp_times,
        toeplitz_matrix=toeplitz,
        prior_means_transformed=np.array([p.mean_transformed for p in priors]),
        prior_stds_transformed=np.array([p.std_transformed for p in priors]),
        sloppy=False,
        save_fitted_curves=False,
    ), model, fitting_defs


def extract_cbf_mtt(result, model, fitting_defs, tissue_labels):
    """Extract CBF/MTT per tissue from batched result."""
    labels_arr = np.array(tissue_labels)
    results_by_tissue = {}

    for ti, (cbf_gt, alpha, beta, name) in enumerate(TISSUES):
        mask = labels_arr == ti
        true_mtt = alpha * beta
        cbfs = []
        mtts = []
        for i in np.where(mask)[0]:
            p_dict = {p.name: float(result.params[p.name][i]) for p in fitting_defs}
            p_list = model.dict_to_params(p_dict)
            cbfs.append(model.get_cbf(p_list))
            mtts.append(model.get_mtt(p_list))
        cbfs = np.array(cbfs)
        mtts = np.array(mtts)
        results_by_tissue[name] = {
            "cbf_gt": cbf_gt,
            "mtt_gt": true_mtt,
            "cbf_mean": float(np.mean(cbfs)),
            "cbf_std": float(np.std(cbfs)),
            "mtt_mean": float(np.mean(mtts)),
            "mtt_std": float(np.std(mtts)),
            "cbf_err_pct": float((np.mean(cbfs) - cbf_gt) / cbf_gt * 100),
            "mtt_err_pct": float((np.mean(mtts) - true_mtt) / true_mtt * 100),
        }
    return results_by_tissue


def print_results(label, results_by_tissue, elapsed):
    """Print a formatted table of results."""
    print(f"\n{'='*70}")
    print(f"  {label}  (elapsed: {elapsed:.1f}s)")
    print(f"{'='*70}")
    print(f"{'Tissue':<14} {'CBF GT':>7} {'CBF rec':>8} {'CBF err%':>9} "
          f"{'MTT GT':>7} {'MTT rec':>8} {'MTT err%':>9}")
    print(f"{'-'*14} {'-'*7} {'-'*8} {'-'*9} {'-'*7} {'-'*8} {'-'*9}")
    for name, r in results_by_tissue.items():
        print(f"{name:<14} {r['cbf_gt']:7.3f} {r['cbf_mean']:8.4f} "
              f"{r['cbf_err_pct']:+8.1f}% "
              f"{r['mtt_gt']:7.1f} {r['mtt_mean']:8.2f} "
              f"{r['mtt_err_pct']:+8.1f}%")


def main():
    noise_frac = 0.0
    if len(sys.argv) > 1:
        noise_frac = float(sys.argv[1])

    print(f"Noise fraction: {noise_frac}")

    tissue_curves, aif, time_sec, mask_coords, spatial_shape, tissue_labels = \
        make_multi_tissue_grid(noise_frac=noise_frac)

    common_kwargs, model, fitting_defs = build_cpi_fit_kwargs(aif, time_sec)

    # --- Standard VI ---
    print("\nRunning standard VI...")
    t0 = time.time()
    result_vi = batched_dsc_vi_fit(
        tissue_curves=tissue_curves,
        **common_kwargs,
    )
    elapsed_vi = time.time() - t0
    vi_results = extract_cbf_mtt(result_vi, model, fitting_defs, tissue_labels)
    print_results("Standard VI (batched)", vi_results, elapsed_vi)

    # --- Batched MAP ---
    print("\nRunning batched MAP...")
    t0 = time.time()
    result_map = batched_dsc_map_fit(
        tissue_curves=tissue_curves,
        **common_kwargs,
        compute_uncertainties=False,
    )
    elapsed_map = time.time() - t0
    map_results = extract_cbf_mtt(result_map, model, fitting_defs, tissue_labels)
    print_results("Batched MAP", map_results, elapsed_map)

    # --- SVB ---
    print("\nRunning SVB (spatial VB)...")
    t0 = time.time()
    result_svb = batched_dsc_spatial_vi_fit(
        tissue_curves=tissue_curves,
        mask_coords=mask_coords,
        spatial_shape=spatial_shape,
        n_spatial_sweeps=8,
        **common_kwargs,
    )
    elapsed_svb = time.time() - t0
    svb_results = extract_cbf_mtt(result_svb, model, fitting_defs, tissue_labels)
    print_results("SVB (spatial VB, FABBER default)", svb_results, elapsed_svb)

    # --- Summary comparison ---
    print(f"\n{'='*70}")
    print("  SUMMARY: CBF % error comparison")
    print(f"{'='*70}")
    print(f"{'Tissue':<14} {'MAP':>10} {'VI':>10} {'SVB':>10}")
    print(f"{'-'*14} {'-'*10} {'-'*10} {'-'*10}")
    for name in [t[3] for t in TISSUES]:
        print(f"{name:<14} "
              f"{map_results[name]['cbf_err_pct']:+9.1f}% "
              f"{vi_results[name]['cbf_err_pct']:+9.1f}% "
              f"{svb_results[name]['cbf_err_pct']:+9.1f}%")

    print(f"\n{'Tissue':<14} {'MAP':>10} {'VI':>10} {'SVB':>10}")
    print(f"{'-'*14} {'-'*10} {'-'*10} {'-'*10}")
    print("  MTT % error comparison:")
    for name in [t[3] for t in TISSUES]:
        print(f"{name:<14} "
              f"{map_results[name]['mtt_err_pct']:+9.1f}% "
              f"{vi_results[name]['mtt_err_pct']:+9.1f}% "
              f"{svb_results[name]['mtt_err_pct']:+9.1f}%")


def run_homogeneous(noise_frac: float = 0.02):
    """Run SVB vs VI on a homogeneous grid (single tissue) to isolate variance reduction."""
    from tests.dro.conftest import add_gaussian_noise

    print(f"\n{'#'*70}")
    print(f"  HOMOGENEOUS GRID TEST — noise={noise_frac}")
    print(f"  (Isolates SVB variance reduction from boundary effects)")
    print(f"{'#'*70}")

    for cbf_gt, alpha, beta, name in TISSUES:
        true_mtt = alpha * beta
        aif, time_sec = generate_dsc_aif_dro(
            n_timepoints=N_TIMEPOINTS, dt_sec=DT_SEC, baseline_frames=BASELINE_FRAMES,
        )
        vascular = VascularModel()
        residue = vascular.residue_function([cbf_gt, alpha, beta, 0.0], time_sec)
        clean_curve = dsc_forward_model(aif, residue, DT_SEC)

        rng = np.random.default_rng(42)
        nx, ny, nz = 8, 8, 1
        n_voxels = nx * ny
        curves = np.tile(clean_curve, (n_voxels, 1))
        if noise_frac > 0:
            peak = float(np.max(np.abs(clean_curve)))
            curves = curves + rng.normal(0, noise_frac * peak, curves.shape)

        x, y = np.mgrid[0:nx, 0:ny]
        mask_coords = (x.ravel(), y.ravel(), np.zeros(n_voxels, dtype=int))

        common_kwargs, model, fitting_defs = build_cpi_fit_kwargs(aif, time_sec)

        # VI
        result_vi = batched_dsc_vi_fit(tissue_curves=curves, **common_kwargs)
        vi_cbfs, vi_mtts = [], []
        for i in range(n_voxels):
            p_dict = {p.name: float(result_vi.params[p.name][i]) for p in fitting_defs}
            p_list = model.dict_to_params(p_dict)
            vi_cbfs.append(model.get_cbf(p_list))
            vi_mtts.append(model.get_mtt(p_list))
        vi_cbfs = np.array(vi_cbfs)
        vi_mtts = np.array(vi_mtts)

        # SVB
        result_svb = batched_dsc_spatial_vi_fit(
            tissue_curves=curves, mask_coords=mask_coords,
            spatial_shape=(nx, ny, nz), n_spatial_sweeps=8, **common_kwargs,
        )
        svb_cbfs, svb_mtts = [], []
        for i in range(n_voxels):
            p_dict = {p.name: float(result_svb.params[p.name][i]) for p in fitting_defs}
            p_list = model.dict_to_params(p_dict)
            svb_cbfs.append(model.get_cbf(p_list))
            svb_mtts.append(model.get_mtt(p_list))
        svb_cbfs = np.array(svb_cbfs)
        svb_mtts = np.array(svb_mtts)

        vi_cbf_cv = float(np.std(vi_cbfs) / np.mean(vi_cbfs) * 100)
        svb_cbf_cv = float(np.std(svb_cbfs) / np.mean(svb_cbfs) * 100)
        vi_mtt_cv = float(np.std(vi_mtts) / np.mean(vi_mtts) * 100)
        svb_mtt_cv = float(np.std(svb_mtts) / np.mean(svb_mtts) * 100)

        print(f"\n  {name} (CBF={cbf_gt}, MTT={true_mtt}s):")
        print(f"    {'':12} {'VI':>12} {'SVB':>12}")
        print(f"    CBF mean:  {np.mean(vi_cbfs):12.4f} {np.mean(svb_cbfs):12.4f}")
        print(f"    CBF err%:  {(np.mean(vi_cbfs)-cbf_gt)/cbf_gt*100:+11.1f}% "
              f"{(np.mean(svb_cbfs)-cbf_gt)/cbf_gt*100:+11.1f}%")
        print(f"    CBF CV%:   {vi_cbf_cv:11.1f}% {svb_cbf_cv:11.1f}%")
        print(f"    MTT mean:  {np.mean(vi_mtts):12.2f} {np.mean(svb_mtts):12.2f}")
        print(f"    MTT err%:  {(np.mean(vi_mtts)-true_mtt)/true_mtt*100:+11.1f}% "
              f"{(np.mean(svb_mtts)-true_mtt)/true_mtt*100:+11.1f}%")
        print(f"    MTT CV%:   {vi_mtt_cv:11.1f}% {svb_mtt_cv:11.1f}%")


if __name__ == "__main__":
    main()

    # Also run homogeneous test at 2% and 5% noise
    run_homogeneous(noise_frac=0.02)
    run_homogeneous(noise_frac=0.05)
