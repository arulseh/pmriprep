#!/usr/bin/env python
"""Validate anatomical pipeline against ground truth for sub-0001/ses-0002."""

import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

# Input paths
BIDS_ROOT = Path("/Users/arulseh/Developer/perf-plus/bidsroot")
INPUT_DIR = BIDS_ROOT / "rawdata/sub-0001/ses-0002/anat"
GROUND_TRUTH_DIR = BIDS_ROOT / "derivatives/anat_pipeline/sub-0001/ses-0002"

# Input files
T1W_PRECON = INPUT_DIR / "sub-0001_ses-0002_acq-mprage_run-1_T1w.nii.gz"
T1W_POSTCON = INPUT_DIR / "sub-0001_ses-0002_acq-mprage_ce-GAD_run-1_T1w.nii.gz"
T2W = INPUT_DIR / "sub-0001_ses-0002_acq-tse_run-1_T2w.nii.gz"
FLAIR = INPUT_DIR / "sub-0001_ses-0002_acq-space_run-1_FLAIR.nii.gz"
PDW = INPUT_DIR / "sub-0001_ses-0002_acq-tse_run-1_PDw.nii.gz"

# Output directory
OUTPUT_DIR = Path("/Users/arulseh/Developer/pmriprep/test_outputs/anat_validation")
WORK_DIR = OUTPUT_DIR / "work"


def compute_dice(mask1, mask2):
    """Compute DICE coefficient between two binary masks."""
    intersection = np.sum(mask1 & mask2)
    return 2 * intersection / (np.sum(mask1) + np.sum(mask2))


def compute_correlation(img1, img2, mask=None):
    """Compute correlation coefficient between two images."""
    if mask is not None:
        data1 = img1[mask > 0]
        data2 = img2[mask > 0]
    else:
        data1 = img1.flatten()
        data2 = img2.flatten()

    # Remove zeros/NaNs
    valid = (data1 > 0) & (data2 > 0) & np.isfinite(data1) & np.isfinite(data2)
    if np.sum(valid) < 100:
        return np.nan

    return np.corrcoef(data1[valid], data2[valid])[0, 1]


def compare_images(test_path, gt_path, name, mask=None):
    """Compare test image to ground truth."""
    if not Path(test_path).exists():
        print(f"  {name}: TEST FILE MISSING - {test_path}")
        return None

    if not Path(gt_path).exists():
        print(f"  {name}: GROUND TRUTH MISSING - {gt_path}")
        return None

    test_img = nib.load(test_path)
    gt_img = nib.load(gt_path)

    test_data = test_img.get_fdata()
    gt_data = gt_img.get_fdata()

    # Check dimensions
    if test_data.shape != gt_data.shape:
        print(f"  {name}: SHAPE MISMATCH - test {test_data.shape} vs gt {gt_data.shape}")
        return None

    # Compute correlation
    corr = compute_correlation(test_data, gt_data, mask)

    # For binary masks, also compute DICE
    if "mask" in name.lower() or "probseg" in name.lower():
        if "probseg" in name.lower():
            # Threshold probability maps
            dice = compute_dice(test_data > 0.5, gt_data > 0.5)
        else:
            dice = compute_dice(test_data > 0, gt_data > 0)
        print(f"  {name}: corr={corr:.4f}, DICE={dice:.4f}")
        return {"corr": corr, "dice": dice}
    else:
        print(f"  {name}: corr={corr:.4f}")
        return {"corr": corr}


def run_validation():
    """Run full validation."""
    print("=" * 70)
    print("ANATOMICAL PIPELINE VALIDATION")
    print("=" * 70)

    # Check inputs exist
    print("\n1. Checking input files...")
    inputs = {
        "T1w precon": T1W_PRECON,
        "T1w postcon": T1W_POSTCON,
        "T2w": T2W,
        "FLAIR": FLAIR,
        "PDw": PDW,
    }

    for name, path in inputs.items():
        exists = path.exists()
        print(f"  {name}: {'OK' if exists else 'MISSING'} - {path.name}")
        if not exists:
            print(f"    ERROR: Input file not found!")
            return

    # Create output directory
    print("\n2. Setting up directories...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Work: {WORK_DIR}")

    # Initialize workflow
    print("\n3. Initializing anatomical workflow...")
    from pmriprep.config.anat import AnatConfig
    from pmriprep.workflows.anat.base import init_anat_preproc_wf

    config = AnatConfig(
        tool="hdbet",
        denoise=False,
        skip_acpc=False,
        coregister_modalities=True,
        use_bbr=True,
        create_subtraction=True,
    )

    wf = init_anat_preproc_wf(
        bids_dir=str(BIDS_ROOT / "rawdata"),
        subject_id="0001",
        session_id="0002",
        config=config,
        name="anat_validation_wf",
    )

    wf.base_dir = str(WORK_DIR)

    # Set inputs
    print("\n4. Setting workflow inputs...")
    wf.inputs.inputnode.t1w = str(T1W_PRECON)
    wf.inputs.inputnode.t1w_postcon = str(T1W_POSTCON)
    wf.inputs.inputnode.t2w = str(T2W)
    wf.inputs.inputnode.flair = str(FLAIR)
    wf.inputs.inputnode.pdw = str(PDW)

    print(f"  t1w: {T1W_PRECON.name}")
    print(f"  t1w_postcon: {T1W_POSTCON.name}")
    print(f"  t2w: {T2W.name}")
    print(f"  flair: {FLAIR.name}")
    print(f"  pdw: {PDW.name}")

    # Run workflow
    print("\n5. Running workflow...")
    print("  This may take several minutes...")

    try:
        result = wf.run()
        print("  Workflow completed successfully!")
    except Exception as e:
        print(f"  ERROR: Workflow failed with: {e}")
        import traceback
        traceback.print_exc()
        return

    # Get outputs
    print("\n6. Collecting outputs...")
    outputnode = [n for n in result.nodes() if n.name == "outputnode"][0]
    outputs = outputnode.result.outputs

    print(f"  t1w_preproc: {outputs.t1w_preproc}")
    print(f"  brain_mask: {outputs.brain_mask}")
    print(f"  wm_probseg: {outputs.wm_probseg}")
    print(f"  t1w_postcon_coreg: {outputs.t1w_postcon_coreg}")
    print(f"  t2w_coreg: {outputs.t2w_coreg}")
    print(f"  flair_coreg: {outputs.flair_coreg}")
    print(f"  pdw_coreg: {outputs.pdw_coreg}")
    print(f"  t1w_subtraction: {outputs.t1w_subtraction}")
    print(f"  is_postcon_only: {outputs.is_postcon_only}")

    # Compare to ground truth
    print("\n7. Comparing to ground truth...")

    # Load brain mask for masked comparisons
    gt_mask_path = GROUND_TRUTH_DIR / "anat/sub-0001_ses-0002_space-T1w_desc-brain_mask.nii.gz"
    gt_mask = nib.load(gt_mask_path).get_fdata() if gt_mask_path.exists() else None

    comparisons = [
        (outputs.brain_mask,
         GROUND_TRUTH_DIR / "anat/sub-0001_ses-0002_space-T1w_desc-brain_mask.nii.gz",
         "Brain mask"),
        (outputs.wm_probseg,
         GROUND_TRUTH_DIR / "anat/sub-0001_ses-0002_space-T1w_label-WM_probseg.nii.gz",
         "WM probseg"),
        (outputs.gm_probseg,
         GROUND_TRUTH_DIR / "anat/sub-0001_ses-0002_space-T1w_label-GM_probseg.nii.gz",
         "GM probseg"),
        (outputs.csf_probseg,
         GROUND_TRUTH_DIR / "anat/sub-0001_ses-0002_space-T1w_label-CSF_probseg.nii.gz",
         "CSF probseg"),
        (outputs.t1w_preproc,
         GROUND_TRUTH_DIR / "anat/sub-0001_ses-0002_space-T1w_desc-preproc_T1w.nii.gz",
         "T1w preproc"),
        (outputs.t1w_postcon_coreg,
         GROUND_TRUTH_DIR / "anat/sub-0001_ses-0002_ce-GAD_space-T1w_desc-preproc_T1w.nii.gz",
         "T1w postcon coreg"),
        (outputs.t2w_coreg,
         GROUND_TRUTH_DIR / "anat/sub-0001_ses-0002_space-T1w_desc-preproc_T2w.nii.gz",
         "T2w coreg"),
        (outputs.flair_coreg,
         GROUND_TRUTH_DIR / "anat/sub-0001_ses-0002_space-T1w_desc-preproc_FLAIR.nii.gz",
         "FLAIR coreg"),
        (outputs.pdw_coreg,
         GROUND_TRUTH_DIR / "anat/sub-0001_ses-0002_space-T1w_desc-preproc_PDw.nii.gz",
         "PDw coreg"),
        (outputs.t1w_subtraction,
         GROUND_TRUTH_DIR / "anat/sub-0001_ses-0002_space-T1w_desc-subtraction_T1w.nii.gz",
         "T1w subtraction"),
    ]

    results = {}
    for test_path, gt_path, name in comparisons:
        if test_path is not None:
            result = compare_images(test_path, gt_path, name, gt_mask)
            if result:
                results[name] = result
        else:
            print(f"  {name}: OUTPUT IS NONE")

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    pass_threshold = 0.9  # Correlation threshold for pass
    dice_threshold = 0.85  # DICE threshold for masks

    all_pass = True
    for name, metrics in results.items():
        corr = metrics.get("corr", 0)
        dice = metrics.get("dice", None)

        status = "PASS" if corr >= pass_threshold else "FAIL"
        if dice is not None and dice < dice_threshold:
            status = "FAIL"

        if status == "FAIL":
            all_pass = False

        if dice is not None:
            print(f"  {name}: {status} (corr={corr:.4f}, DICE={dice:.4f})")
        else:
            print(f"  {name}: {status} (corr={corr:.4f})")

    print()
    if all_pass:
        print("  OVERALL: ALL TESTS PASSED")
    else:
        print("  OVERALL: SOME TESTS FAILED")

    # QC metrics
    print("\n8. QC Metrics from workflow...")
    print(f"  {outputs.qc_metrics}")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_validation()
