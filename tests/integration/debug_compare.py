#!/usr/bin/env python3
"""Quick diagnostic: compare all pmriprep outputs vs perf-plus ground truth."""
import nibabel as nib
import numpy as np
from pathlib import Path

work = Path("/Users/arulseh/Developer/pmriprep/work/pmriprep_sub0001_ses-0002_wf/dsc_wf/compute_maps")
gt = Path("/Users/arulseh/Developer/perf-plus/bidsroot/derivatives/dsc_pipeline/sub-0001/ses-0002/perf")
mask_path = "/Users/arulseh/Developer/perf-plus/bidsroot/derivatives/dsc_pipeline/sub-0001/ses-0002/anat/sub-0001_ses-0002_space-DSC_res-1.87x1.88x4.00_desc-brain_mask.nii.gz"

mask = nib.load(mask_path).get_fdata() > 0.5
print(f"mask {mask.shape} nvox={int(np.sum(mask))}")

pairs = [
    ("rel_cbf", "desc-rel_cbf.nii.gz"),
    ("rel_cbv", "desc-rel_cbv.nii.gz"),
    ("mtt", "desc-dscproc_mtt.nii.gz"),
    ("tmax", "desc-dscproc_tmax.nii.gz"),
    ("ph", "desc-dscproc_ph.nii.gz"),
    ("psr", "desc-dscproc_psr.nii.gz"),
    ("ttp", "desc-dscproc_ttp.nii.gz"),
    ("cnr", "desc-dscproc_cnr.nii.gz"),
    ("snr", "desc-dscproc_snr.nii.gz"),
    ("tsnr", "desc-dscproc_tsnr.nii.gz"),
]

for our_suf, gt_pat in pairs:
    ours = sorted(work.glob(f"*_{our_suf}.nii.gz"))
    gts = sorted(gt.glob(f"*{gt_pat}"))
    if not ours or not gts:
        print(f"{our_suf}: MISSING ours={len(ours)} gt={len(gts)}")
        continue
    od = nib.load(str(ours[0])).get_fdata()
    gd = nib.load(str(gts[0])).get_fdata()
    if od.shape != gd.shape:
        print(f"{our_suf}: SHAPE MISMATCH {od.shape} vs {gd.shape}")
        continue
    om = od[mask]
    gm = gd[mask]
    v = np.isfinite(om) & np.isfinite(gm)
    om, gm = om[v], gm[v]
    nz = (om != 0) | (gm != 0)
    nnz = int(np.sum(nz))
    if nnz > 10:
        c = float(np.corrcoef(om[nz], gm[nz])[0, 1])
        gm_mean = float(np.mean(gm[nz]))
        r = float(np.mean(om[nz])) / gm_mean if gm_mean != 0 else 0.0
    else:
        c, r = 0.0, 0.0
    print(
        f"{our_suf:10s} corr={c:.4f} ratio={r:.4f} "
        f"ours=[{float(np.min(om)):.2f},{float(np.max(om)):.2f}] "
        f"gt=[{float(np.min(gm)):.2f},{float(np.max(gm)):.2f}] "
        f"ours_mean={float(np.mean(om)):.4f} gt_mean={float(np.mean(gm)):.4f}"
    )
