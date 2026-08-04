# PMRIPrep

**Automated preprocessing and quantification of DSC and DCE MRI**

PMRIPrep is a BIDS-App for preprocessing and quantifying Dynamic Susceptibility
Contrast (DSC) and Dynamic Contrast Enhanced (DCE) MRI. It follows NiPreps
workflow conventions and writes BIDS-Derivatives-style outputs with per-session
HTML QC reports.

PMRIPrep provides:

- DSC deconvolution with sSVD, oSVD, Tikhonov, CPI, or vascular models;
- DCE pharmacokinetic modelling with Patlak, Tofts, Extended Tofts, or 2CXM;
- Automatic or external arterial input functions;
- VFA T1 mapping with optional B1+ correction;
- Anatomical preprocessing, optional HD-GLIO tumor segmentation, and NAWM
  normalization; 
- Provenance-rich derivatives and visual QC.

## Installation

### Containers

For Docker:

Install:
```bash
docker pull ghcr.io/arulseh/pmriprep:latest
```

Run:
```bash
docker run --rm --init \
  -v /path/to/bids:/data:ro \
  -v /path/to/output:/out \
  -v /path/to/work:/work \
  ghcr.io/arulseh/pmriprep:latest \
  /data /out participant --participant-label 01 -w /work
```

Add `--gpus all` to the Docker command for NVIDIA GPU support. HD-BET falls
back to CPU when needed; HD-GLIO tumor segmentation is skipped when GPU is
unavailable. TemplateFlow resources are included in the image.

For Apptainer:

Install:
```bash
apptainer pull pmriprep.sif docker://ghcr.io/arulseh/pmriprep:latest
```

Run:
```bash
apptainer run --cleanenv \
  --bind /path/to/bids:/data:ro \
  --bind /path/to/output:/out \
  --bind /path/to/work:/work \
  pmriprep.sif \
  /data /out participant --participant-label 01 -w /work
```

Add `--nv` for NVIDIA GPU support. TemplateFlow resources are included in the
image.

### Source installation

```bash
python -m pip install .
```

Python 3.11–3.12, FSL, and ANTs are required. The container is the supported
way to obtain the complete runtime environment.

### System Requirements

Container execution requires Docker or Apptainer. Source installations require
Python 3.11–3.12, FSL, and ANTs. An NVIDIA GPU is optional; CPU, memory, and disk
requirements depend on the input data and enabled pipelines.

## Usage

```bash
pmriprep bids_dir output_dir participant [options]
```

Common options:

| Option | Description |
|---|---|
| `--participant-label LABEL [...]` | Select participants (without `sub-` prefix) |
| `--session-label LABEL [...]` | Select sessions (without `ses-` prefix) |
| `-w PATH`, `--work-dir PATH` | Set the Nipype working directory |
| `--nprocs N` | Limit workflow and fitting processes (default: 1) |
| `--omp-nthreads N` | Limit threads per process |
| `--mem-gb GB` | Set the Nipype memory limit |
| `--config-file PATH` | Load options from a YAML or JSON configuration file |
| `--skip-tumor-seg`, `--skip-dsc`, `--skip-dce` | Disable optional pipelines |
| `--report-only` | Rebuild HTML reports from existing derivatives |

Key DSC options:

| Option | Description |
|---|---|
| `--dsc-deconv-method {sSVD,oSVD,tikhonov,vascular,cpi}` | Select deconvolution (default: oSVD) |
| `--dsc-inference {vi,svb,nlls}` | Select inference for CPI/vascular models (default: vi) |
| `--dsc-dose FLOAT` | Set a positive dose multiplier (for example, `0.5` for half dose) |
| `--dsc-aif-method {auto,synthetic}` | Select detected or synthetic AIF |
| `--dsc-aif-file FILE` | Supply an external ΔR2* AIF in s⁻¹ |

Key DCE options:

| Option | Description |
|---|---|
| `--dce-model {auto,patlak,tofts,extended_tofts,2cxm}` | Select PK model; `auto` uses voxelwise BIC |
| `--dce-inference {nlls,vi}` | Select inference (default: nlls) |
| `--dce-aif-method {auto,population}` | Select detected or population AIF |
| `--dce-aif-file FILE` | Supply a plasma-concentration AIF in mM |
| `--dce-dose FLOAT` | Set a positive dose multiplier (for example, `0.5` for half dose) |
| `--dce-b1-method {auto,afi,precomputed,none}` | Control B1+ correction |
| `--dce-no-t1` | Use a uniform field-strength reference T1 |
| `--dce-t1-map FILE` | Supply a millisecond T1 map world-aligned to DCE |

Use `pmriprep --help` for the complete execution interface. `--use-plugin`
accepts a Nipype plugin YAML/JSON for cluster schedulers. `--clean-workdir`,
`--resource-monitor`, `--write-graph`, `--stop-on-first-crash`, and `--sloppy`
follow the corresponding NiPreps conventions.

## Configuration

The file schema has four optional top-level mappings: `dsc`, `dce`, `anat`, and
`tumor_seg`. Options within each section are flat. CLI scientific options
override their file equivalents.

```yaml
dsc:
  method: cpi
  inference: svb
  dose: 0.75
  lesion_mask: /path/to/lesion_mask.nii.gz
  smooth_fwhm: 3.0
  underlays: [t1w_postcon, flair_precon]

dce:
  model: auto
  inference: nlls
  b1_method: auto
  t1_method: nlls
  smooth_fwhm: 3.0

anat:
  tool: hdbet
  denoise: false
  create_subtraction: true

tumor_seg:
  nawm_erosion: 2
  nawm_min_voxels: 500
```

DSC fields:

| Field | Default | Description |
|---|---:|---|
| `method` | `oSVD` | `sSVD`, `oSVD`, `tikhonov`, `vascular`, or `cpi` |
| `inference` | `vi` | `vi`, `svb` or `nlls` for CPI/vascular |
| `cpi_control_points` | `0` | CPI control points; 0 selects automatically |
| `cpi_infer_cpt` | `true` | Infer ordered CPI control-point times |
| `cpi_infer_delay` | `false` | Fit voxelwise CPI bolus delay |
| `dose` | `1.0` | Contrast-dose multiplier (float; for example, 0.5 for half dose) |
| `agent_standard_dose` | `0.1` | Agent standard dose in mmol Gd/kg (float) |
| `preload` | `unknown` | Administered preload dose fraction, or `unknown`; provenance only |
| `aif_method` | `auto` | `auto` or `synthetic` |
| `aif_file` | `null` | External ΔR2* AIF; bypasses detection |
| `aif_mask_file` | `null` | AIF search-space mask (native-DSC space) |
| `lesion_distance` | `15.0` | Minimum distance from lesion in mm for AIF candidates (float) |
| `n_aif_candidates` | `10` | Maximum averaged candidates; minimum configuration is 3 |
| `disable_aif_fallback` | `false` | Fail or retain a flagged AIF instead of substituting synthetic |
| `lesion_mask` | `null` | Lesion mask (statistics and AIF lesion exclusion; native-DSC space) |
| `nawm_mask` | `null` | NAWM mask (native-DSC space) |
| `roi_mask` | `null` | Statistics-only ROI mask (native-DSC space) |
| `leakage_method` | `bsw` | `bsw` leakage correction or `none` for baseline centering without leakage correction |
| `pre_bolus` | `5` | Initial baseline frames; final baseline is AIF-informed |
| `motion_correct` | `true` | Apply motion correction |
| `motion_thresh` | `0.5` | Motion-QC threshold in mm (float) |
| `smooth_fwhm` | `0.0` | Additional smoothed outputs in mm (float); 0 disables |
| `underlays` | `[]` | Anatomical images to save in DSC space (`t1w_precon`, `t1w_postcon`, `t1w_subtraction`, `flair_precon`, `flair_postcon`, `t2w`) |
| `num_processes` | `0` | Per-pipeline process cap; 0 inherits `--nprocs` |

DCE fields:

| Field | Default | Description |
|---|---:|---|
| `model` | `auto` | BIC selection (`auto`) or a fixed PK model (`patlak`, `tofts`, `extended_tofts`, `2cxm`) |
| `inference` | `nlls` | `nlls` or `vi` |
| `dose` | `1.0` | Contrast-dose multiplier (float; for example, 0.5 for half dose) |
| `r1_relaxivity` | `null` | Contrast-agent r1 override in s⁻¹ mM⁻¹ (float) |
| `hematocrit` | `0.42` | Arterial hematocrit for plasma conversion (float) |
| `aif_method` | `auto` | `auto` or `population` |
| `aif_file` | `null` | External plasma AIF; bypasses detection and scaling |
| `aif_mask_file` | `null` | AIF search-space mask (native-DCE space) |
| `lesion_distance` | `15.0` | Minimum distance from lesion in mm for AIF candidates (float) |
| `n_aif_candidates` | `10` | Maximum averaged candidates; minimum configuration is 3 |
| `aif_peak_floor` | `true` | Raise (only) low detected AIF peaks to the dose-adjusted floor |
| `target_peak` | `null` | Explicit blood-scale AIF peak calibration in mM |
| `aif_bat_alignment` | `auto` | Guarded AIF-to-tissue BAT alignment or `off` |
| `lesion_mask` | `null` | Lesion mask (statistics and AIF lesion exclusion; native-DCE space) |
| `nawm_mask` | `null` | NAWM mask (native-DCE space) |
| `roi_mask` | `null` | Statistics-only ROI mask (native-DCE space) |
| `no_t1` | `false` | Skip VFA fitting and use field-strength default T1 |
| `t1_map_file` | `null` | External T1 map in milliseconds, world-aligned to DCE |
| `t1_method` | `nlls` | WLLS-initialized nonlinear fitting, or faster `wlls` |
| `b1_method` | `auto` | `auto`, `afi`, `precomputed`, or `none` |
| `b1_map_encoding` | `auto` | Infer (`auto`) or specify precomputed B1+ pixel encoding (`actual_flip_deci_deg`, `actual_flip_deg`, `percent`, or `ratio`) |
| `baseline_frames` | `0` | Baseline frames (0 = auto-detect; minimum 5) |
| `motion_correct` | `true` | Apply motion correction |
| `motion_thresh` | `0.5` | Motion-QC threshold in mm (float) |
| `tsnr_threshold` | `0.0` | Optional voxel tSNR floor; 0 disables |
| `ser_early_time_s` | `60.0` | Early SER sample time after the selected global contrast reference (s) |
| `ser_late_time_s` | `300.0` | Late SER sample time after the selected global contrast reference (s) |
| `cbv_constraint_mode` | `auto` | `auto` uses QC-accepted DSC nCBV as a broad Bayesian vp prior (60% relative SD); `skip` disables it. NLLS is always unconstrained |
| `no_dsc_validation` | `false` | Skip DSC–DCE cross-validation |
| `smooth_fwhm` | `0.0` | Additional smoothed outputs in mm (float); 0 disables |
| `underlays` | `[]` | Anatomical images to save in DCE space (`t1w_precon`, `t1w_postcon`, `t1w_subtraction`, `flair_precon`, `flair_postcon`, `t2w`) |
| `num_processes` | `0` | Per-pipeline process cap; 0 inherits `--nprocs` |

Anatomical fields:

| Field | Default | Description |
|---|---:|---|
| `tool` | `hdbet` | `hdbet` or `ants` brain extraction |
| `denoise` | `false` | Apply ANTs denoising |
| `create_subtraction` | `true` | Create post-minus-precontrast T1w difference image (not quantitative) |

Tumor-segmentation fields:

| Field | Default | Description |
|---|---:|---|
| `tool` | `hdglio` | Segmentation tool; currently HD-GLIO only |
| `nawm_erosion` | `2` | NAWM-mask erosion iterations (int) |
| `nawm_min_voxels` | `500` | Minimum accepted NAWM voxels (int) |

## BIDS inputs

A normal run requires `dataset_description.json` and an anatomical T1w image.
Pre-contrast T1w is preferred; post-contrast-only anatomy is supported. DSC and
DCE run when corresponding 4D `*_dsc.nii.gz` and `*_dce.nii.gz` images are
present. HD-GLIO requires pre- and post-contrast T1w, T2w, and
FLAIR.

| Input | Required metadata |
|---|---|
| DSC | `RepetitionTime`, `EchoTime`, `MagneticFieldStrength` |
| DCE | `FlipAngle`, `MagneticFieldStrength`, volume TR, and excitation TR |
| VFA | `RepetitionTime`, `FlipAngle`; at least two distinct angles |
| AFI | `RepetitionTime`, `FlipAngle` |

For 3D DCE, `RepetitionTime` is the volume TR and
`RepetitionTimeExcitation` is the excitation TR. For 2D DCE,
`RepetitionTime` may provide both.

B1+ correction accepts AFI (`*_TB1AFI`), Siemens precomputed flip-angle maps
(`*_TB1TFL`), and vendor-neutral maps (`*_TB1map`). The default
`--dce-b1-map-encoding auto` infers precomputed-map encoding from the suffix,
`Units`, and pixel scale. When inference is ambiguous, set encoding explicitly.
AFI magnitude data are registered automatically. A precomputed map has no
independent magnitude image and must already be world-aligned with the DCE
series. When VFA T1 mapping is used, PMRIPrep applies the inverse of the
accepted VFA-to-DCE transform to transfer that map into VFA space.

When `ContrastBolusAgent` is absent or unrecognized, PMRIPrep uses Gd-DTPA as
the fallback agent: field-strength-specific Gd-DTPA r1 for DCE and a standard
dose of 0.1 mmol Gd/kg for dose-adjusted AIF handling. DSC and DCE dose options
are multipliers of the agent's standard administered dose.

## Outputs and QC

Derivatives are written under `output_dir/pmriprep/`:

```text
pmriprep/
  dataset_description.json
  sub-XX[/ses-YY]/
    anat/      # preprocessed anatomy, masks, segmentations, transforms
    perf/      # DSC/DCE series, maps, AIFs, confounds, statistics
    figures/   # reportlets
    sub-XX[_ses-YY].html
```

Output availability depends on the supplied modalities and configuration:

- Anatomical outputs include preprocessed images, masks, tumor segmentation,
  and transforms. Secondary T1w, T2w, and FLAIR images are prepared only when
  required for tumor segmentation, T1w subtraction or reference handling, or a
  requested DSC/DCE underlay.
- DSC outputs include relative and NAWM-normalized perfusion maps, AUC-CBV,
  MTT, Tmax, PSR, leakage and signal-quality diagnostics, masks, and regional
  statistics.
- DCE outputs include PK, model-selection, model-free, timing, fit-quality,
  QC, mask, and regional-statistics derivatives. Quantitative maps remain
  uncensored; regional statistics retain the native finite-value summaries and
  add explicitly labelled QC-filtered companions when a DCE QC mask is
  available.

Accepted VFA T1 mapping emits a millisecond `T1map` registered to the native
DCE reference grid. Population-T1 fallbacks and external T1 maps are consumed
without being republished. B1+ maps are likewise used for correction but are
not emitted as derivatives.

Applicable NIfTI derivatives have JSON sidecars describing units, conventions,
processing, and provenance. AIF and motion time series are TSV files; regional
statistics are JSON files.

DCE parameter-uncertainty derivatives contain NLLS standard errors or Bayesian
posterior standard deviations conditional on the fitted inputs and model.
DSC parametric CBF uncertainty derivatives use the corresponding conditional
standard error or posterior standard deviation.

Smoothed maps are additional derivatives. DCE kep is recomputed from smoothed
Ktrans and ve; uncertainty, timing, categorical, and fit-quality maps remain at
native resolution. Unsmoothed maps are used for quantitative regional analysis.

The HTML report summarizes anatomy, masks, registration, motion, AIF selection,
T1/B1 processing, leakage correction, model fits, parameter maps, voxel
accounting, and cross-modal checks when available. Quantitative acceptance and
fallback decisions are also recorded in derivative sidecars. Reports support
review; they do not replace acquisition- or study-specific quality criteria.

## License

PMRIPrep is licensed under Apache-2.0. Bundled atlas resources are CC0-1.0 and
documented in [`pmriprep/data/atlases`](pmriprep/data/atlases/README.md). See
[LICENSE](LICENSE) and [NOTICE](NOTICE).

## Citation

Use [`CITATION.cff`](CITATION.cff) or cite:

```bibtex
@software{rulseh2026pmriprep,
  title = {PMRIPrep: automated preprocessing and quantification of DSC and DCE MRI},
  author = {Rulseh, Aaron M. and Rajan, Archith and Curl, Brandon and Mohan, Suyash and Vymazal, Josef and Chawla, Sanjeev},
  year = {2026},
  url = {https://github.com/arulseh/pmriprep}
}
```

## Support

Report problems at <https://github.com/arulseh/pmriprep/issues>. For development
and contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).
