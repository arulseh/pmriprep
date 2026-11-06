# PMRIPrep

**Automated preprocessing and quantification of DSC and DCE MRI**

PMRIPrep is a BIDS-App for preprocessing and quantifying Dynamic Susceptibility
Contrast (DSC) and Dynamic Contrast Enhanced (DCE) MRI. It follows NiPreps
workflow conventions and writes BIDS-Derivatives-style outputs with per-session
HTML QC reports.

PMRIPrep provides:

- DSC deconvolution with sSVD, oSVD, Tikhonov, or CPI;
- DCE pharmacokinetic modelling with Patlak, Tofts, Extended Tofts, or 2CXM;
- Automatic or external arterial input functions;
- VFA T1 mapping with optional B1+ correction;
- Anatomical preprocessing, optional HD-GLIO tumor segmentation, and NAWM
  normalization;
- Provenance-rich derivatives and visual QC.

## Installation

### Containers

Containers provide the supported complete runtime environment and include
TemplateFlow resources.

#### Docker

Install:
```bash
docker pull ghcr.io/arulseh/pmriprep:latest
```

Run:
```bash
docker run --rm --init --gpus 1 \
  -v /path/to/bids:/data:ro \
  -v /path/to/output:/out \
  -v /path/to/work:/work \
  ghcr.io/arulseh/pmriprep:latest \
  /data /out participant --participant-label 01 -w /work
```

Remove `--gpus 1` when running without an NVIDIA GPU. HD-BET falls
back to CPU; HD-GLIO tumor segmentation is skipped when GPU is
unavailable.

#### Apptainer

Install:
```bash
apptainer pull pmriprep.sif docker://ghcr.io/arulseh/pmriprep:latest
```

Run:
```bash
apptainer run --cleanenv --nv \
  --bind /path/to/bids:/data:ro \
  --bind /path/to/output:/out \
  --bind /path/to/work:/work \
  pmriprep.sif \
  /data /out participant --participant-label 01 -w /work
```

Remove `--nv` when running without an NVIDIA GPU. HD-BET falls back to CPU;
HD-GLIO tumor segmentation is skipped when GPU is unavailable.

### Source installation

```bash
python -m pip install .
```

Building from source requires a C compiler and Python development headers.
Source installations require Python 3.11–3.12, FSL, and ANTs. HD-BET and
HD-GLIO are optional: missing HD-BET falls back to ANTs for anatomical images
and to FSL BET for DSC/DCE images, while missing HD-GLIO skips tumor
segmentation.

Runtime depends on image dimensions, time points, enabled pipelines, inference
method, and available hardware.

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
| `--nprocs N` | CPU budget for workflow scheduling and fitting (default: 1) |
| `--omp-nthreads N` | Limit threads per process |
| `--mem-gb GB` | Set the Nipype memory limit |
| `--config-file PATH` | Load options from a YAML or JSON configuration file |
| `--skip-tumor-seg`, `--skip-dsc`, `--skip-dce` | Disable optional pipelines |
| `--report-only` | Rebuild HTML reports from existing derivatives |

Key DSC options:

| Option | Description |
|---|---|
| `--dsc-deconv-method {sSVD,oSVD,tikhonov,cpi}` | Select deconvolution (default: oSVD) |
| `--dsc-inference {vi,svb}` | Select inference for CPI models (default: vi) |
| `--dsc-dose FLOAT` | Set a positive dose multiplier (for example, `0.5` for half dose) |
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
| `--dce-no-t1` | Use the uniform fallback T1 |
| `--dce-t1-map FILE` | Supply a millisecond T1 map world-aligned to DCE |

Use `pmriprep --help` for the complete execution interface. `--use-plugin`
accepts a Nipype plugin YAML/JSON for cluster schedulers. `--clean-workdir`,
`--resource-monitor`, `--write-graph`, `--stop-on-first-crash`, and `--sloppy`
follow the corresponding NiPreps conventions.

CPI fitting uses single-threaded workers up to the process and node-memory
limits; `--sloppy` reduces its fitting iterations for testing. HD-BET uses
test-time augmentation on CUDA and MPS; CPU and sloppy execution
disable it for speed.

Automatic DCE selection compares Patlak and extended Tofts; 2CXM is eligible
at volume intervals ≤2 s with additional evidence checks. Explicit 2CXM is
available at any interval, but coarse sampling limits parameter identification.
VI refines least-squares parameter estimates with the model and delay held fixed.

PMRIPrep automatically creates lesion and normal-appearing white-matter (NAWM)
masks when tumor segmentation runs with the required anatomical inputs. Supplying
masks through the configuration is optional.

## Configuration

The file schema has four optional top-level mappings: `dsc`, `dce`, `anat`, and
`tumor_seg`. Options within each section are flat. CLI options override their
file equivalents. Example configuration:

```yaml
dsc:
  dose: 0.75
  method: cpi
  lesion_mask: /path/to/lesion_mask.nii.gz
  smooth_fwhm: 3.5
  underlays: [t1w_postcon, flair_precon]

dce:
  model: 2cxm
  inference: vi
  t1_default_ms: 1400
  smooth_fwhm: 3

anat:
  create_subtraction: true

tumor_seg:
  nawm_erosion: 3
```

DSC fields:

| Field | Default | Description |
|---|---:|---|
| `method` | `oSVD` | `sSVD`, `oSVD`, `tikhonov`, `cpi` |
| `inference` | `vi` | `vi` or `svb` for CPI |
| `preload` | `unknown` | Administered preload dose fraction, or `unknown`; provenance only |
| `dose` | `1.0` | Contrast-dose multiplier (float; for example, 0.5 for half dose) |
| `agent_standard_dose` | `0.1` | Agent standard dose in mmol Gd/kg (float) |
| `pre_bolus` | `5` | Initial baseline frames; final baseline may be refined |
| `motion_correct` | `true` | Apply motion correction; false estimates motion for QC without resampling |
| `motion_thresh` | `0.5` | Motion-QC threshold in mm (float) |
| `aif_file` | `null` | External ΔR2* AIF; bypasses detection |
| `aif_mask_file` | `null` | AIF search mask (native-DSC space). If exhausted, AIF-independent maps only |
| `lesion_distance` | `15.0` | Minimum distance from lesion in mm for AIF candidates (float) |
| `n_aif_candidates` | `10` | Maximum averaged candidates; minimum configuration is 3 |
| `leakage_method` | `bsw` | BSW correction (or estimation for CPI); `none` disables leakage correction/estimation |
| `cpi_control_points` | `0` | Number of CPI residue control points (0 = automatic; otherwise 5–12) |
| `cpi_infer_delay` | `true` | Fit voxelwise arrival delay; `false` fixes delay at zero |
| `cpi_dispersion` | `none` | `none` omits arterial-dispersion fitting; `auto` or `gamma` enables (substantially slower) |
| `cpi_fallback` | `osvd` | Replace rejected CPI fits with oSVD; `none` leaves those measurements missing |
| `nawm_mask` | `null` | Optional user-supplied NAWM mask (native-DSC space) |
| `lesion_mask` | `null` | Optional user-supplied lesion mask (statistics and AIF lesion exclusion; native-DSC space) |
| `roi_mask` | `null` | Statistics-only ROI mask (native-DSC space); binary or integer labels |
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
| `baseline_frames` | `0` | Baseline frames (0 = auto-detect; minimum 5) |
| `motion_correct` | `true` | Apply motion correction; false estimates motion for QC without resampling |
| `motion_thresh` | `0.5` | Motion-QC threshold in mm (float) |
| `b1_method` | `auto` | `auto`, `afi`, `precomputed`, or `none` |
| `b1_map_encoding` | `auto` | Infer (`auto`) or specify precomputed B1+ pixel encoding (`actual_flip_deci_deg`, `actual_flip_deg`, `percent`, or `ratio`) |
| `t1_map_file` | `null` | External T1 map in milliseconds, world-aligned to DCE |
| `no_t1` | `false` | Skip VFA fitting and use the uniform fallback T1 |
| `t1_default_ms` | `null` | Uniform fallback T1 in ms; null uses field defaults. Usable measured/external maps take precedence |
| `t1_method` | `nlls` | WLLS-initialized nonlinear fitting, or faster `wlls` |
| `aif_file` | `null` | External plasma AIF; bypasses detection and scaling |
| `aif_method` | `auto` | `auto` or `population` |
| `aif_mask_file` | `null` | AIF search mask (native-DCE space). If exhausted, the population AIF is used |
| `lesion_distance` | `15.0` | Minimum distance from lesion in mm for AIF candidates (float) |
| `n_aif_candidates` | `10` | Maximum averaged candidates; minimum configuration is 3 |
| `aif_peak_floor` | `true` | Raise (only) low detected AIF peaks to the dose-adjusted floor |
| `target_peak` | `null` | Explicit blood-scale AIF peak calibration in mM |
| `hematocrit` | `0.42` | Arterial hematocrit for plasma conversion (float) |
| `delay_correction` | `true` | Local correction of the effective AIF–tissue model offset; `false` fixes delay at zero |
| `tsnr_threshold` | `0.0` | Optional voxel tSNR floor; 0 disables |
| `cbv_constraint_mode` | `auto` | `auto` uses QC-accepted DSC nCBV as a broad vp prior for VI; `skip` disables it. NLLS is always unconstrained |
| `no_dsc_validation` | `false` | Skip descriptive DSC–DCE vascular concordance measurement |
| `ser_early_time_s` | `60.0` | Early SER sample time after the selected global contrast reference (s) |
| `ser_late_time_s` | `300.0` | Late SER sample time after the selected global contrast reference (s) |
| `nawm_mask` | `null` | Optional user-supplied NAWM mask (native-DCE space) |
| `lesion_mask` | `null` | Optional user-supplied lesion mask (statistics and AIF lesion exclusion; native-DCE space) |
| `roi_mask` | `null` | Statistics-only ROI mask (native-DCE space); binary or integer labels |
| `smooth_fwhm` | `0.0` | Additional smoothed outputs in mm (float); 0 disables |
| `underlays` | `[]` | Anatomical images to save in DCE space (`t1w_precon`, `t1w_postcon`, `t1w_subtraction`, `flair_precon`, `flair_postcon`, `t2w`) |
| `num_processes` | `0` | Per-pipeline process cap; 0 inherits `--nprocs` |

Anatomical fields:

| Field | Default | Description |
|---|---:|---|
| `tool` | `hdbet` | `hdbet` or `ants` brain extraction |
| `denoise` | `false` | Apply ANTs denoising |
| `create_subtraction` | `false` | Create post-minus-precontrast T1w difference image from a sequence-compatible pair; implied when a DSC or DCE `t1w_subtraction` underlay is requested |

Tumor-segmentation fields:

| Field | Default | Description |
|---|---:|---|
| `tool` | `hdglio` | Segmentation tool; currently HD-GLIO only |
| `nawm_erosion` | `2` | NAWM-mask erosion iterations (int) |
| `nawm_min_voxels` | `500` | Minimum accepted NAWM voxels (int) |

## BIDS inputs

A normal run requires `dataset_description.json` and an anatomical T1w image.
Post-contrast-only T1w anatomy is supported, but downstream options requiring a
pre-contrast T1w reference are unavailable. DSC and DCE run when corresponding
4D `*_dsc.nii.gz` and `*_dce.nii.gz` images are present. HD-GLIO requires pre-
and post-contrast T1w, T2w, and FLAIR.

DSC requires a single-echo gradient-echo time series; DCE requires a single-echo
spoiled gradient-echo time series. Each enabled modality requires exactly one
candidate per subject/session in the input dataset. Multiple candidates must be
resolved in the input dataset before processing; `--skip-dsc` and `--skip-dce`
bypass validation of the skipped modality. A supplied `EchoTime` must be a finite
scalar in seconds; it is required for DSC. Multi-echo and combined gradient- and
spin-echo acquisitions are currently unsupported for both DSC and DCE.

| Input | Required metadata |
|---|---|
| DSC | `RepetitionTime`, `EchoTime`, `MagneticFieldStrength` |
| DCE | `FlipAngle`, `MagneticFieldStrength`, volume TR, and excitation TR |
| VFA | `RepetitionTime`, `FlipAngle` (at least two distinct angles) |
| AFI | `RepetitionTime`, `FlipAngle` |

For 3D-excitation DCE (MRAcquisitionType = 3D), RepetitionTime is
the volume TR and RepetitionTimeExcitation is the excitation TR
used in the SPGR signal equation. For 2D-excitation DCE
(MRAcquisitionType = 2D), supply RepetitionTimeExcitation as well,
unless each slice is excited once per volume (single-shot, e.g. EPI),
where RepetitionTime serves both. Supply the correct excitation TR explicitly
whenever it differs from the volume TR.

B1+ correction accepts AFI (`*_TB1AFI`), Siemens precomputed flip-angle maps
(`*_TB1TFL`), and vendor-neutral maps (`*_TB1map`). The default
`--dce-b1-map-encoding auto` infers precomputed-map encoding from the suffix,
`Units`, and pixel scale. When inference is ambiguous, set encoding explicitly.
AFI magnitude data are registered automatically. A precomputed map has no
independent magnitude image and must already be world-aligned with the DCE
series. When VFA T1 mapping is used, PMRIPrep applies the inverse of the
accepted VFA-to-DCE transform to transfer that map into VFA space.

DCE resolves r1 from `ContrastBolusAgent` and `MagneticFieldStrength`, using
the 1.5 T entry for fields ≤2.25 T and the 3 T entry for fields >2.25 T through
5 T. Fields >5 T, or a missing/unrecognized agent, require an explicit finite
positive dce `r1_relaxivity` value in the configuration file (or `--dce-r1`), in
s⁻¹ mM⁻¹ per Gd. DSC retains a standard-dose fallback of 0.1 mmol Gd/kg with
a warning when the agent is unresolved. DSC and DCE dose options are
multipliers of the agent's standard administered dose in mmol Gd/kg.

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
  MTT, SVD-family Tmax or parametric AIF-relative arrival delay, PSR, leakage
  and signal-quality diagnostics, masks, and regional
  statistics.
- DCE outputs include PK, model-selection, model-free, timing, and fit-quality
  derivatives, a binary QC mask for fit quality and parameter identification, and
  regional statistics. Quantitative maps are not filtered. Statistics are reported
  for all finite voxels and, when a QC mask is available, separately for voxels
  that pass QC.

Accepted VFA T1 mapping emits a millisecond `T1map` registered to the native
DCE reference grid. Population-T1 fallbacks and external T1 maps are consumed
without being republished. B1+ maps are likewise used for correction but are
not emitted as derivatives.

Applicable NIfTI derivatives have JSON sidecars describing units, conventions,
processing, and provenance. AIF and motion time series are TSV files; regional
statistics are JSON files. When a lesion mask is available, detected-AIF
provenance reports tumor laterality and the final candidate counts relative to
a unilateral lesion; these fields do not affect candidate selection.
DSC regional statistics distinguish residue-integral CBV (`cbv`) from
AUC-CBV (`cbv_auc`); each map's sidecar reports its own regional values.
Normalized CBV (`ncbv`) uses post-baseline tissue concentration integrals.
DSC uses measured AIFs for deconvolution. If no acceptable AIF is found,
tissue-based nCBV, PSR, TTP, CNR and tSNR remain available, together with K2
when BSW correction is applied; AIF-dependent perfusion maps are withheld.

DCE parameter-uncertainty derivatives contain NLLS standard errors or Bayesian
posterior standard deviations conditional on the fitted inputs and model.
CPI CBF uncertainty derivatives contain posterior standard deviations
conditional on the fitted inputs and model.

Smoothed maps are additional derivatives. DCE kep is recomputed from smoothed
Ktrans and ve; uncertainty, timing, categorical, and fit-quality maps remain at
native resolution. Unsmoothed maps are used for quantitative regional analysis.

DSC/CPI and DCE fitting use equal temporal weights; motion estimates are used for QC,
not as a measure of residual motion after correction.
DCE also outputs the selected model delay in seconds at native resolution.
This AIF-relative fitted offset is not a measure of vascular transit time.

The HTML report summarizes anatomy, masks, registration, motion, AIF selection,
T1/B1 processing, leakage correction, model fits, parameter maps, voxel
accounting, and cross-modal checks when available. Quantitative acceptance and
fallback decisions are also recorded in derivative sidecars. Reports support
review; they do not replace acquisition- or study-specific quality criteria.

## License

PMRIPrep is licensed under Apache-2.0. Bundled atlas resources are CC0-1.0 and
documented in [`pmriprep/data/atlases`](pmriprep/data/atlases/README.md). See
[LICENSE](LICENSE) and [NOTICE](NOTICE). The native DCE solver includes
[SciPy-derived code under BSD-3-Clause](pmriprep/models/dce/NOTICE_SCIPY.md).

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
