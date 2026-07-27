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

Python 3.10–3.12, FSL, and ANTs are required. The container is the supported
way to obtain the complete runtime environment.

### System Requirements

Resource needs vary substantially with image size, inference method, and
enabled pipelines. Run a representative subject before sizing a cohort job.
Keep the work directory until outputs have been checked; use a fresh work
directory when changing versions or scientific configuration if cache reuse is
not intended.

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
| `--config-file PATH` | Load advanced scientific options from YAML or JSON |
| `--skip-tumor-seg`, `--skip-dsc`, `--skip-dce` | Disable optional pipelines |
| `--report-only` | Rebuild HTML reports from existing derivatives |

Key DSC options:

| Option | Description |
|---|---|
| `--dsc-deconv-method {sSVD,oSVD,tikhonov,vascular,cpi}` | Select deconvolution (default: oSVD) |
| `--dsc-inference {vi,svb,nlls}` | Select inference for CPI/vascular models (default: vi) |
| `--dsc-preload {auto\|FLOAT}` | Auto-classify preload adequacy or provide a dose multiplier |
| `--dsc-dose DOSE` | Set dose multiplier (`half`, `full`, `double`, or float) |
| `--dsc-aif-method {auto,synthetic}` | Select detected or synthetic AIF |
| `--dsc-aif-file FILE` | Supply an external ΔR2* AIF in s⁻¹ |

Key DCE options:

| Option | Description |
|---|---|
| `--dce-model {auto,patlak,tofts,extended_tofts,2cxm}` | Select PK model; `auto` uses voxelwise BIC |
| `--dce-inference {nlls,vi}` | Select inference (default: nlls) |
| `--dce-aif-method {auto,population}` | Select detected or population AIF |
| `--dce-aif-file FILE` | Supply a plasma-concentration AIF in mM |
| `--dce-dose DOSE` | Set dose multiplier (`half`, `full`, `double`, or float) |
| `--dce-b1-method {auto,afi,precomputed,none}` | Control B1+ correction |
| `--dce-no-t1` | Use the field-strength default tissue T1 |
| `--dce-t1-map FILE` | Supply an external T1 map in milliseconds |

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
  t1_method: hybrid
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
| `preload` | `auto` | K2-based classification (auto) or administered-dose multiplier (float) |
| `dose` | `1.0` | Contrast-dose multiplier (float) |
| `agent_standard_dose` | `0.1` | Agent standard dose in mmol Gd/kg (float) |
| `aif_method` | `auto` | `auto` or `synthetic` |
| `aif_file` | `null` | External ΔR2* AIF; bypasses detection |
| `aif_mask_file` | `null` | DSC-space AIF search mask |
| `lesion_distance` | `15.0` | Minimum lesion distance in mm for AIF candidates (float) |
| `n_aif_candidates` | `10` | Maximum averaged candidates; minimum configuration is 3 |
| `disable_aif_fallback` | `false` | Fail or retain a flagged AIF instead of substituting synthetic |
| `lesion_mask` | `null` | Native-DSC lesion mask (statistics and AIF lesion exclusion) |
| `nawm_mask` | `null` | Native-DSC NAWM mask |
| `roi_mask` | `null` | Native-DSC statistics-only ROI mask |
| `leakage_method` | `bsw` | `bsw` or `bidirectional` |
| `kep` | `null` | Bidirectional-leakage kep override in s⁻¹ |
| `pre_bolus` | `5` | Initial baseline frames; final baseline is AIF-informed |
| `first_pass_cbv` | `false` | Add first-pass AUC-rCBV |
| `motion_correct` | `true` | Apply motion correction |
| `motion_thresh` | `0.5` | Motion-QC threshold in mm (float) |
| `smooth_fwhm` | `0.0` | Additional smoothed outputs in mm; 0 (float) disables |
| `underlays` | `[]` | Anatomical images to save in DSC space |
| `num_processes` | `0` | Per-pipeline process cap; 0 inherits `--nprocs` |

DCE fields:

| Field | Default | Description |
|---|---:|---|
| `model` | `auto` | BIC selection or a fixed PK model (`patlak`, `tofts`, `extended_tofts`, `2cxm`) |
| `inference` | `nlls` | `nlls` or `vi` |
| `dose` | `1.0` | Contrast-dose multiplier (float) |
| `aif_method` | `auto` | `auto` or `population` |
| `aif_peak_floor` | `true` | Raise (only) low detected AIF peaks to the dose-adjusted floor |
| `target_peak` | `null` | Explicit blood-scale AIF peak calibration in mM |
| `aif_file` | `null` | External plasma AIF; bypasses detection and scaling |
| `aif_mask_file` | `null` | DCE-space AIF search mask |
| `lesion_distance` | `15.0` | Minimum lesion distance in mm for AIF candidates (float) |
| `n_aif_candidates` | `10` | Maximum averaged candidates; minimum configuration is 3 |
| `aif_bat_alignment` | `auto` | Guarded AIF-to-tissue BAT alignment or `off` |
| `lesion_mask` | `null` | Native-DCE lesion mask (statistics and AIF lesion exclusion) |
| `nawm_mask` | `null` | Native-DCE NAWM mask |
| `roi_mask` | `null` | Native-DCE statistics-only ROI mask |
| `t1_method` | `hybrid` | VFA fitting with `hybrid`, `wlls`, or `nlls` |
| `t1_correction_mode` | `auto` | Retain plausible measured T1 support; `fallback` forces uniform T1 |
| `t1_map_file` | `null` | External T1 map in milliseconds |
| `no_t1` | `false` | Skip VFA fitting and use field-strength default T1 |
| `b1_method` | `auto` | `auto`, `afi`, `precomputed`, or `none` |
| `b1_map_encoding` | `auto` | Infer or specify precomputed B1+ pixel encoding |
| `baseline_frames` | `0` | Baseline frames (0 = auto-detect; minimum 5) |
| `motion_correct` | `true` | Apply motion correction |
| `motion_thresh` | `0.5` | Motion-QC threshold in mm (float) |
| `tsnr_threshold` | `0.0` | Optional voxel tSNR floor; 0 disables |
| `r1_relaxivity` | `null` | Contrast-agent r1 override in s⁻¹ mM⁻¹ (float) |
| `cbv_constraint_mode` | `auto` | Use reliable DSC CBV as a soft vp prior, or `skip` |
| `hematocrit` | `0.42` | Arterial hematocrit for plasma conversion (float) |
| `no_dsc_validation` | `false` | Skip DSC–DCE cross-validation |
| `smooth_fwhm` | `0.0` | Additional smoothed outputs in mm (float); 0 disables |
| `underlays` | `[]` | Anatomical underlays used in QC |
| `num_processes` | `0` | Per-pipeline process cap; 0 inherits `--nprocs` |

Anatomical fields:

| Field | Default | Description |
|---|---:|---|
| `tool` | `hdbet` | `hdbet` or `ants` brain extraction |
| `denoise` | `false` | Apply ANTs denoising |
| `create_subtraction` | `true` | Create post-minus-precontrast T1w subtraction |

Tumor-segmentation fields:

| Field | Default | Description |
|---|---:|---|
| `tool` | `hdglio` | Segmentation tool; currently HD-GLIO only |
| `nawm_erosion` | `2` | NAWM-mask erosion iterations (int) |
| `nawm_min_voxels` | `500` | Minimum accepted NAWM voxels (int) |

Masks (`lesion_mask`, `nawm_mask`, `roi_mask`) must already be in the native
perfusion space. `underlays` accepts `t1w_precon`, `t1w_postcon`,
`t1w_subtraction`, `flair_precon`, `flair_postcon`, and `t2w`. Unknown sections
or fields fail before workflow construction.

## BIDS inputs

A normal run requires `dataset_description.json` and an anatomical T1w image.
Pre-contrast T1w is preferred; post-contrast-only anatomy is supported. DSC and
DCE run when corresponding 4D `*_dsc.nii.gz` and `*_dce.nii.gz` images are
present. HD-GLIO additionally requires pre- and post-contrast T1w, T2w, and
FLAIR.

| Input | Required metadata |
|---|---|
| DSC | `RepetitionTime`, `EchoTime`, `MagneticFieldStrength` |
| DCE | `FlipAngle`, `MagneticFieldStrength`, volume TR, and excitation TR |
| VFA | `RepetitionTime`, `FlipAngle`; at least two distinct angles |
| AFI | `RepetitionTime`, `FlipAngle` |

For 3D DCE, use `RepetitionTime` for volume TR and
`RepetitionTimeExcitation` for pulse-sequence TR. For 2D DCE,
`RepetitionTime` may serve both roles.

B1+ correction recognizes AFI (`*_TB1AFI`), Siemens precomputed flip-angle
maps (`*_TB1TFL`), and vendor-neutral maps (`*_TB1map`). Encoding is inferred
from suffix, `Units`, and pixel scale; use `--dce-b1-map-encoding` when metadata
are ambiguous.

`ContrastBolusAgent` is recommended for relaxivity and AIF threshold lookup.
When absent, PMRIPrep records and warns about its assumptions. DSC and DCE dose
options are multipliers of the agent's standard administered dose.

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

DSC outputs include relative CBF/CBV, AUC-CBV, MTT, Tmax, leakage, timing, and
signal-quality maps. DCE outputs include PK, model-selection, model-free,
timing, fit-quality, and QC maps. Applicable NIfTI derivatives have JSON
sidecars describing units, conventions, processing, and provenance; AIF and
motion time series are TSV files.

Smoothed maps are additional derivatives. DCE Kep is recomputed from smoothed
Ktrans and ve; uncertainty, timing, categorical, and fit-quality maps remain at
native resolution. Native maps are used for quantitative ROI analysis.

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

Report problems at <https://github.com/arulseh/pmriprep/issues>.
