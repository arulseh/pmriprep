# PMRIPrep

**Perfusion MRI Preprocessing Pipeline**

PMRIPrep is a BIDS-App for processing Dynamic Susceptibility Contrast (DSC) and Dynamic Contrast Enhanced (DCE) data. It follows the [NiPreps](https://www.nipreps.org/) framework conventions and produces BIDS-Derivatives compliant outputs with HTML QC reports.

## Features

- **DSC Processing**: Five deconvolution methods (sSVD, oSVD, Tikhonov, CPI, Vascular) and K2 leakage correction
- **DCE Processing**: Four pharmacokinetic models (Tofts, Extended Tofts, Patlak, 2CXM) plus model-free maps
- **Bayesian Inference**: Variational inference (VI), spatial variational Bayes (SVB), and maximum a posteriori (MAP) estimation with uncertainty quantification
- **Automatic AIF Detection**: Multi-level arterial input function detection with population fallback, or user-supplied external AIF
- **T1 Mapping**: Variable flip angle T1 mapping with optional B1+ correction
- **Tumor Segmentation**: HD-GLIO integration for glioma segmentation
- **NAWM Normalization**: Automatic NAWM ROI derivation for relative perfusion metrics
- **QC Reports**: Per-session HTML reports with AIF quality, motion, registration, and parameter map visualizations

## Installation

### Docker (Recommended)

```bash
docker pull ghcr.io/arulseh/pmriprep:latest
```

Run with Docker:

```bash
docker run --rm \
    -v /path/to/bids:/data:ro \
    -v /path/to/output:/out \
    -v /path/to/work:/work \
    ghcr.io/arulseh/pmriprep:latest \
    /data /out participant \
    --participant-label 01 \
    -w /work
```

For GPU support (required for tumor segmentation):

```bash
docker run --gpus all --rm \
    -v /path/to/bids:/data:ro \
    -v /path/to/output:/out \
    -v /path/to/work:/work \
    ghcr.io/arulseh/pmriprep:latest \
    /data /out participant \
    --participant-label 01 \
    -w /work
```

> **Note:** Tumor segmentation (HD-GLIO) requires an NVIDIA GPU. When no GPU is detected, tumor segmentation is automatically skipped and the remaining pipelines proceed normally. Use `--skip-tumor-seg` to suppress the warning.

### Apptainer

```bash
apptainer pull pmriprep.sif docker://ghcr.io/arulseh/pmriprep:latest
```

Run with Apptainer:

```bash
apptainer run --cleanenv \
    --bind /path/to/bids:/data:ro \
    --bind /path/to/output:/out \
    --bind /path/to/work:/work \
    pmriprep.sif \
    /data /out participant \
    --participant-label 01 \
    -w /work
```

For GPU support (required for tumor segmentation), add `--nv`:

```bash
apptainer run --cleanenv --nv \
    --bind /path/to/bids:/data:ro \
    --bind /path/to/output:/out \
    --bind /path/to/work:/work \
    pmriprep.sif \
    /data /out participant \
    --participant-label 01 \
    -w /work
```

TemplateFlow templates are pre-populated at `/opt/templateflow` inside the image — no extra bind mounts needed.

### pip (Development)

```bash
pip install pmriprep
```

Requires FSL and ANTs on `$PATH`. See the Dockerfile for the full list of system dependencies.

## Quick Start

```bash
pmriprep /path/to/bids /path/to/output participant \
    --participant-label 01 02 03
```

## CLI Reference

```
pmriprep bids_dir output_dir {participant} [options]
```

### Core Options

| Flag | Description |
|------|-------------|
| `--participant-label LABEL [...]` | Participant(s) to process (without `sub-` prefix) |
| `--session-label LABEL [...]` | Session(s) to process (without `ses-` prefix) |
| `--bids-validation` | Run the `bids-validator` pre-flight check (off by default; perfusion suffixes are not yet in the stable BIDS spec) |
| `--nprocs N` | Maximum number of parallel processes (default: 1) |
| `--omp-nthreads N` | Per-process thread limit (OMP/MKL/ITK) |
| `--mem-gb GB` | Upper bound memory limit in GB |
| `-w PATH`, `--work-dir PATH` | Working directory for intermediate files |
| `--config-file PATH` | YAML or JSON configuration file for advanced options |
| `-v`, `--verbose` | Increase log verbosity (repeat for more: `-v`, `-vv`, `-vvv`) |

### Pipeline Skip Flags

| Flag | Description |
|------|-------------|
| `--skip-anat` | Skip anatomical preprocessing |
| `--skip-tumor-seg` | Skip tumor segmentation |
| `--skip-dsc` | Skip DSC perfusion pipeline |
| `--skip-dce` | Skip DCE perfusion pipeline |

### DSC Options

| Flag | Description |
|------|-------------|
| `--dsc-deconv-method {sSVD,oSVD,tikhonov,vascular,cpi}` | Deconvolution method (default: oSVD) |
| `--dsc-inference {nlls,map,vi,svb}` | Inference engine for parametric methods (cpi, vascular). Default: `vi` (FABBER variational Bayes). SVD/Tikhonov ignore. |
| `--dsc-preload {auto\|FLOAT}` | Preload: `auto` (detect from signal) or float dose multiplier (0.0–2.0, e.g. `0.5` for half-dose preload) |
| `--dsc-dose DOSE` | Dose multiplier (default: 1.0). Accepts a float or presets: `half`, `full`, `double` |
| `--dsc-agent-standard-dose DOSE` | Override agent standard dose in mmol Gd/kg (auto-detected from `ContrastBolusAgent` sidecar field when available). Default: `0.1` |
| `--dsc-aif-method {auto,synthetic}` | AIF detection method (default: auto) |
| `--dsc-aif-file FILE` | External AIF curve file (ΔR2* in s⁻¹); bypasses detection |

### DCE Options

| Flag | Description |
|------|-------------|
| `--dce-model {auto,tofts,extended_tofts,patlak,2cxm}` | Pharmacokinetic model. `auto` performs per-voxel BIC model selection from Patlak, Extended Tofts, and 2CXM (default: auto) |
| `--dce-inference {nlls,vi,map}` | Inference engine for DCE: `nlls` (default, scipy least_squares), `vi`, `map` |
| `--dce-aif-method {auto,population}` | AIF detection method (default: auto) |
| `--dce-aif-file FILE` | External AIF curve file (true plasma [Gd] in mM, already Hct-corrected); bypasses detection — the pipeline does not re-scale |
| `--dce-dose DOSE` | Dose multiplier (default: 1.0). Accepts a float or presets: `half`, `full`, `double` |
| `--dce-r1 R1` | Override r1 relaxivity (s⁻¹/mM) for concentration conversion |
| `--dce-b1-method {auto,afi,tfl_b1map,none}` | B1+ correction method (default: auto) |
| `--dce-no-t1` | Skip T1 mapping (use default T1 for field strength) |
| `--dce-t1-map FILE` | External T1 map (NIfTI, units: milliseconds). Bypasses VFA T1 fitting. Useful for MP2RAGE, MOLLI, or ground-truth T1 maps. |

### Anatomical Options

| Flag | Description |
|------|-------------|
| `--brain-extraction-tool {hdbet,ants}` | Brain extraction tool (default: hdbet) |
| `--denoise` | Apply ANTs denoising to anatomical images |

### Other Options

| Flag | Description |
|------|-------------|
| `--report-only` | Regenerate HTML QC reports from existing outputs |
| `--sloppy` | Use fast, low-quality settings for testing (not for production) |
| `--random-seed SEED` | Initialize random seed for reproducibility |
| `--stop-on-first-crash` | Stop on first crash |
| `--write-graph` | Write workflow graph |
| `--resource-monitor` | Enable Nipype resource monitoring |
| `--clean-workdir` | Delete working directory after processing |

## Configuration File

For options not exposed on the CLI (mask paths, AIF tuning, smoothing, etc.), use a YAML configuration file:

```bash
pmriprep /path/to/bids /path/to/output participant \
    --config-file config.yaml
```

### Example config.yaml

```yaml
dsc:
  method: cpi
  inference: svb
  preload: 0.75
  dose: 0.75
  lesion_mask: /path/to/lesion_mask.nii.gz
  nawm_mask: /path/to/nawm_mask.nii.gz
  underlays: [t1w_postcon, flair]

dce:
  model: extended_tofts
  inference: vi
  dose: 0.75
  lesion_mask: /path/to/lesion_mask.nii.gz
  nawm_mask: /path/to/nawm_mask.nii.gz
  underlays: [t1w_postcon, flair]
```

### DSC Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | `"oSVD"` | Deconvolution method: `sSVD`, `oSVD`, `tikhonov`, `vascular`, `cpi` |
| `inference` | str | `"vi"` | Inference engine: `nlls`, `map`, `vi`, `svb` (CPI and Vascular only; SVD and Tikhonov ignore) |
| `cpi_control_points` | int | `0` | Control points for CPI (0 = auto) |
| `preload` | str or float | `"auto"` | Preload dose fraction: `"auto"` (detect from signal) or a float (0.0–2.0) |
| `dose` | float | `1.0` | Dose multiplier (default: 1.0). Accepts a float or presets: `half`, `full`, `double` |
| `agent_standard_dose` | float | `0.1` | Override agent standard dose in mmol Gd/kg (auto-detected from `ContrastBolusAgent` sidecar field when available) |
| `aif_method` | str | `"auto"` | AIF method: `auto`, `synthetic` |
| `aif_file` | path | `null` | External AIF curve (ΔR2* in s⁻¹). Bypasses AIF detection |
| `aif_mask_file` | path | `null` | Custom AIF mask file (search constraint; DSC space) |
| `n_aif_candidates` | int | `10` | Number of AIF candidate voxels to average |
| `lesion_distance` | float | `15.0` | Minimum distance (mm) from lesion for AIF candidates |
| `disable_aif_fallback` | bool | `false` | Disable synthetic AIF fallback |
| `leakage_method` | str | `"bsw"` | K2 leakage correction: `bsw` (standard), `bidirectional` |
| `kep` | float | `0.005` | EES efflux rate for bidirectional correction (s⁻¹); ignored for bsw |
| `pre_bolus` | int | `5` | Initial baseline for motion correction. Baseline is re-estimated after AIF detection |
| `motion_thresh` | float | `0.5` | Framewise displacement threshold in mm |
| `lesion_mask` | path | `null` | Lesion mask for AIF exclusion and lesion statistics (DSC space) |
| `nawm_mask` | path | `null` | NAWM mask for normalization (DSC space) |
| `roi_mask` | path | `null` | ROI mask for statistics extraction. Multi-label integer masks produce per-label stats (DSC space) |
| `smooth_fwhm` | float | `0.0` | Smoothing kernel FWHM in mm (0 = none); additional outputs |
| `underlays` | list | `[]` | Structural images to save in perfusion space: `t1w_precon`, `t1w_postcon`, `flair`, `t2w` |

### DCE Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `"auto"` | PK model. `auto` = per-voxel BIC selection (Patlak, Extended Tofts, 2CXM). Fixed: `tofts`, `extended_tofts`, `patlak`, `2cxm` |
| `inference` | str | `"nlls"` | Inference engine: `nlls`, `vi`, `map` (and `mcmc` via YAML only) |
| `dose` | float | `1.0` | Dose multiplier (default: 1.0). Accepts a float or presets: `half`, `full`, `double` |
| `r1_relaxivity` | float | `null` | Override r1 relaxivity (s⁻¹/mM); null = lookup from BIDS sidecar |
| `hematocrit` | float | `0.42` | Arterial hematocrit for AIF plasma correction |
| `aif_method` | str | `"auto"` | AIF method: `auto`, `population` |
| `scale_aif` | bool | `true` | Scale AIF to population reference |
| `target_peak` | float | `null` | Target AIF peak (mM); null = field-strength default |
| `aif_file` | path | `null` | External AIF file (plasma [Gd] in mM, Hct-corrected). Bypasses AIF detection |
| `aif_mask_file` | path | `null` | Custom AIF mask file (search constraint; DCE space) |
| `n_aif_candidates` | int | `10` | Number of AIF candidate voxels to average |
| `lesion_distance` | float | `15.0` | Minimum distance (mm) from lesion for AIF candidates |
| `no_t1` | bool | `false` | Skip T1 mapping |
| `t1_method` | str | `"hybrid"` | VFA T1 fitting: `hybrid`, `wlls`, `nlls` |
| `t1_map_file` | path | `null` | External T1 map (NIfTI, ms). Bypasses VFA fitting |
| `t1_correction_mode` | str | `"auto"` | T1 correction mode: `auto`, `none`, `offset`, `fallback` |
| `b1_method` | str | `"auto"` | B1+ correction: `auto`, `afi`, `tfl_b1map`, `none` |
| `cbv_constraint_mode` | str | `"auto"` | DSC→DCE vp constraint. `auto`: `soft` prior when DSC CBV is available, `skip` otherwise. Options: `auto`, `soft`, `skip` |
| `no_dsc_validation` | bool | `false` | Skip DSC-DCE cross-validation. Requires a NAWM mask; automatically skipped when unavailable |
| `baseline_frames` | int | `0` | Baseline frames (0 = auto-detect) |
| `tsnr_threshold` | float | `0.0` | tSNR threshold for voxel inclusion (0 = none) |
| `lesion_mask` | path | `null` | Lesion mask for AIF exclusion and lesion statistics (DCE space) |
| `nawm_mask` | path | `null` | NAWM mask for normalization (DCE space) |
| `roi_mask` | path | `null` | ROI mask for statistics extraction. Multi-label integer masks produce per-label stats (DCE space) |
| `smooth_fwhm` | float | `0.0` | Smoothing kernel FWHM in mm (0 = none); additional outputs |
| `underlays` | list | `[]` | Structural images to save in perfusion space: `t1w_precon`, `t1w_postcon`, `flair`, `t2w` |

### Anatomical Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tool` | str | `"hdbet"` | Brain extraction tool: `hdbet`, `ants` |
| `denoise` | bool | `false` | Apply ANTs denoising |
| `create_subtraction` | bool | `true` | Create post-pre contrast subtraction image |

### Tumor Segmentation Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tool` | str | `"hdglio"` | Segmentation tool (currently only `hdglio`) |
| `nawm_erosion` | int | `2` | Erosion iterations for NAWM mask (voxels) |
| `nawm_min_voxels` | int | `500` | Minimum voxels required in NAWM ROI |

## BIDS Input Requirements

PMRIPrep expects input data in BIDS format:

```
bids_dir/
  dataset_description.json
  sub-01/
    ses-01/
      anat/
        sub-01_ses-01_run-1_T1w.nii.gz                       # Pre-contrast T1w
        sub-01_ses-01_run-1_T1w.json
        sub-01_ses-01_ce-GAD_run-1_T1w.nii.gz                # Post-contrast T1w
        sub-01_ses-01_ce-GAD_run-1_T1w.json
        sub-01_ses-01_run-1_FLAIR.nii.gz
        sub-01_ses-01_run-1_FLAIR.json
        sub-01_ses-01_run-1_T2w.nii.gz
        sub-01_ses-01_run-1_T2w.json
        sub-01_ses-01_flip-05_run-1_VFA.nii.gz               # VFA T1 mapping
        sub-01_ses-01_flip-05_run-1_VFA.json
        sub-01_ses-01_flip-15_run-1_VFA.nii.gz
        sub-01_ses-01_flip-15_run-1_VFA.json
        sub-01_ses-01_flip-25_run-1_VFA.nii.gz
        sub-01_ses-01_flip-25_run-1_VFA.json
      perf/
        sub-01_ses-01_ce-GAD_run-1_dsc.nii.gz                # DSC
        sub-01_ses-01_ce-GAD_run-1_dsc.json
        sub-01_ses-01_ce-GAD_run-1_dce.nii.gz                # DCE
        sub-01_ses-01_ce-GAD_run-1_dce.json
      fmap/
        sub-01_ses-01_run-1_TB1AFI.nii.gz                    # B1+ mapping (AFI)
        sub-01_ses-01_run-1_TB1AFI.json
        sub-01_ses-01_acq-famp_run-1_TB1TFL.nii.gz           # B1+ mapping (Siemens tfl_b1map, alternative to AFI)
        sub-01_ses-01_acq-famp_run-1_TB1TFL.json
```

> The `acq-<label>` entity is optional but used to rank candidates when multiple files share the same suffix: T1w prefers `acq-mprage` over other sequences and excludes `acq-se`; FLAIR prefers `acq-3d`; B1+ mapping matches `acq-famp` for Siemens tfl_b1map.

### Required Files

| Pipeline | Required | Optional |
|----------|----------|----------|
| DSC | `*_dsc.nii.gz`, T1w | — |
| DCE | `*_dce.nii.gz`, T1w | VFA (T1 mapping), B1+ map (`*_TB1AFI` or `*_TB1TFL`) |
| HD-GLIO | T1w pre-contrast, T1w post-contrast, T2w, FLAIR | — |

B1+ correction accepts either AFI (`*_TB1AFI`) or Siemens pre-computed flip angle maps (`*_TB1TFL`). Auto-detected via `--dce-b1-method auto`.

### JSON Sidecar Fields

**DSC sidecar (`*_dsc.json`):**
```json
{
  "RepetitionTime": 1.5,
  "EchoTime": 0.03,
  "FlipAngle": 60,
  "MagneticFieldStrength": 3,
  "ContrastBolusAgent": "gadobutrol"
}
```
Required: `RepetitionTime`, `EchoTime`, `FlipAngle`, `MagneticFieldStrength`. Recommended: `ContrastBolusAgent` (used for AIF peak threshold scaling).

**DCE sidecar (`*_dce.json`):**
```json
{
  "RepetitionTime": 5.0,
  "RepetitionTimeExcitation": 0.00313,
  "EchoTime": 0.001,
  "FlipAngle": 15,
  "MagneticFieldStrength": 3,
  "ContrastBolusAgent": "gadobutrol"
}
```
Required: `FlipAngle`, `MagneticFieldStrength`. For 3D sequences (VIBE/LAVA/THRIVE), both `RepetitionTime` (volume TR) and `RepetitionTimeExcitation` (pulse TR) are required. For 2D sequences, `RepetitionTime` serves as both. Recommended: `ContrastBolusAgent` (used for relaxivity lookup and AIF peak threshold scaling).

> **Note:** If `ContrastBolusAgent` is not provided, the pipeline defaults to Gd-DTPA (r1=3.8 s⁻¹/mM at 3T). The pipeline warns about defaults in the HTML QC report and records assumed values in output JSON sidecars under `PerfusionProcessing`. Use `--dce-r1` to override.
>
> **Dose:** `--dsc-dose` and `--dce-dose` are multipliers of the agent's recommended dose (e.g., `0.5` for half dose, `1.0` for full). If you administered less than a full dose (common with DSC preload protocols), set the multiplier accordingly. Incorrect dose settings cause AIF detection failure because peak concentration thresholds are scaled by dose.

**VFA sidecar (`*_VFA.json`):**
```json
{
  "RepetitionTime": 0.00313,
  "FlipAngle": 5
}
```
Required: `RepetitionTime`, `FlipAngle` (per file, minimum 2 distinct angles, 3+ recommended).

**AFI sidecar (`*_TB1AFI.json`):**
```json
{
  "RepetitionTime": [0.02, 0.1],
  "FlipAngle": 60
}
```
Required: `RepetitionTime` (scalar or 2-element array for 4D), `FlipAngle`.

**TB1TFL sidecar (`*_TB1TFL.json`):**
```json
{
  "RepetitionTime": 0.008,
  "FlipAngle": 80
}
```
Required: `RepetitionTime`, `FlipAngle`.

## Output Structure

PMRIPrep outputs follow BIDS-Derivatives conventions:

```
output_dir/pmriprep/
  dataset_description.json
  sub-01/ses-01/
    anat/
      sub-01_ses-01_space-T1w_desc-preproc_T1w.nii.gz
      sub-01_ses-01_space-T1w_desc-brain_mask.nii.gz
      sub-01_ses-01_space-T1w_label-GM_probseg.nii.gz
      sub-01_ses-01_space-T1w_label-WM_probseg.nii.gz
      sub-01_ses-01_space-T1w_label-CSF_probseg.nii.gz
      sub-01_ses-01_space-dceref_T1map.nii.gz
      sub-01_ses-01_space-dceref_B1plusmap.nii.gz
      sub-01_ses-01_space-T1w_desc-tumorseg_dseg.nii.gz
      sub-01_ses-01_space-T1w_desc-enhancing_mask.nii.gz
      sub-01_ses-01_space-T1w_desc-nawm_mask.nii.gz
      # ... tissue segmentations, transforms, space-specific masks
    perf/
      # DSC preprocessing
      sub-01_ses-01_desc-dscPreproc_dsc.nii.gz
      sub-01_ses-01_desc-dscMean_stat.nii.gz
      sub-01_ses-01_desc-dscConfounds_timeseries.tsv
      # DSC maps
      sub-01_ses-01_desc-dscRelCbf_stat.nii.gz
      sub-01_ses-01_desc-dscRelCbv_stat.nii.gz
      sub-01_ses-01_desc-dscRelAUCCbv_stat.nii.gz
      sub-01_ses-01_desc-dscRelAuc_stat.nii.gz
      sub-01_ses-01_desc-dscMtt_stat.nii.gz
      sub-01_ses-01_desc-dscTmax_stat.nii.gz
      sub-01_ses-01_desc-dscTtp_stat.nii.gz
      sub-01_ses-01_desc-dscK2_stat.nii.gz
      sub-01_ses-01_desc-dscPh_stat.nii.gz
      sub-01_ses-01_desc-dscPsr_stat.nii.gz
      sub-01_ses-01_desc-dscSnr_stat.nii.gz
      sub-01_ses-01_desc-dscTsnr_stat.nii.gz
      sub-01_ses-01_desc-dscCnr_stat.nii.gz
      sub-01_ses-01_desc-dscAifCorrectedTtp_stat.nii.gz
      sub-01_ses-01_desc-dscRelCorrectedAuc_stat.nii.gz
      sub-01_ses-01_desc-dscLeakageAuc_stat.nii.gz
      sub-01_ses-01_desc-dscNawmNormCbf_stat.nii.gz
      sub-01_ses-01_desc-dscNawmNormCbv_stat.nii.gz
      sub-01_ses-01_desc-dscNawmRelTtp_stat.nii.gz
      # DSC parametric uncertainty (inference: vi/svb/map only)
      sub-01_ses-01_desc-dscUncertaintyCbf_stat.nii.gz
      sub-01_ses-01_desc-dscCVCbf_stat.nii.gz
      sub-01_ses-01_desc-dscCI95lowerCbf_stat.nii.gz
      sub-01_ses-01_desc-dscCI95upperCbf_stat.nii.gz
      # DSC vascular model (optional)
      sub-01_ses-01_desc-dscAlpha_stat.nii.gz
      sub-01_ses-01_desc-dscBeta_stat.nii.gz
      sub-01_ses-01_desc-dscCth_stat.nii.gz
      # DCE preprocessing
      sub-01_ses-01_desc-dcePreproc_dce.nii.gz
      sub-01_ses-01_desc-dceMean_stat.nii.gz
      sub-01_ses-01_desc-dceConfounds_timeseries.tsv
      # DCE PK maps
      sub-01_ses-01_desc-dceKtrans_stat.nii.gz
      sub-01_ses-01_desc-dceVe_stat.nii.gz
      sub-01_ses-01_desc-dceVp_stat.nii.gz
      sub-01_ses-01_desc-dceKep_stat.nii.gz
      sub-01_ses-01_desc-dceRsquared_stat.nii.gz
      sub-01_ses-01_desc-dceNawmNormKtrans_stat.nii.gz
      # DCE PK uncertainty (inference: vi/map/mcmc only)
      sub-01_ses-01_desc-dceUncertaintyKtrans_stat.nii.gz
      sub-01_ses-01_desc-dceCVKtrans_stat.nii.gz
      sub-01_ses-01_desc-dceCI95lowerKtrans_stat.nii.gz
      sub-01_ses-01_desc-dceCI95upperKtrans_stat.nii.gz
      # DCE model selection (auto mode)
      sub-01_ses-01_desc-dceModelSelection_dseg.nii.gz
      sub-01_ses-01_desc-dceDeltaBIC_stat.nii.gz
      sub-01_ses-01_desc-dceQualityFlag_mask.nii.gz
      # DCE model-free maps
      sub-01_ses-01_desc-dceIAUC60_stat.nii.gz
      sub-01_ses-01_desc-dcePeakEnhancement_stat.nii.gz
      sub-01_ses-01_desc-dceSER_stat.nii.gz
      sub-01_ses-01_desc-dceWashInSlope_stat.nii.gz
      sub-01_ses-01_desc-dceWashOutSlope_stat.nii.gz
      sub-01_ses-01_desc-dceBAT_stat.nii.gz
      sub-01_ses-01_desc-dceCurveType_dseg.nii.gz
      sub-01_ses-01_desc-dceTtp_stat.nii.gz
      sub-01_ses-01_desc-dceTsnr_stat.nii.gz
      sub-01_ses-01_desc-dceNawmNormIAUC60_stat.nii.gz
      # AIF
      sub-01_ses-01_desc-dscAif_timeseries.tsv
      sub-01_ses-01_desc-dceAif_timeseries.tsv
    # HTML report
    sub-01_ses-01.html
```

> When `smooth_fwhm` is set (DSC or DCE config), smoothed variants of perfusion maps are also written with a `Sm{fwhm}` suffix in the `desc-` entity (e.g., `desc-dscRelSm3Cbf_stat` for 3 mm FWHM). All NIfTI outputs have paired `.json` sidecars with descriptions, units, and provenance.

### Output File Descriptions

**DSC preprocessing:**

| File Pattern | Description |
|--------------|-------------|
| `*_desc-dscPreproc_dsc` | Motion-corrected 4D DSC time series |
| `*_desc-dscMean_stat` | Mean DSC reference image |
| `*_desc-dscConfounds_timeseries.tsv` | Framewise displacement and motion parameters |

**DSC maps:**

| File Pattern | Description | Units |
|--------------|-------------|-------|
| `*_desc-dscRelCbf_stat` | Cerebral blood flow (relative) | 1/min |
| `*_desc-dscRelCbv_stat` | Cerebral blood volume (relative, deconvolution) | arbitrary |
| `*_desc-dscRelAUCCbv_stat` | Cerebral blood volume (K2-corrected full-bolus AUC, relative) | arbitrary |
| `*_desc-dscRelAuc_stat` | Relative area under curve | arbitrary |
| `*_desc-dscMtt_stat` | Mean transit time | seconds |
| `*_desc-dscTmax_stat` | Time to maximum of residue function | seconds |
| `*_desc-dscTtp_stat` | Time to peak concentration | seconds |
| `*_desc-dscAifCorrectedTtp_stat` | Time to peak, corrected for AIF bolus arrival | seconds |
| `*_desc-dscNawmRelTtp_stat` | Time to peak relative to contralateral NAWM | seconds |
| `*_desc-dscK2_stat` | K2 leakage parameter | 1/s |
| `*_desc-dscPh_stat` | Peak height | % |
| `*_desc-dscPsr_stat` | Percent signal recovery | % |
| `*_desc-dscSnr_stat` | Signal-to-noise ratio (baseline) | arbitrary |
| `*_desc-dscTsnr_stat` | Temporal signal-to-noise ratio | arbitrary |
| `*_desc-dscCnr_stat` | Contrast-to-noise ratio | arbitrary |
| `*_desc-dscRelCorrectedAuc_stat` | Leakage-corrected relative AUC | arbitrary |
| `*_desc-dscLeakageAuc_stat` | Leakage contribution (AUC_raw - AUC_corr) | arbitrary |
| `*_desc-dscNawmNormCbf_stat` | CBF normalized by NAWM | ratio |
| `*_desc-dscNawmNormCbv_stat` | CBV normalized by NAWM | ratio |
| `*_desc-dscAlpha_stat` | Vascular model alpha (vascular method only) | — |
| `*_desc-dscBeta_stat` | Vascular model beta (vascular method only) | — |
| `*_desc-dscCth_stat` | Capillary transit heterogeneity (vascular method only) | seconds |
| `*_desc-dscUncertainty{Param}_stat` | Posterior std (Bayesian only) | varies |
| `*_desc-dscCV{Param}_stat` | Coefficient of variation (Bayesian only) | fraction |
| `*_desc-dscCI95lower{Param}_stat` | 95% credible interval lower bound (Bayesian only) | varies |
| `*_desc-dscCI95upper{Param}_stat` | 95% credible interval upper bound (Bayesian only) | varies |

**DCE preprocessing:**

| File Pattern | Description |
|--------------|-------------|
| `*_desc-dcePreproc_dce` | Motion-corrected 4D concentration time series |
| `*_desc-dceMean_stat` | Mean DCE reference image |
| `*_desc-dceConfounds_timeseries.tsv` | Framewise displacement and motion parameters |

**DCE maps:**

| File Pattern | Description | Units |
|--------------|-------------|-------|
| `*_desc-dceKtrans_stat` | Volume transfer constant | 1/min |
| `*_desc-dceVe_stat` | Extravascular extracellular volume | fraction |
| `*_desc-dceVp_stat` | Plasma volume fraction | fraction |
| `*_desc-dceKep_stat` | Efflux rate constant | 1/min |
| `*_desc-dceRsquared_stat` | Goodness of fit (R-squared) | fraction |
| `*_desc-dceIAUC60_stat` | Initial area under curve at 60s | mM\*min |
| `*_desc-dceIAUC90_stat` | Initial area under curve at 90s | mM\*min |
| `*_desc-dceIAUC120_stat` | Initial area under curve at 120s | mM\*min |
| `*_desc-dcePeakEnhancement_stat` | Peak enhancement percentage | % |
| `*_desc-dceTtp_stat` | Time to peak | seconds |
| `*_desc-dceSER_stat` | Signal enhancement ratio | ratio |
| `*_desc-dceWashInSlope_stat` | Maximum upslope during wash-in | mM/min |
| `*_desc-dceWashOutSlope_stat` | Linear washout slope (negative=washout) | mM/min |
| `*_desc-dceBAT_stat` | Bolus arrival time | seconds |
| `*_desc-dceCurveType_dseg` | Curve type (1=persistent, 2=plateau, 3=washout) | categorical |
| `*_desc-dceTsnr_stat` | Temporal signal-to-noise ratio | arbitrary |
| `*_desc-dceNawmNormKtrans_stat` | NAWM-normalized Ktrans | ratio |
| `*_desc-dceNawmNormIAUC60_stat` | NAWM-normalized IAUC60 | ratio |
| `*_desc-dceUncertainty{Param}_stat` | Posterior std (Bayesian only) | varies |
| `*_desc-dceCV{Param}_stat` | Coefficient of variation (Bayesian only) | fraction |
| `*_desc-dceCI95lower{Param}_stat` | 95% credible interval lower bound (Bayesian only) | varies |
| `*_desc-dceCI95upper{Param}_stat` | 95% credible interval upper bound (Bayesian only) | varies |
| `*_desc-dceModelSelection_dseg` | Per-voxel model selection (auto mode only) | categorical |
| `*_desc-dceDeltaBIC_stat` | Delta-BIC confidence (auto mode only) | — |
| `*_desc-dceQualityFlag_mask` | Voxel quality flags (auto mode only) | categorical |


## QC Reports

PMRIPrep generates a per-session HTML report (`sub-XX_ses-XX.html`) in the output directory. The report includes:

- **Anatomical**: Brain mask overlay, tissue segmentation, MNI alignment, cross-modal coregistration (T1w post-contrast, T2w, FLAIR)
- **Tumor segmentation**: Segmentation overlays on T1w post-contrast and FLAIR, NAWM ROI visualization, volume statistics
- **DSC**: Brain extraction, registration overlay, motion parameters, preprocessing QC (tSNR, CNR), AIF curve and quality metrics, leakage correction, example voxel fit, parameter map montages (CBF, CBV, MTT, Tmax)
- **DCE**: Brain extraction, registration overlay, B1+ mapping QC, T1 map quality, motion parameters, preprocessing QC (tSNR, enhancement), AIF curve and quality metrics, example voxel fit, parameter map montages (Ktrans, ve, vp), model selection visualization, DSC-DCE cross-validation

Use `--report-only` to regenerate reports from existing outputs without re-running the pipeline.

## License

PMRIPrep is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full text.

## Citation

If you use PMRIPrep in your research, please cite:

```
@software{pmriprep,
  title = {PMRIPrep: Perfusion MRI Preprocessing Pipeline},
  author = {PMRIPrep Developers},
  year = {2026},
  url = {https://github.com/arulseh/pmriprep}
}
```

## Support

- **Issues**: https://github.com/arulseh/pmriprep/issues
