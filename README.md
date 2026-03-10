# PMRIPrep

**Perfusion MRI Preprocessing Pipeline**

PMRIPrep is a BIDS-App for processing Dynamic Susceptibility Contrast (DSC) and Dynamic Contrast Enhanced (DCE) perfusion MRI data. It follows the [NiPreps](https://www.nipreps.org/) framework conventions and produces BIDS-Derivatives compliant outputs with HTML QC reports.

## Features

- **DSC Processing**: Six deconvolution methods (sSVD, cSVD, oSVD, Tikhonov, CPI, Vascular) and K2 leakage correction
- **DCE Processing**: Four pharmacokinetic models (Tofts, Extended Tofts, Patlak, 2CXM) plus model-free maps
- **Bayesian Inference**: Optional MAP, VI, or MCMC fitting with uncertainty quantification
- **Automatic AIF Detection**: Multi-level arterial input function detection with population fallback, or user-supplied external AIF
- **T1 Mapping**: Variable flip angle T1 mapping with optional B1+ correction
- **Tumor Segmentation**: HD-GLIO integration for glioma segmentation
- **NAWM Normalization**: Relative perfusion metrics using contralateral normal-appearing white matter
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
| `--nprocs N` | Maximum number of parallel processes (default: 1) |
| `--omp-nthreads N` | Threads per process for ANTs/FSL |
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
| `--dsc-deconv-method {sSVD,cSVD,oSVD,tikhonov,vascular,cpi}` | Deconvolution method (default: oSVD) |
| `--dsc-bayesian {none,map,vi,mcmc}` | Bayesian inference method (default: none) |
| `--dsc-preload` | Indicate contrast preload is present |
| `--dsc-dose {half,full,double}` | Contrast agent dose (default: full) |
| `--dsc-aif-method {auto,synthetic}` | AIF detection method (default: auto) |
| `--dsc-aif-file FILE` | External AIF curve file (ΔR2* in s⁻¹); bypasses detection |

### DCE Options

| Flag | Description |
|------|-------------|
| `--dce-model {auto,tofts,extended_tofts,patlak,2cxm}` | Pharmacokinetic model. `auto` performs per-voxel BIC model selection, where each voxel uses the best-fitting model (default: auto) |
| `--dce-bayesian {none,map,vi,mcmc}` | Bayesian inference method (default: none) |
| `--dce-aif-method {auto,population}` | AIF detection method (default: auto) |
| `--dce-aif-file FILE` | External AIF curve file (plasma concentration in mM); bypasses detection |
| `--dce-dose {half,full,double}` | Contrast agent dose (default: full) |
| `--dce-r1 R1` | Override r1 relaxivity (s⁻¹/mM) for concentration conversion |
| `--dce-b1-method {auto,afi,tfl_b1map,none}` | B1+ correction method (default: auto) |
| `--dce-no-t1` | Skip T1 mapping (use default T1 for field strength) |

### Anatomical Options

| Flag | Description |
|------|-------------|
| `--brain-extraction-tool {hdbet,ants}` | Brain extraction tool (default: hdbet) |
| `--denoise` | Apply ANTs denoising to anatomical images |

### Other Options

| Flag | Description |
|------|-------------|
| `--use-gpu` | Enable GPU acceleration for DSC and DCE |
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
  bayesian: map
  aif_criteria: relaxed
  lesion_mask: /path/to/lesion_mask.nii.gz
  nawm_mask: /path/to/nawm_mask.nii.gz
  underlays: [t1w_postcon, flair]

dce:
  model: extended_tofts
  bayesian: map
  lesion_mask: /path/to/lesion_mask.nii.gz
  nawm_mask: /path/to/nawm_mask.nii.gz
  underlays: [t1w_postcon, flair]
```

### DSC Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `preload_present` | bool | `false` | Contrast preload present |
| `dose` | str | `"full"` | Dose: `half`, `full`, `double` |
| `method` | str | `"oSVD"` | Deconvolution method: `sSVD`, `cSVD`, `oSVD`, `tikhonov`, `vascular`, `cpi` |
| `bayesian` | str | `"none"` | Bayesian mode: `none`, `map`, `vi`, `mcmc` |
| `mcmc_samples` | int | `500` | MCMC samples if bayesian=mcmc |
| `cpi_control_points` | int | `0` | Control points for CPI (0 = auto) |
| `aif_criteria` | str | `"strict"` | AIF selection: `strict`, `relaxed` |
| `n_aif_candidates` | int | `10` | Number of AIF candidate voxels to average |
| `lesion_distance` | float | `15.0` | Minimum distance (mm) from lesion for AIF candidates |
| `aif_method` | str | `"auto"` | AIF method: `auto`, `synthetic` |
| `aif_mask_file` | path | `null` | Custom AIF mask file |
| `gamma_fit` | str | `"auto"` | Gamma-variate AIF fitting: `auto`, `on`, `off` |
| `disable_aif_fallback` | bool | `false` | Disable synthetic AIF fallback |
| `pre_bolus` | int | `5` | Pre-bolus frames for S0 estimation (AIF timing auto-detected) |
| `smooth_fwhm` | float | `0.0` | Smoothing kernel FWHM in mm (0 = none) |
| `motion_thresh` | float | `0.5` | Framewise displacement threshold in mm |
| `registration` | str | `"auto"` | Registration method: `auto`, `flirt`, `bbr` |
| `lesion_mask` | path | `null` | Lesion mask for AIF exclusion and lesion statistics |
| `nawm_mask` | path | `null` | NAWM mask for normalization |
| `roi_mask` | path | `null` | ROI mask for statistics extraction |
| `underlays` | list | `[]` | Structural images to transform to DSC space for QC overlays: `t1w_precon`, `t1w_postcon`, `flair`, `t2w` |

### DCE Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `"auto"` | PK model. `auto` = per-voxel BIC selection. Fixed: `tofts`, `extended_tofts`, `patlak`, `2cxm` |
| `bayesian` | str | `"none"` | Bayesian mode: `none`, `map`, `vi`, `mcmc` |
| `mcmc_samples` | int | `500` | MCMC samples if bayesian=mcmc |
| `aif_method` | str | `"auto"` | AIF method: `auto`, `population` |
| `aif_file` | path | `null` | External AIF curve file (`.txt` or `.npz` with `Cp`/`aif` key); plasma concentration in mM, temporally aligned with DCE volumes. Bypasses AIF detection |
| `scale_aif` | bool | `true` | Scale AIF to population reference |
| `target_peak` | float | `null` | Target AIF peak (mM); null = field-strength default |
| `aif_mask_file` | path | `null` | Custom AIF mask file |
| `n_aif_candidates` | int | `10` | Number of AIF candidate voxels to average |
| `lesion_distance` | float | `15.0` | Minimum distance (mm) from lesion for AIF candidates |
| `dose` | str | `"full"` | Dose: `half`, `full`, `double` |
| `r1_relaxivity` | float | `null` | Override r1 relaxivity (s⁻¹/mM); null = lookup from BIDS sidecar |
| `b1_method` | str | `"auto"` | B1+ correction: `auto`, `afi`, `tfl_b1map`, `none` |
| `no_t1` | bool | `false` | Skip T1 mapping |
| `skip_flip_validation` | bool | `false` | Skip flip angle validation in VFA T1 mapping |
| `lenient_t1_qc` | bool | `false` | Use lenient T1 QC ranges (500-2500 ms vs strict) |
| `baseline_frames` | int | `0` | Baseline frames (0 = auto-detect) |
| `tsnr_threshold` | float | `0.0` | tSNR threshold for voxel inclusion (0 = none) |
| `smooth_fwhm` | float | `0.0` | Smoothing kernel FWHM in mm (0 = none) |
| `t1_correction` | str | `"auto"` | T1 correction mode: `auto`, `none`, `offset`, `fallback` |
| `no_tortuosity` | bool | `false` | Skip DWI-based tortuosity correction |
| `no_dsc_validation` | bool | `false` | Skip DSC-DCE cross-validation |
| `cbv_constraint_mode` | str | `"auto"` | DSC CBV constraint: `auto`, `hard`, `soft`, `skip` |
| `hematocrit` | float | `0.40` | Blood hematocrit for CBV-to-vp conversion |
| `lesion_mask` | path | `null` | Lesion mask for AIF exclusion and lesion statistics |
| `nawm_mask` | path | `null` | NAWM mask for normalization |
| `roi_mask` | path | `null` | ROI mask for statistics extraction |
| `underlays` | list | `[]` | Structural images to transform to DCE space for QC overlays: `t1w_precon`, `t1w_postcon`, `flair`, `t2w` |

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
        sub-01_ses-01_flip-02_run-1_VFA.nii.gz               # VFA T1 mapping
        sub-01_ses-01_flip-02_run-1_VFA.json
        sub-01_ses-01_flip-15_run-1_VFA.nii.gz
        sub-01_ses-01_flip-15_run-1_VFA.json
      perf/
        sub-01_ses-01_ce-GAD_run-1_dsc.nii.gz                # DSC perfusion
        sub-01_ses-01_ce-GAD_run-1_dsc.json
        sub-01_ses-01_ce-GAD_run-1_dce.nii.gz                # DCE perfusion
        sub-01_ses-01_ce-GAD_run-1_dce.json
      dwi/
        sub-01_ses-01_run-1_dwi.nii.gz                       # DWI (for ADC / tortuosity)
        sub-01_ses-01_run-1_dwi.json
        sub-01_ses-01_run-1_dwi.bvec
        sub-01_ses-01_run-1_dwi.bval
      fmap/
        sub-01_ses-01_run-1_TB1AFI.nii.gz                    # B1+ mapping (AFI)
        sub-01_ses-01_run-1_TB1AFI.json
        sub-01_ses-01_acq-famp_run-1_TB1TFL.nii.gz           # B1+ mapping (Siemens tfl_b1map, alternative to AFI)
        sub-01_ses-01_acq-famp_run-1_TB1TFL.json
```

> The `acq-<label>` entity is optional. If your BIDS dataset includes it (e.g., `acq-mprage`, `acq-3d`), it will be recognized automatically. The pipeline discovers files by suffix (`_T1w`, `_FLAIR`, `_dsc`, etc.) and functional entities (`ce-`, `flip-`), not by `acq-` label.

### Required Files

| Pipeline | Required | Optional |
|----------|----------|----------|
| DSC | `*_dsc.nii.gz`, T1w | — |
| DCE | `*_dce.nii.gz`, T1w | VFA (T1 mapping), B1+ map (`*_TB1AFI` or `*_TB1TFL`), DWI (tortuosity) |
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
Required: `RepetitionTime`, `EchoTime`, `FlipAngle`. Optional: `MagneticFieldStrength` (default 3T), `ContrastBolusAgent`.

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
Required: `FlipAngle`. For 3D sequences (VIBE/LAVA/THRIVE), both `RepetitionTime` (volume TR) and `RepetitionTimeExcitation` (pulse TR) are required. For 2D sequences, `RepetitionTime` serves as both. Optional: `MagneticFieldStrength`, `ContrastBolusAgent`.

**VFA sidecar (`*_VFA.json`):**
```json
{
  "RepetitionTime": 0.00313,
  "FlipAngle": 2
}
```
Required: `RepetitionTime`, `FlipAngle` (per file, minimum 2 distinct angles).

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
      sub-01_ses-01_space-anat_desc-preproc_T1w.nii.gz
      sub-01_ses-01_space-anat_desc-brain_mask.nii.gz
      sub-01_ses-01_space-anat_label-WM_probseg.nii.gz
      sub-01_ses-01_space-DCE_T1map.nii.gz
      sub-01_ses-01_space-DCE_B1plusmap.nii.gz
      sub-01_ses-01_space-anat_desc-hdglio_dseg.nii.gz
      sub-01_ses-01_space-anat_desc-enhancing_mask.nii.gz
      sub-01_ses-01_space-anat_desc-nawm_mask.nii.gz
      # ... tissue segmentations, transforms, space-specific masks
    perf/
      # DSC maps
      sub-01_ses-01_space-DSC_desc-dscRel_cbf.nii.gz
      sub-01_ses-01_space-DSC_desc-dscRel_cbv.nii.gz
      sub-01_ses-01_space-DSC_desc-dscRel_auc.nii.gz
      sub-01_ses-01_space-DSC_desc-dsc_mtt.nii.gz
      sub-01_ses-01_space-DSC_desc-dsc_tmax.nii.gz
      sub-01_ses-01_space-DSC_desc-dsc_ttp.nii.gz
      sub-01_ses-01_space-DSC_desc-dsc_k2.nii.gz
      sub-01_ses-01_space-DSC_desc-dsc_ph.nii.gz
      sub-01_ses-01_space-DSC_desc-dsc_psr.nii.gz
      sub-01_ses-01_space-DSC_desc-dsc_snr.nii.gz
      sub-01_ses-01_space-DSC_desc-dsc_tsnr.nii.gz
      sub-01_ses-01_space-DSC_desc-dsc_cnr.nii.gz
      sub-01_ses-01_space-DSC_desc-dscAifCorrected_ttp.nii.gz
      sub-01_ses-01_space-DSC_desc-dscRelCorrected_auc.nii.gz
      sub-01_ses-01_space-DSC_desc-dscLeakage_auc.nii.gz
      sub-01_ses-01_space-DSC_desc-dscNawmNorm_cbf.nii.gz
      sub-01_ses-01_space-DSC_desc-dscNawmNorm_cbv.nii.gz
      sub-01_ses-01_space-DSC_desc-dscNawmRel_ttp.nii.gz
      # DSC Bayesian (optional)
      sub-01_ses-01_space-DSC_desc-dscUncertainty_cbf.nii.gz
      sub-01_ses-01_space-DSC_desc-dscCV_cbf.nii.gz
      # DSC vascular model (optional)
      sub-01_ses-01_space-DSC_desc-dsc_alpha.nii.gz
      sub-01_ses-01_space-DSC_desc-dsc_beta.nii.gz
      sub-01_ses-01_space-DSC_desc-dsc_cth.nii.gz
      # DCE PK maps
      sub-01_ses-01_space-DCE_model-auto_desc-dce_Ktrans.nii.gz
      sub-01_ses-01_space-DCE_model-auto_desc-dce_ve.nii.gz
      sub-01_ses-01_space-DCE_model-auto_desc-dce_vp.nii.gz
      sub-01_ses-01_space-DCE_model-auto_desc-dce_kep.nii.gz
      sub-01_ses-01_space-DCE_model-auto_desc-dce_R2.nii.gz
      sub-01_ses-01_space-DCE_model-auto_desc-dceNawmNorm_Ktrans.nii.gz
      sub-01_ses-01_space-DCE_model-auto_desc-dceTortuosityCorr_Ktrans.nii.gz
      # DCE Bayesian (optional)
      sub-01_ses-01_space-DCE_model-auto_desc-dceUncertainty_Ktrans.nii.gz
      sub-01_ses-01_space-DCE_model-auto_desc-dceCV_Ktrans.nii.gz
      sub-01_ses-01_space-DCE_model-auto_desc-dceCI95lower_Ktrans.nii.gz
      sub-01_ses-01_space-DCE_model-auto_desc-dceCI95upper_Ktrans.nii.gz
      # DCE model selection (auto mode)
      sub-01_ses-01_space-DCE_model-auto_desc-dceModelSelection_dseg.nii.gz
      sub-01_ses-01_space-DCE_model-auto_desc-dceDeltaBIC_stat.nii.gz
      sub-01_ses-01_space-DCE_model-auto_desc-dceQualityFlag_mask.nii.gz
      # DCE model-free maps
      sub-01_ses-01_space-DCE_desc-dce_IAUC60.nii.gz
      sub-01_ses-01_space-DCE_desc-dce_peakEnhancement.nii.gz
      sub-01_ses-01_space-DCE_desc-dce_SER.nii.gz
      sub-01_ses-01_space-DCE_desc-dce_ttp.nii.gz
      sub-01_ses-01_space-DCE_desc-dce_tsnr.nii.gz
      sub-01_ses-01_space-DCE_desc-dceNawmNorm_IAUC60.nii.gz
      # AIF
      sub-01_ses-01_desc-dscAif_timeseries.txt
      sub-01_ses-01_desc-dceAif_timeseries.txt
    # QC and statistics
    sub-01_ses-01_desc-anatQC_quality.json
    sub-01_ses-01_desc-dscQC_quality.json
    sub-01_ses-01_desc-dceQC_quality.json
    sub-01_ses-01_desc-hdglioQC_quality.json
    sub-01_ses-01_space-DSC_desc-lesionStats.json
    sub-01_ses-01_space-DCE_desc-lesionStats.json
    # HTML report
    sub-01_ses-01.html
```

### Output File Descriptions

**DSC maps:**

| File Pattern | Description | Units |
|--------------|-------------|-------|
| `*_desc-dscRel_cbf` | Cerebral blood flow (relative) | 1/min |
| `*_desc-dscRel_cbv` | Cerebral blood volume (relative) | arbitrary |
| `*_desc-dscRel_auc` | Relative area under curve | arbitrary |
| `*_desc-dsc_mtt` | Mean transit time | seconds |
| `*_desc-dsc_tmax` | Time to maximum of residue function | seconds |
| `*_desc-dsc_ttp` | Time to peak concentration | seconds |
| `*_desc-dscAifCorrected_ttp` | Time to peak, corrected for AIF bolus arrival | seconds |
| `*_desc-dscNawmRel_ttp` | Time to peak relative to contralateral NAWM | seconds |
| `*_desc-dsc_k2` | K2 leakage parameter | 1/s |
| `*_desc-dsc_ph` | Peak height | arbitrary |
| `*_desc-dsc_psr` | Percent signal recovery | % |
| `*_desc-dsc_snr` | Signal-to-noise ratio (baseline) | arbitrary |
| `*_desc-dsc_tsnr` | Temporal signal-to-noise ratio | arbitrary |
| `*_desc-dsc_cnr` | Contrast-to-noise ratio | arbitrary |
| `*_desc-dscRelCorrected_auc` | Leakage-corrected relative AUC | arbitrary |
| `*_desc-dscLeakage_auc` | Leakage contribution (AUC_raw - AUC_corr) | arbitrary |
| `*_desc-dscNawmNorm_cbf` | CBF normalized by NAWM | ratio |
| `*_desc-dscNawmNorm_cbv` | CBV normalized by NAWM | ratio |
| `*_desc-dsc_alpha` | Vascular model alpha (vascular method only) | — |
| `*_desc-dsc_beta` | Vascular model beta (vascular method only) | — |
| `*_desc-dsc_cth` | Capillary transit heterogeneity (vascular method only) | seconds |
| `*_desc-dscUncertainty_{param}` | Posterior std (Bayesian only) | varies |
| `*_desc-dscCV_{param}` | Coefficient of variation (Bayesian only) | fraction |
| `*_desc-dscCI95lower_{param}` | 95% credible interval lower bound (Bayesian only) | varies |
| `*_desc-dscCI95upper_{param}` | 95% credible interval upper bound (Bayesian only) | varies |

**DCE maps:**

| File Pattern | Description | Units |
|--------------|-------------|-------|
| `*_desc-dce_Ktrans` | Volume transfer constant | 1/min |
| `*_desc-dce_ve` | Extravascular extracellular volume | fraction |
| `*_desc-dce_vp` | Plasma volume fraction | fraction |
| `*_desc-dce_kep` | Efflux rate constant | 1/min |
| `*_desc-dce_R2` | Goodness of fit (R-squared) | fraction |
| `*_desc-dce_IAUC60` | Initial area under curve at 60s | mM\*min |
| `*_desc-dce_IAUC90` | Initial area under curve at 90s | mM\*min |
| `*_desc-dce_IAUC120` | Initial area under curve at 120s | mM\*min |
| `*_desc-dce_peakEnhancement` | Peak enhancement percentage | % |
| `*_desc-dce_ttp` | Time to peak | seconds |
| `*_desc-dce_SER` | Signal enhancement ratio | ratio |
| `*_desc-dce_tsnr` | Temporal signal-to-noise ratio | ratio |
| `*_desc-dceNawmNorm_Ktrans` | NAWM-normalized Ktrans | ratio |
| `*_desc-dceNawmNorm_IAUC60` | NAWM-normalized IAUC60 | ratio |
| `*_desc-dceTortuosityCorr_Ktrans` | Tortuosity-corrected Ktrans (requires DWI) | 1/min |
| `*_desc-dceUncertainty_{param}` | Posterior std (Bayesian only) | varies |
| `*_desc-dceCV_{param}` | Coefficient of variation (Bayesian only) | fraction |
| `*_desc-dceCI95lower_{param}` | 95% credible interval lower bound (Bayesian only) | varies |
| `*_desc-dceCI95upper_{param}` | 95% credible interval upper bound (Bayesian only) | varies |
| `*_desc-dceModelSelection_dseg` | Per-voxel model selection (auto mode only) | categorical |
| `*_desc-dceDeltaBIC_stat` | Delta-BIC confidence (auto mode only) | — |
| `*_desc-dceQualityFlag_mask` | Voxel quality flags (auto mode only) | categorical |

All NIfTI outputs have paired `.json` sidecars with descriptions, units, and provenance.

## QC Reports

PMRIPrep generates a per-session HTML report (`sub-XX_ses-XX.html`) in the output directory. The report includes:

- **Anatomical**: Brain mask overlays, tissue segmentation, registration quality
- **Tumor segmentation**: Segmentation overlays on T1w post-contrast and FLAIR, volume statistics
- **DSC**: AIF curve and quality metrics, motion parameters, parameter map montages (CBF, CBV, MTT, Tmax), K2 leakage maps
- **DCE**: AIF curve, T1 map quality, tSNR map, parameter map montages (Ktrans, ve, vp, kep), model-free maps, R-squared goodness-of-fit

Use `--report-only` to regenerate reports from existing outputs without re-running the pipeline.

## License

PMRIPrep is licensed under the BSD 3-Clause License.

## Citation

If you use PMRIPrep in your research, please cite:

```
@software{pmriprep,
  title = {PMRIPrep: Perfusion MRI Preprocessing Pipeline},
  author = {PMRIPrep Developers},
  year = {2026},
  url = {https://github.com/pmriprep/pmriprep}
}
```

## Support

- **Issues**: https://github.com/pmriprep/pmriprep/issues
- **Documentation**: https://pmriprep.readthedocs.io
