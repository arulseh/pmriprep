# PMRIPrep

**Perfusion MRI Preprocessing Pipeline** (v0.2.0)

PMRIPrep is a BIDS-App for processing Dynamic Susceptibility Contrast (DSC) and Dynamic Contrast Enhanced (DCE) perfusion MRI data. It follows the [NiPreps](https://www.nipreps.org/) framework conventions and produces BIDS-Derivatives compliant outputs with HTML QC reports.

## Features

- **DSC Processing**: Six deconvolution methods (sSVD, cSVD, oSVD, Tikhonov, CPI, Vascular) with CBF, CBV, MTT, and Tmax maps
- **DCE Processing**: Four pharmacokinetic models (Tofts, Extended Tofts, Patlak, 2CXM) plus model-free maps (IAUC, SER, peak enhancement, TTP)
- **Bayesian Inference**: Optional MAP, VI, or MCMC fitting with uncertainty quantification
- **Automatic AIF Detection**: Atlas-based arterial input function detection with population fallback
- **Leakage Correction**: K2 correction for BBB disruption in DSC
- **T1 Mapping**: Variable flip angle T1 mapping with optional B1+ correction
- **Tumor Segmentation**: HD-GLIO integration for glioma segmentation with enhancing/non-enhancing labels
- **NAWM Normalization**: Relative CBF/CBV and normalized Ktrans/IAUC using contralateral NAWM
- **QC Reports**: Per-session HTML reports with AIF quality, motion, registration, and parameter map visualizations
- **GPU Acceleration**: Optional CuPy-based GPU acceleration for DSC and DCE processing
- **Preload Correction**: K2-based leakage correction for contrast-preloaded DSC acquisitions

## Installation

### Docker (Recommended)

```bash
docker pull pmriprep/pmriprep:latest
```

Run with Docker:

```bash
docker run --rm \
    -v /path/to/bids:/data:ro \
    -v /path/to/output:/out \
    -v /path/to/work:/work \
    pmriprep/pmriprep:latest \
    /data /out participant \
    --participant-label 01 \
    -w /work
```

> **Apple Silicon (M1/M2/M3):** Add `--platform linux/amd64` to the `docker run` command. The container is built for x86_64 and runs under Rosetta emulation.

For GPU support (NVIDIA):

```bash
docker run --gpus all --rm \
    -v /path/to/bids:/data:ro \
    -v /path/to/output:/out \
    -v /path/to/work:/work \
    pmriprep/pmriprep:latest \
    /data /out participant \
    --use-gpu -w /work
```

### Singularity / Apptainer

```bash
# Pull from Docker Hub
singularity pull pmriprep.sif docker://pmriprep/pmriprep:latest

# Or from local Docker daemon
singularity pull pmriprep.sif docker-daemon://pmriprep:latest
```

Run with Singularity:

```bash
singularity run --cleanenv \
    --bind /path/to/bids:/data:ro \
    --bind /path/to/output:/out \
    --bind /path/to/work:/work \
    --bind /tmp:/home/pmriprep/.cache \
    pmriprep.sif \
    /data /out participant \
    --participant-label 01 \
    -w /work
```

The `--bind /tmp:/home/pmriprep/.cache` mount is needed for TemplateFlow template caching. For GPU support, add `--nv` (NVIDIA) to the `singularity run` command.

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

### Pipeline Skip Flags

| Flag | Description |
|------|-------------|
| `--skip-anat` | Skip anatomical preprocessing |
| `--skip-tumor-seg` | Skip HD-GLIO tumor segmentation |
| `--skip-dsc` | Skip DSC perfusion pipeline |
| `--skip-dce` | Skip DCE perfusion pipeline |

### DSC Options

| Flag | Description |
|------|-------------|
| `--dsc-deconv-method {sSVD,cSVD,oSVD,tikhonov,vascular,cpi}` | Deconvolution method (default: oSVD) |
| `--dsc-bayesian {none,map,vi,mcmc}` | Bayesian inference method (default: none) |
| `--dsc-preload` | Indicate contrast preload is present |
| `--dsc-dose {half,full,double}` | Contrast agent dose (default: full) |
| `--dsc-aif-method {auto,population}` | AIF detection method (default: auto) |

### DCE Options

| Flag | Description |
|------|-------------|
| `--dce-model {auto,tofts,extended_tofts,patlak,2cxm}` | Pharmacokinetic model (default: auto) |
| `--dce-bayesian {none,map,vi,mcmc}` | Bayesian inference method (default: none) |
| `--dce-aif-method {auto,population}` | AIF detection method (default: auto) |
| `--dce-dose {half,full,double}` | Contrast agent dose (default: full) |
| `--dce-r1 R1` | Override r1 relaxivity (s⁻¹/mM) for concentration conversion |
| `--dce-no-b1` | Skip B1+ mapping (use nominal flip angles) |
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
  preload_present: true
  dose: half
  method: oSVD
  bayesian: map
  mcmc_samples: 500
  aif_criteria: relaxed
  n_aif_candidates: 10
  pre_bolus: 12
  use_gpu: false
  lesion_mask: /path/to/lesion_mask.nii.gz
  nawm_mask: /path/to/nawm_mask.nii.gz
  roi_mask: /path/to/roi_mask.nii.gz
  aif_mask_file: /path/to/aif_mask.nii.gz

dce:
  model: extended_tofts
  bayesian: map
  aif_method: auto
  scale_aif: true
  dose: full
  no_b1: false
  no_t1: false
  baseline_frames: 0
  tsnr_threshold: 0.0
  use_gpu: false
  lesion_mask: /path/to/lesion_mask.nii.gz
  nawm_mask: /path/to/nawm_mask.nii.gz
  roi_mask: /path/to/roi_mask.nii.gz
  aif_mask_file: /path/to/aif_mask.nii.gz

anat:
  tool: hdbet
  denoise: false

tumor_seg:
  lesion_dilation: 15.0
  nawm_erosion: 2
  nawm_min_voxels: 500
  device: cuda
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
| `aif_method` | str | `"auto"` | AIF method: `auto`, `population` |
| `aif_mask_file` | path | `null` | Custom AIF mask file |
| `gamma_fit` | bool | `false` | Apply gamma-variate fitting to AIF |
| `disable_aif_fallback` | bool | `false` | Disable synthetic AIF fallback |
| `lesion_distance` | float | `15.0` | Minimum distance (mm) from lesion for AIF candidates |
| `pre_bolus` | int | `12` | Pre-bolus baseline frames |
| `smooth_fwhm` | float | `0.0` | Smoothing kernel FWHM in mm (0 = none) |
| `motion_thresh` | float | `0.5` | Framewise displacement threshold in mm |
| `registration` | str | `"auto"` | Registration method: `auto`, `flirt`, `bbr` |
| `lesion_mask` | path | `null` | Lesion mask for registration weighting and AIF exclusion |
| `nawm_mask` | path | `null` | NAWM mask for CBV normalization |
| `roi_mask` | path | `null` | ROI mask for statistics extraction |
| `use_gpu` | bool | `false` | Enable GPU acceleration |

### DCE Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `"auto"` | PK model: `auto`, `tofts`, `extended_tofts`, `patlak`, `2cxm` |
| `bayesian` | str | `"none"` | Bayesian mode: `none`, `map`, `vi`, `mcmc` |
| `mcmc_samples` | int | `500` | MCMC samples if bayesian=mcmc |
| `aif_method` | str | `"auto"` | AIF method: `auto`, `population` |
| `scale_aif` | bool | `true` | Scale AIF to population reference |
| `target_peak` | float | `null` | Target AIF peak (mM); null = field-strength default |
| `aif_mask_file` | path | `null` | Custom AIF mask file |
| `lesion_distance` | float | `15.0` | Minimum distance (mm) from lesion for AIF candidates |
| `n_aif_candidates` | int | `10` | Number of AIF candidate voxels to average |
| `dose` | str | `"full"` | Dose: `half`, `full`, `double` |
| `r1_relaxivity` | float | `null` | Override r1 relaxivity (s⁻¹/mM); null = lookup from BIDS sidecar |
| `no_b1` | bool | `false` | Skip B1+ mapping |
| `no_t1` | bool | `false` | Skip T1 mapping |
| `baseline_frames` | int | `0` | Baseline frames (0 = auto-detect) |
| `tsnr_threshold` | float | `0.0` | tSNR threshold for voxel inclusion (0 = none) |
| `smooth_fwhm` | float | `0.0` | Smoothing kernel FWHM in mm (0 = none) |
| `t1_correction` | str | `"auto"` | T1 correction mode: `auto`, `none`, `offset`, `fallback` |
| `cbv_constraint_mode` | str | `"auto"` | DSC CBV constraint: `auto`, `hard`, `soft`, `skip` |
| `hematocrit` | float | `0.40` | Blood hematocrit for CBV-to-vp conversion |
| `lesion_mask` | path | `null` | Lesion mask for registration and AIF exclusion |
| `nawm_mask` | path | `null` | NAWM mask for Ktrans/IAUC normalization |
| `roi_mask` | path | `null` | ROI mask for statistics extraction |
| `use_gpu` | bool | `false` | Enable GPU acceleration |

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
| `lesion_dilation` | float | `15.0` | Dilation radius (mm) for lesion mask |
| `nawm_erosion` | int | `2` | Erosion iterations for NAWM mask (voxels) |
| `nawm_min_voxels` | int | `500` | Minimum voxels required in NAWM ROI |
| `device` | str | `"cuda"` | Inference device: `cuda`, `mps`, `cpu` |
| `conda_env` | str | `null` | Conda environment for the segmentation tool |

## BIDS Input Requirements

PMRIPrep expects input data in BIDS format:

```
bids_dir/
  dataset_description.json
  sub-01/
    ses-01/
      anat/
        sub-01_ses-01_T1w.nii.gz
        sub-01_ses-01_T1w.json
        sub-01_ses-01_acq-postcon_T1w.nii.gz  # Post-contrast T1w
        sub-01_ses-01_FLAIR.nii.gz
        sub-01_ses-01_T2w.nii.gz
      perf/
        sub-01_ses-01_dsc.nii.gz
        sub-01_ses-01_dsc.json
        sub-01_ses-01_dce.nii.gz
        sub-01_ses-01_dce.json
      fmap/
        sub-01_ses-01_acq-vfa_flip-5_T1map.nii.gz   # VFA T1 mapping
        sub-01_ses-01_acq-vfa_flip-15_T1map.nii.gz
        sub-01_ses-01_acq-afi_B1plusmap.nii.gz       # B1+ mapping
```

### Required Files

| Pipeline | Required | Optional |
|----------|----------|----------|
| DSC | `*_dsc.nii.gz`, T1w | Lesion mask, FLAIR |
| DCE | `*_dce.nii.gz`, T1w | VFA images, B1+ map, lesion mask |
| HD-GLIO | T1w-pre, T1w-postcon, T2w, FLAIR | - |

### JSON Sidecar Fields

**DSC sidecar (`*_dsc.json`):**
```json
{
  "RepetitionTime": 1.5,
  "EchoTime": 0.03,
  "FlipAngle": 60,
  "MagneticFieldStrength": 3
}
```

**DCE sidecar (`*_dce.json`):**
```json
{
  "RepetitionTime": 5.0,
  "EchoTime": 0.002,
  "FlipAngle": 15,
  "MagneticFieldStrength": 3
}
```

## Output Structure

PMRIPrep outputs follow BIDS-Derivatives conventions:

```
output_dir/pmriprep/
  dataset_description.json
  sub-01/ses-01/
    anat/
      sub-01_ses-01_desc-preproc_T1w.nii.gz
      sub-01_ses-01_desc-brain_mask.nii.gz
      sub-01_ses-01_label-WM_probseg.nii.gz
      sub-01_ses-01_label-GM_probseg.nii.gz
      sub-01_ses-01_label-CSF_probseg.nii.gz
      sub-01_ses-01_T1map.nii.gz
      sub-01_ses-01_B1plusmap.nii.gz
      # Tumor segmentation
      sub-01_ses-01_space-anat_desc-tumorseg_dseg.nii.gz
      sub-01_ses-01_space-anat_desc-enhancing_mask.nii.gz
      sub-01_ses-01_space-anat_desc-nonenhancing_mask.nii.gz
      sub-01_ses-01_space-anat_desc-wholetumor_mask.nii.gz
      sub-01_ses-01_space-anat_desc-lesion_mask.nii.gz
      sub-01_ses-01_space-anat_desc-nawm_mask.nii.gz
    perf/
      # DSC maps (relative)
      sub-01_ses-01_space-DSC_desc-dscRel_cbf.nii.gz
      sub-01_ses-01_space-DSC_desc-dscRel_cbv.nii.gz
      sub-01_ses-01_space-DSC_desc-dsc_mtt.nii.gz
      sub-01_ses-01_space-DSC_desc-dsc_tmax.nii.gz
      sub-01_ses-01_space-DSC_desc-dsc_k2.nii.gz
      # DSC maps (NAWM-normalized)
      sub-01_ses-01_space-DSC_desc-dscNawmNorm_cbv.nii.gz
      sub-01_ses-01_space-DSC_desc-dscNawmNorm_cbf.nii.gz
      # DSC ancillary
      sub-01_ses-01_desc-dscAif_timeseries.txt
      sub-01_ses-01_space-DSC_desc-dscAif_mask.nii.gz
      sub-01_ses-01_desc-dscConfounds_timeseries.tsv
      # DCE PK maps (model-dependent)
      sub-01_ses-01_space-DCE_model-extendedTofts_desc-dce_Ktrans.nii.gz
      sub-01_ses-01_space-DCE_model-extendedTofts_desc-dce_ve.nii.gz
      sub-01_ses-01_space-DCE_model-extendedTofts_desc-dce_vp.nii.gz
      sub-01_ses-01_space-DCE_model-extendedTofts_desc-dce_kep.nii.gz
      # DCE model-free maps
      sub-01_ses-01_space-DCE_desc-dce_IAUC60.nii.gz
      sub-01_ses-01_space-DCE_desc-dce_IAUC90.nii.gz
      sub-01_ses-01_space-DCE_desc-dce_IAUC120.nii.gz
      sub-01_ses-01_space-DCE_desc-dce_peakEnhancement.nii.gz
      sub-01_ses-01_space-DCE_desc-dce_ttp.nii.gz
      sub-01_ses-01_space-DCE_desc-dce_SER.nii.gz
      sub-01_ses-01_space-DCE_desc-dce_tsnr.nii.gz
      # DCE ancillary
      sub-01_ses-01_desc-dceAif_timeseries.txt
      sub-01_ses-01_space-DCE_desc-dceAif_mask.nii.gz
      sub-01_ses-01_desc-dceConfounds_timeseries.tsv
    # QC files (session level)
    sub-01_ses-01_desc-dscQC_quality.json
    sub-01_ses-01_desc-dceQC_quality.json
    sub-01_ses-01_desc-tumorsegQC_quality.json
    # HTML report
    sub-01_ses-01.html
```

### Output File Descriptions

| File Pattern | Description | Units |
|--------------|-------------|-------|
| `*_desc-dscRel_cbf` | Cerebral blood flow (relative) | 1/min |
| `*_desc-dscRel_cbv` | Cerebral blood volume (relative) | arbitrary |
| `*_desc-dscNawmNorm_cbf` | CBF normalized by NAWM | ratio |
| `*_desc-dscNawmNorm_cbv` | CBV normalized by NAWM | ratio |
| `*_desc-dsc_mtt` | Mean transit time | seconds |
| `*_desc-dsc_tmax` | Time to maximum of residue function | seconds |
| `*_desc-dsc_k2` | K2 leakage parameter | 1/s |
| `*_desc-dce_Ktrans` | Volume transfer constant | 1/min |
| `*_desc-dce_ve` | Extravascular extracellular volume | fraction |
| `*_desc-dce_vp` | Plasma volume fraction | fraction |
| `*_desc-dce_kep` | Efflux rate constant | 1/min |
| `*_desc-dce_IAUC60` | Initial area under curve at 60s | mM\*min |
| `*_desc-dce_IAUC90` | Initial area under curve at 90s | mM\*min |
| `*_desc-dce_IAUC120` | Initial area under curve at 120s | mM\*min |
| `*_desc-dce_peakEnhancement` | Peak enhancement percentage | % |
| `*_desc-dce_ttp` | Time to peak | seconds |
| `*_desc-dce_SER` | Signal enhancement ratio | ratio |
| `*_desc-dce_tsnr` | Temporal signal-to-noise ratio | ratio |
| `*_desc-dce_R2` | Goodness of fit (R-squared) | fraction |

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
