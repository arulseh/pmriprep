# PMRIPrep

**Perfusion MRI Preprocessing Pipeline**

PMRIPrep is a BIDS-App for processing Dynamic Susceptibility Contrast (DSC) and Dynamic Contrast Enhanced (DCE) perfusion MRI data. It follows the NiPreps framework conventions and produces BIDS-Derivatives compliant outputs.

## Features

- **DSC Processing**: Multiple deconvolution methods (sSVD, cSVD, oSVD, Tikhonov, CPI, Vascular)
- **DCE Processing**: Multiple pharmacokinetic models (Tofts, Extended Tofts, Patlak, 2CXM)
- **Bayesian Inference**: Optional Bayesian fitting with uncertainty quantification (MAP, VI, MCMC)
- **Automatic AIF Detection**: Atlas-based arterial input function detection with population fallback
- **Leakage Correction**: K2 correction for BBB disruption
- **Tumor Segmentation**: HD-GLIO integration for glioma segmentation
- **GPU Acceleration**: Optional CuPy-based GPU acceleration

## Installation

### pip (Recommended for Development)

```bash
pip install pmriprep
```

### Docker

```bash
docker pull pmriprep/pmriprep:latest

# Run with Docker
docker run -v /path/to/bids:/data:ro \
           -v /path/to/output:/out \
           pmriprep/pmriprep:latest \
           /data /out participant
```

### Singularity

```bash
# Convert from Docker Hub
singularity pull pmriprep.sif docker://pmriprep/pmriprep:latest

# Or build from local Docker image
singularity build pmriprep.sif docker-daemon://pmriprep/pmriprep:latest

# Run with Singularity
singularity run --cleanenv \
    -B /path/to/bids:/data:ro \
    -B /path/to/output:/out \
    pmriprep.sif /data /out participant
```

## CLI Usage

### Basic Usage

```bash
pmriprep /path/to/bids /path/to/output participant \
    --participant-label 01 02 03
```

### DSC Processing Options

```bash
pmriprep /path/to/bids /path/to/output participant \
    --dsc-deconv-method oSVD \
    --dsc-bayesian map \
    --dsc-preload \
    --dsc-dose half
```

#### DSC Deconvolution Methods

| Method | Description |
|--------|-------------|
| `sSVD` | Standard SVD (default) |
| `cSVD` | Circulant SVD (delay-insensitive) |
| `oSVD` | Oscillation-index limited SVD |
| `tikhonov` | Tikhonov regularization with GCV |
| `vascular` | Gamma-variate vascular model |
| `cpi` | Control Point Interpolation |

#### GBM Preset

For glioma patients, use the `--dsc-gbm-preset` flag which sets:
- Method: oSVD
- Pre-bolus frames: 10
- AIF criteria: relaxed
- AIF candidates: 5
- Save K2 map: true

```bash
pmriprep /path/to/bids /path/to/output participant \
    --dsc-gbm-preset \
    --dsc-preload \
    --dsc-dose half
```

### DCE Processing Options

```bash
pmriprep /path/to/bids /path/to/output participant \
    --dce-model extended_tofts \
    --dce-bayesian map \
    --dce-scale-aif
```

#### DCE Pharmacokinetic Models

| Model | Parameters | Description |
|-------|------------|-------------|
| `tofts` | Ktrans, ve, kep | Standard Tofts model |
| `extended_tofts` | Ktrans, ve, vp, kep | Extended Tofts with vascular term |
| `patlak` | Ktrans, vp | Patlak model (no backflux) |
| `2cxm` | Ktrans, ve, vp, Fp, PS | Two-compartment exchange model |
| `auto` | (varies) | Auto-select based on temporal resolution |

### Skip Flags

```bash
pmriprep /path/to/bids /path/to/output participant \
    --skip-anat \        # Skip anatomical preprocessing
    --skip-tumor-seg \   # Skip HD-GLIO tumor segmentation
    --skip-dsc \         # Skip DSC processing
    --skip-dce           # Skip DCE processing
```

### Resource Options

```bash
pmriprep /path/to/bids /path/to/output participant \
    --nprocs 8 \         # Number of parallel processes
    --mem-gb 16 \        # Memory limit in GB
    -w /path/to/work     # Working directory
```

## Configuration File

For advanced options, use a YAML or JSON configuration file:

```bash
pmriprep /path/to/bids /path/to/output participant \
    --config-file config.yaml
```

### Example config.yaml

```yaml
# DSC configuration
dsc:
  preload_detected: true
  dose: half
  method: oSVD
  bayesian: map
  mcmc_samples: 500
  aif_criteria: relaxed
  n_aif_candidates: 10
  pre_bolus: 5
  save_k2_map: true
  save_uncertainty: true
  use_gpu: false

# DCE configuration
dce:
  model: extended_tofts
  bayesian: map
  aif_method: auto
  scale_aif: true
  no_b1: false
  no_t1: false
  save_uncertainty: true
  use_gpu: false

  # Advanced settings
  registration:
    skip_bbr: true

  aif:
    method: auto
    target_peak: 6.0

  processing:
    tsnr_threshold: 0.0
    baseline_frames: 10

# Anatomical configuration
anat:
  brain_extraction:
    tool: hdbet
  denoise: true
  skip_acpc: false

# HD-GLIO configuration
hdglio:
  lesion_dilation: 15.0
  nawm_erosion: 2
  nawm_min_voxels: 500
  device: cuda
```

### Configuration Sections

#### DSC Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `preload_detected` | bool | false | Contrast preload present |
| `dose` | str | "full" | Dose: half, full, double |
| `method` | str | "sSVD" | Deconvolution method |
| `bayesian` | str | "none" | Bayesian mode: none, map, vi, mcmc |
| `mcmc_samples` | int | 500 | MCMC samples if bayesian=mcmc |
| `aif_criteria` | str | "strict" | AIF selection: strict, relaxed |
| `n_aif_candidates` | int | 10 | Number of AIF candidates |
| `pre_bolus` | int | 5 | Pre-bolus baseline frames |
| `save_k2_map` | bool | false | Save K2 leakage map |
| `save_uncertainty` | bool | false | Save uncertainty maps |
| `use_gpu` | bool | false | Use GPU acceleration |

#### DCE Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | "auto" | PK model: tofts, extended_tofts, patlak, 2cxm, auto |
| `bayesian` | str | "none" | Bayesian mode: none, map, mcmc |
| `aif_method` | str | "auto" | AIF method: auto, population |
| `scale_aif` | bool | false | Scale AIF to population reference |
| `no_b1` | bool | false | Skip B1+ mapping |
| `no_t1` | bool | false | Skip T1 mapping |
| `save_uncertainty` | bool | false | Save uncertainty maps |
| `use_gpu` | bool | false | Use GPU acceleration |

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
        sub-01_ses-01_acq-afi_B1plusmap.nii.gz      # B1+ mapping
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
output_dir/
  dataset_description.json
  sub-01/
    ses-01/
      anat/
        sub-01_ses-01_desc-preproc_T1w.nii.gz
        sub-01_ses-01_desc-brain_mask.nii.gz
        sub-01_ses-01_label-WM_probseg.nii.gz
        sub-01_ses-01_label-GM_probseg.nii.gz
        sub-01_ses-01_label-CSF_probseg.nii.gz
        sub-01_ses-01_T1map.nii.gz              # T1 map from VFA
        sub-01_ses-01_B1plusmap.nii.gz          # B1+ map
      perf/
        # DSC outputs
        sub-01_ses-01_desc-dsc_cbf.nii.gz
        sub-01_ses-01_desc-dsc_cbf.json
        sub-01_ses-01_desc-dsc_cbv.nii.gz
        sub-01_ses-01_desc-dsc_mtt.nii.gz
        sub-01_ses-01_desc-dsc_tmax.nii.gz
        sub-01_ses-01_desc-dsc_k2.nii.gz        # K2 leakage map
        sub-01_ses-01_desc-dsc_aif.json         # AIF quality metrics
        # DCE outputs
        sub-01_ses-01_desc-dce_Ktrans.nii.gz
        sub-01_ses-01_desc-dce_ve.nii.gz
        sub-01_ses-01_desc-dce_vp.nii.gz
        sub-01_ses-01_desc-dce_kep.nii.gz
        sub-01_ses-01_desc-dce_aif.json
      tumor/
        sub-01_ses-01_desc-hdglio_dseg.nii.gz   # Tumor segmentation
        sub-01_ses-01_desc-lesion_mask.nii.gz   # Binary lesion mask
        sub-01_ses-01_desc-nawm_mask.nii.gz     # NAWM reference mask
```

### Output File Descriptions

| File Pattern | Description | Units |
|--------------|-------------|-------|
| `*_cbf.nii.gz` | Cerebral blood flow | mL/100g/min |
| `*_cbv.nii.gz` | Cerebral blood volume | mL/100g |
| `*_mtt.nii.gz` | Mean transit time | seconds |
| `*_tmax.nii.gz` | Time to maximum | seconds |
| `*_k2.nii.gz` | K2 leakage parameter | 1/s |
| `*_Ktrans.nii.gz` | Volume transfer constant | 1/min |
| `*_ve.nii.gz` | Extravascular extracellular volume | fraction |
| `*_vp.nii.gz` | Plasma volume fraction | fraction |
| `*_kep.nii.gz` | Efflux rate constant | 1/min |

## License

PMRIPrep is licensed under the Apache License 2.0.

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
