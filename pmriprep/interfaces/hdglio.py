"""HD-GLIO tumor segmentation interfaces for PMRIPrep.

This module implements Nipype interfaces for HD-GLIO deep learning tumor
segmentation preprocessing and inference. HD-GLIO requires minimal preprocessing:
brain extraction only, NO bias correction, NO intensity normalization.

Interfaces
----------
HDGLIOPrep
    Minimal preprocessing (brain extraction only) for HD-GLIO input
HDGLIOSegment
    Deep learning tumor segmentation using HD-GLIO
"""

import platform
from pathlib import Path
from typing import Any, Literal, Optional

from nipype.interfaces.base import (  # type: ignore[import-untyped]
    BaseInterface,
    BaseInterfaceInputSpec,
    CommandLine,
    CommandLineInputSpec,
    File,
    TraitedSpec,
    traits,
)


class HDGLIOPrepInputSpec(BaseInterfaceInputSpec):
    """Input specification for HDGLIOPrep interface."""

    t1w_pre = File(
        exists=True,
        mandatory=True,
        desc="T1-weighted pre-contrast image",
    )
    t1w_postcon = File(
        exists=True,
        mandatory=True,
        desc="T1-weighted post-contrast (ce-GAD) image",
    )
    t2w = File(
        exists=True,
        mandatory=True,
        desc="T2-weighted image",
    )
    flair = File(
        exists=True,
        mandatory=True,
        desc="FLAIR image",
    )
    brain_mask = File(
        exists=True,
        mandatory=True,
        desc="Brain mask from anatomical preprocessing (for brain extraction)",
    )
    output_dir = traits.Directory(
        desc="Output directory for prepped images (default: current directory)",
    )
    check_dimensions = traits.Bool(
        True,
        usedefault=True,
        desc="Verify all images have matching dimensions (default: True)",
    )


class HDGLIOPrepOutputSpec(TraitedSpec):
    """Output specification for HDGLIOPrep interface."""

    t1w_pre_prepped = File(exists=True, desc="Brain-extracted T1w pre-contrast image")
    t1w_postcon_prepped = File(exists=True, desc="Brain-extracted T1w post-contrast image")
    t2w_prepped = File(exists=True, desc="Brain-extracted T2w image")
    flair_prepped = File(exists=True, desc="Brain-extracted FLAIR image")
    dimensions_matched = traits.Bool(desc="Whether all image dimensions matched")
    qc_metrics = traits.Dict(desc="QC metrics dictionary")


class HDGLIOPrep(BaseInterface):
    """Minimal preprocessing for HD-GLIO tumor segmentation.

    This interface prepares input images for HD-GLIO deep learning segmentation.
    HD-GLIO requires minimal preprocessing to preserve image characteristics
    that the neural network was trained on:

    - Brain extraction: Applied using provided brain mask
    - NO bias correction: HD-GLIO is robust to B1 inhomogeneity
    - NO intensity normalization: Neural network handles normalization internally

    All 4 modalities (T1w-pre, T1w-ce-GAD, T2w, FLAIR) must be present and
    have matching dimensions for HD-GLIO inference.

    Parameters
    ----------
    t1w_pre : str
        Path to T1-weighted pre-contrast image
    t1w_postcon : str
        Path to T1-weighted post-contrast (ce-GAD) image
    t2w : str
        Path to T2-weighted image
    flair : str
        Path to FLAIR image
    brain_mask : str
        Path to brain mask from anatomical preprocessing
    check_dimensions : bool
        If True (default), verify all images have matching dimensions

    Returns
    -------
    t1w_pre_prepped : str
        Path to brain-extracted T1w pre-contrast image
    t1w_postcon_prepped : str
        Path to brain-extracted T1w post-contrast image
    t2w_prepped : str
        Path to brain-extracted T2w image
    flair_prepped : str
        Path to brain-extracted FLAIR image
    dimensions_matched : bool
        Whether all image dimensions matched
    qc_metrics : dict
        QC metrics including dimension check results

    Examples
    --------
    >>> prep = HDGLIOPrep()
    >>> prep.inputs.t1w_pre = 'T1w_pre.nii.gz'
    >>> prep.inputs.t1w_postcon = 'T1w_ce-GAD.nii.gz'
    >>> prep.inputs.t2w = 'T2w.nii.gz'
    >>> prep.inputs.flair = 'FLAIR.nii.gz'
    >>> prep.inputs.brain_mask = 'brain_mask.nii.gz'
    >>> result = prep.run()  # doctest: +SKIP

    Notes
    -----
    HD-GLIO (High-Definition GlioMa) requirements:

    1. **All 4 modalities required**: T1w-pre, T1w-ce-GAD, T2w, FLAIR
    2. **Brain extraction**: Images should be skull-stripped
    3. **NO bias correction**: The network is trained on uncorrected images
    4. **NO intensity normalization**: Internal normalization handles this
    5. **Matching dimensions**: All images must have same shape

    The brain mask is applied via multiplication to preserve original
    voxel intensities within the brain region. Voxels outside the brain
    mask are set to zero.

    This interface only prepares images; actual segmentation is performed
    by the HDGLIOSegment interface.

    See Also
    --------
    HDGLIOSegment : HD-GLIO deep learning segmentation interface
    HD-GLIO: https://github.com/MIC-DKFZ/HD-GLIO
    """

    input_spec = HDGLIOPrepInputSpec
    output_spec = HDGLIOPrepOutputSpec

    def _get_output_dir(self) -> Path:
        """Get output directory for prepped images."""
        if self.inputs.output_dir:
            return Path(self.inputs.output_dir)
        return Path.cwd()

    def _gen_output_filename(self, input_file: str, suffix: str = "_hdglio") -> str:
        """Generate output filename for prepped image."""
        in_path = Path(input_file)
        out_dir = self._get_output_dir()

        # Handle .nii.gz extension
        if in_path.name.endswith(".nii.gz"):
            base = in_path.name[:-7]
            ext = ".nii.gz"
        elif in_path.name.endswith(".nii"):
            base = in_path.stem
            ext = ".nii"
        else:
            base = in_path.stem
            ext = in_path.suffix

        return str(out_dir / f"{base}{suffix}{ext}")

    def _check_dimensions(
        self,
        images: dict[str, str],
    ) -> tuple[bool, dict[str, Any]]:
        """Check if all images have matching dimensions.

        Parameters
        ----------
        images : dict
            Dictionary mapping modality name to file path

        Returns
        -------
        matched : bool
            True if all dimensions match
        details : dict
            Dictionary with dimension details for each image
        """
        import nibabel as nib  # type: ignore[import-untyped]
        import numpy as np

        details: dict[str, Any] = {}
        shapes: list[tuple[int, ...]] = []
        affines: list[Any] = []

        for name, filepath in images.items():
            img = nib.load(filepath)
            shape = img.shape[:3]  # type: ignore[attr-defined]
            affine = img.affine  # type: ignore[attr-defined]
            zooms = img.header.get_zooms()[:3]  # type: ignore[attr-defined]

            details[name] = {
                "shape": shape,
                "zooms": tuple(float(z) for z in zooms),
                "file": filepath,
            }
            shapes.append(shape)
            affines.append(affine)

        # Check if all shapes match
        reference_shape = shapes[0]
        shapes_match = all(s == reference_shape for s in shapes)

        # Check if affines are similar (within tolerance for floating point)
        reference_affine = affines[0]
        affines_match = all(
            np.allclose(a, reference_affine, atol=1e-4) for a in affines
        )

        matched = shapes_match and affines_match
        details["shapes_match"] = shapes_match
        details["affines_match"] = affines_match
        details["all_matched"] = matched

        return matched, details

    def _apply_brain_mask(
        self,
        input_file: str,
        mask_file: str,
        output_file: str,
    ) -> None:
        """Apply brain mask to image (multiply image by mask).

        Parameters
        ----------
        input_file : str
            Path to input image
        mask_file : str
            Path to brain mask
        output_file : str
            Path to output brain-extracted image
        """
        import nibabel as nib  # type: ignore[import-untyped]
        import numpy as np

        # Load image and mask
        img = nib.load(input_file)
        mask_img = nib.load(mask_file)

        data = img.get_fdata()  # type: ignore[attr-defined]
        mask_data = mask_img.get_fdata()  # type: ignore[attr-defined]

        # Binarize mask (in case it's a probability map)
        mask_binary = mask_data > 0.5

        # Apply mask
        masked_data = data * mask_binary

        # Save output
        out_img = nib.Nifti1Image(  # type: ignore[attr-defined]
            masked_data.astype(np.float32),
            img.affine,  # type: ignore[attr-defined]
            img.header,  # type: ignore[attr-defined]
        )
        nib.save(out_img, output_file)

    def _run_interface(self, runtime: Any) -> Any:
        """Execute the interface."""
        import nibabel as nib  # type: ignore[import-untyped]

        # Collect input images
        images = {
            "t1w_pre": str(self.inputs.t1w_pre),
            "t1w_postcon": str(self.inputs.t1w_postcon),
            "t2w": str(self.inputs.t2w),
            "flair": str(self.inputs.flair),
        }
        mask_file = str(self.inputs.brain_mask)

        # Check dimensions if requested
        qc_metrics: dict[str, Any] = {}
        dimensions_matched = True

        if self.inputs.check_dimensions:
            # Add mask to dimension check
            images_with_mask = {**images, "brain_mask": mask_file}
            dimensions_matched, dim_details = self._check_dimensions(images_with_mask)
            qc_metrics["dimensions"] = dim_details

            if not dimensions_matched:
                raise ValueError(
                    f"Image dimensions do not match. Check QC metrics for details: {dim_details}"
                )

        # Create output directory if needed
        out_dir = self._get_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)

        # Apply brain mask to each image (NO bias correction, NO normalization)
        output_files: dict[str, str] = {}

        for modality, input_file in images.items():
            output_file = self._gen_output_filename(input_file, f"_{modality}_hdglio")
            self._apply_brain_mask(input_file, mask_file, output_file)
            output_files[modality] = output_file

        # Store results for _list_outputs
        self._results = {
            "t1w_pre_prepped": output_files["t1w_pre"],
            "t1w_postcon_prepped": output_files["t1w_postcon"],
            "t2w_prepped": output_files["t2w"],
            "flair_prepped": output_files["flair"],
            "dimensions_matched": dimensions_matched,
            "qc_metrics": qc_metrics,
        }

        return runtime

    def _list_outputs(self) -> dict[str, Any]:  # type: ignore[override]
        """List expected outputs."""
        outputs: dict[str, Any] = self.output_spec().get()  # type: ignore[misc]

        if hasattr(self, "_results"):
            outputs.update(self._results)
        else:
            # Generate expected output filenames before run
            outputs["t1w_pre_prepped"] = self._gen_output_filename(
                str(self.inputs.t1w_pre), "_t1w_pre_hdglio"
            )
            outputs["t1w_postcon_prepped"] = self._gen_output_filename(
                str(self.inputs.t1w_postcon), "_t1w_postcon_hdglio"
            )
            outputs["t2w_prepped"] = self._gen_output_filename(
                str(self.inputs.t2w), "_t2w_hdglio"
            )
            outputs["flair_prepped"] = self._gen_output_filename(
                str(self.inputs.flair), "_flair_hdglio"
            )
            outputs["dimensions_matched"] = True
            outputs["qc_metrics"] = {}

        return outputs


# --------------------------------------------------------------------------
# HDGLIOSegment: Deep learning tumor segmentation
# --------------------------------------------------------------------------


class HDGLIOSegmentInputSpec(CommandLineInputSpec):
    """Input specification for HDGLIOSegment interface."""

    t1w_pre = File(
        exists=True,
        mandatory=True,
        argstr="-t1 %s",
        position=1,
        desc="Brain-extracted T1-weighted pre-contrast image",
    )
    t1w_postcon = File(
        exists=True,
        mandatory=True,
        argstr="-t1c %s",
        position=2,
        desc="Brain-extracted T1-weighted post-contrast (ce-GAD) image",
    )
    t2w = File(
        exists=True,
        mandatory=True,
        argstr="-t2 %s",
        position=3,
        desc="Brain-extracted T2-weighted image",
    )
    flair = File(
        exists=True,
        mandatory=True,
        argstr="-flair %s",
        position=4,
        desc="Brain-extracted FLAIR image",
    )
    output_dir = traits.Directory(
        argstr="-o %s",
        position=5,
        desc="Output directory for segmentation results",
        genfile=True,
    )
    device = traits.Enum(
        "cuda",
        "mps",
        "cpu",
        usedefault=True,
        desc="Device for inference: cuda (NVIDIA GPU), mps (Apple Silicon), or cpu",
    )
    conda_env = traits.Str(
        desc=(
            "Conda environment name to run hd_glio_predict in via "
            "'conda run -n <env>'. Required when HD-GLIO is installed in a "
            "separate environment due to dependency conflicts (e.g., "
            "hdglio-mps for Apple Silicon)."
        ),
    )
    validate_output = traits.Bool(
        True,
        usedefault=True,
        desc="Validate that output contains tumor voxels (default: True)",
    )


class HDGLIOSegmentOutputSpec(TraitedSpec):
    """Output specification for HDGLIOSegment interface."""

    segmentation = File(exists=True, desc="HD-GLIO segmentation output NIfTI file")
    has_tumor = traits.Bool(desc="Whether segmentation contains tumor voxels")
    tumor_volume_ml = traits.Float(desc="Total tumor volume in milliliters")
    enhancing_volume_ml = traits.Float(desc="Enhancing tumor volume (label 2) in milliliters")
    non_enhancing_volume_ml = traits.Float(desc="Non-enhancing tumor volume (label 1) in milliliters")
    qc_metrics = traits.Dict(desc="QC metrics dictionary")


class HDGLIOSegment(BaseInterface):
    """HD-GLIO deep learning tumor segmentation interface.

    This interface wraps the HD-GLIO (High-Definition GlioMa) deep learning
    model for automated brain tumor segmentation. HD-GLIO produces multi-class
    segmentation masks with the following labels:

    - 0: Background (non-tumor)
    - 1: Non-enhancing tumor (necrosis, non-enhancing core)
    - 2: Enhancing tumor (contrast-enhancing tumor)

    Parameters
    ----------
    t1w_pre : str
        Path to brain-extracted T1-weighted pre-contrast image
    t1w_postcon : str
        Path to brain-extracted T1-weighted post-contrast (ce-GAD) image
    t2w : str
        Path to brain-extracted T2-weighted image
    flair : str
        Path to brain-extracted FLAIR image
    output_dir : str, optional
        Output directory for segmentation results
    device : {'cuda', 'mps', 'cpu'}
        Device for inference:
        - 'cuda': NVIDIA GPU (requires CUDA and appropriate drivers)
        - 'mps': Apple Silicon GPU (macOS with M1/M2/M3 chips)
        - 'cpu': CPU only (slowest, but always available)
    validate_output : bool
        If True (default), validate that output contains tumor voxels

    Returns
    -------
    segmentation : str
        Path to HD-GLIO segmentation output NIfTI file
    has_tumor : bool
        Whether segmentation contains tumor voxels (labels 1 or 2)
    tumor_volume_ml : float
        Total tumor volume in milliliters
    enhancing_volume_ml : float
        Enhancing tumor (label 2) volume in milliliters
    non_enhancing_volume_ml : float
        Non-enhancing tumor (label 1) volume in milliliters
    qc_metrics : dict
        QC metrics including volume statistics and validation results

    Examples
    --------
    >>> segment = HDGLIOSegment()
    >>> segment.inputs.t1w_pre = 'T1w_pre_brain.nii.gz'
    >>> segment.inputs.t1w_postcon = 'T1w_ce-GAD_brain.nii.gz'
    >>> segment.inputs.t2w = 'T2w_brain.nii.gz'
    >>> segment.inputs.flair = 'FLAIR_brain.nii.gz'
    >>> segment.inputs.device = 'cuda'
    >>> result = segment.run()  # doctest: +SKIP

    Notes
    -----
    HD-GLIO requirements:

    1. **Input images**: Must be brain-extracted (skull-stripped)
    2. **All 4 modalities**: T1w-pre, T1w-ce-GAD, T2w, FLAIR required
    3. **Matching dimensions**: All images must have same shape
    4. **NO bias correction**: Images should NOT be bias-corrected
    5. **GPU recommended**: CUDA or MPS for reasonable inference speed

    Device selection:

    - CUDA is fastest if NVIDIA GPU is available
    - MPS works on Apple Silicon (M1/M2/M3) Macs
    - CPU is always available but much slower (~10-30x slower)

    The interface automatically detects Apple Silicon and uses MPS if
    device='mps' is specified. On non-Apple systems, 'mps' falls back to CPU.

    See Also
    --------
    HDGLIOPrep : Preprocessing interface to prepare images for HD-GLIO
    HD-GLIO: https://github.com/MIC-DKFZ/HD-GLIO

    References
    ----------
    .. [1] Isensee F, Kickingereder P, Wick W, Bendszus M, Maier-Hein KH.
           "Brain Tumor Segmentation and Radiomics Survival Prediction:
           Contribution to the BRATS 2017 Challenge." arXiv:1802.10508.
    """

    input_spec = HDGLIOSegmentInputSpec
    output_spec = HDGLIOSegmentOutputSpec

    def _get_output_dir(self) -> Path:
        """Get output directory for segmentation results."""
        if self.inputs.output_dir:
            return Path(self.inputs.output_dir)
        return Path.cwd()

    def _get_device_flag(self) -> str:
        """Get appropriate device flag for hd_glio_predict command.

        Returns
        -------
        str
            Device flag string for the command line

        Notes
        -----
        Device handling:
        - 'cuda': Use NVIDIA GPU (--device cuda)
        - 'mps': Use Apple Silicon GPU, but HD-GLIO uses --device mps
        - 'cpu': Use CPU only (--device cpu)

        On non-Apple systems with 'mps' specified, falls back to 'cpu'.
        """
        device: str = str(self.inputs.device)

        if device == "mps":
            # Check if running on Apple Silicon
            is_apple_silicon = (
                platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")
            )
            if not is_apple_silicon:
                # Fall back to CPU on non-Apple systems
                return "cpu"
            return "mps"

        return device

    def _get_output_file(self) -> Path:
        """Get the output file path for hd_glio_predict.

        hd_glio_predict requires ``-o`` to be a filename ending in
        ``.nii.gz``, not a directory.

        Returns
        -------
        Path
            Full path to the output segmentation file.
        """
        out_dir = self._get_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / "segmentation.nii.gz"

    def _build_command(self) -> str:
        """Build the hd_glio_predict command string.

        When ``conda_env`` is set, the command is wrapped with
        ``conda run -n <env> --no-capture-output`` so that
        hd_glio_predict executes inside the correct environment.

        Returns
        -------
        str
            Full command string for hd_glio_predict

        Notes
        -----
        hd_glio_predict does **not** accept a ``--device`` flag. Device
        selection (cuda/mps/cpu) is controlled by the conda environment
        and PyTorch installation within it. The ``device`` input on this
        interface is retained for metadata/QC purposes only.
        """
        from nipype.interfaces.base import isdefined  # type: ignore[import-untyped]

        out_file = self._get_output_file()

        cmd_parts = [
            "hd_glio_predict",
            f"-t1 {self.inputs.t1w_pre}",
            f"-t1c {self.inputs.t1w_postcon}",
            f"-t2 {self.inputs.t2w}",
            f"-flair {self.inputs.flair}",
            f"-o {out_file}",
        ]

        cmd = " ".join(cmd_parts)

        # Wrap with conda run if a separate environment is specified
        conda_env = self.inputs.conda_env
        if isdefined(conda_env) and conda_env:
            cmd = f"conda run -n {conda_env} --no-capture-output {cmd}"

        return cmd

    def _validate_segmentation(
        self,
        segmentation_file: str,
    ) -> tuple[bool, dict[str, Any]]:
        """Validate segmentation output and compute volume metrics.

        Parameters
        ----------
        segmentation_file : str
            Path to segmentation NIfTI file

        Returns
        -------
        has_tumor : bool
            Whether segmentation contains any tumor voxels
        metrics : dict
            Dictionary with volume metrics and validation info
        """
        import nibabel as nib  # type: ignore[import-untyped]
        import numpy as np

        img = nib.load(segmentation_file)
        data = img.get_fdata()  # type: ignore[attr-defined]
        zooms = img.header.get_zooms()[:3]  # type: ignore[attr-defined]

        # Calculate voxel volume in mm³, convert to mL
        voxel_volume_mm3 = float(np.prod(zooms))
        voxel_volume_ml = voxel_volume_mm3 / 1000.0

        # Count voxels by label
        # Label 0: background, Label 1: non-enhancing, Label 2: enhancing
        non_enhancing_voxels = int(np.sum(data == 1))
        enhancing_voxels = int(np.sum(data == 2))
        total_tumor_voxels = non_enhancing_voxels + enhancing_voxels

        # Compute volumes in mL
        non_enhancing_volume_ml = non_enhancing_voxels * voxel_volume_ml
        enhancing_volume_ml = enhancing_voxels * voxel_volume_ml
        total_volume_ml = total_tumor_voxels * voxel_volume_ml

        has_tumor = total_tumor_voxels > 0

        metrics: dict[str, Any] = {
            "non_enhancing_voxels": non_enhancing_voxels,
            "enhancing_voxels": enhancing_voxels,
            "total_tumor_voxels": total_tumor_voxels,
            "non_enhancing_volume_ml": non_enhancing_volume_ml,
            "enhancing_volume_ml": enhancing_volume_ml,
            "total_tumor_volume_ml": total_volume_ml,
            "voxel_volume_mm3": voxel_volume_mm3,
            "has_tumor": has_tumor,
            "shape": tuple(int(s) for s in data.shape[:3]),
            "unique_labels": sorted([int(u) for u in np.unique(data)]),
        }

        return has_tumor, metrics

    def _find_segmentation_output(self) -> Path:
        """Find the segmentation output file from HD-GLIO.

        First checks the exact output file path passed to ``-o``.
        Falls back to searching the output directory for common patterns
        in case the tool writes to a different name.

        Returns
        -------
        Path
            Path to the segmentation output file

        Raises
        ------
        FileNotFoundError
            If no segmentation output is found
        """
        # Check the known output file path first
        expected = self._get_output_file()
        if expected.exists():
            return expected

        out_dir = self._get_output_dir()

        # HD-GLIO may output under a slightly different name
        patterns = [
            "segmentation.nii.gz",
            "segmentation.nii",
            "*_seg.nii.gz",
            "*_seg.nii",
            "*segmentation*.nii.gz",
            "*segmentation*.nii",
        ]

        for pattern in patterns:
            matches = list(out_dir.glob(pattern))
            if matches:
                # Return the most recently modified match
                return max(matches, key=lambda p: p.stat().st_mtime)

        # If no match found, list what's in the output directory
        existing_files = list(out_dir.glob("*.nii*"))
        if existing_files:
            # Return the first NIfTI file found
            return existing_files[0]

        raise FileNotFoundError(
            f"No segmentation output found in {out_dir}. "
            f"Expected file: {expected}"
        )

    def _run_interface(self, runtime: Any) -> Any:
        """Execute the interface."""
        import subprocess

        # Ensure output directory exists
        out_dir = self._get_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build and run the command
        cmd = self._build_command()

        # Execute HD-GLIO
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"hd_glio_predict failed with return code {result.returncode}.\n"
                f"Command: {cmd}\n"
                f"Stdout: {result.stdout}\n"
                f"Stderr: {result.stderr}"
            )

        # Find the segmentation output file
        segmentation_file = self._find_segmentation_output()

        # Validate output if requested
        qc_metrics: dict[str, Any] = {
            "command": cmd,
            "device": self._get_device_flag(),
        }

        if self.inputs.validate_output:
            has_tumor, validation_metrics = self._validate_segmentation(
                str(segmentation_file)
            )
            qc_metrics["validation"] = validation_metrics

            if not has_tumor:
                # Warning but don't fail - some images may legitimately have no tumor
                qc_metrics["warning"] = "No tumor voxels detected in segmentation"
        else:
            # Still compute basic metrics even without full validation
            has_tumor, validation_metrics = self._validate_segmentation(
                str(segmentation_file)
            )
            qc_metrics["validation"] = validation_metrics

        # Store results
        self._results = {
            "segmentation": str(segmentation_file),
            "has_tumor": has_tumor,
            "tumor_volume_ml": validation_metrics["total_tumor_volume_ml"],
            "enhancing_volume_ml": validation_metrics["enhancing_volume_ml"],
            "non_enhancing_volume_ml": validation_metrics["non_enhancing_volume_ml"],
            "qc_metrics": qc_metrics,
        }

        return runtime

    def _list_outputs(self) -> dict[str, Any]:  # type: ignore[override]
        """List expected outputs."""
        outputs: dict[str, Any] = self.output_spec().get()  # type: ignore[misc]

        if hasattr(self, "_results"):
            outputs.update(self._results)
        else:
            # Generate expected output before run
            out_dir = self._get_output_dir()
            outputs["segmentation"] = str(out_dir / "segmentation.nii.gz")
            outputs["has_tumor"] = True
            outputs["tumor_volume_ml"] = 0.0
            outputs["enhancing_volume_ml"] = 0.0
            outputs["non_enhancing_volume_ml"] = 0.0
            outputs["qc_metrics"] = {}

        return outputs
