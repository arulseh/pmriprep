"""Unit tests for report utility functions.

Tests for QC metric classification helpers used in HTML report generation.
"""

import pytest

from pmriprep.utils.report_utils import (
    _classify_b1,
    _classify_metric,
    _classify_metric_inverse,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(25.0, "pass", id="above-warn"),
        pytest.param(20.0, "pass", id="at-warn"),
        pytest.param(10.0, "warn", id="between"),
        pytest.param(5.0, "warn", id="at-minimum"),
        pytest.param(3.0, "poor", id="below-minimum"),
        pytest.param(None, "na", id="missing"),
        pytest.param(float("nan"), "na", id="nan"),
    ],
)
def test_classify_metric(value: float | None, expected: str) -> None:
    """Higher-is-better metrics retain threshold and missing-value semantics."""
    assert _classify_metric(value, 20.0, 5.0) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(1.0, "pass", id="within-range"),
        pytest.param(0.7, "pass", id="at-low"),
        pytest.param(1.3, "pass", id="at-high"),
        pytest.param(0.6, "warn", id="below-low"),
        pytest.param(1.4, "warn", id="above-high"),
        pytest.param(None, "na", id="missing"),
        pytest.param(float("nan"), "na", id="nan"),
    ],
)
def test_classify_b1(value: float | None, expected: str) -> None:
    """B1 classification retains inclusive bounds and missing-value semantics."""
    assert _classify_b1(value, 0.7, 1.3) == expected


@pytest.mark.parametrize(
    ("value", "warn", "poor", "expected"),
    [
        pytest.param(3.0, 5.0, 10.0, "pass", id="below-warn"),
        pytest.param(5.0, 5.0, 10.0, "warn", id="at-warn"),
        pytest.param(7.0, 5.0, 10.0, "warn", id="between"),
        pytest.param(10.0, 5.0, 10.0, "poor", id="at-poor"),
        pytest.param(15.0, 5.0, 10.0, "poor", id="above-poor"),
        pytest.param(None, 5.0, 10.0, "na", id="missing"),
        pytest.param(float("nan"), 5.0, 10.0, "na", id="nan"),
        pytest.param(0.05, 0.10, float("inf"), "pass", id="warn-cap-pass"),
        pytest.param(0.50, 0.10, float("inf"), "warn", id="warn-cap-warn"),
        pytest.param(1.00, 0.10, float("inf"), "warn", id="warn-cap-high"),
        pytest.param(float("nan"), 0.10, float("inf"), "na", id="warn-cap-nan"),
    ],
)
def test_classify_metric_inverse(
    value: float | None,
    warn: float,
    poor: float,
    expected: str,
) -> None:
    """Lower-is-better metrics retain boundaries and the warning-only mode."""
    assert _classify_metric_inverse(value, warn, poor) == expected


# ---------------------------------------------------------------------------
# BIDS run-entity handling in report sidecar discovery
# ---------------------------------------------------------------------------


class TestRunEntityHandling:
    """Anat/tumor_seg sidecar lookups must accept either the bare-prefix
    BIDS form (``sub-XX_ses-YY_space-...``) or the form with the optional
    ``_run-N`` entity propagated from raw input
    (``sub-XX_ses-YY_run-1_space-...``). NiPreps convention propagates
    raw run entities into derivatives verbatim.
    """

    PREFIX = "sub-09_ses-01"

    def _make_anat(self, tmp_path, *names: str) -> "Path":  # noqa: F821
        anat = tmp_path / "anat"
        anat.mkdir(parents=True, exist_ok=True)
        for n in names:
            (anat / n).write_text("{}")
        return tmp_path

    def test_anchor_resolves_with_run_entity(self, tmp_path) -> None:
        from pmriprep.utils.report_utils import _resolve_bids_anchor
        ses = self._make_anat(
            tmp_path,
            f"{self.PREFIX}_run-1_space-T1w_desc-preproc_T1w.json",
        )
        result = _resolve_bids_anchor(
            ses, f"anat/{self.PREFIX}*_space-T1w_desc-preproc_T1w.json",
        )
        assert result.exists()
        assert "_run-1_" in result.name

    def test_anchor_resolves_without_run_entity(self, tmp_path) -> None:
        """Bare prefix (no `_run-`) must also resolve."""
        from pmriprep.utils.report_utils import _resolve_bids_anchor
        ses = self._make_anat(
            tmp_path,
            f"{self.PREFIX}_space-T1w_desc-preproc_T1w.json",
        )
        result = _resolve_bids_anchor(
            ses, f"anat/{self.PREFIX}*_space-T1w_desc-preproc_T1w.json",
        )
        assert result.exists()
        assert "_run-" not in result.name

    def test_anchor_excludes_contrast_variant_when_native_present(
        self, tmp_path,
    ) -> None:
        """When both `_ce-GAD_run-1_*preproc_T1w` and `_run-1_*preproc_T1w`
        coexist, the helper MUST return the native (non-postcon) variant.
        Without the `_ce-` exclusion, sorted-ascending picks the postcon
        (alphabetically `_ce-` < `_run-`) — which would silently render
        the wrong sidecar's QC.
        """
        from pmriprep.utils.report_utils import _resolve_bids_anchor
        ses = self._make_anat(
            tmp_path,
            f"{self.PREFIX}_ce-GAD_run-1_space-T1w_desc-preproc_T1w.json",
            f"{self.PREFIX}_run-1_space-T1w_desc-preproc_T1w.json",
        )
        result = _resolve_bids_anchor(
            ses, f"anat/{self.PREFIX}*_space-T1w_desc-preproc_T1w.json",
        )
        assert result.exists()
        assert "_ce-" not in result.name
        assert "_run-1_" in result.name

    def test_anchor_falls_back_to_contrast_variant_when_only_postcon(
        self, tmp_path,
    ) -> None:
        """Postcon-only acquisitions: when the only anat sidecar is the
        `_ce-GAD_*preproc_T1w` variant, the helper MUST fall back to it
        rather than returning a non-existent bare path (which would drop
        the anat section and its HD-BET citation from the report).
        """
        from pmriprep.utils.report_utils import _resolve_bids_anchor
        ses = self._make_anat(
            tmp_path,
            f"{self.PREFIX}_ce-GAD_run-1_space-T1w_desc-preproc_T1w.json",
        )
        result = _resolve_bids_anchor(
            ses, f"anat/{self.PREFIX}*_space-T1w_desc-preproc_T1w.json",
        )
        assert result.exists()
        assert "_ce-GAD_" in result.name

    def test_anchor_lowest_run_when_multiple(self, tmp_path) -> None:
        """Multiple legitimate runs (`_run-1` and `_run-2`) — return run-1."""
        from pmriprep.utils.report_utils import _resolve_bids_anchor
        ses = self._make_anat(
            tmp_path,
            f"{self.PREFIX}_run-1_space-T1w_desc-preproc_T1w.json",
            f"{self.PREFIX}_run-2_space-T1w_desc-preproc_T1w.json",
        )
        result = _resolve_bids_anchor(
            ses, f"anat/{self.PREFIX}*_space-T1w_desc-preproc_T1w.json",
        )
        assert "_run-1_" in result.name

    def test_anat_brain_extraction_with_run_entity(self, tmp_path) -> None:
        """`_anat_nested_from_sidecars` must surface BrainExtraction*
        fields from the brain mask sidecar even when the filename
        carries `_run-1`."""
        import json as _json
        from pmriprep.utils.report_utils import _anat_nested_from_sidecars
        anat = tmp_path / "anat"
        anat.mkdir(parents=True)
        (anat / f"{self.PREFIX}_run-1_space-T1w_desc-brain_mask.json").write_text(
            _json.dumps({
                "BrainExtractionTool": "HD-BET",
                "BrainExtractionVolume": 1234.94,
                "BrainExtractionVolumeMm3": 1234940.0,
            })
        )
        out = _anat_nested_from_sidecars(tmp_path, self.PREFIX, anchor={})
        assert "brain_extraction" in out
        be = out["brain_extraction"]
        assert be["tool"] == "hdbet"
        assert be["brain_volume_ml"] == 1234.94

    def test_anat_tissue_segmentation_with_run_entity(self, tmp_path) -> None:
        """Probseg sidecars with `_run-1` must populate
        `tissue_segmentation.{WM,GM,CSF}`."""
        import json as _json
        from pmriprep.utils.report_utils import _anat_nested_from_sidecars
        anat = tmp_path / "anat"
        anat.mkdir(parents=True)
        for tissue in ("WM", "GM", "CSF"):
            (
                anat / f"{self.PREFIX}_run-1_space-T1w_label-{tissue}_probseg.json"
            ).write_text(_json.dumps({"TissueType": tissue}))
        out = _anat_nested_from_sidecars(tmp_path, self.PREFIX, anchor={})
        assert "tissue_segmentation" in out
        ts = out["tissue_segmentation"]
        assert ts == {"WM": {}, "GM": {}, "CSF": {}}

    def test_anat_coregistration_uses_structured_flag(self, tmp_path) -> None:
        """A real coregistered derivative activates the methods clause."""
        import json as _json
        from pmriprep.utils.report_utils import _anat_nested_from_sidecars

        anat = tmp_path / "anat"
        anat.mkdir(parents=True)
        (anat / f"{self.PREFIX}_run-1_desc-preproc_T2w.json").write_text(
            _json.dumps({
                "CoregistrationApplied": True,
                "ProcessingSteps": [
                    "Reorientation to RAS",
                    "Rigidly aligned to the anatomical reference",
                ],
            })
        )

        out = _anat_nested_from_sidecars(tmp_path, self.PREFIX, anchor={})

        assert out["coregistration"]["secondary_modalities"] is True

    def test_load_pipeline_sidecars_anat_native_over_postcon(
        self, tmp_path,
    ) -> None:
        """Integration: a session with BOTH native and postcon T1w
        preproc sidecars + a native brain mask must pick the native
        anchor and surface the native brain extraction QC."""
        import json as _json
        from pmriprep.utils.report_utils import _load_pipeline_sidecars
        anat = tmp_path / "anat"
        anat.mkdir(parents=True)
        # Native + postcon preproc anchor sidecars
        (anat / f"{self.PREFIX}_run-1_space-T1w_desc-preproc_T1w.json").write_text(
            _json.dumps({"PreprocessingDenoising": True})
        )
        (
            anat / f"{self.PREFIX}_ce-GAD_run-1_space-T1w_desc-preproc_T1w.json"
        ).write_text(_json.dumps({"PreprocessingDenoising": False}))
        # Native brain mask with QC
        (anat / f"{self.PREFIX}_run-1_space-T1w_desc-brain_mask.json").write_text(
            _json.dumps({
                "BrainExtractionTool": "HD-BET",
                "BrainExtractionVolume": 1234.94,
            })
        )
        out = _load_pipeline_sidecars(tmp_path, self.PREFIX, "anat")
        # Anchor resolved to native → preprocessing.denoising == True
        assert out.get("preprocessing", {}).get("denoising") is True
        # Brain extraction surfaced
        assert out.get("brain_extraction", {}).get("brain_volume_ml") == 1234.94


# ---------------------------------------------------------------------------
# DSC loader: reads PipelineQC from anchor sidecar
# ---------------------------------------------------------------------------


class TestDscLoader:
    PREFIX = "sub-09_ses-01"

    def _make_perf(self, tmp_path) -> "Path":  # noqa: F821
        perf = tmp_path / "perf"
        perf.mkdir(parents=True, exist_ok=True)
        return tmp_path

    def test_dsc_loader_reads_pipeline_qc(self, tmp_path) -> None:
        import json as _json
        from pmriprep.utils.report_utils import _load_pipeline_sidecars
        ses = self._make_perf(tmp_path)
        (ses / "perf" / f"{self.PREFIX}_desc-dscPreproc_dsc.json").write_text(_json.dumps({
            "PipelineQC": {
                "preprocessing": {"motion_status": "GOOD"},
                "preload": {"percent_leaky_voxels": 5},
                "aif_detection": {"detection_method": "atlas_relaxed"},
                "deconvolution": {"method": "oSVD", "qc_voxel": "[1,2,3]"},
                "registration": {"registration_method": "flirt"},
                "normalization_factor": 1.234,
                "config": {"aif_source": "auto"},
            },
        }))
        out = _load_pipeline_sidecars(ses, self.PREFIX, "dsc")
        assert out.get("preprocessing", {}).get("motion_status") == "GOOD"
        assert out.get("preload", {}).get("percent_leaky_voxels") == 5
        assert out.get("aif_detection", {}).get("detection_method") == "atlas_relaxed"
        assert out.get("deconvolution", {}).get("method") == "oSVD"
        assert out.get("normalization_factor") == 1.234

    def test_perfusion_loaders_read_entity_preserving_anchors(self, tmp_path) -> None:
        import json as _json
        from pmriprep.utils.report_utils import _load_pipeline_sidecars
        ses = self._make_perf(tmp_path)
        anchor = ses / "perf" / (
            f"{self.PREFIX}_acq-gre_run-2_desc-dscPreproc_dsc.json"
        )
        anchor.write_text(_json.dumps({
            "PipelineQC": {"preprocessing": {"motion_status": "GOOD"}},
        }))

        out = _load_pipeline_sidecars(ses, self.PREFIX, "dsc")

        assert out["preprocessing"]["motion_status"] == "GOOD"

        dce_anchor = ses / "perf" / (
            f"{self.PREFIX}_acq-vfa_run-2_desc-dcePreproc_dce.json"
        )
        dce_anchor.write_text(_json.dumps({
            "PipelineQC": {"preprocessing": {"motion_status": "ACCEPTABLE"}},
        }))

        dce_out = _load_pipeline_sidecars(ses, self.PREFIX, "dce")

        assert dce_out["preprocessing"]["motion_status"] == "ACCEPTABLE"

    def test_dsc_loader_empty_pipeline_qc(self, tmp_path) -> None:
        import json as _json
        from pmriprep.utils.report_utils import _load_pipeline_sidecars
        ses = self._make_perf(tmp_path)
        (ses / "perf" / f"{self.PREFIX}_desc-dscPreproc_dsc.json").write_text(_json.dumps({}))
        out = _load_pipeline_sidecars(ses, self.PREFIX, "dsc")
        assert "preprocessing" not in out
        assert "deconvolution" not in out
