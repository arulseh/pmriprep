"""Tool-version provenance: dataset-level GeneratedBy + methods boilerplate.

Covers per-suite GeneratedBy entries, idempotency across datasinks, and
version rendering in the methods boilerplate.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Part A — GeneratedBy tool entries + idempotency
# ---------------------------------------------------------------------------
class TestGeneratedByToolEntries:
    def test_entries_are_per_suite_with_versions(self) -> None:
        from pmriprep.utils.sidecar_builders import build_tool_generatedby_entries

        entries = build_tool_generatedby_entries()
        names = [e["Name"] for e in entries]
        assert names == ["FSL", "ANTs", "Nipype", "HD-BET", "HD-GLIO"]
        # HD-BET/HD-GLIO carry the Dockerfile-pinned fallback version.
        by_name = {e["Name"]: e for e in entries}
        assert by_name["HD-BET"]["Version"] == "2.0.1"
        assert by_name["HD-GLIO"]["Version"] == "2.0"
        # Never the literal "unknown".
        assert all(e.get("Version") != "unknown" for e in entries)
        # Per-suite: binaries named in Description, not as separate entries.
        assert "FAST" in by_name["FSL"]["Description"]
        assert "N4BiasFieldCorrection" in by_name["ANTs"]["Description"]

    def test_write_dataset_description_idempotent_across_4_calls(
        self, tmp_path: Path,
    ) -> None:
        from pmriprep.utils.outputs import write_dataset_description

        pdir = tmp_path / "pmriprep"
        pdir.mkdir()
        # A pre-existing foreign entry from a co-resident NiPreps tool.
        (pdir / "dataset_description.json").write_text(
            json.dumps({"GeneratedBy": [{"Name": "aslprep", "Version": "0.7"}]})
        )
        for _ in range(4):  # the 4 datasinks (anat/dsc/dce/tumor-seg)
            write_dataset_description(pdir, str(tmp_path))

        gb = json.loads((pdir / "dataset_description.json").read_text())["GeneratedBy"]
        names = [e["Name"] for e in gb]
        # Exactly one of each — no accumulation across calls.
        assert all(c == 1 for c in Counter(names).values()), Counter(names)
        assert "pmriprep" in names and "FSL" in names and "HD-GLIO" in names
        # Foreign entry preserved.
        assert "aslprep" in names

    def test_normalize_tool_version(self) -> None:
        from pmriprep.utils.sidecar_builders import _normalize_tool_version

        assert _normalize_tool_version("fsl-6.0.7.1") == "6.0.7.1"
        assert _normalize_tool_version("ANTs Version: 2.6.5") == "2.6.5"
        assert _normalize_tool_version("unknown") == ""
        assert _normalize_tool_version("") == ""

    def test_hd_tool_versions_fall_back_to_pin(self) -> None:
        from pmriprep.utils.sidecar_builders import (
            get_hdbet_version,
            get_hdglio_version,
        )

        # HD-BET/HD-GLIO aren't installed in the test env → PackageNotFoundError
        # → Dockerfile-pinned fallback.
        assert get_hdbet_version() == "2.0.1"
        assert get_hdglio_version() == "2.0"

    def test_version_probe_does_not_import_package(self) -> None:
        """Regression: version probing must NOT import HD_BET/hd_glio.

        HD-BET prints a citation banner to stdout on import; reading the version
        from package metadata must avoid executing the package ``__init__``.
        """
        import sys

        from pmriprep.utils.sidecar_builders import (
            build_tool_generatedby_entries,
            get_hdbet_version,
            get_hdglio_version,
        )

        for mod in ("HD_BET", "hd_glio"):
            sys.modules.pop(mod, None)
        get_hdbet_version()
        get_hdglio_version()
        build_tool_generatedby_entries()
        assert "HD_BET" not in sys.modules
        assert "hd_glio" not in sys.modules


# ---------------------------------------------------------------------------
# Part B — boilerplate version rendering
# ---------------------------------------------------------------------------
class TestBoilerplateVersionRendering:
    def _data(self):
        ran = {"anat": True, "tumor_seg": True, "dsc": True, "dce": True}
        d = {
            "anat": {"qc_metrics": {"brain_extraction": {"tool": "hdbet"},
                                    "tissue_segmentation": True}},
            "dsc": {"qc_metrics": {"preprocessing": {"motion_applied": True, "motion_tool": "ANTs"},
                                   "deconvolution": {"method": "oSVD"}}},
            "dce": {"qc_metrics": {"preprocessing": {"motion_applied": True, "motion_tool": "ANTs"},
                                   "t1_mapping": {"method": "nlls", "b1_correction": "afi"}}},
        }
        return ran, d

    def test_versions_rendered_on_software_only(self) -> None:
        from pmriprep.utils.report_utils import _generate_boilerplate_text

        ran, d = self._data()
        vers = {"FSL": "6.0.7.1", "ANTs": "2.6.5", "Nipype": "1.10.0",
                "HD-BET": "2.0.1", "HD-GLIO": "2.0"}
        txt = _generate_boilerplate_text("0.78.0", ran, d, versions=vers)
        assert "a Nipype v1.10.0-based" in txt
        assert "N4BiasFieldCorrection (v2.6.5)" in txt
        assert "FAST (FSL v6.0.7.1)" in txt
        assert "ANTs (v2.6.5)" in txt
        assert "HD-BET (v2.0.1)" in txt
        assert "HD-GLIO (v2.0)" in txt
        d["anat"]["qc_metrics"]["brain_extraction"]["tool"] = "ants"
        ants = _generate_boilerplate_text("0.78.0", ran, d, versions=vers)
        assert "ANTs brain extraction (v2.6.5) [@tustison_2014; @tustison_2021]" in ants
        # Method citations get NO version.
        assert "Tofts (" not in txt and "Patlak (" not in txt

    def test_unknown_versions_are_byte_identical(self) -> None:
        from pmriprep.utils.report_utils import _generate_boilerplate_text

        ran, d = self._data()
        none = _generate_boilerplate_text("0.78.0", ran, d, versions=None)
        empty = _generate_boilerplate_text("0.78.0", ran, d, versions={})
        assert none == empty
        # No version tokens leaked in.
        for tok in ("(FSL ", "(ANTs ", "(2.0.1)", "Nipype 1."):
            assert tok not in none

    def test_motion_and_anatomical_tools_use_their_own_versions(self) -> None:
        from pmriprep.utils.report_utils import _inject_tool_versions

        md = "using ANTs [@tustison_2021; @avants_2014] and using FLIRT [@jenkinson_2002]."
        out = _inject_tool_versions(md, {"FSL": "6.0", "ANTs": "2.6.5"})
        assert "ANTs (v2.6.5) [@tustison_2021; @avants_2014]" in out
        assert "using FLIRT (FSL v6.0) [@jenkinson_2002]" in out

    def test_load_tool_versions_reads_dataset_description(
        self, tmp_path: Path,
    ) -> None:
        from pmriprep.utils.report_utils import _load_tool_versions

        pdir = tmp_path / "pmriprep"
        ses = pdir / "sub-01" / "ses-01"
        ses.mkdir(parents=True)
        (pdir / "dataset_description.json").write_text(json.dumps({
            "GeneratedBy": [
                {"Name": "pmriprep", "Version": "0.78.0"},
                {"Name": "FSL", "Version": "6.0.7.1"},
                {"Name": "ANTs", "Version": "2.6.5"},
            ]
        }))
        versions = _load_tool_versions(ses, "01")
        assert versions["FSL"] == "6.0.7.1"
        assert versions["ANTs"] == "2.6.5"
        assert "pmriprep" not in versions  # the pipeline itself is handled separately
