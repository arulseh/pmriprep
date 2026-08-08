"""HTML report integration: rendered HTML carries expected sections.

Section-presence test (NOT byte-equality) — the aggregator
reads canonical per-image sidecars rather than ``desc-*QC_quality.json``
files. This test wires up a fake session tree, runs
``generate_unified_report``, and confirms key HTML banners (overall
recommendation, warnings, per-pipeline section headings) render.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from pmriprep.data import get_boilerplate_bib
from pmriprep.utils import report_utils as ru
from pmriprep.utils.report_utils import generate_unified_report


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def test_unified_report_renders_recommendation_and_warnings(tmp_path: Path) -> None:
    """End-to-end render carries recommendation banner, warnings, and section headings."""
    ses = tmp_path / "pmriprep" / "sub-01" / "ses-0001"
    prefix = "sub-01_ses-0001"

    # Anat anchor with two warnings
    _write(
        ses / "anat" / f"{prefix}_space-T1w_desc-preproc_T1w.json",
        {
            "QCWarnings": [
                "Tissue segmentation (WM/GM/CSF probability maps) not available",
                "FLAIR not available",
            ],
        },
    )
    _write(
        ses / "perf" / f"{prefix}_desc-dscPreproc_dsc.json",
        {
            "QCRecommendationAction": "review",
            "QCRecommendationReason": "AIF detection marginal",
            "PipelineQC": {
                "aif_detection": {"detection_method": "atlas_relaxed"},
                "preload": {},
                "preprocessing": {"fd_mean": 0.1, "fd_max": 0.4,
                                  "n_volumes": 60, "tr_ms": 1500.0},
            },
        },
    )

    output_html = ses / f"{prefix}.html"
    figures_dir = ses / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    generate_unified_report(
        subject_id="01", session_id="0001",
        output_path=str(output_html), ses_dir=str(ses),
        figures_dir=str(figures_dir),
        ran_pipelines={"anat": True, "tumor_seg": False,
                       "dsc": True, "dce": False},
        command_string="pmriprep ...",
    )

    html = output_html.read_text()
    # Per-pipeline section headings rendered
    assert "Anatomical" in html or "anatomical" in html.lower()
    assert "DSC" in html
    # Recommendation banner reflects DSC review action
    assert "review" in html.lower() or "AIF detection marginal" in html
    # Warnings rendered
    assert "Tissue segmentation" in html
    assert "FLAIR not available" in html


def test_boilerplate_bib_resolves_every_cited_key() -> None:
    """The bundled ``boilerplate.bib`` is the single source of truth for cites.

    The methods boilerplate carries Pandoc ``@key`` markers and
    ``boilerplate.bib`` holds the entries. A typo'd marker (which pandoc only
    *warns* about, silently leaving ``[@key]`` in the report) or a duplicate
    ``.bib`` key is a hard failure here. Keys are read from inside ``[@key]``
    groups so a typo of any shape (wrong case, missing underscore, multi-token
    surname) is caught rather than silently skipped.
    """
    bib_text = get_boilerplate_bib().read_text(encoding="utf-8")
    keys = re.findall(r"@\w+\{\s*([^,\s]+)\s*,", bib_text)

    dupes = sorted({k for k in keys if keys.count(k) > 1})
    assert not dupes, f"duplicate keys in boilerplate.bib: {dupes}"

    # Pull keys from inside Pandoc citation groups ([@key], [@a; @b]) so a
    # typo'd marker of ANY shape is caught: a shape-restricted regex would
    # silently drop e.g. [@Wu_2003] or [@van_osch_2018] from ``used`` and let
    # the phantom check pass while pandoc emits a literal "[@key]". Bracketing
    # also excludes the bare ``@key`` doc-mentions and any Python decorator.
    src = Path(ru.__file__).read_text(encoding="utf-8")
    used: set[str] = set()
    for group in re.findall(r"\[([^\]]*@[^\]]*)\]", src):
        used |= set(re.findall(r"@([^\s;,\]]+)", group))
    assert used, "no @key citation markers found in report_utils (regex drift?)"
    phantom = sorted(used - set(keys))
    assert not phantom, f"@key markers with no boilerplate.bib entry: {phantom}"
    orphaned = sorted(set(keys) - used)
    assert not orphaned, f"boilerplate.bib entries with no generated citation: {orphaned}"


def test_boilerplate_bib_has_unique_fields_per_entry() -> None:
    """BibTeX entries do not repeat fields that renderers may resolve ambiguously."""
    bib_text = get_boilerplate_bib().read_text(encoding="utf-8")
    for entry in re.findall(r"@\w+\{[^,]+,(.*?)\n\}", bib_text, re.DOTALL):
        fields = re.findall(r"^\s*([A-Za-z][A-Za-z0-9_-]*)\s*=", entry, re.MULTILINE)
        duplicates = sorted({field for field in fields if fields.count(field) > 1})
        assert not duplicates, f"duplicate BibTeX fields: {duplicates}"


def test_boilerplate_scientific_wording_matches_configurable_methods() -> None:
    """Boilerplate avoids unsupported sources and optional-method overclaims."""
    synthetic = ru._build_dsc_boilerplate(
        {"config": {"aif_source": "synthetic"}, "deconvolution": {"method": "oSVD"}}
    )
    assert "dose- and field-strength-scaled synthetic gamma-variate AIF" in synthetic
    assert "@parker_2006" not in synthetic
    assert "@mouridsen_2006" not in synthetic

    tikhonov = ru._build_dsc_boilerplate(
        {
            "aif_detection": {"aif_method": "auto"},
            "preload": {"preload_correction_applied": True},
            "deconvolution": {"method": "tikhonov"},
        }
    )
    assert "Morozov discrepancy principle" in tikhonov
    assert "L-curve" not in tikhonov
    assert "@morozov_1966" in tikhonov
    assert "@boxerman_1995" in tikhonov
    assert "@boxerman_2006" in tikhonov
    assert "@weisskoff_1994" not in tikhonov
    assert "@mouches_2019" in tikhonov

    dce = ru._build_dce_boilerplate(
        {
            "aif_detection": {"aif_source": "auto"},
            "pk_modeling": {
                "model": "auto",
                "candidate_models": ["patlak", "extended_tofts"],
            },
        }
    )
    assert "DCE images were motion-corrected" in dce
    assert "dose-aware candidate assessment" in dce
    assert ", and the extended Tofts model" not in dce
    assert "@mouches_2019" in dce
    assert "@schabel_2008" in dce
    assert "@tofts_1999" in dce
    assert "@schwarz_1978" in dce
    assert "@sourbron_2007" in dce
    assert "product integration" in dce


def test_dsc_boilerplate_reports_retained_aif_and_leakage_paths() -> None:
    """Runtime outcomes override configured intent in DSC methods text."""
    synthetic = ru._build_dsc_boilerplate(
        {
            "config": {"aif_source": "auto"},
            "aif_detection": {
                "detection_method": "synthetic",
                "synthetic_fallback_reason": "gamma_r2_floor",
            },
            "preload": {
                "preload_correction_applied": False,
                "k2_method": "bsw",
                "k2_reference_available": False,
            },
        }
    )
    assert "synthetic gamma-variate AIF was used as a fallback" in synthetic
    assert "automatically detected" not in synthetic
    assert "eligible reference curve could not be constructed" in synthetic

    disabled = ru._build_dsc_boilerplate(
        {
            "config": {"aif_source": "external"},
            "preload": {
                "preload_correction_applied": False,
                "k2_method": "none",
                "leakage_correction_disabled": True,
            },
            "normalization_factor": 1.0,
        }
    )
    assert "user-supplied arterial input function" in disabled
    assert "BSW leakage correction was disabled" in disabled
    assert "automatically segmented" not in disabled


def test_dce_boilerplate_reports_t1_b1_plasma_and_retained_prior() -> None:
    """DCE methods text reflects quantitative inputs used by retained fits."""
    text = ru._build_dce_boilerplate(
        {
            "t1_mapping": {
                "method": "assumed",
                "assumed_t1_ms": 1200.0,
                "assumed_t1_field_strength_t": 3.0,
                "b1_correction": "precomputed",
            },
            "signal_conversion": {"b1_correction_applied": True},
            "aif_detection": {
                "aif_source": "population",
                "hematocrit": 0.42,
            },
            "pk_modeling": {
                "model": "2cxm",
                "inference": "vi",
                "n_constrained": 42,
                "dsc_cbv_constraint": {
                    "mode": "soft",
                    "vp_uncertainty": 0.6,
                },
            },
        }
    )
    assert "uniform field-strength-dependent T1 of 1200 ms at 3 T" in text
    assert "with B1+ transmit field correction" in text
    assert "Parker population AIF" in text
    assert "hematocrit of 0.42" in text
    assert "@sourbron_2013" in text
    assert "piecewise-linear plasma arterial input function" in text
    assert "relative standard deviation of 60%" in text
    assert "used by retained Bayesian fits" in text

    no_prior = ru._build_dce_boilerplate(
        {
            "pk_modeling": {
                "model": "extended_tofts",
                "inference": "vi",
                "n_constrained": 0,
                "dsc_cbv_constraint": {
                    "mode": "soft",
                    "vp_uncertainty": 0.6,
                },
            }
        }
    )
    assert "DSC-informed vp prior" not in no_prior

    rejected_b1 = ru._build_dce_boilerplate(
        {
            "t1_mapping": {
                "method": "external",
                "b1_correction": "precomputed",
            },
            "signal_conversion": {"b1_correction_applied": False},
        }
    )
    assert "user-supplied T1 map" in rejected_b1
    assert "B1+ transmit field correction" not in rejected_b1


def test_render_citations_falls_back_to_markdown(tmp_path: Path, monkeypatch) -> None:
    """When pandoc fails, raw ``@key`` markdown is embedded so Methods is never lost.

    Guards the degradation path for platforms without a pandoc binary: the
    HTML fragment falls back to the raw markdown and ``CITATION.md`` is still
    written for manuscript reuse.
    """
    import pypandoc

    def _boom(*_args, **_kwargs):
        raise OSError("pandoc not found")

    monkeypatch.setattr(pypandoc, "convert_text", _boom)

    ses = tmp_path / "pmriprep" / "sub-01" / "ses-0001"
    ses.mkdir(parents=True)
    md = "Methods text [@wu_2003]."

    html, raw = ru._render_citations(md, ses)

    assert html == md and raw == md
    assert (ses / "logs" / "CITATION.md").read_text(encoding="utf-8") == md
