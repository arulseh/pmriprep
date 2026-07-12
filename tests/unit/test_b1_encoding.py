"""Unit tests for the units-aware B1+ encoding resolver.

Locks the anti-mis-scale guarantee (all encodings of the same physical field
collapse to one ratio) and the ``auto`` hybrid fallback (resolve when
unambiguous, raise on genuine ambiguity).
"""
from __future__ import annotations

import numpy as np
import pytest

from pmriprep.utils.b1_encoding import (
    B1Encoding,
    B1EncodingIndeterminateError,
    expected_pixel_max,
    resolve_b1_divisor,
)

_FA = 70.0  # nominal prep flip angle (deg)
_TRUE_RATIO = 1.05
_DUMMY = np.array([1.0])  # unused by explicit encodings


def _ratio(pixel: float, divisor: float) -> float:
    return pixel / divisor


class TestRoundTripMatrix:
    """Same physical field, four encodings → one identical ratio (no 10× trap)."""

    @pytest.mark.parametrize(
        "encoding,pixel",
        [
            (B1Encoding.ACTUAL_FLIP_DECI_DEG, _TRUE_RATIO * 10 * _FA),  # 735
            (B1Encoding.ACTUAL_FLIP_DEG, _TRUE_RATIO * _FA),           # 73.5
            (B1Encoding.PERCENT, _TRUE_RATIO * 100),                   # 105
            (B1Encoding.RATIO, _TRUE_RATIO),                           # 1.05
        ],
    )
    def test_all_encodings_recover_same_ratio(self, encoding, pixel):
        """Each encoding of the same field recovers the identical B1+ ratio."""
        res = resolve_b1_divisor(
            encoding=encoding, nominal_prep_fa=_FA, b1_raw=_DUMMY
        )
        assert res.resolved_encoding is encoding
        assert res.source == "explicit"
        assert _ratio(pixel, res.divisor) == pytest.approx(_TRUE_RATIO)

    def test_deci_deg_divisor_is_exactly_10fa(self):
        """TB1TFL byte-identity: divisor must stay exactly 10.0 * FA (float)."""
        res = resolve_b1_divisor(
            encoding=B1Encoding.ACTUAL_FLIP_DECI_DEG, nominal_prep_fa=_FA, b1_raw=_DUMMY
        )
        assert res.divisor == 10.0 * _FA
        assert expected_pixel_max(res.divisor) == 10.0 * _FA


class TestAutoFallback:
    """The ``auto`` hybrid: resolve well-separated, raise on ambiguity."""

    def test_ratio_median_resolves(self):
        """A ratio-scale median (~1) resolves to the ratio encoding."""
        res = resolve_b1_divisor(
            encoding=B1Encoding.AUTO,
            nominal_prep_fa=_FA,
            b1_raw=np.full((4, 4, 4), 1.05),
        )
        assert res.resolved_encoding is B1Encoding.RATIO
        assert res.source == "median_fallback"

    def test_deci_deg_median_resolves(self):
        """A 10xFA-scale median resolves to the deci-degree encoding."""
        res = resolve_b1_divisor(
            encoding=B1Encoding.AUTO,
            nominal_prep_fa=_FA,
            b1_raw=np.full((4, 4, 4), 700.0),  # 10*FA → B1+≈1.0
        )
        assert res.resolved_encoding is B1Encoding.ACTUAL_FLIP_DECI_DEG

    def test_deg_percent_overlap_raises(self):
        """median≈85 with FA=70: percent(0.85) and deg(1.21) both plausible."""
        with pytest.raises(B1EncodingIndeterminateError, match="ambiguous"):
            resolve_b1_divisor(
                encoding=B1Encoding.AUTO,
                nominal_prep_fa=_FA,
                b1_raw=np.full((4, 4, 4), 85.0),
            )

    def test_no_fa_ratio_vs_percent_separable(self):
        """Without FA, only ratio/percent candidates — always separable."""
        r_ratio = resolve_b1_divisor(
            encoding=B1Encoding.AUTO, nominal_prep_fa=None, b1_raw=np.full((3, 3, 3), 1.0)
        )
        r_pct = resolve_b1_divisor(
            encoding=B1Encoding.AUTO, nominal_prep_fa=None, b1_raw=np.full((3, 3, 3), 100.0)
        )
        assert r_ratio.resolved_encoding is B1Encoding.RATIO
        assert r_pct.resolved_encoding is B1Encoding.PERCENT

    def test_implausible_median_raises(self):
        """A median implausible under every encoding raises."""
        with pytest.raises(B1EncodingIndeterminateError):
            resolve_b1_divisor(
                encoding=B1Encoding.AUTO,
                nominal_prep_fa=None,
                b1_raw=np.full((3, 3, 3), 5000.0),
            )

    def test_brain_mask_restricts_median(self):
        """Median is taken over brain voxels; out-of-mask junk is ignored."""
        raw = np.full((4, 4, 4), 1.02)
        raw[0, 0, 0] = 9999.0  # would wreck a whole-volume mean, not the median
        mask = np.ones((4, 4, 4))
        mask[0, 0, 0] = 0
        res = resolve_b1_divisor(
            encoding=B1Encoding.AUTO, nominal_prep_fa=_FA, b1_raw=raw, brain_mask=mask
        )
        assert res.resolved_encoding is B1Encoding.RATIO


class TestMissingFlipAngle:
    """Actual-flip encodings need FA; percent/ratio do not."""

    @pytest.mark.parametrize(
        "encoding", [B1Encoding.ACTUAL_FLIP_DECI_DEG, B1Encoding.ACTUAL_FLIP_DEG]
    )
    def test_actual_flip_without_fa_raises(self, encoding):
        """Actual-flip encodings without a flip angle raise a clear error."""
        with pytest.raises(ValueError, match="requires a positive nominal"):
            resolve_b1_divisor(encoding=encoding, nominal_prep_fa=None, b1_raw=_DUMMY)

    @pytest.mark.parametrize(
        "encoding,divisor", [(B1Encoding.PERCENT, 100.0), (B1Encoding.RATIO, 1.0)]
    )
    def test_normalized_encodings_need_no_fa(self, encoding, divisor):
        """Percent/ratio encodings resolve without a flip angle."""
        res = resolve_b1_divisor(encoding=encoding, nominal_prep_fa=None, b1_raw=_DUMMY)
        assert res.divisor == divisor


class TestInterfacePrecomputed:
    """End-to-end B1Mapping precomputed path across encodings."""

    def _run(self, tmp_path, pixel, encoding, fa=None):
        import nibabel as nib

        from pmriprep.interfaces.dce import B1Mapping

        arr = np.full((20, 20, 20), pixel, dtype=np.float32)
        p = tmp_path / f"b1_{encoding}.nii.gz"
        nib.save(nib.Nifti1Image(arr, np.eye(4)), str(p))
        kw = dict(precomputed_b1_file=str(p), b1_map_encoding=encoding, smoothing_fwhm=0.0)
        if fa is not None:
            kw["b1_nominal_prep_fa"] = fa
        return B1Mapping(**kw).run().outputs

    def test_percent_recovers_ratio(self, tmp_path):
        """A percent-encoded map recovers a B1+ ratio near 1.0."""
        out = self._run(tmp_path, 105.0, "percent")
        assert out.mean_b1 == pytest.approx(1.05, abs=1e-3)
        assert out.qc_metrics["method"] == "precomputed"
        assert out.qc_metrics["encoding"] == "percent"

    def test_deci_deg_locks_tfl_numbers(self, tmp_path):
        """The legacy Siemens tfl path (735 px, FA=70) still yields ratio 1.05."""
        out = self._run(tmp_path, 735.0, "actual_flip_deci_deg", fa=70.0)
        assert out.mean_b1 == pytest.approx(1.05, abs=1e-3)
        assert out.qc_metrics["encoding"] == "actual_flip_deci_deg"
        assert out.qc_metrics["divisor"] == 700.0


class TestReportBoilerplate:
    """Guard against the silent B1-sentence drop after the token rename.

    The renamed 'precomputed' token must key the Methods B1+ correction sentence.
    """

    def test_precomputed_token_yields_b1_sentence(self):
        """The renamed token keys a non-empty B1+ correction sentence."""
        from pmriprep.utils.report_utils import _B1_CORRECTION_TEXT

        # QC-present path (qc_metrics["method"]) and QC-absent path (enum value)
        # both key with "precomputed" — a rename miss would return "".
        assert _B1_CORRECTION_TEXT.get("precomputed", "").strip()
        assert "tfl_b1map" not in _B1_CORRECTION_TEXT
