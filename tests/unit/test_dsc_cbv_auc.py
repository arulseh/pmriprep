"""Unit tests for AUC-based CBV computation (full-bolus window)."""

import numpy as np
import pytest


class TestFullBolusWindow:
    """Tests for full-bolus window = BAT to end of acquisition."""

    @staticmethod
    def _compute_auc_cbv(
        conc_data: np.ndarray,
        measured_aif: np.ndarray,
        dt: float,
        bat: int,
        n_bl: int | None = None,
    ) -> np.ndarray:
        """Replicate production AUC CBV (full-bolus ASFNR/NCI consensus).

        Matches dsc_compute.py: window = [BAT, n_timepoints); measured AIF
        baseline-subtracted; conc_data already zero-baseline upstream;
        both numerator and denominator use same bolus_slice.
        """
        EPSILON = 1e-6
        if n_bl is None:
            n_bl = max(bat, 4)
        n_timepoints = measured_aif.shape[-1]
        bolus_slice = slice(bat, n_timepoints)

        aif_for_auc = measured_aif - float(np.mean(measured_aif[:n_bl]))
        aif_auc = float(np.sum(aif_for_auc[bolus_slice]) * dt)
        tissue_auc = np.sum(conc_data[..., bolus_slice], axis=-1) * dt

        if aif_auc < EPSILON:
            return np.full(conc_data.shape[:3], np.nan)
        return np.clip(tissue_auc / aif_auc, 0.0, 10.0)

    def test_uniform_tissue_matches_ratio(self) -> None:
        """Tissue at constant fraction of AIF produces uniform CBV."""
        n_frames = 40
        dt = 1.5
        bat = 5

        # AIF: gamma-shaped with zero baseline
        t = np.arange(n_frames, dtype=np.float64)
        aif = np.zeros(n_frames)
        bolus = t[bat:] - bat
        aif[bat:] = bolus * np.exp(-bolus / 5.0)
        aif /= np.max(aif)

        # Tissue = half the AIF, baseline-subtracted
        ratio = 0.5
        conc = np.zeros((3, 3, 3, n_frames))
        conc[1, 1, 1, :] = ratio * (aif - np.mean(aif[:bat]))

        cbv_auc = self._compute_auc_cbv(conc, aif, dt, bat)
        # Full-bolus integration of half-AIF tissue → CBV ≈ ratio
        assert cbv_auc[1, 1, 1] == pytest.approx(ratio, abs=0.05)
        # Background voxels (zero tissue) → CBV ≈ 0
        assert cbv_auc[0, 0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_zero_aif_returns_nan(self) -> None:
        """Near-zero AIF AUC produces NaN CBV."""
        conc = np.ones((2, 2, 2, 20)) * 0.5
        aif = np.full(20, 1e-9)
        cbv_auc = self._compute_auc_cbv(conc, aif, 1.5, 5)
        assert np.all(np.isnan(cbv_auc))

    def test_negative_tissue_clipped(self) -> None:
        """Negative tissue AUC clipped to 0 by lower bound of np.clip."""
        n_frames = 30
        dt = 1.5
        bat = 5

        aif = np.zeros(n_frames)
        aif[bat:15] = 1.0

        conc = np.zeros((2, 2, 2, n_frames))
        conc[0, 0, 0, bat:15] = -1.0

        cbv_auc = self._compute_auc_cbv(conc, aif, dt, bat)
        assert cbv_auc[0, 0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_upper_clip_at_ten(self) -> None:
        """Very large tissue/AIF ratio clipped at 10.0."""
        n_frames = 20
        dt = 1.5
        bat = 5

        aif = np.zeros(n_frames)
        aif[bat:bat + 3] = 0.1  # small AIF

        conc = np.zeros((1, 1, 1, n_frames))
        conc[0, 0, 0, bat:bat + 3] = 10.0  # huge tissue → ratio ~100

        cbv_auc = self._compute_auc_cbv(conc, aif, dt, bat)
        assert cbv_auc[0, 0, 0] == pytest.approx(10.0)

    def test_same_window_cancels_recirculation(self) -> None:
        """Recirculation in both numerator and denominator cancels in ratio."""
        n_frames = 60
        dt = 1.5
        bat = 5

        # AIF with first-pass + recirculation bump
        t = np.arange(n_frames, dtype=np.float64)
        aif = np.zeros(n_frames)
        bolus = t[bat:] - bat
        aif[bat:] = bolus * np.exp(-bolus / 5.0)
        aif /= np.max(aif)
        # Add a small recirculation at t ~ 30
        recirc_idx = 30
        recirc_range = t[recirc_idx:] - recirc_idx
        aif[recirc_idx:] += 0.3 * recirc_range * np.exp(-recirc_range / 6.0) / (
            np.max(recirc_range * np.exp(-recirc_range / 6.0))
        )

        ratio = 0.4
        conc = np.zeros((1, 1, 1, n_frames))
        conc[0, 0, 0, :] = ratio * (aif - np.mean(aif[:bat]))

        cbv_auc = self._compute_auc_cbv(conc, aif, dt, bat)
        # Recirculation adds to both; ratio should still ≈ `ratio`
        assert cbv_auc[0, 0, 0] == pytest.approx(ratio, abs=0.05)


class TestMeasuredAIFAlwaysProduced:
    """Verify AIFDetection always produces measured AIF output."""

    def test_output_spec_has_measured_aif(self) -> None:
        from pmriprep.interfaces.dsc_aif import AIFDetectionOutputSpec
        spec = AIFDetectionOutputSpec()
        assert "out_measured_aif_curve" in spec.traits()

    def test_output_spec_still_has_gamma_aif(self) -> None:
        """Gamma AIF output kept for QC overlay and future use."""
        from pmriprep.interfaces.dsc_aif import AIFDetectionOutputSpec
        spec = AIFDetectionOutputSpec()
        assert "out_gamma_aif_curve" in spec.traits()

    def test_gamma_fit_no_longer_in_config(self) -> None:
        from pmriprep.config.dsc import DSCConfig
        config = DSCConfig()
        assert not hasattr(config, "gamma_fit")

    def test_gamma_fit_no_longer_in_aif_input_spec(self) -> None:
        from pmriprep.interfaces.dsc_aif import AIFDetectionInputSpec
        spec = AIFDetectionInputSpec()
        assert "gamma_fit" not in spec.traits()

    def test_compute_maps_has_measured_aif_input(self) -> None:
        from pmriprep.interfaces.dsc_compute import ComputeDSCMapsInputSpec
        spec = ComputeDSCMapsInputSpec()
        assert "measured_aif_file" in spec.traits()
        assert "gamma_aif_file" not in spec.traits()

    def test_compute_maps_has_auc_cbv_output(self) -> None:
        from pmriprep.interfaces.dsc_compute import ComputeDSCMapsOutputSpec
        spec = ComputeDSCMapsOutputSpec()
        assert "out_rel_cbv_auc" in spec.traits()


class TestMeasuredAIFPathCoverage:
    """Guard against regressions where a code path forgets to populate
    out_measured_aif_curve. Every ``self._results = {...}`` block in
    AIFDetection must include this key so downstream AUC CBV always has
    a denominator.
    """

    def test_all_results_blocks_include_measured_aif(self) -> None:
        import ast
        import inspect
        from pmriprep.interfaces.dsc_aif import AIFDetection

        source = inspect.getsource(AIFDetection)
        tree = ast.parse(source)

        # Find all assignments to `self._results = {...}` with a dict literal.
        results_blocks: list[dict] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for tgt in node.targets:
                if (
                    isinstance(tgt, ast.Attribute)
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "self"
                    and tgt.attr == "_results"
                    and isinstance(node.value, ast.Dict)
                ):
                    keys = {
                        k.value for k in node.value.keys
                        if isinstance(k, ast.Constant) and isinstance(k.value, str)
                    }
                    results_blocks.append({"keys": keys, "lineno": node.lineno})

        assert len(results_blocks) >= 2, (
            f"Expected >=2 _results blocks (external AIF + main path); "
            f"found {len(results_blocks)}"
        )
        for block in results_blocks:
            assert "out_measured_aif_curve" in block["keys"], (
                f"_results block at line {block['lineno']} missing "
                f"'out_measured_aif_curve'. Every AIF code path must populate "
                f"this output for AUC CBV denominator. "
                f"Keys found: {sorted(block['keys'])}"
            )


class TestPreloadConfig:
    """Verify DSCConfig.preload `'auto' | float` semantics."""

    def test_default_is_auto(self) -> None:
        from pmriprep.config.dsc import DSCConfig
        config = DSCConfig()
        assert config.preload == "auto"

    def test_valid_float_dose(self) -> None:
        from pmriprep.config.dsc import DSCConfig
        for frac in [0.0, 0.25, 0.5, 1.0, 1.5]:
            config = DSCConfig(preload=frac)
            assert config.preload == frac

    def test_invalid_string_raises(self) -> None:
        """Old enum values now rejected — users must migrate to float."""
        from pmriprep.config.dsc import DSCConfig
        for enum_val in ["half_dose", "significant", "severe", "minimal", "none", "yes"]:
            with pytest.raises(ValueError, match="Invalid preload"):
                DSCConfig(preload=enum_val)  # type: ignore[arg-type]

    def test_out_of_range_raises(self) -> None:
        from pmriprep.config.dsc import DSCConfig
        with pytest.raises(ValueError, match="outside valid range"):
            DSCConfig(preload=2.5)
        with pytest.raises(ValueError, match="outside valid range"):
            DSCConfig(preload=-0.1)

    def test_preload_present_field_removed(self) -> None:
        from pmriprep.config.dsc import DSCConfig
        config = DSCConfig()
        assert not hasattr(config, "preload_present")


class TestCLIPreloadType:
    """Parsing of --dsc-preload as 'auto' | float via _preload_type()."""

    def test_auto_passes_through(self) -> None:
        from pmriprep.cli.run import _preload_type
        assert _preload_type("auto") == "auto"

    def test_zero_is_user_specified(self) -> None:
        """0.0 must be preserved (not falsy-coerced to None/auto)."""
        from pmriprep.cli.run import _preload_type
        val = _preload_type("0.0")
        assert val == 0.0
        assert val is not None
        # Downstream config-merge path uses `is not None`, so this is a
        # regression guard against accidental truthiness checks.

    def test_valid_floats(self) -> None:
        from pmriprep.cli.run import _preload_type
        for s, expected in [("0.25", 0.25), ("0.5", 0.5), ("1.0", 1.0), ("2.0", 2.0)]:
            assert _preload_type(s) == expected

    def test_invalid_string_raises(self) -> None:
        import argparse
        from pmriprep.cli.run import _preload_type
        with pytest.raises(argparse.ArgumentTypeError, match="must be 'auto'"):
            _preload_type("half_dose")
        with pytest.raises(argparse.ArgumentTypeError, match="must be 'auto'"):
            _preload_type("yes")

    def test_out_of_range_raises(self) -> None:
        import argparse
        from pmriprep.cli.run import _preload_type
        with pytest.raises(argparse.ArgumentTypeError, match=r"\[0\.0, 2\.0\]"):
            _preload_type("2.5")
        with pytest.raises(argparse.ArgumentTypeError, match=r"\[0\.0, 2\.0\]"):
            _preload_type("-0.1")

    def test_non_finite_raises(self) -> None:
        import argparse
        from pmriprep.cli.run import _preload_type
        with pytest.raises(argparse.ArgumentTypeError):
            _preload_type("inf")
        with pytest.raises(argparse.ArgumentTypeError):
            _preload_type("nan")
