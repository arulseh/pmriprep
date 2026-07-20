"""Lock-in tests for the ``cbv_auc_preload_warning`` indicator set.

After the K2-adequacy decouple (v0.74.7) the DSC auto path emits K2-adequacy
labels (``adequate`` / ``partial`` / ``inadequate`` / ``unknown``) as the
effective ``preload_classification``. The full-bolus AUC warning must stay
silent on those (they are not preload doses) and fire only on a real
user-specified preload dose (``half_dose`` / ``significant`` / ``severe``).

The warning is driven by a local ``_PRELOAD_INDICATORS`` set inside
``ComputeDSCMaps._run_interface``; this test extracts that literal by AST so
the guarantee is locked without exercising the full nipype interface.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from pmriprep.interfaces.dsc_compute import ComputeDSCMaps


def _extract_preload_indicators() -> set[str]:
    """Return the ``_PRELOAD_INDICATORS`` set literal from ``_run_interface``."""
    src = textwrap.dedent(inspect.getsource(ComputeDSCMaps._compute_auc_maps))
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_PRELOAD_INDICATORS":
                    return {ast.literal_eval(elt) for elt in node.value.elts}  # type: ignore[attr-defined]
    raise AssertionError("_PRELOAD_INDICATORS not found in ComputeDSCMaps._compute_auc_maps")


class TestCBVAUCPreloadWarning:
    """Full-bolus AUC preload warning fires only for real preload doses."""

    def test_auto_k2_adequacy_labels_are_silent(self) -> None:
        """K2-adequacy labels (auto path) do NOT trigger the AUC preload warning."""
        indicators = _extract_preload_indicators()
        for adequacy in ("adequate", "partial", "inadequate", "unknown"):
            assert adequacy not in indicators, (
                f"K2-adequacy label {adequacy!r} must not be a preload indicator"
            )

    def test_user_preload_doses_fire(self) -> None:
        """Real user-specified preload doses DO trigger the AUC preload warning."""
        indicators = _extract_preload_indicators()
        assert {"half_dose", "significant", "severe"} <= indicators
