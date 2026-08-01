#!/usr/bin/env python3
"""Regenerate the validation-only bounded-vp serial numerical fixture.

Computes Patlak serial-fit output (vp, Ktrans) on a deterministic 200-voxel
synthetic cohort, then writes it to
``tests/fixtures/dce_bounded_vp_serial_reference.json``.

Run this ONLY when intentional semantics changes in the serial NLLS
bounded-vp API require rebaselining. Accidental drift is exactly what the
numerical contract test is designed to catch — if the test fails,
revert the offending change rather than regenerating this fixture.

Usage:
    python tests/fixtures/generate_dce_bounded_vp_reference.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.unit.test_dce_bounded_vp_validation import _compute_serial_outputs

_FIXTURE_COMMENT = (
    "Numerical reference arrays for the serial-path Patlak NLLS fit (vp, "
    "Ktrans) on a deterministic cohort. Compared with tight tolerances for "
    "portability across supported Python and numerical-library versions."
)


def _aif_and_time():
    n_time = 60
    time_points = np.linspace(0.0, 5.0, n_time)
    peak_time = 0.5
    aif = (
        5.0 * np.exp(-0.5 * ((time_points - peak_time) / 0.15) ** 2)
        + 1.5 * np.exp(-0.3 * np.maximum(time_points - peak_time, 0.0))
    )
    return aif, time_points


def main() -> None:
    aif, time_points = _aif_and_time()
    outputs = _compute_serial_outputs(aif, time_points)

    payload = {
        "_comment": _FIXTURE_COMMENT,
        **{key: values.tolist() for key, values in outputs.items()},
    }
    fixture_path = Path(__file__).parent / "dce_bounded_vp_serial_reference.json"
    fixture_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {fixture_path}")
    for key, values in outputs.items():
        print(f"  {key}: {values.size} values")


if __name__ == "__main__":
    main()
