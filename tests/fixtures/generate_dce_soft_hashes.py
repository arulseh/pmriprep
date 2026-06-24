#!/usr/bin/env python3
"""Regenerate the DSC-CBV SOFT-constraint serial-path byte-identity fixture.

Computes SHA-256 hashes of Patlak serial-fit output (vp, Ktrans) on a
deterministic 200-voxel synthetic cohort, then writes them to
``tests/fixtures/dce_soft_serial_hashes.json``.

Run this ONLY when intentional semantics changes in the serial NLLS
SOFT path require rebaselining. Accidental drift is exactly what the
byte-identity contract test is designed to catch — if the test fails,
revert the offending change rather than regenerating this fixture.

Usage:
    python tests/fixtures/generate_dce_soft_hashes.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tests.unit.test_dce_soft_constraint_parity import _compute_serial_hashes

_FIXTURE_COMMENT = (
    "SHA-256 byte-identity hashes of the serial-path Patlak NLLS fit (vp, "
    "Ktrans) on the deterministic _compute_serial_hashes cohort. Regenerate "
    "with this script only on intentional serial-path changes; otherwise a "
    "mismatch is accidental drift to revert."
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
    hashes = _compute_serial_hashes(aif, time_points)

    payload = {"_comment": _FIXTURE_COMMENT, **hashes}
    fixture_path = Path(__file__).parent / "dce_soft_serial_hashes.json"
    fixture_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {fixture_path}")
    for key, value in hashes.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
