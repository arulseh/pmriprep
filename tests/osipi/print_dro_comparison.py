"""Print GT vs PMRIPrep Ktrans comparison for OSIPI DRO sessions.

Usage:
    python tests/osipi/print_dro_comparison.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy import stats

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.osipi.conftest import (
    DRO_SESSIONS,
    load_ci_map,
    load_ground_truth_ktrans,
    load_map,
    load_tumor_mask,
)


def ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient."""
    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x), np.var(y)
    cov = np.mean((x - mx) * (y - my))
    denom = vx + vy + (mx - my) ** 2
    return float(2.0 * cov / denom) if denom > 1e-15 else 0.0


def main() -> None:
    for subject, session in DRO_SESSIONS:
        gt = load_ground_truth_ktrans(subject, session)
        calc = load_map(subject, session, "Ktrans")
        mask = load_tumor_mask(subject, session)

        if gt is None or calc is None or mask is None:
            missing = [
                name
                for name, val in [("GT", gt), ("calc", calc), ("mask", mask)]
                if val is None
            ]
            print(f"{subject} {session}: SKIPPED (missing: {missing})")
            continue

        g = gt[mask]
        c = calc[mask]
        valid = np.isfinite(g) & np.isfinite(c) & (g > 0)
        g, c = g[valid], c[valid]

        r = float(np.corrcoef(g, c)[0, 1])
        rho, rho_p = stats.spearmanr(g, c)
        con = ccc(g, c)
        mae = float(np.mean(np.abs(c - g)))
        bias = float(np.mean(c - g))
        ratio = float(np.median(c)) / float(np.median(g))

        print(f"--- {subject} {session} (n={len(g)} voxels) ---")
        print(
            f"  GT  median={np.median(g):.6f}  mean={np.mean(g):.6f}"
            f"  std={np.std(g):.6f}"
            f"  [P5={np.percentile(g, 5):.6f}, P95={np.percentile(g, 95):.6f}]"
        )
        print(
            f"  PMR median={np.median(c):.6f}  mean={np.mean(c):.6f}"
            f"  std={np.std(c):.6f}"
            f"  [P5={np.percentile(c, 5):.6f}, P95={np.percentile(c, 95):.6f}]"
        )
        print(
            f"  Pearson r={r:.4f}  Spearman rho={float(rho):.4f}"
            f" (p={float(rho_p):.1e})  CCC={con:.4f}"
        )
        print(
            f"  MAE={mae:.6f}  Bias={bias:+.6f}  Median ratio={ratio:.3f}x"
        )

        ci_lo = load_ci_map(subject, session, "lower", "Ktrans")
        ci_hi = load_ci_map(subject, session, "upper", "Ktrans")
        if ci_lo is not None and ci_hi is not None:
            lo_v, hi_v = ci_lo[mask][valid], ci_hi[mask][valid]
            cov_pct = float(np.mean((g >= lo_v) & (g <= hi_v))) * 100
            print(
                f"  CI coverage={cov_pct:.1f}%"
                f"  median_width={float(np.median(hi_v - lo_v)):.6f}"
            )
        print()


if __name__ == "__main__":
    main()
