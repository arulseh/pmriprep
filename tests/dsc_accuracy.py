"""Expected accuracy limitations for specific deterministic DSC fixtures."""

import pytest


P2_NO_CORNER = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "This noiseless fixture has no interior curvature maximum. P2 selects "
        "its upper-endpoint default (10*sigma_max), substantially underestimating "
        "CBF and residue area."
    ),
)

P2_UPPER_CORNER = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "This noiseless dispersed fixture has two curvature maxima. P2 selects "
        "the maximum nearest the upper lambda bound, underestimating CBF by "
        "about 21% and exceeding the 16% recovery target."
    ),
)

OSVD_RESIDUE_MTT = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "For this noiseless fixture at 1.5-s sampling, the 1% truncation floor "
        "and block-circulant discretization underestimate the residue peak. "
        "Full-period area divided by that peak exceeds the MTT recovery target."
    ),
)
