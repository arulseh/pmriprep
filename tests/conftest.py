"""Pytest suite markers and a synthetic AIF generator."""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command-line options."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run slow tests (multi-seed noise sweeps with 50 seeds)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Classify validation suites and skip slow tests unless requested."""
    tests_root = Path(__file__).parent.resolve()
    local_root = (tests_root / "osipi").resolve()
    synthetic_roots = {(tests_root / name).resolve() for name in ("dro", "qiba")}

    for item in items:
        item_path = Path(str(item.path)).resolve()
        if item_path.is_relative_to(local_root):
            item.add_marker(pytest.mark.local_validation)
        elif (
            not item.get_closest_marker("local_validation")
            and any(item_path.is_relative_to(root) for root in synthetic_roots)
        ):
            item.add_marker(pytest.mark.synthetic_dro)

    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


def generate_parker_aif(
    t_min: npt.NDArray[Any],
    arrival_time_min: float = 0.0,
    peak_concentration: float = 5.0,
) -> npt.NDArray[Any]:
    """Generate a synthetic AIF from two Gaussian components.

    The first component starts at its maximum after arrival; the second
    is delayed and broader. These are fixture-specific parameters.

    Parameters
    ----------
    t_min : npt.NDArray[Any]
        Time array in minutes.
    arrival_time_min : float
        Bolus arrival time in minutes.
    peak_concentration : float
        Amplitude of the first Gaussian component (mM).

    Returns
    -------
    npt.NDArray[Any]
        AIF concentration array (mM).
    """
    aif = np.zeros_like(t_min)

    mask = t_min > arrival_time_min
    if np.any(mask):
        dt = t_min[mask] - arrival_time_min

        # Sum the two Gaussian components
        # First pass (early peak)
        sigma1 = 0.1  # minutes
        A1 = peak_concentration

        # Recirculation (broader, delayed)
        sigma2 = 0.3
        A2 = peak_concentration * 0.3
        t_recirc = 0.3  # delay for recirculation

        aif[mask] = A1 * np.exp(-dt**2 / (2 * sigma1**2)) + A2 * np.exp(
            -(dt - t_recirc) ** 2 / (2 * sigma2**2)
        )

    return aif
