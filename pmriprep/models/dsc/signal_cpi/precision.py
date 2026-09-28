"""Numerical error budgets, independent of physiological or fitting thresholds.

Transport tolerances scale with the measured AIF polynomial envelope. The
independent final signal check scales with fitted baseline signal, not noise
or the potentially attenuated signal in an individual frame.
"""

TRANSPORT_RAW_RELATIVE_ERROR = 4e-9
TRANSPORT_INTERPOLATION_RELATIVE_ERROR = 4e-8
SIGNAL_BASELINE_RELATIVE_ERROR = 1e-8
