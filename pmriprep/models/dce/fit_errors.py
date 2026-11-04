"""Explicitly classified failures of an individual kinetic fit."""


class NumericalFitError(RuntimeError):
    """A known numerical failure that makes this voxel/family unavailable.

    Backend incompatibility, invalid shared inputs, allocation failures and
    unexpected programming exceptions must not use this exception category.
    """
