"""PMRIPrep: Perfusion MRI Preprocessing Pipeline.

A NiPreps-compatible BIDS-App for perfusion MRI processing,
supporting DSC and DCE perfusion analysis.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pmriprep")
except PackageNotFoundError:
    __version__ = "0+unknown"
