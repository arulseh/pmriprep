"""PMRIPrep: automated preprocessing and quantification of DSC and DCE MRI.

A NiPreps-compatible BIDS-App for perfusion MRI processing,
supporting DSC and DCE perfusion analysis.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pmriprep")
except PackageNotFoundError:
    __version__ = "0+unknown"
