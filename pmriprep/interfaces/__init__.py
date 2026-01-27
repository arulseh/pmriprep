"""Nipype interfaces for PMRIPrep."""

from pmriprep.interfaces.anat import (
    ACPCAlignment,
    BiasCorrection,
    BrainExtraction,
    Coregistration,
    ReorientAndCrop,
    TissueSegmentation,
)
from pmriprep.interfaces.dce import (
    B1Mapping,
    T1Mapping,
)
from pmriprep.interfaces.dce_aif import (
    AIFDetection as DCEAIFDetection,
)
from pmriprep.interfaces.dsc import (
    DSCRegistration,
    MotionCorrection,
)
from pmriprep.interfaces.dsc_aif import (
    AIFDetection as DSCAIFDetection,
)
from pmriprep.interfaces.dsc_compute import (
    ComputeDSCMaps,
)

# Alias for backward compatibility - DSC AIF was the original
AIFDetection = DSCAIFDetection

__all__ = [
    "ACPCAlignment",
    "AIFDetection",
    "B1Mapping",
    "BiasCorrection",
    "BrainExtraction",
    "ComputeDSCMaps",
    "Coregistration",
    "DCEAIFDetection",
    "DSCAIFDetection",
    "DSCRegistration",
    "MotionCorrection",
    "ReorientAndCrop",
    "T1Mapping",
    "TissueSegmentation",
]
