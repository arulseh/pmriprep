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
    DCEPreprocess,
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
from pmriprep.interfaces.hdglio import (
    HDGLIOPrep,
    HDGLIOSegment,
)

__all__ = [
    "ACPCAlignment",
    "B1Mapping",
    "BiasCorrection",
    "BrainExtraction",
    "ComputeDSCMaps",
    "Coregistration",
    "DCEAIFDetection",
    "DCEPreprocess",
    "DSCAIFDetection",
    "DSCRegistration",
    "HDGLIOPrep",
    "HDGLIOSegment",
    "MotionCorrection",
    "ReorientAndCrop",
    "T1Mapping",
    "TissueSegmentation",
]
