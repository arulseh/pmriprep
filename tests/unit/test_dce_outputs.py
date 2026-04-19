"""Tests for DCE outputs sidecar generation."""

from pmriprep.interfaces.dce_outputs import DCEDerivativeWriter


def _minimal_interface_with_qc(hematocrit: float) -> DCEDerivativeWriter:
    iface = DCEDerivativeWriter()
    iface.inputs.provenance_dict = {"PerfusionProcessing": {"modality": "DCE"}}
    iface.inputs.qc_metrics = {"aif_detection": {"hematocrit": hematocrit}}
    return iface


def test_sidecar_ktrans_convention_key_default_hct() -> None:
    iface = _minimal_interface_with_qc(hematocrit=0.42)
    sidecar = iface._build_base_sidecar(
        bids_dir="/tmp/bids", voxel_sizes=(1.0, 1.0, 1.0),
    )
    assert sidecar["KtransConvention"] == "plasma_Tofts1999"
    assert sidecar["KtransConventionHct"] == 0.42


def test_sidecar_ktrans_convention_key_honours_hct_override() -> None:
    iface = _minimal_interface_with_qc(hematocrit=0.38)
    sidecar = iface._build_base_sidecar(
        bids_dir="/tmp/bids", voxel_sizes=(1.0, 1.0, 1.0),
    )
    assert sidecar["KtransConvention"] == "plasma_Tofts1999"
    assert sidecar["KtransConventionHct"] == 0.38


def test_minimal_sidecar_omits_convention_keys() -> None:
    iface = _minimal_interface_with_qc(hematocrit=0.42)
    sidecar = iface._build_base_sidecar(
        bids_dir="/tmp/bids", voxel_sizes=(1.0, 1.0, 1.0), minimal=True,
    )
    assert "KtransConvention" not in sidecar
    assert "KtransConventionHct" not in sidecar
