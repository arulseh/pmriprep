"""Run one HD-BET image with bounded CPU threads and export workers.

This file is executable in the HD-BET environment without importing PMRIPrep.
"""

import argparse
import os
from pathlib import Path


def main() -> None:
    """Use HD-BET's predictor and masking with one worker per auxiliary stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", required=True)
    parser.add_argument("-o", required=True)
    parser.add_argument("-device", choices=("cpu", "cuda", "mps"), required=True)
    parser.add_argument("--disable_tta", action="store_true")
    parser.add_argument("--threads", type=int, required=True)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if not args.o.endswith(".nii.gz"):
        parser.error("-o must name a .nii.gz image")

    # Spawned preprocessing/export workers stay single-threaded. The parent
    # uses the node's reservation for model initialization and brain masking.
    for variable in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
    ):
        os.environ[variable] = "1"
    os.environ["nnUNet_def_n_proc"] = "1"

    import SimpleITK as sitk  # type: ignore[import-not-found]
    import torch
    from HD_BET.checkpoint_download import maybe_download_parameters  # type: ignore[import-not-found]
    from HD_BET.hd_bet_prediction import apply_bet, get_hdbet_predictor  # type: ignore[import-not-found]

    torch.set_num_threads(args.threads)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(args.threads)
    maybe_download_parameters()
    predictor = get_hdbet_predictor(
        use_tta=not args.disable_tta, device=torch.device(args.device), verbose=False,
    )
    # HD-BET resets CPU-mode threads to the host CPU count during initialization.
    torch.set_num_threads(args.threads)
    output = Path(args.o)
    output.parent.mkdir(parents=True, exist_ok=True)
    mask = str(output)[:-7] + "_bet.nii.gz"
    predictor.predict_from_files(
        [[args.i]], [mask], save_probabilities=False, overwrite=True,
        num_processes_preprocessing=1, num_processes_segmentation_export=1,
        folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0,
    )
    apply_bet(args.i, mask, str(output))
    Path(mask).unlink()
    for filename in ("dataset.json", "plans.json", "predict_from_raw_data_args.json"):
        (output.parent / filename).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
