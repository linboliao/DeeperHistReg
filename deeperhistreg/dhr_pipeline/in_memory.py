"""In-memory deformation helpers for DeeperHistReg.

The standard full-resolution pipeline estimates a displacement field at a
working resolution and normally materializes the final image through pyvips.
This module keeps registration unchanged but applies the estimated field to the
original RGB array with PyTorch grid_sample.
"""

import copy
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image
import torch as tc

from . import registration_params as rp
from .full_resolution import DeeperHistReg_FullResolution
from ..dhr_utils import utils as u
from ..dhr_utils import warping as w


_PRESETS = {
    "default_nonrigid": rp.default_nonrigid,
    "default_nonrigid_fast": rp.default_nonrigid_fast,
    "default_nonrigid_high_resolution": rp.default_nonrigid_high_resolution,
    "create_identity": rp.create_identity,
}


def _recursive_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _recursive_update(base[key], value)
        else:
            base[key] = value
    return base


def build_registration_parameters(
    preset: str = "default_nonrigid_fast",
    device: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    disable_disk_results: bool = True,
) -> Dict[str, Any]:
    """Build a registration config from a standard preset plus nested overrides."""
    if preset not in _PRESETS:
        raise ValueError("Unknown DeeperHistReg preset: %s" % preset)

    params = copy.deepcopy(_PRESETS[preset]())
    if overrides:
        _recursive_update(params, copy.deepcopy(overrides))

    if device is not None:
        params["device"] = device
        if isinstance(params.get("initial_registration_params"), dict):
            initial = params["initial_registration_params"]
            if "device" in initial:
                initial["device"] = device
            if "cuda" in initial:
                initial["cuda"] = device != "cpu"
        if isinstance(params.get("nonrigid_registration_params"), dict):
            nonrigid = params["nonrigid_registration_params"]
            if "device" in nonrigid:
                nonrigid["device"] = device

    if isinstance(params.get("preprocessing_params"), dict):
        params["preprocessing_params"].setdefault("flip_intensity", False)

    if disable_disk_results:
        if isinstance(params.get("preprocessing_params"), dict):
            params["preprocessing_params"].setdefault("flip_intensity", False)
        params["save_final_images"] = False
        params["save_final_displacement_field"] = False
        for section in (
            "preprocessing_params",
            "initial_registration_params",
            "nonrigid_registration_params",
        ):
            if isinstance(params.get(section), dict) and "save_results" in params[section]:
                params[section]["save_results"] = False

    params["logging_path"] = None
    return params


def register_and_warp_arrays(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    registration_parameters: Optional[Dict[str, Any]] = None,
    preset: str = "default_nonrigid_fast",
    device: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    temporary_root: Optional[str] = None,
    keep_temporary: bool = False,
    padding_mode: str = "zeros",
    return_displacement_field: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Register source to target and warp the original source array in memory.

    The array API intentionally requires source and target to have the same
    spatial size. This keeps the full-resolution deformation path free of
    hidden padding corrections.
    """
    source = np.asarray(source_rgb)
    target = np.asarray(target_rgb)
    if source.ndim != 3 or target.ndim != 3 or source.shape[2] != 3 or target.shape[2] != 3:
        raise ValueError("source_rgb and target_rgb must be HxWx3 arrays")
    if source.shape[:2] != target.shape[:2]:
        raise ValueError(
            "In-memory array API requires equal spatial size, got %s vs %s"
            % (source.shape[:2], target.shape[:2])
        )

    if registration_parameters is None:
        params = build_registration_parameters(
            preset=preset,
            device=device,
            overrides=overrides,
            disable_disk_results=True,
        )
    else:
        params = copy.deepcopy(registration_parameters)
        if device is not None:
            params["device"] = device
            if isinstance(params.get("nonrigid_registration_params"), dict):
                if "device" in params["nonrigid_registration_params"]:
                    params["nonrigid_registration_params"]["device"] = device
        params["save_final_images"] = False
        params["save_final_displacement_field"] = False
        for section in (
            "preprocessing_params",
            "initial_registration_params",
            "nonrigid_registration_params",
        ):
            if isinstance(params.get(section), dict) and "save_results" in params[section]:
                params[section]["save_results"] = False
        params["logging_path"] = None

    root = Path(temporary_root) if temporary_root else Path(tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="dhr_inmemory_", dir=str(root)))

    try:
        source_path = temp_dir / "source.tiff"
        target_path = temp_dir / "target.tiff"
        work_path = temp_dir / "work"
        work_path.mkdir(parents=True, exist_ok=True)

        Image.fromarray(source.astype(np.uint8), mode="RGB").save(
            source_path, format="TIFF", compression="tiff_deflate"
        )
        Image.fromarray(target.astype(np.uint8), mode="RGB").save(
            target_path, format="TIFF", compression="tiff_deflate"
        )

        params["case_name"] = "InMemory"
        pipeline = DeeperHistReg_FullResolution(params)
        pipeline.run_registration(str(source_path), str(target_path), str(work_path))

        displacement = pipeline.current_displacement_field.detach()
        source_tensor = u.image_to_tensor(source.astype(np.uint8)).to(
            device=displacement.device, dtype=tc.float32
        )
        full_displacement = u.resample_displacement_field_to_size(
            displacement, source_tensor.shape[2:]
        )
        with tc.set_grad_enabled(False):
            warped_tensor = w.warp_tensor(
                source_tensor,
                full_displacement,
                padding_mode=padding_mode,
            )

        warped = np.clip(u.tensor_to_image(warped_tensor), 0, 255).astype(np.uint8)
        metadata = {
            "preset": preset,
            "device": str(displacement.device),
            "input_shape": list(source.shape),
            "working_displacement_shape": list(displacement.shape),
            "full_displacement_shape": list(full_displacement.shape),
            "registration_time_seconds": float(
                getattr(pipeline, "total_registration_time", 0.0)
            ),
        }
        if return_displacement_field:
            metadata["displacement_field"] = full_displacement.detach().cpu()
        return warped, metadata
    finally:
        if not keep_temporary:
            shutil.rmtree(temp_dir, ignore_errors=True)
