"""Generic in-memory registration helpers for DeeperHistReg.

The standard full-resolution pipeline estimates a displacement field at a
working resolution and normally materializes its final result through pyvips.
For patch-oriented callers this module keeps the registration optimization
unchanged, but applies the estimated field directly to an RGB array with
PyTorch grid_sample.

This module deliberately contains no dataset-, cohort-, or project-specific
logic. Callers own WSI sampling, affine pre-alignment, manifests, and QC
threshold policy.
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


def _set_device(params: Dict[str, Any], device: Optional[str]) -> None:
    if device is None:
        return
    params["device"] = device
    initial = params.get("initial_registration_params")
    if isinstance(initial, dict):
        if "device" in initial:
            initial["device"] = device
        if "cuda" in initial:
            initial["cuda"] = device != "cpu"
    nonrigid = params.get("nonrigid_registration_params")
    if isinstance(nonrigid, dict) and "device" in nonrigid:
        nonrigid["device"] = device


def _disable_disk_results(params: Dict[str, Any]) -> None:
    params["save_final_images"] = False
    params["save_final_displacement_field"] = False
    for section in (
        "preprocessing_params",
        "initial_registration_params",
        "nonrigid_registration_params",
    ):
        section_params = params.get(section)
        if isinstance(section_params, dict) and "save_results" in section_params:
            section_params["save_results"] = False
    params["logging_path"] = None


def _validate_parameters(params: Dict[str, Any]) -> None:
    loading = params.get("loading_params")
    if not isinstance(loading, dict):
        raise ValueError("registration parameters must contain loading_params")

    source_ratio = loading.get("source_resample_ratio")
    target_ratio = loading.get("target_resample_ratio")
    if source_ratio is None or target_ratio is None:
        raise ValueError(
            "loading_params must define source_resample_ratio and target_resample_ratio"
        )
    if float(source_ratio) <= 0 or float(target_ratio) <= 0:
        raise ValueError("resample ratios must be positive")
    if not np.isclose(float(source_ratio), float(target_ratio)):
        raise ValueError(
            "in-memory equal-grid registration requires matching source/target "
            "resample ratios; got %s and %s" % (source_ratio, target_ratio)
        )


def build_registration_parameters(
    preset: str = "default_nonrigid_fast",
    device: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    disable_disk_results: bool = True,
) -> Dict[str, Any]:
    """Build a validated config from a DHR preset plus nested overrides."""
    if preset not in _PRESETS:
        raise ValueError(
            "Unknown DeeperHistReg preset %r. Available: %s"
            % (preset, ", ".join(sorted(_PRESETS)))
        )
    if overrides is not None and not isinstance(overrides, dict):
        raise TypeError("overrides must be a dictionary or None")

    params = copy.deepcopy(_PRESETS[preset]())
    if overrides:
        _recursive_update(params, copy.deepcopy(overrides))

    preprocessing = params.get("preprocessing_params")
    if isinstance(preprocessing, dict):
        preprocessing.setdefault("flip_intensity", False)

    _set_device(params, device)
    if disable_disk_results:
        _disable_disk_results(params)
    _validate_parameters(params)
    return params


def _as_uint8_rgb(image: np.ndarray, name: str) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("%s must be an HxWx3 RGB array" % name)
    if image.dtype != np.uint8:
        raise TypeError("%s must have dtype uint8; got %s" % (name, image.dtype))
    if not image.flags.c_contiguous or not image.flags.writeable:
        image = np.array(image, dtype=np.uint8, copy=True, order='C')
    return image


def _as_unit_mask(mask: Optional[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=np.float32)
    mask = np.asarray(mask)
    if mask.shape != shape:
        raise ValueError(
            "source_valid_mask must have shape %s; got %s" % (shape, mask.shape)
        )
    if not np.issubdtype(mask.dtype, np.number) and mask.dtype != np.bool_:
        raise TypeError("source_valid_mask must be bool or numeric")
    mask = mask.astype(np.float32, copy=False)
    if not np.isfinite(mask).all():
        raise ValueError("source_valid_mask contains non-finite values")
    if mask.min() < 0 or mask.max() > 1:
        raise ValueError("source_valid_mask values must be in [0, 1]")
    return np.ascontiguousarray(mask)


def _pixel_displacement(displacement_field: tc.Tensor) -> np.ndarray:
    if displacement_field.ndim != 4 or displacement_field.shape[-1] != 2:
        raise ValueError(
            "Expected displacement field BxHxWx2; got %s"
            % (tuple(displacement_field.shape),)
        )
    if displacement_field.shape[0] != 1:
        raise ValueError("Only batch size 1 is supported by the array API")
    return u.tc_df_to_np_df(displacement_field)


def displacement_field_qc(displacement_field: tc.Tensor) -> Dict[str, float]:
    """Compute displacement magnitude and topology statistics."""
    field = _pixel_displacement(displacement_field)
    if not np.isfinite(field).all():
        raise ValueError("displacement field contains non-finite values")

    dx, dy = field[0], field[1]
    magnitude = np.sqrt(dx * dx + dy * dy)

    height, width = dx.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    map_x = grid_x + dx
    map_y = grid_y + dy
    dmap_x_dy, dmap_x_dx = np.gradient(map_x)
    dmap_y_dy, dmap_y_dx = np.gradient(map_y)
    jacobian = dmap_x_dx * dmap_y_dy - dmap_x_dy * dmap_y_dx

    return {
        "displacement_mean_px": float(np.mean(magnitude)),
        "displacement_median_px": float(np.median(magnitude)),
        "displacement_p95_px": float(np.percentile(magnitude, 95)),
        "displacement_max_px": float(np.max(magnitude)),
        "jacobian_min": float(np.min(jacobian)),
        "jacobian_p01": float(np.percentile(jacobian, 1)),
        "jacobian_p05": float(np.percentile(jacobian, 5)),
        "jacobian_median": float(np.median(jacobian)),
        "jacobian_p95": float(np.percentile(jacobian, 95)),
        "jacobian_max": float(np.max(jacobian)),
        "folding_fraction": float(np.mean(jacobian < 0)),
        "nonpositive_jacobian_fraction": float(np.mean(jacobian <= 0)),
    }


def warp_array_with_displacement(
    source_rgb: np.ndarray,
    displacement_field: tc.Tensor,
    source_valid_mask: Optional[np.ndarray] = None,
    padding_mode: str = "zeros",
    valid_mask_threshold: float = 0.999,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Warp an RGB array and an optional source-validity mask."""
    source = _as_uint8_rgb(source_rgb, "source_rgb")
    if padding_mode not in {"zeros", "border", "reflection"}:
        raise ValueError("Unsupported padding_mode: %s" % padding_mode)
    if valid_mask_threshold < 0 or valid_mask_threshold > 1:
        raise ValueError("valid_mask_threshold must be in [0, 1]")

    source_tensor = u.image_to_tensor(source).to(
        device=displacement_field.device, dtype=tc.float32
    )
    full_displacement = u.resample_displacement_field_to_size(
        displacement_field, source_tensor.shape[2:]
    )

    mask = _as_unit_mask(source_valid_mask, source.shape[:2])
    mask_tensor = (
        tc.from_numpy(mask)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device=full_displacement.device, dtype=tc.float32)
    )

    with tc.set_grad_enabled(False):
        warped_tensor = w.warp_tensor(
            source_tensor,
            full_displacement,
            padding_mode=padding_mode,
        )
        warped_mask_tensor = w.warp_tensor(
            mask_tensor,
            full_displacement,
            mode="bilinear",
            padding_mode="zeros",
        )

    warped = np.clip(u.tensor_to_image(warped_tensor), 0, 255).astype(np.uint8)
    warped_mask = warped_mask_tensor[0, 0].detach().cpu().numpy()
    valid_mask = warped_mask >= float(valid_mask_threshold)

    qc = displacement_field_qc(full_displacement)
    qc["valid_fraction"] = float(np.mean(valid_mask))
    qc["valid_weight_mean"] = float(np.mean(warped_mask))
    return warped, valid_mask, qc


def register_and_warp_arrays(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    registration_parameters: Optional[Dict[str, Any]] = None,
    preset: str = "default_nonrigid_fast",
    device: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    source_valid_mask: Optional[np.ndarray] = None,
    temporary_root: Optional[str] = None,
    keep_temporary: bool = False,
    padding_mode: str = "zeros",
    valid_mask_threshold: float = 0.999,
    return_valid_mask: bool = False,
    return_displacement_field: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Register source to target and warp the original source array in memory."""
    source = _as_uint8_rgb(source_rgb, "source_rgb")
    target = _as_uint8_rgb(target_rgb, "target_rgb")
    if source.shape[:2] != target.shape[:2]:
        raise ValueError(
            "In-memory equal-grid registration requires matching spatial size; "
            "got %s vs %s" % (source.shape[:2], target.shape[:2])
        )
    _as_unit_mask(source_valid_mask, source.shape[:2])

    if registration_parameters is None:
        params = build_registration_parameters(
            preset=preset,
            device=device,
            overrides=overrides,
            disable_disk_results=True,
        )
    else:
        params = copy.deepcopy(registration_parameters)
        preprocessing = params.get("preprocessing_params")
        if isinstance(preprocessing, dict):
            preprocessing.setdefault("flip_intensity", False)
        _set_device(params, device)
        _disable_disk_results(params)
        _validate_parameters(params)

    root = Path(temporary_root) if temporary_root else Path(tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="dhr_inmemory_", dir=str(root)))

    try:
        source_path = temp_dir / "source.tiff"
        target_path = temp_dir / "target.tiff"
        work_path = temp_dir / "work"
        work_path.mkdir(parents=True, exist_ok=True)

        Image.fromarray(source, mode="RGB").save(
            source_path, format="TIFF", compression="tiff_deflate", dpi=(100.0, 100.0)
        )
        Image.fromarray(target, mode="RGB").save(
            target_path, format="TIFF", compression="tiff_deflate", dpi=(100.0, 100.0)
        )

        params["case_name"] = "InMemory"
        pipeline = DeeperHistReg_FullResolution(params)
        try:
            pipeline.run_registration(
                str(source_path), str(target_path), str(work_path)
            )
        except Exception as exc:
            detail = (
                "DeeperHistReg in-memory registration failed "
                "(preset=%s, device=%s, shape=%s)"
                % (preset, params.get("device"), source.shape)
            )
            if keep_temporary:
                detail += "; temporary data kept at %s" % temp_dir
            raise RuntimeError(detail) from exc

        displacement = pipeline.current_displacement_field
        if displacement is None:
            raise RuntimeError("DeeperHistReg produced no displacement field")
        displacement = displacement.detach()
        if not tc.isfinite(displacement).all():
            raise RuntimeError("DeeperHistReg produced a non-finite displacement field")

        warped, valid_mask, deformation_qc = warp_array_with_displacement(
            source,
            displacement,
            source_valid_mask=source_valid_mask,
            padding_mode=padding_mode,
            valid_mask_threshold=valid_mask_threshold,
        )

        full_displacement = u.resample_displacement_field_to_size(
            displacement, source.shape[:2]
        )
        metadata: Dict[str, Any] = {
            "preset": preset,
            "device": str(displacement.device),
            "input_shape": list(source.shape),
            "working_displacement_shape": list(displacement.shape),
            "full_displacement_shape": list(full_displacement.shape),
            "registration_time_seconds": float(
                getattr(pipeline, "total_registration_time", 0.0)
            ),
            "deformation_qc": deformation_qc,
        }
        if return_valid_mask:
            metadata["valid_mask"] = valid_mask
        if return_displacement_field:
            metadata["displacement_field"] = full_displacement.detach().cpu()
        return warped, metadata
    finally:
        if not keep_temporary:
            shutil.rmtree(temp_dir, ignore_errors=True)
