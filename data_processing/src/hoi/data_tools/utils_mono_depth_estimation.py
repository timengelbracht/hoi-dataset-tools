import torch
from mapanything.models import MapAnything
from mapanything.utils.image import load_images, preprocess_inputs
import numpy as np
from PIL import Image
from typing import Any
import cv2
from pathlib import Path

def load_map_anything_model(model_name: str = "facebook/map-anything", device: str = "cuda") -> Any:
    """
    Load the MapAnything model from the specified pretrained weights.
    """
    model = MapAnything.from_pretrained(model_name).to(device)
    return model

def run_map_anything_multimodal_inference(
    model: Any,
    image_path: list[str] | list[np.ndarray],
    intrinsics: list[np.ndarray] | None = None,
    extrinsics: list[np.ndarray] | None = None,
    depth: list[np.ndarray] | None = None,
    is_metric_scale: list[bool] | None = None,
):
    """
    Run MapAnything multi-modal inference with strict list-based inputs.
    All non-None arguments must be lists of the same length.
    """
    device = next(model.parameters()).device
    # --- Validate inputs ---
    num_images = len(image_path)
    for name, val in [
        ("intrinsics", intrinsics),
        ("extrinsics", extrinsics),
        ("depth", depth),
        ("is_metric_scale", is_metric_scale),
    ]:
        if val is not None and len(val) != num_images:
            raise ValueError(f"Length of '{name}' ({len(val)}) must match number of images ({num_images}).")

    # # --- Load model ---
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = MapAnything.from_pretrained("facebook/map-anything").to(device)

    # --- Load images into (H, W, 3) numpy arrays ---
    images = []
    for x in image_path:
        if isinstance(x, str):
            img = np.asarray(Image.open(x).convert("RGB"))
        elif isinstance(x, np.ndarray):
            img = x
        else:
            raise TypeError(f"Unsupported image type: {type(x)}")
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Image must have shape (H, W, 3), got {img.shape}")
        images.append(img)

    # --- Build per-view dictionaries ---
    views = []
    for i in range(num_images):
        view = {"img": images[i]}
        if intrinsics is not None and intrinsics[i] is not None:
            view["intrinsics"] = intrinsics[i]
        if extrinsics is not None and extrinsics[i] is not None:
            view["camera_poses"] = extrinsics[i]
        if depth is not None and depth[i] is not None:
            view["depth_z"] = depth[i]
        if is_metric_scale is not None and is_metric_scale[i] is not None:
            view["is_metric_scale"] = torch.tensor([is_metric_scale[i]], device=device)
        views.append(view)

    # --- Preprocess and run inference ---
    processed_views = preprocess_inputs(views)
    predictions = model.infer(
        processed_views,
        memory_efficient_inference=True,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
        apply_confidence_mask=False,
        confidence_percentile=10,
    )

    return predictions

def npy_depth_to_png(npy_path: str | Path, png_path: str | Path, meters_to_mm: float = 1000.0):
    depth_m = np.load(npy_path)                          # float depth in meters
    depth_mm = depth_m * meters_to_mm                    # meters -> millimeters
    depth_mm[~np.isfinite(depth_mm)] = 0                 # invalid -> 0
    depth_mm[depth_mm <= 0] = 0                          # nonpositive -> 0
    depth_mm = np.clip(depth_mm, 0, np.iinfo(np.uint16).max)
    depth_u16 = depth_mm.astype(np.uint16)               # uint16
    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(png_path), depth_u16)
