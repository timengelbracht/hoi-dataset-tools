from __future__ import annotations

import json
import logging
import os
import cv2  # Requires opencv-python
import numpy as np
import torch
import torchvision
from collections import Counter
from pathlib import Path
from typing import List, Optional, Dict, Sequence, Tuple
from tqdm import tqdm

logger = logging.getLogger(__name__)

class EgoBlurFaceAnonymizer:
    """
    EgoBlur Face-only anonymizer for a *flat list of image paths*.
    
    Self-contained: No external utility dependencies.
    """

    labels_filename = "labels.json"

    # Face detection score range
    min_face_score = 0.4
    max_face_score = 0.9

    def __init__(self, anonymization_dir: Optional[Path] = None, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if anonymization_dir is None:
            # Fallback path logic
            anonym_path = Path(__file__).parent.parent.parent / "anonymization"
        else:
            anonym_path = Path(anonymization_dir)

        # Look for the model file
        face_path = anonym_path / "ego_blur_face_gen1.jit"
        if not face_path.exists():
            # Try fallback name if gen1 specific name not found
            face_path = anonym_path / "ego_blur_face.jit"
        
        if not face_path.exists():
            raise FileNotFoundError(
                f"Could not find EgoBlur Face model at {anonym_path}. "
                "Download it from https://www.projectaria.com/tools/egoblur/"
            )

        print(f"[EgoBlur] Loading model from {face_path} on {self.device}...")
        self.face_detector = torch.jit.load(face_path, map_location="cpu").to(self.device).eval()

    # ------------------------------------------------------------------ #
    # Internal Utilities (Formerly external dependencies)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _read_image(path: Path | str) -> np.ndarray:
        """Reads image as BGR (OpenCV default)."""
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to read image: {path}")
        return img

    @staticmethod
    def _write_image(path: Path | str, image: np.ndarray) -> None:
        """Writes BGR image to disk."""
        cv2.imwrite(str(path), image)

    @staticmethod
    def _resize_image_max(image_tensor: torch.Tensor, max_size: int) -> Tuple[torch.Tensor, float]:
        """
        Resizes a (C, H, W) tensor so its longest side is at most max_size.
        Returns: (resized_tensor, scale_factor)
        """
        _, h, w = image_tensor.shape
        scale = max_size / max(h, w)
        
        if scale < 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            # Unsqueeze to (1, C, H, W) for interpolate, then squeeze back
            image_tensor = torch.nn.functional.interpolate(
                image_tensor.unsqueeze(0), 
                size=(new_h, new_w), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            return image_tensor, scale
        
        return image_tensor, 1.0

    @staticmethod
    def _get_box_area_ratio(box: List[float], image_shape: Tuple[int, int, int]) -> float:
        """Calculates what percentage of the image the box covers."""
        h, w, _ = image_shape
        x1, y1, x2, y2 = box
        area_box = (x2 - x1) * (y2 - y1)
        area_img = h * w
        return area_box / area_img if area_img > 0 else 0.0

    @staticmethod
    def _score_threshold(area_ratio: float, score_min: float, score_max: float) -> float:
        """
        Dynamic thresholding.
        Small faces (low area ratio) require higher confidence (score_max) to avoid noise.
        Large faces (high area ratio) are accepted with lower confidence (score_min).
        """
        # Linear interpolation:
        # If area_ratio < 0.005 (0.5% of screen), require max_score.
        # If area_ratio > 0.05 (5% of screen), accept min_score.
        lower_bound = 0.005
        upper_bound = 0.05
        
        if area_ratio < lower_bound:
            return score_max
        elif area_ratio > upper_bound:
            return score_min
        else:
            # Linear decay between bounds
            progress = (area_ratio - lower_bound) / (upper_bound - lower_bound)
            return score_max - (score_max - score_min) * progress

    @staticmethod
    def _blur_detections(image: np.ndarray, boxes: List[List[float]]) -> np.ndarray:
        """
        Applies Gaussian Blur to the regions defined by boxes.
        The kernel size scales with the box size.
        """
        img_h, img_w = image.shape[:2]
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            
            # Clip to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue

            # Calculate ROI
            roi = image[y1:y2, x1:x2]
            
            # Dynamic Kernel Size: ~1/3rd of the face width/height
            # Must be an odd number
            box_w, box_h = x2 - x1, y2 - y1
            k_size = max(1, int(min(box_w, box_h) * 0.3))
            if k_size % 2 == 0:
                k_size += 1
                
            # Sigma calculation (standard OpenCV formula based on ksize)
            sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8
            
            # Apply blur
            blurred_roi = cv2.GaussianBlur(roi, (k_size, k_size), sigma)
            image[y1:y2, x1:x2] = blurred_roi
            
        return image

    # ------------------------------------------------------------------ #
    # Detection Logic
    # ------------------------------------------------------------------ #

    def get_face_detections(
        self,
        image_tensor: torch.Tensor,
        nms_iou_threshold: float = 0.3,
        max_image_size: Optional[int] = None,
    ):
        """Run EgoBlur face detector on a single image tensor CxHxW."""
        
        scale = 1.0
        if max_image_size is not None:
            image_tensor, scale = self._resize_image_max(image_tensor, max_image_size)

        with torch.no_grad():
            # Forward pass
            boxes, _, scores, _ = self.face_detector(image_tensor)

        # NMS
        keep_idx = torchvision.ops.nms(boxes, scores, nms_iou_threshold)
        boxes = boxes[keep_idx] / scale # Project boxes back to original size
        scores = scores[keep_idx]

        boxes = boxes.cpu().numpy().tolist()
        scores = scores.cpu().numpy().tolist()

        return [dict(bounding_box=b, score=s) for b, s in zip(boxes, scores)]

    def face_is_valid(self, face, image_shape) -> bool:
        """Apply area-dependent score threshold."""
        area_ratio = self._get_box_area_ratio(face["bounding_box"], image_shape)
        thr = self._score_threshold(
            area_ratio,
            score_min=self.min_face_score,
            score_max=self.max_face_score,
        )
        return face["score"] > thr

    # ------------------------------------------------------------------ #
    # Core blur logic
    # ------------------------------------------------------------------ #

    def _blur_image_group(
            self,
            input_paths: List[Path],
            tmp_dir: Path,
            output_paths: Optional[List[Path]] = None,
            max_face_image_size: int = 640,
        ) -> Counter:
            """Run detection + blurring on a batch of images."""
            
            labels_path = tmp_dir / self.labels_filename
            
            # --- Cache Loading Logic ---
            labels = []
            cached_frames = None
            
            if labels_path.exists():
                try:
                    data = json.loads(labels_path.read_text())
                    cached_frames = data.get("frames", [])
                    
                    # Verify cache alignment
                    if len(input_paths) != len(cached_frames):
                        logger.warning(
                            f"Cache mismatch: {len(input_paths)} images vs {len(cached_frames)} cached entries. "
                            "Ignoring cache and re-running detection."
                        )
                        cached_frames = None
                    else:
                        logger.info("Using cached detections from %s", labels_path)
                except Exception as e:
                    logger.warning(f"Failed to read cache: {e}. Re-running detection.")
            
            # ---------------------------

            inplace = output_paths is None
            counts = Counter()

            for idx, image_path in enumerate(tqdm(input_paths, desc="[EgoBlur] Processing", unit="img")):
                try:
                    image = self._read_image(image_path)
                except Exception as e:
                    print(f"Skipping {image_path}: {e}")
                    if cached_frames is None: labels.append(dict(faces=[]))
                    continue

                # --- Detection Phase ---
                if cached_frames is None:
                    if image.ndim == 3:
                        # CV2 is BGR (H,W,C) -> Transpose (C,H,W) -> Flip to RGB
                        tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)).copy()).flip(0)
                    else:
                        # Handle Grayscale (HW)
                        tensor = torch.from_numpy(image.copy())
                        tensor = tensor.unsqueeze(0).repeat(3, 1, 1) # Make 3 channel

                    # --- CRITICAL FIX HERE ---
                    # PyTorch interpolate/resize requires Float, not Byte (uint8).
                    # We cast to Float, but we keep values in 0-255 range.
                    tensor = tensor.to(self.device).float()

                    faces = self.get_face_detections(tensor, max_image_size=max_face_image_size)
                    labels.append(dict(faces=faces))
                else:
                    faces = cached_frames[idx]["faces"]

                # --- Filtering Phase ---
                valid_faces = [f for f in faces if f is not None and self.face_is_valid(f, image.shape)]

                # --- Blurring Phase ---
                if valid_faces:
                    box_list = [f["bounding_box"] for f in valid_faces]
                    blurred = self._blur_detections(image, box_list)
                else:
                    blurred = image

                counts["faces"] += len(valid_faces)

                # --- Saving Phase ---
                if inplace:
                    if len(valid_faces) > 0:
                        self._write_image(image_path, blurred)
                else:
                    out = output_paths[idx]
                    out.parent.mkdir(parents=True, exist_ok=True)
                    self._write_image(out, blurred)

            # Write new cache if we generated new labels
            if cached_frames is None:
                labels_path.write_text(json.dumps(dict(frames=labels)))

            logger.info("Finished face anonymization in %s", tmp_dir)
            return counts

    # ------------------------------------------------------------------ #
    # Public unified API
    # ------------------------------------------------------------------ #

    def run_anonymization(
        self,
        image_paths: Sequence[str | Path],
        *,
        tmp_dir: Path,
        outdir: Optional[Path] = None,
        inplace: bool = False,
        overwrite: bool = False,
        max_face_image_size: int = 640,
    ) -> Dict[str, int]:
        """
        Anonymize face detections in a list of images.
        """
        paths = [Path(p) for p in image_paths]
        if len(paths) == 0:
            logger.warning("No images supplied for anonymization.")
            return {"faces": 0}

        # Determine output layout
        output_paths = None
        if not inplace:
            if outdir is None:
                common_root = Path(os.path.commonpath([str(p) for p in paths]))
                outdir = common_root / "anonymization_egoblur_faces"
            outdir = Path(outdir)
            
            # safe resolve common path
            try:
                common_root = Path(os.path.commonpath([str(p) for p in paths]))
                output_paths = [outdir / p.relative_to(common_root) for p in paths]
            except ValueError:
                # If paths are on different drives or totally disjoint, flatten structure
                output_paths = [outdir / p.name for p in paths]

            if not overwrite and all(out.exists() for out in output_paths):
                logger.info("All anonymized outputs already exist in %s; skipping.", outdir)
                return {"faces": 0}

        tmp_dir.mkdir(parents=True, exist_ok=True)

        counts = self._blur_image_group(
            input_paths=paths,
            tmp_dir=tmp_dir,
            output_paths=output_paths,
            max_face_image_size=max_face_image_size,
        )

        logger.info("Detected %d faces across %d images.", counts["faces"], len(paths))
        return {"faces": counts["faces"]}

if __name__ == "__main__":
    # Example Usage
    files = sorted(Path("/data/ikea_recordings/test").glob("*.jpg"))
    
    # Ensure this path points to where you saved the 'ego_blur_face.jit'
    model_dir = Path("/data/ikea_recordings/ego_blur_weights/")
    
    an = EgoBlurFaceAnonymizer(anonymization_dir=model_dir)
    
    print(f"Found {len(files)} images to process.")
    
    an.run_anonymization(
        files,
        tmp_dir=Path("/data/tmp/egoblur_cache"),
        inplace=True, # Be careful, this overwrites!
    )