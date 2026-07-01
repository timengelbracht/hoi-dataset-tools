#!/usr/bin/env python3
# sam2_service.py
# FastAPI service exposing SAM-2 segmentation with box/point prompts.
# Returns: polygon + base64-encoded PNG mask.
# Deps: torch (GPU recommended), ultralytics, fastapi, uvicorn, opencv-python-headless, numpy

import os, cv2, base64, numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from ultralytics import SAM
import torch

app = FastAPI()

# Pick weights (tiny/small recommended for interactivity): sam2_t.pt / sam2_s.pt / sam2_b.pt / sam2_l.pt
WEIGHTS = os.getenv("SAM2_WEIGHTS", "sam2_s.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SAM(WEIGHTS)
model.to(device)
model.model.eval()

class Req(BaseModel):
    image_path: str
    bbox: list | None = None          # [x1,y1,x2,y2] in pixels
    points: list[list[float]] = []    # [[x,y], ...]
    labels: list[int] = []            # 1=fg, 0=bg (len == points)

def mask_to_polygon(mask: np.ndarray) -> list[list[int]]:
    # mask: HxW (bool/uint8). Return simplified outer contour.
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    mask = (mask > 0).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    cnt = max(cnts, key=cv2.contourArea)
    cnt = cv2.approxPolyDP(cnt, epsilon=1.5, closed=True)
    return cnt.reshape(-1, 2).tolist()

def mask_to_png_b64(mask: np.ndarray) -> str:
    # mask: HxW (bool/uint8) -> PNG bytes -> base64 string
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    mask = (mask > 0).astype(np.uint8) * 255
    ok, buf = cv2.imencode(".png", mask)  # single-channel PNG
    if not ok:
        return ""
    return base64.b64encode(buf).decode("ascii")

@app.get("/health")
def health():
    return {"ok": True, "device": device, "weights": WEIGHTS}

@app.post("/segment")
def segment(req: Req):
    kwargs = {}
    if req.bbox:
        kwargs["bboxes"] = [req.bbox]
    if req.points:
        kwargs["points"] = req.points
        kwargs["labels"] = req.labels or [1] * len(req.points)

    print(f"Debug: bbox shape: {np.array(req.bbox).shape if req.bbox else 'None'}")
    print(f"Debug: points shape: {np.array(req.points).shape if req.points else 'None'}")
    print(f"Debug: labels length: {len(req.labels) if req.labels else 'None'}")

    res = model(req.image_path, **kwargs)[0]
    if not getattr(res, "masks", None) or len(res.masks) == 0:
        return {"ok": False, "polygons": [], "mask_png_b64": ""}

    # Best mask (index 0). Ultralytics Masks: (N,H,W) tensor
    mask = res.masks.data[0].detach().cpu().numpy()
    poly = mask_to_polygon(mask)
    mask_png_b64 = mask_to_png_b64(mask)

    return {"ok": True, "polygons": [poly], "mask_png_b64": mask_png_b64}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)