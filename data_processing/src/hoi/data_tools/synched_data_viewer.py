#!/usr/bin/env python3
import math, time, os
from pathlib import Path

import cv2, numpy as np
import pandas as pd

# ----------------------------------------------------  VIDEO PART  --
SYNCED_DIR = Path("/bags/spot-aria-recordings/dlab_recordings/extracted/door_6")

BASE_STREAMS = {
    "aria_rgb"   : SYNCED_DIR / "aria_human_ego/camera_rgb/data.mp4",
    "zed_right"  : SYNCED_DIR / "gripper_right/zedm/zed_node/right/image_rect_color/data.mp4",
    "digit_right": SYNCED_DIR / "gripper_right/digit/right/image_raw/data.mp4",
    "iphone_rgb" : SYNCED_DIR / "iphone_left/camera_rgb/data.mp4",
    "depth"      : SYNCED_DIR / "gripper_right/zedm/zed_node/depth/depth_registered/data.mp4",
}
AUTO_LEFT = {"zed_left": "zed_right", "digit_left": "digit_right"}
VIDEO_PATHS = BASE_STREAMS | {
    k: Path(str(BASE_STREAMS[v]).replace("/right/", "/left/"))
    for k, v in AUTO_LEFT.items()
}

FRAME_SZ  = (320, 240)          # size per tile
GRID_COL  = math.ceil(math.sqrt(len(VIDEO_PATHS)))
GRID_ROW  = math.ceil(len(VIDEO_PATHS)/GRID_COL)
TILE_W, TILE_H = FRAME_SZ
SIG_HEIGHT = 120                # pixel height for graphs
WIN_NAME   = "Videos + Signals (q to quit)"

# ----------------------------------------------------  CSV PART  ----
CSV_DIR = SYNCED_DIR / "gripper_right"
CSV_FILES = {
    "joint":   CSV_DIR / "joint_states/data.csv",
    "force":   CSV_DIR / "gripper_force_trigger/data.csv",
}
# Column to show for each CSV:
CSV_COL = {
    "joint":  "effort",       # first joint angle
    "force":  "data",            # thresholded force value
}

def strlist_to_float(series, key):
    """If a cell looks like '[1.23]', return 1.23; otherwise leave as-is."""
    def _convert(x):
        if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
            try:
                val = float(x.strip("[]").split()[0])
                if val > 176.0 and key=="joint":
                    return 0.0
                return val
            except ValueError:
                return np.nan
        return x
    return series.apply(_convert).astype(float)

csv_df, csv_ptr = {}, {}
for key, f in CSV_FILES.items():
    df = (pd.read_csv(f)
            .sort_values("timestamp")
            .reset_index(drop=True))

    col = CSV_COL[key]
    df[col] = strlist_to_float(df[col], key)   # <-- convert here

    base = df["timestamp"].iloc[0]        # first sample time
    df["timestamp"] = df["timestamp"] - base

    csv_df[key]  = df
    csv_ptr[key] = 0

# scale signals to 0-100 vertically (simple auto-range)
sig_minmax = {k: (df[c].min(), df[c].max()) for k, df in csv_df.items() for c in [CSV_COL[k]]}

# -------------------------------------------------  DEPTH PALETTE  --
def vis_depth(img):
    if len(img.shape)==3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    norm = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)

# -------------------------------------------------  OPEN VIDEOS  ---
caps, period, next_t, frame_last, ended = {}, {}, {}, {}, {}
for name, path in VIDEO_PATHS.items():
    caps[name] = cv2.VideoCapture(str(path))
    fps = caps[name].get(cv2.CAP_PROP_FPS) or 30
    period[name] = 1.0/fps
    next_t[name] = 0.0
    ended[name]  = False
    frame_last[name] = np.zeros((TILE_H, TILE_W, 3), np.uint8)

# -------------------------------------------------  MOSAIC HELPER -
def mosaic(frames):
    blank = np.zeros_like(frames[0])
    pads  = frames + [blank]*(GRID_ROW*GRID_COL-len(frames))
    rows  = [np.hstack(pads[r*GRID_COL:(r+1)*GRID_COL]) for r in range(GRID_ROW)]
    return np.vstack(rows)

# -------------------------------------------------  WINDOW --------
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
full_w = TILE_W*GRID_COL
full_h = TILE_H*GRID_ROW + SIG_HEIGHT
cv2.resizeWindow(WIN_NAME, full_w, full_h)

# -------------------------------------------------  MAIN LOOP -----
t0 = time.time()
order = list(VIDEO_PATHS.keys())
sig_img = np.zeros((SIG_HEIGHT, full_w, 3), np.uint8)

while True:
    now_s   = time.time() - t0
    now_ns  = int(now_s*1e9)

    # ---------- video frames -------------
    for name, cap in caps.items():
        if ended[name] or now_s < next_t[name]:
            continue
        ok, fr = cap.read()
        if not ok:
            ended[name] = True
            continue
        if "depth" in name: fr = vis_depth(fr)
        frame_last[name] = cv2.resize(fr, FRAME_SZ)
        next_t[name]    += period[name]

    # ---------- CSV signals --------------
    sig_img[:] = 0
    for idx, key in enumerate(["force", "joint"]):
        df = csv_df[key]; col = CSV_COL[key]
        ptr = csv_ptr[key]
        while ptr < len(df) and df.iloc[ptr]["timestamp"] <= now_ns:
            ptr += 1
        csv_ptr[key] = ptr
        hist = df.iloc[:ptr][col].values
        if len(hist) < 2: continue
        # map value -> y pixel
        lo, hi = sig_minmax[key]
        ys = np.interp(hist, (lo, hi), (SIG_HEIGHT-10, 10)).astype(int)
        xs = np.linspace(0, full_w-1, len(ys)).astype(int)
        pts = np.vstack([xs, ys]).T.reshape(-1,1,2)
        color = (255,255,255) if key=="force" else (255,255,0)
        cv2.polylines(sig_img, [pts], False, color, 1, cv2.LINE_AA)
        cv2.putText(sig_img, key, (10,20+20*idx), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 1, cv2.LINE_AA)

    # ---------- display ------------------
    if all(ended.values()) and all(ptr>=len(df) for ptr,df in zip(csv_ptr.values(),csv_df.values())):
        break

    grid = mosaic([frame_last[k] for k in order])
    full = np.vstack([grid, sig_img])
    cv2.imshow(WIN_NAME, full)
    if cv2.waitKey(1)&0xFF==ord('q'): break
    time.sleep(0.001)

for cap in caps.values(): cap.release()
cv2.destroyAllWindows()
print("Done.")
