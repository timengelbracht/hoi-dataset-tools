#!/usr/bin/env python3
import math, time
from pathlib import Path
import cv2, numpy as np

# ------------------------------------------------------------------ #
# 1.  ROOT FOLDER AND BASE STREAMS ********************************** #
# ------------------------------------------------------------------ #
SYNCED_DIR = Path("/bags/spot-aria-recordings/dlab_recordings/extracted/door_6")
SAVE_VIDEO = True
OUT_PATH = SYNCED_DIR / "multistream_mosaic.mp4"
VIDEO_FPS = 120
video_writer = None

BASE_STREAMS = {
    "aria_rgb"   : SYNCED_DIR / "aria_human_ego/camera_rgb/data.mp4",
    "zed_right"  : SYNCED_DIR / "gripper_right/zedm/zed_node/right/image_rect_color/data.mp4",
    "digit_right": SYNCED_DIR / "gripper_right/digit/right/image_raw/data.mp4",
    "iphone_rgb" : SYNCED_DIR / "iphone_left/camera_rgb/data.mp4",

}

# ------------------------------------------------------------------ #
# 2.  AUTO-CREATE “LEFT” COUNTERPARTS ******************************* #
# ------------------------------------------------------------------ #
AUTO_LEFT = {
    "zed_left"  : "zed_right",
    "digit_left": "digit_right",
}
VIDEO_PATHS = BASE_STREAMS.copy()
for new_name, ref_name in AUTO_LEFT.items():
    VIDEO_PATHS.setdefault(
        new_name,
        Path(str(VIDEO_PATHS[ref_name]).replace("/right/", "/left/"))
    )

# ------------------------------------------------------------------ #
# 3.  DEPTH VISUALISATION HELPER ************************************ #
# ------------------------------------------------------------------ #
def visualise_depth(gray_or_bgr):
    """Convert metric depth (0–6 m) to a coloured image for display."""
    gray = gray_or_bgr
    if len(gray.shape) == 3:                    # if already BGR, take gray
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)

# ------------------------------------------------------------------ #
# 4.  OPEN FILES & QUERY FPS **************************************** #
# ------------------------------------------------------------------ #
FRAME_SIZE, WINDOW_NAME = (320, 240), "Multi-stream sync (q to quit)"
caps, period, next_t, last_frame, ended = {}, {}, {}, {}, {}

for name, path in VIDEO_PATHS.items():
    if not path.exists():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    caps[name] = cap
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if fps <= 1e-2: fps = 30
    period[name] = 1.0 / fps
    next_t[name] = 0.0
    ended[name]  = False
    last_frame[name] = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), np.uint8)

print("Streams & FPS:", {k: f"{1/p:.1f}" for k, p in period.items()})

# ------------------------------------------------------------------ #
# 5.  DYNAMIC GRID *************************************************** #
# ------------------------------------------------------------------ #
N = len(VIDEO_PATHS)
cols = math.ceil(math.sqrt(N))
rows = math.ceil(N / cols)

def mosaic(frames, rows, cols):
    blank = np.zeros_like(frames[0])
    padded = frames + [blank]*(rows*cols - len(frames))
    return np.vstack([np.hstack(padded[r*cols:(r+1)*cols]) for r in range(rows)])

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, FRAME_SIZE[0]*cols, FRAME_SIZE[1]*rows)

# ------------------------------------------------------------------ #
# 6.  MAIN LOOP ***************************************************** #
# ------------------------------------------------------------------ #

t0, order = time.time(), list(VIDEO_PATHS.keys())
while True:
    now = time.time() - t0

    for name, cap in caps.items():
        if ended[name] or now < next_t[name]:
            continue
        ok, frame = cap.read()
        if not ok:
            ended[name] = True
            continue

        # ---- depth stream gets coloured ----------------------------
        if "depth" in name:
            frame = visualise_depth(frame)

        last_frame[name] = cv2.resize(frame, FRAME_SIZE)
        next_t[name]    += period[name]

    if all(ended.values()):
        break

    mosaic_frame = mosaic([last_frame[k] for k in order], rows, cols)
    cv2.imshow(WINDOW_NAME, mosaic_frame)

    # initialize writer once
    if SAVE_VIDEO and video_writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        h, w = mosaic_frame.shape[:2]
        video_writer = cv2.VideoWriter(str(OUT_PATH), fourcc, VIDEO_FPS, (w, h))    

        # write frame
    if SAVE_VIDEO:
        video_writer.write(mosaic_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(0.001)

for cap in caps.values(): cap.release()
cv2.destroyAllWindows()

if SAVE_VIDEO and video_writer is not None:
    video_writer.release()
    print(f"[✓] Saved mosaic video to: {OUT_PATH}")

print("Done.")
