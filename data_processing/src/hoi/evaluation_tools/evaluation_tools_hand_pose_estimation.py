from hoi.data_tools.data_loader_aria import AriaData
from hoi.data_tools.data_loader_iphone import IPhoneData
from hoi.data_tools.data_indexer import RecordingIndex
from pathlib import Path
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
import json
import shutil
import random
import math
import re
import yaml

# ARIA hand landmarks order to MANO order
ARIA_TO_MANO_INDEX = [5, 20, 6, 7, 0, 8, 9, 10, 1, 11, 12, 13, 2, 14, 15, 16, 3, 17, 18, 19, 4]
#kp_mano = kp_aria[aria_to_mano_idx]


def transform_point_to_camera(point_3d: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Map a 3D point from its source frame (device/world) to the camera frame
    using a 4x4 extrinsic matrix T (source -> camera). No projection.
    """
    p = np.append(point_3d, 1.0)             # [x,y,z,1]
    pc = T @ p                               # homogeneous
    if pc[3] != 0:
        pc = pc / pc[3]
    return pc[:3]                            # [x_cam, y_cam, z_cam]

def project_point_into_image(point_3d, intrinsic, extrinsic):
    """Project a 3D point into the image plane using intrinsic and extrinsic matrices.
    
    Args:
        point_3d (np.array): A 3D point in world coordinates (shape: [3,]).
        intrinsic (np.array): The camera intrinsic matrix (shape: [3, 3]).
        extrinsic (np.array): The camera extrinsic matrix (shape: [4, 4]).  

    Returns:
        np.array: The 2D point in image coordinates (shape: [2,]).
    """
    # Convert point to homogeneous coordinates
    point_homogeneous = np.append(point_3d, 1)  # Shape: [4,]

    # Transform the point from world coordinates to camera coordinates
    point_camera = extrinsic @ point_homogeneous  # Shape: [4,]
    point_camera = point_camera[:3] / point_camera[3]  # Normalize if needed

    # Project the point onto the image plane
    point_image_homogeneous = intrinsic @ point_camera  # Shape: [3,]
    point_image = point_image_homogeneous[:2] / point_image_homogeneous[2]  # Normalize

    return point_image


def filter_points_inside_image(df: pd.DataFrame, intrinsic: np.array, extrinsic: np.array, image_width: int, image_height: int, frame: str):
    """Filter out rows in the dataframe where the projected points are outside the image boundaries.
    
    Args:
        df (pd.DataFrame): DataFrame containing 3D points in columns
        intrinsic (np.array): The camera intrinsic matrix (shape: [3, 3]).
        extrinsic (np.array): The camera extrinsic matrix (shape: [4,
    4]).    
        image_width (int): Width of the image in pixels.
        image_height (int): Height of the image in pixels.  
    Returns:
        pd.DataFrame: Filtered DataFrame with points inside image boundaries.
    """
    def is_point_inside_image(point_3d):
        point_2d = project_point_into_image(point_3d, intrinsic, extrinsic)
        x, y = point_2d
        return 0 <= x < image_width and 0 <= y < image_height

    if frame not in ["device", "world"]:
        raise ValueError("frame must be one of ['device', 'world']")

    mask_right = []
    mask_left = []

    for idx, row in df.iterrows():
        column_names_landmarks_tx_right = [f"tx_right_landmark_{i}_{frame}" for i in range(21)]
        column_names_landmarks_ty_right = [f"ty_right_landmark_{i}_{frame}" for i in range(21)]
        column_names_landmarks_tz_right = [f"tz_right_landmark_{i}_{frame}" for i in range(21)]
        right_landmarks_tx = row[column_names_landmarks_tx_right].to_numpy()
        right_landmarks_ty = row[column_names_landmarks_ty_right].to_numpy()
        right_landmarks_tz = row[column_names_landmarks_tz_right].to_numpy()
        right_landmarks = np.stack([right_landmarks_tx, right_landmarks_ty, right_landmarks_tz], axis=-1) # (21, 3)
        right_wrist_device = right_landmarks[5]
        right_palm_device = right_landmarks[20]

        # left 
        column_names_landmarks_tx_left = [f"tx_left_landmark_{i}_{frame}" for i in range(21)]
        column_names_landmarks_ty_left = [f"ty_left_landmark_{i}_{frame}" for i in range(21)]
        column_names_landmarks_tz_left = [f"tz_left_landmark_{i}_{frame}" for i in range(21)]
        left_landmarks_tx = row[column_names_landmarks_tx_left].to_numpy()
        left_landmarks_ty = row[column_names_landmarks_ty_left].to_numpy()
        left_landmarks_tz = row[column_names_landmarks_tz_left].to_numpy()
        left_landmarks = np.stack([left_landmarks_tx, left_landmarks_ty, left_landmarks_tz], axis=-1) # (21, 3)
        left_wrist_device = left_landmarks[5]
        left_palm_device = left_landmarks[20]
        

        wrist_inside_right = is_point_inside_image(right_wrist_device)
        palm_inside_right = is_point_inside_image(right_palm_device)
        mask_right.append(wrist_inside_right and palm_inside_right)

        wrist_inside_left = is_point_inside_image(left_wrist_device)
        palm_inside_left = is_point_inside_image(left_palm_device)
        mask_left.append(wrist_inside_left and palm_inside_left)

    # add both masks as seperate columns
    if frame == "device":
        df["mask_inside_image_aria_right"] = mask_right
        df["mask_inside_image_aria_left"] = mask_left
    else:
        df["mask_inside_image_iphone_right"] = mask_right
        df["mask_inside_image_iphone_left"] = mask_left

    filtered_df = df[mask_right].reset_index(drop=True)
    return filtered_df

def visualize_keypoints_debug(row, kps, side, frame, img_w, img_h):
    """
    Quick temporary visualization of 2D keypoints.
    Saves image overlays to ./debug_vis/.
    """
    try:
        # get image path from row
        img_path = None
        if frame == "device" and "frame_path_aria" in row:
            img_path = row["frame_path_aria"]
        elif frame == "world" and "frame_path_iphone" in row:
            img_path = row["frame_path_iphone"]
        if img_path is None:
            print(f"[VIS] no image path found in row for frame={frame}")
            return

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[VIS] could not read {img_path}")
            return

        vis = img.copy()
        for j, (u, v) in enumerate(kps):
            if np.isfinite(u) and np.isfinite(v):
                color = (0, 255, 0)
                cv2.circle(vis, (int(round(u)), int(round(v))), 3, color, -1)
                cv2.putText(vis, str(j), (int(round(u)) + 4, int(round(v)) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        out_dir = Path("debug_vis")
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"{Path(str(img_path)).stem}_{frame}_{side}.jpg"
        cv2.imwrite(str(out_path), vis)
        print(f"[VIS] wrote {out_path}")
    except Exception as e:
        print(f"[VIS ERROR] {e}")



def visualize_on_view(merged_df: pd.DataFrame,
                      view: str,                  # 'ego' or 'exo'
                      n_views: int = 2,
                      mode: str = "wrist_palm",   # 'wrist_palm' or 'landmarks'
                      aria_calibration: dict = None,
                      iphone_calibration: dict = None,
                      iphone_extrinsic: np.ndarray = None):
    """
    Unified visualization for ego (Aria) and exo (iPhone).
    - Uses your existing `project_point_into_image(point_3d, K, T)` as-is.
    - For ego: points are in DEVICE frame; builds Device->Pinhole: T = inv(pinhole_T_device_camera) @ inv(T_device_camera)
    - For exo: points are in WORLD frame; uses iphone_extrinsic directly (World->Camera).
    """

    if mode not in ["wrist_palm", "landmarks"]:
        raise ValueError("mode must be one of ['wrist_palm', 'landmarks']")
    if view not in ["ego", "exo"]:
        raise ValueError("view must be 'ego' or 'exo'")

    # Select intrinsics and extrinsics, and which columns to read
    if view == "ego":
        if aria_calibration is None:
            raise ValueError("aria_calibration required for view='ego'")
        K = aria_calibration["PINHOLE"]["K"]  # 3x3
        T_device_camera = aria_calibration["PINHOLE"]["T_device_camera"]               # 4x4
        T_camera_rectified = aria_calibration["PINHOLE"]["pinhole_T_device_camera"]    # 4x4
        T = np.linalg.inv(T_camera_rectified) @ np.linalg.inv(T_device_camera)         # Device -> Pinhole
        suffix = "device"
    else:  # exo
        if (iphone_calibration is None) or (iphone_extrinsic is None):
            raise ValueError("iphone_calibration and iphone_extrinsic required for view='exo'")
        K = iphone_calibration["PINHOLE"]["K"]  # 3x3
        T = iphone_extrinsic                    # 4x4, World -> Camera
        suffix = "world"

    # Sample frames
    sample_df = merged_df.sample(n=n_views)

    # Visualize
    for _, row in sample_df.iterrows():
        # Collect 21 landmarks in the selected frame (device or world)
        col_tx = [f"tx_right_landmark_{i}_{suffix}" for i in range(21)]
        col_ty = [f"ty_right_landmark_{i}_{suffix}" for i in range(21)]
        col_tz = [f"tz_right_landmark_{i}_{suffix}" for i in range(21)]

        right_landmarks_tx = row[col_tx].to_numpy()
        right_landmarks_ty = row[col_ty].to_numpy()
        right_landmarks_tz = row[col_tz].to_numpy()
        right_landmarks = np.stack([right_landmarks_tx, right_landmarks_ty, right_landmarks_tz], axis=-1)  # (21,3)

        right_wrist = right_landmarks[5]
        right_palm  = right_landmarks[20]

        # Load image
        image_path = row["frame_path"]
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        # Plot
        plt.figure(figsize=(10, 10))
        plt.imshow(image)

        if mode == "landmarks":
            proj = np.array([project_point_into_image(p, K, T) for p in right_landmarks])
            plt.scatter(proj[:, 0], proj[:, 1], c='r', s=20, label='Right Landmarks')
        else:
            uw = project_point_into_image(right_wrist, K, T)
            up = project_point_into_image(right_palm,  K, T)
            plt.scatter(uw[0], uw[1], c='r', s=100, label='Right Wrist')
            plt.scatter(up[0], up[1], c='g', s=100, label='Right Palm')

        plt.legend()
        plt.title(f"{view.upper()} | {os.path.basename(image_path)}")
        plt.axis('off')
        plt.show()

def visualize_pair_side_by_side(
    df_merged_total: pd.DataFrame,
    extracted_iphone_frames_df: pd.DataFrame,   # columns: frame_path_iphone, timestamp (sorted ascending)
    K_ego: np.ndarray,  T_ego: np.ndarray,       # Device -> Pinhole
    K_exo: np.ndarray,  T_exo: np.ndarray,       # World  -> Camera
    n_views: int = 2,
):
    """
    For each sampled ARIA+iPhone match, show:
        [ARIA] | [iPhone prev] | [iPhone matched] | [iPhone next]
    Projections follow your original logic.
    Panel titles show Δt (ns) of iPhone frames relative to the ARIA frame.
    """

    extracted_iphone_frames_df = extracted_iphone_frames_df.sort_values("timestamp").reset_index(drop=True)
    path_to_index = {r["frame_path_iphone"]: i for i, r in extracted_iphone_frames_df.iterrows()}

    sample_df = df_merged_total.sample(n=n_views) if n_views > 0 else df_merged_total.iloc[[0]]

    for _, row in sample_df.iterrows():
        # --- Landmarks (device & world) ---
        idxs = range(21)
        lmk_dev = np.stack([
            row[[f"tx_right_landmark_{i}_device" for i in idxs]],
            row[[f"ty_right_landmark_{i}_device" for i in idxs]],
            row[[f"tz_right_landmark_{i}_device" for i in idxs]]
        ], axis=-1).astype(float)

        lmk_w = np.stack([
            row[[f"tx_right_landmark_{i}_world" for i in idxs]],
            row[[f"ty_right_landmark_{i}_world" for i in idxs]],
            row[[f"tz_right_landmark_{i}_world" for i in idxs]]
        ], axis=-1).astype(float)

        # --- Images & timestamps ---
        ego_path = row["frame_path_aria"]
        img_ego = cv2.cvtColor(cv2.imread(ego_path), cv2.COLOR_BGR2RGB)

        mid_path = row["frame_path_iphone"]
        img_mid = cv2.cvtColor(cv2.imread(mid_path), cv2.COLOR_BGR2RGB)

        j_mid = path_to_index.get(mid_path, None)

        def load_neighbor(j):
            if j is None or j < 0 or j >= len(extracted_iphone_frames_df):
                return None, None, None
            r = extracted_iphone_frames_df.iloc[j]
            img = cv2.imread(r["frame_path_iphone"])
            if img is None:
                return None, None, None
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), r["frame_path_iphone"], int(r["timestamp"])

        img_prev, prev_path, ts_prev = load_neighbor(j_mid - 1 if j_mid is not None else None)
        img_next, next_path, ts_next = load_neighbor(j_mid + 1 if j_mid is not None else None)

        ts_aria = int(row["timestamp"])
        ts_mid  = int(extracted_iphone_frames_df.iloc[j_mid]["timestamp"]) if j_mid is not None else None

        # --- Projection (unchanged) ---
        proj_ego = np.array([project_point_into_image(p, K_ego, T_ego) for p in lmk_dev])  # device -> cam
        proj_exo = np.array([project_point_into_image(p, K_exo, T_exo) for p in lmk_w])    # world  -> cam

        # --- Δt relative to ARIA ---
        dt_prev = f"{ts_prev - ts_aria:+d}" if ts_prev is not None else "n/a"
        dt_mid  = f"{ts_mid  - ts_aria:+d}" if ts_mid  is not None else "n/a"
        dt_next = f"{ts_next - ts_aria:+d}" if ts_next is not None else "n/a"

        # --- Plot ---
        fig, axes = plt.subplots(1, 4, figsize=(22, 6))
        panels = [
            ("EGO", img_ego, proj_ego),
            (f"EXO prev (Δt={dt_prev} ns)", img_prev, proj_exo),
            (f"EXO mid (Δt={dt_mid} ns)",  img_mid,  proj_exo),
            (f"EXO next (Δt={dt_next} ns)", img_next, proj_exo),
        ]

        for ax, (title, img, proj) in zip(axes, panels):
            if img is None:
                ax.set_title(f"{title}\n(n/a)")
                ax.axis("off")
                continue
            ax.imshow(img)
            ax.scatter(proj[:, 0], proj[:, 1], s=20, c='r', label='Right Landmarks')
            ax.set_title(title)
            ax.axis('off')
            if "EGO" in title or "mid" in title:
                ax.legend(loc='lower right')

        ego_name = os.path.basename(ego_path)
        mid_name = os.path.basename(mid_path)
        fig.suptitle(f"EGO: {ego_name} | iPhone(mid): {mid_name}", y=0.98, fontsize=11)
        plt.tight_layout()
        plt.show()

def visualize_ego_exo_pair(
    df_merged_total: pd.DataFrame,            # must contain frame_path_aria, frame_path_iphone, timestamp and landmark cols
    K_ego: np.ndarray,  T_ego: np.ndarray,    # Device -> Pinhole (ARIA)
    K_exo: np.ndarray,  T_exo: np.ndarray,    # World  -> Camera (iPhone)
    n_views: int = 2,
):
    """
    For sampled ARIA+iPhone matches, show: [ARIA] | [iPhone matched]
    Panel titles show Δt (ns) of the iPhone frame relative to the ARIA frame.
    Uses device-frame landmarks for ARIA and world-frame landmarks for iPhone.
    """

    # choose samples
    sample_df = df_merged_total.sample(n=n_views) if n_views > 0 else df_merged_total.iloc[[0]]

    for _, row in sample_df.iterrows():
        # --- Landmarks ---
        idxs = range(21)
        lmk_dev = np.stack([
            row[[f"tx_right_landmark_{i}_device" for i in idxs]],
            row[[f"ty_right_landmark_{i}_device" for i in idxs]],
            row[[f"tz_right_landmark_{i}_device" for i in idxs]]
        ], axis=-1).astype(float)

        lmk_w = np.stack([
            row[[f"tx_right_landmark_{i}_world" for i in idxs]],
            row[[f"ty_right_landmark_{i}_world" for i in idxs]],
            row[[f"tz_right_landmark_{i}_world" for i in idxs]]
        ], axis=-1).astype(float)

        # --- Images & timestamps ---
        ego_path = row["frame_path_aria"]
        mid_path = row["frame_path_iphone"]
        img_ego = cv2.cvtColor(cv2.imread(ego_path), cv2.COLOR_BGR2RGB) if os.path.exists(ego_path) else None
        img_mid = cv2.cvtColor(cv2.imread(mid_path), cv2.COLOR_BGR2RGB) if os.path.exists(mid_path) else None

        ts_aria = int(row["timestamp"])
        # if your df has a separate iphone timestamp column, use that; otherwise reuse row["timestamp"]
        ts_mid = int(row["timestamp"])  # replace if you keep a distinct iphone ts column

        # --- Projections ---
        proj_ego = np.array([project_point_into_image(p, K_ego, T_ego) for p in lmk_dev])  # device -> cam
        proj_exo = np.array([project_point_into_image(p, K_exo, T_exo) for p in lmk_w])    # world  -> cam

        # --- Δt (iPhone relative to ARIA) ---
        dt_mid = f"{ts_mid - ts_aria:+d}"

        # --- Plot ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        panels = [
            ("EGO (ARIA)", img_ego, proj_ego),
            (f"EXO matched (Δt={dt_mid} ns)", img_mid, proj_exo),
        ]

        for ax, (title, img, proj) in zip(axes, panels):
            if img is None:
                ax.set_title(f"{title}\n(n/a)")
                ax.axis("off")
                continue
            ax.imshow(img)
            ax.scatter(proj[:, 0], proj[:, 1], s=20, c='r', label='Right Landmarks')
            ax.set_title(title)
            ax.axis('off')
            ax.legend(loc='lower right')

        ego_name = os.path.basename(ego_path)
        mid_name = os.path.basename(mid_path)
        fig.suptitle(f"EGO: {ego_name} | iPhone: {mid_name}", y=0.98, fontsize=11)
        plt.tight_layout()
        plt.show()

def row_to_newdays_json(
    row,
    side: str,
    img_w: int,
    img_h: int,
    *,
    K: np.ndarray,
    T: np.ndarray,
    frame: str,                     # "device" (ego/Aria) or "world" (exo/iPhone)
    conf_col: str | None = None,
    conf_thr: float = 0.99
):
    """
    Build NEWDays JSON by projecting 3D hand landmarks using existing
    project_point_into_image(point_3d, K, T).

    Reads columns:
        tx_{side}_landmark_{i}_{frame}, ty_..., tz_...
    where frame ∈ {"device", "world"}.
    No inside-image filtering (assumed done beforehand).
    """
    if frame not in ("device", "world"):
        raise ValueError("frame must be one of ['device', 'world']")
    if side not in ("right", "left"):
        raise ValueError("side must be 'right' or 'left'")

    # collect 3D landmarks from the correct frame
    col_tx = [f"tx_{side}_landmark_{i}_{frame}" for i in range(21)]
    col_ty = [f"ty_{side}_landmark_{i}_{frame}" for i in range(21)]
    col_tz = [f"tz_{side}_landmark_{i}_{frame}" for i in range(21)]

    tx = row[col_tx].to_numpy(dtype=float, copy=False)
    ty = row[col_ty].to_numpy(dtype=float, copy=False)
    tz = row[col_tz].to_numpy(dtype=float, copy=False)
    P  = np.stack([tx, ty, tz], axis=-1)  # (21,3)

    # optional tracking confidence
    conf_val = None
    if conf_col is not None and conf_col in row:
        try:
            conf_val = float(row[conf_col])
        except Exception:
            conf_val = None

    kps, scores, exist, occ = [], [], [], []
    hand_points_3d = []

    for i in range(21):
        # 3D -> camera frame (extrinsics only)
        x_cam, y_cam, z_cam = transform_point_to_camera(P[i], T)
        hand_points_3d.append([float(x_cam), float(y_cam), float(z_cam)])

        u, v = project_point_into_image(P[i], K, T)
        # we assume all points are valid (filtered already)
        kps.append([float(u), float(v)])
        exist.append(1.0)
        scores.append(1.0)
        occ.append(1.0 if (conf_val is not None and conf_val <= conf_thr) else 0.0)

    # bbox over all projected points (clipped to image just in case)
    pts = np.array(kps, dtype=float)
    x0, y0 = np.clip(pts.min(axis=0), [0, 0], [img_w - 1, img_h - 1]).astype(int)
    x1, y1 = np.clip(pts.max(axis=0), [0, 0], [img_w - 1, img_h - 1]).astype(int)
    bbox = [[int(x0), int(y0), int(x1), int(y1)]]

    return [{
        "keypoints": kps,
        "bbox": bbox,
        "keypoint_scores": scores,
        "occlusion": occ,
        "existence": exist,
        "sample": "normal",
        "keypoints_3d": hand_points_3d
    }]

def build_npz(img_dir: Path, npz_out: Path):
    """
    Build HaMeR NPZ without occlusion labels.
    - Reads per-image JSON next to each image.
    - Uses bbox for center/scale (pixels).
    - Writes hand_keypoints_2d with vis=1 for all joints.
    """
    img_paths = sorted(list(img_dir.glob("*.jpg"))) or sorted(list(img_dir.glob("*.png")))

    imgname, centers, scales, kp2d, right = [], [], [], [], []
    kp3d = []

    for img in img_paths:
        j = img.with_suffix(".json")
        if not j.exists():
            continue

        ann = json.loads(j.read_text())[0]
        kps = np.asarray(ann["keypoints"], dtype=np.float32)  # (21,2)
        kps = kps[ARIA_TO_MANO_INDEX]  # reorder to MANO

        kps_3d = np.asarray(ann["keypoints_3d"], dtype=np.float32)  # (21,3)
        kps_3d = kps_3d[ARIA_TO_MANO_INDEX]  # reorder to MANO

        # bbox -> center/scale (pixels)
        if "bbox" in ann and len(ann["bbox"]) and len(ann["bbox"][0]) == 4:
            x0, y0, x1, y1 = map(float, ann["bbox"][0])
        else:
            # fallback: bbox from keypoints
            x0, y0 = float(kps[:,0].min()), float(kps[:,1].min())
            x1, y1 = float(kps[:,0].max()), float(kps[:,1].max())
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        box_size = max(x1 - x0, y1 - y0)

        vis = np.ones((21, 1), dtype=np.float32)   # all visible
        # vis[1, 0] = 0.0
        kps_2d = np.concatenate([kps, vis], axis=1)  # (21,3) [u,v,vis]
        kps_3d = np.concatenate([kps_3d, vis], axis=1)  # (21,4) [x,y,z,vis]
        

        imgname.append(img.name)
        centers.append([cx, cy])
        scales.append([box_size, box_size])  # store pixels; HaMeR divides by 200 internally
        kp2d.append(kps_2d)
        kp3d.append(kps_3d)
        right.append(1)

    if not imgname:
        raise RuntimeError(f"No (image,json) pairs found in {img_dir}")

    npz_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_out,
        imgname=np.asarray(imgname),
        center=np.asarray(centers, dtype=np.float32),
        scale=np.asarray(scales, dtype=np.float32),
        hand_keypoints_2d=np.asarray(kp2d, dtype=np.float32),
        hand_keypoints_3d=np.asarray(kp3d, dtype=np.float32),
        right=np.asarray(right, dtype=np.int64),
    )
    print(f"[NPZ BUILDER] Wrote {len(imgname)} samples → {npz_out}")

def visualize_pair_with_neighbor_iphone(
    df_merged_total: pd.DataFrame,              # contains frame_path_aria, frame_path_iphone, neighbor path cols
    extracted_iphone_frames_df: pd.DataFrame,   # columns: frame_path_iphone, timestamp (sorted ascending)
    K_ego: np.ndarray,  T_ego: np.ndarray,      # Device -> Pinhole (ARIA)
    K_exo: np.ndarray,  T_exo: np.ndarray,      # World  -> Camera (iPhone)
    n_views: int = 2,
):
    """
    Shows: [ARIA] | [iPhone -2] | [iPhone -1] | [iPhone mid] | [iPhone +1] | [iPhone +2]
    Titles display Δt (ns) relative to the ARIA timestamp.
    Uses device-frame landmarks for ARIA, world-frame landmarks for all iPhone panels.
    """

    # lookup: path -> timestamp
    iph_sorted = extracted_iphone_frames_df.sort_values("timestamp").reset_index(drop=True)
    ts_by_path = dict(zip(iph_sorted["frame_path_iphone"], iph_sorted["timestamp"]))

    # sampling rows
    sample_df = df_merged_total.sample(n=n_views) if n_views > 0 else df_merged_total.iloc[[0]]

    for _, row in sample_df.iterrows():
        idxs = range(21)

        # --- Landmarks from the merged row ---
        lmk_dev = np.stack([
            row[[f"tx_right_landmark_{i}_device" for i in idxs]],
            row[[f"ty_right_landmark_{i}_device" for i in idxs]],
            row[[f"tz_right_landmark_{i}_device" for i in idxs]],
        ], axis=-1).astype(float)

        lmk_w = np.stack([
            row[[f"tx_right_landmark_{i}_world" for i in idxs]],
            row[[f"ty_right_landmark_{i}_world" for i in idxs]],
            row[[f"tz_right_landmark_{i}_world" for i in idxs]],
        ], axis=-1).astype(float)

        # --- Paths present in this row (already merged) ---
        ego_path = row.get("frame_path_aria", None)
        mid_path = row.get("frame_path_iphone", None)
        p_m2 = row.get("frame_path_iphone_(-2)", None)
        p_m1 = row.get("frame_path_iphone_(-1)", None)
        p_p1 = row.get("frame_path_iphone_(+1)", None)
        p_p2 = row.get("frame_path_iphone_(+2)", None)

        # --- Helper to load image and ts by path ---
        def load_img_and_ts(path):
            if path is None or not isinstance(path, str):
                return None, None
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                return None, ts_by_path.get(path, None)
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), ts_by_path.get(path, None)

        img_ego = cv2.cvtColor(cv2.imread(ego_path), cv2.COLOR_BGR2RGB) if ego_path and os.path.exists(ego_path) else None
        img_m2, ts_m2 = load_img_and_ts(p_m2)
        img_m1, ts_m1 = load_img_and_ts(p_m1)
        img_mid, ts_mid = load_img_and_ts(mid_path)
        img_p1, ts_p1 = load_img_and_ts(p_p1)
        img_p2, ts_p2 = load_img_and_ts(p_p2)

        ts_aria = int(row["timestamp"])

        # --- Projections (same world landmarks reused for all iPhone panels) ---
        proj_ego = np.array([project_point_into_image(p, K_ego, T_ego) for p in lmk_dev])
        proj_exo = np.array([project_point_into_image(p, K_exo, T_exo) for p in lmk_w])

        # --- Δt strings ---
        def dts(t): return f"{(int(t) - ts_aria):+d}" if t is not None else "n/a"
        dt_m2, dt_m1, dt_mid, dt_p1, dt_p2 = map(dts, [ts_m2, ts_m1, ts_mid, ts_p1, ts_p2])

        # --- Plot: 1x6 ---
        fig, axes = plt.subplots(1, 6, figsize=(34, 6))
        panels = [
            ("EGO", img_ego, proj_ego),
            (f"EXO -2 (Δt={dt_m2} ns)",  img_m2,  proj_exo),
            (f"EXO -1 (Δt={dt_m1} ns)",  img_m1,  proj_exo),
            (f"EXO mid (Δt={dt_mid} ns)", img_mid, proj_exo),
            (f"EXO +1 (Δt={dt_p1} ns)",  img_p1,  proj_exo),
            (f"EXO +2 (Δt={dt_p2} ns)",  img_p2,  proj_exo),
        ]

        for ax, (title, img, proj) in zip(axes, panels):
            if img is None:
                ax.set_title(f"{title}\n(n/a)")
                ax.axis("off")
                continue
            ax.imshow(img)
            ax.scatter(proj[:, 0], proj[:, 1], s=20, c='r', label='Right Landmarks')
            ax.set_title(title)
            ax.axis('off')
            if "EGO" in title or "mid" in title:
                ax.legend(loc='lower right')

        ego_name = os.path.basename(ego_path) if ego_path else "n/a"
        mid_name = os.path.basename(mid_path) if mid_path else "n/a"
        fig.suptitle(f"EGO: {ego_name} | iPhone(mid): {mid_name}", y=0.98, fontsize=11)
        plt.tight_layout()
        plt.show()

def interpolate_world_landmarks_to_iphone(
    df_merged_aria: pd.DataFrame,
    extracted_iphone_frames_df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    max_gap_ns: int = int(30e6),   # strict cutoff: only interpolate if t1-t0 <= max_gap_ns
) -> pd.DataFrame:
    """
    Strict interpolation of world-frame hand landmarks from df_merged_aria to iPhone timestamps.
    Rows that fall inside Aria discontinuities (hands not tracked) will get NaNs.

    Returns a copy of extracted_iphone_frames_df with interpolated landmark columns added.
    """
    # 1) pick only world-frame landmark columns (tx/ty/tz for left/right landmarks)
    pat = re.compile(r'^(?:t[xyz])_(?:left|right)_landmark_\d+_world$')
    landmark_cols = [c for c in df_merged_aria.columns if pat.match(c)]
    if not landmark_cols:
        return extracted_iphone_frames_df.copy()

    # 2) Aria source (timestamps + landmarks); preserve right key as 'aria_ts'
    aria_src = df_merged_aria[[timestamp_col] + landmark_cols].copy()
    aria_src = aria_src.sort_values(timestamp_col).drop_duplicates(subset=timestamp_col, keep="first")
    aria_src["aria_ts"] = aria_src[timestamp_col].astype("int64")

    # 3) Targets (iPhone), keep order
    iph = extracted_iphone_frames_df[[timestamp_col]].copy()
    iph["_ord"] = np.arange(len(iph))
    iph_sorted = iph.sort_values(timestamp_col)

    # 4) Bracketing merges (carry 'aria_ts' so we know true prev/next times)
    prev_df = pd.merge_asof(iph_sorted, aria_src, on=timestamp_col, direction="backward") \
               .rename(columns={"aria_ts": "aria_ts_prev"})
    next_df = pd.merge_asof(iph_sorted, aria_src, on=timestamp_col, direction="forward") \
               .rename(columns={"aria_ts": "aria_ts_next"})

    # 5) Side-by-side prev/next tables
    prev_slim = prev_df[["_ord", timestamp_col, "aria_ts_prev"] + landmark_cols]
    next_slim = next_df[["_ord", timestamp_col, "aria_ts_next"] + landmark_cols]
    merged = prev_slim.merge(next_slim, on=["_ord", timestamp_col], how="left", suffixes=("_prev", "_next"))

    # 6) Time math (strict bracketing + gap check)
    t  = merged[timestamp_col].astype("int64").to_numpy()
    t0 = merged["aria_ts_prev"].to_numpy(dtype="float64")
    t1 = merged["aria_ts_next"].to_numpy(dtype="float64")

    has_prev = np.isfinite(t0)
    has_next = np.isfinite(t1)
    denom = t1 - t0

    with np.errstate(divide='ignore', invalid='ignore'):
        alpha = (t - t0) / denom
    alpha = np.where(np.isfinite(alpha), alpha, 0.0)

    gap_ok = np.isfinite(denom) & (denom <= float(max_gap_ns))
    can_interp_time = has_prev & has_next & gap_ok

    # 7) Interpolate per column (strict: require values on both sides)
    out = pd.DataFrame({"_ord": merged["_ord"], timestamp_col: merged[timestamp_col]})
    for col in landmark_cols:
        p = merged[f"{col}_prev"].to_numpy(dtype="float64")
        n = merged[f"{col}_next"].to_numpy(dtype="float64")
        has_prev_val = np.isfinite(p)
        has_next_val = np.isfinite(n)

        can_interp = can_interp_time & has_prev_val & has_next_val

        vals = np.full_like(alpha, np.nan, dtype="float64")
        idx = np.where(can_interp)[0]
        if idx.size:
            a = alpha[idx]
            vals[idx] = (1.0 - a) * p[idx] + a * n[idx]

        out[col] = vals

    # 8) Restore original iPhone order and attach to its df
    out = out.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)
    return extracted_iphone_frames_df.merge(out, on=timestamp_col, how="left")

def merge_iphone_into_aria_dataframe(
    df_merged_aria: pd.DataFrame,
    df_merged_iphone: pd.DataFrame,
    timestamp_col: str = "timestamp",
    tolerance_ns: int = int(30e6),  # 30 ms
) -> pd.DataFrame:
    """
    Drop Aria *_landmark_*_world columns and merge the full iPhone dataframe
    (including its world landmarks) by nearest timestamp.
    Keeps Aria's row count; unmatched iPhone data becomes NaN.
    """
    # 1) Drop Aria *_landmark_*_world columns
    pat = re.compile(r'^(?:t[xyz])_(?:left|right)_landmark_\d+_world$')
    aria = df_merged_aria.drop(columns=[c for c in df_merged_aria.columns if pat.match(c)], errors="ignore")

    # 2) Sort both by timestamp (required for merge_asof)
    aria = aria.sort_values(timestamp_col).reset_index(drop=True)
    iphone = df_merged_iphone.sort_values(timestamp_col).reset_index(drop=True)

    # 3) Merge-asof: align by nearest timestamp (keep Aria row count)
    merged = pd.merge_asof(
        aria,
        iphone,
        on=timestamp_col,
        direction="nearest",
        tolerance=tolerance_ns,
    )

    return merged

def add_iphone_neighbor_paths(
    df_combined: pd.DataFrame,
    extracted_iphone_frames_df: pd.DataFrame,
    path_col: str = "frame_path_iphone",
) -> pd.DataFrame:
    """
    Adds 4 columns to df_combined:
        - frame_path_iphone_(-2)
        - frame_path_iphone_(-1)
        - frame_path_iphone_(+1)
        - frame_path_iphone_(+2)

    Uses the order in extracted_iphone_frames_df (assumed sorted by timestamp).
    Rows without a matched iPhone frame (NaN in frame_path_iphone) or out-of-bounds neighbors get NaN.
    """
    frames_sorted = extracted_iphone_frames_df.sort_values("timestamp").reset_index(drop=True)
    paths = frames_sorted[path_col].to_list()
    path_to_idx = {p: i for i, p in enumerate(paths)}
    N = len(paths)

    # current iPhone index for each row (NaN if no iphone match)
    idx_series = df_combined[path_col].map(path_to_idx)

    def neighbor(idx, offset):
        if pd.isna(idx):
            return np.nan
        j = int(idx) + offset
        if 0 <= j < N:
            return paths[j]
        return np.nan

    out = df_combined.copy()
    out["frame_path_iphone_(-2)"] = idx_series.apply(lambda i: neighbor(i, -2))
    out["frame_path_iphone_(-1)"] = idx_series.apply(lambda i: neighbor(i, -1))
    out["frame_path_iphone_(+1)"] = idx_series.apply(lambda i: neighbor(i, +1))
    out["frame_path_iphone_(+2)"] = idx_series.apply(lambda i: neighbor(i, +2))
    return out

def add_right_landmark5_speed(df: pd.DataFrame, timestamp_col: str = "timestamp", max_gap_ns: float | None = 100e6) -> pd.DataFrame:
    """
    Adds 'speed_right_landmark_5_world' using central differences.
    If max_gap_ns is set, large timestamp gaps yield NaN speeds.
    """
    df = df.sort_values(timestamp_col).reset_index(drop=True).copy()

    pos = df[[f"tx_right_landmark_5_world", 
              f"ty_right_landmark_5_world", 
              f"tz_right_landmark_5_world"]].to_numpy(dtype=float)
    t_ns = df[timestamp_col].to_numpy(dtype=float)
    t = t_ns * 1e-9  # seconds

    vel = np.full_like(pos, np.nan)

    # compute time deltas in ns
    dt_forward = np.diff(t_ns, append=np.nan)
    dt_backward = np.diff(np.insert(t_ns, 0, np.nan))  # prepend NaN for same length

    # central difference
    for i in range(1, len(df) - 1):
        gap_prev = t_ns[i] - t_ns[i - 1]
        gap_next = t_ns[i + 1] - t_ns[i]
        if max_gap_ns is not None and (gap_prev > max_gap_ns or gap_next > max_gap_ns):
            continue  # too large gap → leave as NaN
        dt = t[i + 1] - t[i - 1]
        if dt > 0:
            vel[i] = (pos[i + 1] - pos[i - 1]) / dt

    # forward/backward edges
    if len(df) > 1:
        gap_first = t_ns[1] - t_ns[0]
        gap_last = t_ns[-1] - t_ns[-2]
        if max_gap_ns is None or gap_first <= max_gap_ns:
            vel[0] = (pos[1] - pos[0]) / (t[1] - t[0])
        if max_gap_ns is None or gap_last <= max_gap_ns:
            vel[-1] = (pos[-1] - pos[-2]) / (t[-1] - t[-2])

    speed = np.linalg.norm(vel, axis=1)
    df["speed_right_landmark_5_world"] = speed

    return df

def add_right_landmark5_distance_to_iphone(
    df: pd.DataFrame,
    aligned_iphone_poses_df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    max_gap_ns: float | None = 100e6,
) -> pd.DataFrame:
    """
    Adds a column 'dist_right_landmark_5_to_iphone' giving the Euclidean distance [m]
    between right_landmark_5_world and the iPhone camera center in world coordinates.

    - Matches iPhone pose to each df row by nearest timestamp (merge_asof).
    - Skips (NaN) distances if timestamp gap exceeds max_gap_ns (default 100 ms).
    """
    df = df.sort_values(timestamp_col).reset_index(drop=True).copy()

    # --- prepare iphone poses ---
    poses = aligned_iphone_poses_df.copy()
    poses = poses.sort_values("timestamp").reset_index(drop=True)
    poses["timestamp"] = poses["timestamp"].astype(np.int64)

    # --- minimal columns for merge ---
    poses_slim = poses[
        ["timestamp", "tx_world_cam", "ty_world_cam", "tz_world_cam"]
    ].copy()

    # --- merge-asof to find nearest iphone pose ---
    merged = pd.merge_asof(
        df,
        poses_slim,
        on=timestamp_col,
        direction="nearest",
        tolerance=max_gap_ns if max_gap_ns is not None else np.inf,
    )

    # --- compute 3D distance ---
    p_lmk = merged[
        [f"tx_right_landmark_5_world", f"ty_right_landmark_5_world", f"tz_right_landmark_5_world"]
    ].to_numpy(dtype=float)
    p_cam = merged[
        ["tx_world_cam", "ty_world_cam", "tz_world_cam"]
    ].to_numpy(dtype=float)

    dist = np.linalg.norm(p_lmk - p_cam, axis=1)

    # mark NaN if no matching pose
    mask_valid = merged[["tx_world_cam", "ty_world_cam", "tz_world_cam"]].notna().all(axis=1)
    dist[~mask_valid] = np.nan

    df["dist_right_landmark_5_to_iphone"] = dist

    return df


def split_by_landmark5_distance(df: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Splits the dataframe into four subsets based on 'dist_right_landmark_5_to_iphone':
        - close:     < 2.0 m
        - far:     >= 2.0 m


    Returns: [df_close, df_medium, df_far, df_very_far]
    """
    col = "dist_right_landmark_5_to_iphone"
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in dataframe")

    df_close = df[df[col] < 2.0].copy()
    df_far = df[df[col] >= 2.0].copy()

    return [df_close, df_far]


def append_dataset_to_yaml(
    yaml_file: str | Path,
    rec_location: str,
    interaction_index: int,
    view: str,
    exo_tag: str,
    suffix: str = "",
    dist_suffix: str = "",
    data_out_root: str | Path = ".",
    keypoint_count: int = 21,
):
    """
    Appends a dataset entry to a YAML config file, keeping KEYPOINT_LIST inline.
    Example output:

        kitchen_2_3_exo_close:
            TYPE: ImageDataset
            DATASET_FILE: hamer_evaluation_data/exo_close.npz
            IMG_DIR: /data/output/ground_truth/kitchen_2_3_exo_close
            KEYPOINT_LIST: [0, 1, 2, ..., 20]  # Dummy
    """

    yaml_file = Path(yaml_file)
    data_out_root = Path(data_out_root)

    # Build dataset metadata
    name_data_split = f"{rec_location}_{interaction_index}_{view}{suffix}{dist_suffix}"
    dataset_file = f"hamer_evaluation_data/{exo_tag}{suffix}{dist_suffix}.npz"
    img_dir = data_out_root / "ground_truth" / name_data_split
    keypoint_list = ", ".join(str(i) for i in range(keypoint_count))

    # Append manually to preserve inline format
    with open(yaml_file, "a") as yaml_out:
        yaml_out.write(f"{name_data_split}:\n")
        yaml_out.write(f"    TYPE: ImageDataset\n")
        yaml_out.write(f"    DATASET_FILE: {dataset_file}\n")
        yaml_out.write(f"    IMG_DIR: {img_dir}\n")
        yaml_out.write(f"    KEYPOINT_LIST: [{keypoint_list}]  # Dummy\n")

    print(f"Added dataset entry: {name_data_split}")

def remove_rows_with_landmark5_behind(
    df: pd.DataFrame,
    T_world_to_cam: np.ndarray,
    eps: float = 0.0,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Removes rows where right_landmark_5_world lies behind the image plane
    (i.e. z_cam <= eps after transforming world → camera).

    Returns a pruned copy of df.
    """
    df = df.sort_values(timestamp_col).reset_index(drop=True).copy()

    # Columns for landmark 5
    cols_right = [
        "tx_right_landmark_5_world",
        "ty_right_landmark_5_world",
        "tz_right_landmark_5_world",
    ]

    cols_y = [
        "ty_right_landmark_5_world",
        "tz_right_landmark_5_world",
        "tx_right_landmark_5_world",
    ]

    # Extract and homogenize points
    P_right = df[cols_right].to_numpy(dtype=float)
    Ph_right = np.concatenate([P_right, np.ones((len(df), 1), dtype=float)], axis=1)

    P_left = df[cols_y].to_numpy(dtype=float)
    Ph_left = np.concatenate([P_left, np.ones((len(df), 1), dtype=float)], axis=1)

    # Transform world → camera
    Pc_right = (Ph_right @ T_world_to_cam.T)[:, :3]
    z_cam_right = Pc_right[:, 2]

    Pc_left = (Ph_left @ T_world_to_cam.T)[:, :3]
    z_cam_left = Pc_left[:, 2]

    # add z_cam column 
    df["z_right_landmark_5_to_cam"] = z_cam_right
    df["z_left_landmark_5_to_cam"] = z_cam_left

    # Keep only rows with z > eps
    mask_valid = z_cam_right > eps
    df_pruned = df.loc[mask_valid].reset_index(drop=True)

    print(f"Removed {np.count_nonzero(~mask_valid)} frames where landmark 5 was behind the camera.")
    return df_pruned

def bilinear_sample_depth(depth, u, v):
    H, W = depth.shape[:2]
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    u0 = np.floor(u).astype(int)
    v0 = np.floor(v).astype(int)
    u1 = u0 + 1
    v1 = v0 + 1

    inb = (u0 >= 0) & (u1 < W) & (v0 >= 0) & (v1 < H)
    out = np.full_like(u, np.nan, dtype=float)
    if not np.any(inb):
        return out

    du = u - u0
    dv = v - v0

    f00 = depth[v0[inb], u0[inb]]
    f10 = depth[v0[inb], u1[inb]]
    f01 = depth[v1[inb], u0[inb]]
    f11 = depth[v1[inb], u1[inb]]

    w00 = (1 - du[inb]) * (1 - dv[inb])
    w10 = du[inb]        * (1 - dv[inb])
    w01 = (1 - du[inb])  * dv[inb]
    w11 = du[inb]        * dv[inb]

    neigh = np.stack([f00, f10, f01, f11], axis=-1)
    w = np.stack([w00, w10, w01, w11], axis=-1)

    valid = (neigh > 0) & np.isfinite(neigh)
    w = np.where(valid, w, 0.0)
    wsum = np.sum(w, axis=-1)
    good = wsum > 0

    val = np.sum(w * np.where(valid, neigh, 0.0), axis=-1)
    val = np.where(good, val / wsum, np.nan)

    out[inb] = val
    return out

def rgb_to_depth_coords(u_r, v_r, depth_shape, rgb_shape):
    H_d, W_d = depth_shape[:2]
    H_r, W_r = rgb_shape[:2]
    sx = W_d / W_r
    sy = H_d / H_r
    return u_r * sx, v_r * sy

def add_landmark5_occlusion_flag(
    df: pd.DataFrame,
    aligned_iphone_poses_df: pd.DataFrame,
    K: np.ndarray,
    T_world_to_cam, 
    rgb_shape=(1440, 1920), # (H, W) of RGB
    tau=0.03,               # meters; tune 0.02–0.05
    ensure_distance=True,
) -> pd.DataFrame:
    """
    Adds columns:
      - lm5_u, lm5_v: projected RGB pixel
      - lm5_ud, lm5_vd: mapped depth pixel
      - lm5_depth_meas: sampled LiDAR depth [m]
      - lm5_depth_expected: expected range [m] (cam->landmark5)
      - lm5_depth_diff: measured - expected
      - lm5_occluded: bool, True if measured + tau < expected

    Assumptions:
      - depth maps are RGB-aligned downsamples (same orientation).
      - df has 'frame_path_depth' with .npy paths.
      - df has landmark5 world coords in columns:
          tx_right_landmark_5_world, ty_right_landmark_5_world, tz_right_landmark_5_world
      - T_world_to_cam_getter(row) returns extrinsics for this row/frame (used by your projection)
      - project_point_into_image(P_world, K, T) exists and returns (u,v)
    """
    out = df.copy()

    # 1) make sure we have the expected RANGE (Euclidean cam->point), using your helper if needed
    # if ensure_distance and "dist_right_landmark_5_to_iphone" not in out.columns:
    #     out = add_right_landmark5_distance_to_iphone(
    #         out, aligned_iphone_poses_df, timestamp_col="timestamp"
    #     )
    # right
    lm5_u_right = np.full(len(out), np.nan)
    lm5_v_right = np.full(len(out), np.nan)
    lm5_ud_right = np.full(len(out), np.nan)
    lm5_vd_right = np.full(len(out), np.nan)
    d_meas_right = np.full(len(out), np.nan)
    d_exp_right  = np.full(len(out), np.nan)

    #left 
    lm5_u_left = np.full(len(out), np.nan)
    lm5_v_left = np.full(len(out), np.nan)
    lm5_ud_left = np.full(len(out), np.nan)
    lm5_vd_left = np.full(len(out), np.nan)
    d_meas_left = np.full(len(out), np.nan)
    d_exp_left  = np.full(len(out), np.nan)

    # 2) iterate rows (depth path differs per frame)
    for i, row in out.iterrows():
        depth_path = row.get("frame_path_depth", None)
        if not depth_path or not Path(depth_path).exists():
            continue

        # expected range (meters) from your precomputed column
        exp_range_right = row.get("z_right_landmark_5_to_cam", np.nan)
        exp_range_left = row.get("z_left_landmark_5_to_cam", np.nan)

        # landmark 5 in world coords
        Pw_right = np.array([
            row["tx_right_landmark_5_world"],
            row["ty_right_landmark_5_world"],
            row["tz_right_landmark_5_world"],
        ], dtype=float)

        Pw_left = np.array([
            row["ty_right_landmark_5_world"],
            row["tz_right_landmark_5_world"],
            row["tx_right_landmark_5_world"],
        ], dtype=float)

        # get extrinsics for this frame (whatever your projection expects)
        T = T_world_to_cam

        # project to RGB pixel (your function)
        try:
            ur_right, vr_right = project_point_into_image(Pw_right, K, T) 
            ur_left, vr_left = project_point_into_image(Pw_left, K, T)
        except Exception:
            continue

        # map to depth coords and sample
        depth = np.load(depth_path)
        ud_right, vd_right = rgb_to_depth_coords(ur_right, vr_right, depth.shape, rgb_shape)
        ud_left, vd_left = rgb_to_depth_coords(ur_left, vr_left, depth.shape, rgb_shape)
        D_right = bilinear_sample_depth(depth, np.array([ud_right]), np.array([vd_right]))[0]
        D_left = bilinear_sample_depth(depth, np.array([ud_left]), np.array([vd_left]))[0]

        # store
        lm5_u_right[i], lm5_v_right[i] = ur_right, vr_right
        lm5_ud_right[i], lm5_vd_right[i] = ud_right, vd_right
        d_meas_right[i] = D_right
        d_exp_right[i]  = exp_range_right

        lm5_u_left[i], lm5_v_left[i] = ur_left, vr_left
        lm5_ud_left[i], lm5_vd_left[i] = ud_left, vd_left
        d_meas_left[i] = D_left
        d_exp_left[i]  = exp_range_left

    # write results
    out["lm5_u_right"] = lm5_u_right
    out["lm5_v_right"] = lm5_v_right
    out["lm5_ud_right"] = lm5_ud_right
    out["lm5_vd_right"] = lm5_vd_right
    out["lm5_depth_meas_right"] = d_meas_right
    out["lm5_depth_expected_right"] = d_exp_right
    out["lm5_depth_diff_right"] = out["lm5_depth_meas_right"] - out["lm5_depth_expected_right"]
    out["lm5_occluded_right"] = (out["lm5_depth_meas_right"] + 0.1 < out["lm5_depth_expected_right"])

    out["lm5_u_left"] = lm5_u_left
    out["lm5_v_left"] = lm5_v_left
    out["lm5_ud_left"] = lm5_ud_left
    out["lm5_vd_left"] = lm5_vd_left
    out["lm5_depth_meas_left"] = d_meas_left
    out["lm5_depth_expected_left"] = d_exp_left
    out["lm5_depth_diff_left"] = out["lm5_depth_meas_left"] - out["lm5_depth_expected_left"]
    out["lm5_occluded_left"] = (out["lm5_depth_meas_left"] + 0.1 < out["lm5_depth_expected_left"])

    out = out.loc[~out["lm5_occluded_right"]].reset_index(drop=True)

    return out


def generate_ego_exo_hand_pose_ground_truth(interaction_index: str, 
                                                            rec_location: str, 
                                                            base_path: Path, 
                                                            data_out_root: Path,
                                                            format: str,
                                                            data_indexer: RecordingIndex,
                                                            visualize: bool = False):
    """Goal: Generate pseudo ground truth hand pose annotations from aligned 
    Aria MPS data in the NewDays dataset format to plug into Hamer evaluation tools.
    For both ego (aria human) ande exo (iphone) views.
    
    """
    assert format in ["newdays"], "Only 'newdays' format supported for now"

    rec_type = "hand"

    queries_at_loc = data_indexer.query(
        location=rec_location, 
        interaction=rec_type, 
        recorder=None,
        interaction_index=interaction_index
    )

    # must be exactely one aria data loader and one or 2 iphone data loader
    # extract all recording module for the give UMI recording
    aria_data_loader = None
    iphone_data_loaders = []

    for loc, inter, rec, ii, path in queries_at_loc:
        print(f"Found recorder: {rec} at {path}")

        rec_type = inter
        rec_module = rec
        interaction_indices = ii
        if "aria" in rec:
            aria_data = AriaData(base_path, rec_location, rec_type, rec_module, interaction_indices, data_indexer)
        elif "iphone" in rec:
            iphone_data = IPhoneData(base_path, rec_location, rec_type, rec_module, interaction_indices, data_indexer)
            iphone_data_loaders.append(iphone_data)

    assert aria_data is not None, "No Aria data loader found"
    assert len(iphone_data_loaders) > 0, "No iPhone data loader found"

    ################################################################################
    # ARIA DATA
    ################################################################################

    # debugging with wrist and palm poses only
    # get aligned palm and wrist poses for debugging (timestamp col)
    conf_threshold = 0.99
    aligned_hand_tracking_df = aria_data.get_hand_tracking_aligned_df()
    # drop row if left_tracking_confidence or right_tracking_confidence is <= 0.8
    aligned_hand_tracking_df = aligned_hand_tracking_df[(aligned_hand_tracking_df['right_tracking_confidence'] > conf_threshold)]
    aligned_hand_tracking_df = aligned_hand_tracking_df.fillna(0)
    aligned_hand_tracking_df = aligned_hand_tracking_df.sort_values(by='timestamp').reset_index(drop=True)

    aligned_aria_poses_df = aria_data.get_closed_loop_trajectory_aligned()
    extracted_aria_frames = aria_data.get_extracted_frames()
    extracted_aria_frames_df = pd.DataFrame({
                                                        'frame_path_aria': [str(p) for p in extracted_aria_frames],
                                                        'timestamp': [int(Path(p).stem) for p in extracted_aria_frames],
                                                        })
    extracted_aria_frames_df = extracted_aria_frames_df.sort_values(by='timestamp').reset_index(drop=True)
    aria_device_to_camera_calibration = aria_data.get_calibration()
    df_merged_aria = pd.merge_asof(
                    aligned_hand_tracking_df, extracted_aria_frames_df,
                    on='timestamp', direction='nearest', tolerance=int(1e6)
                    )
    df_merged_aria = df_merged_aria.dropna().reset_index(drop=True)

    # filter out rows, where wrist and palm are projected outside of the aria image
    # necessary because aria mps tracking also tracks outside of the rectified camera view
    image_width_aria = aria_device_to_camera_calibration["PINHOLE"]["w"]
    image_height_aria = aria_device_to_camera_calibration["PINHOLE"]["h"]
    K_aria_pinhole = aria_device_to_camera_calibration["PINHOLE"]["K"] # 3x3
    T_device_camera = aria_device_to_camera_calibration["PINHOLE"]["T_device_camera"] # 4x4
    T_camera_rectified = aria_device_to_camera_calibration["PINHOLE"]["pinhole_T_device_camera"] # 4x4
    aria_T_rectified_device = np.linalg.inv(T_camera_rectified) @ np.linalg.inv(T_device_camera)
    T_extrinsic_aria = np.linalg.inv(T_camera_rectified) @ np.linalg.inv(T_device_camera)
    df_merged_aria = filter_points_inside_image(df=df_merged_aria, 
                                                intrinsic=K_aria_pinhole, 
                                                extrinsic=aria_T_rectified_device,
                                                frame="device", 
                                                image_width=image_width_aria, 
                                                image_height=image_height_aria)

    ################################################################################
    # IPHONE DATA
    ################################################################################
    iphone_data = iphone_data_loaders[1]
    aligned_iphone_poses_df = iphone_data.get_trajectory_aligned()
    iphone_camera_calibration = iphone_data.calibration
    image_width_iphone = iphone_camera_calibration["PINHOLE"]["w"]
    image_height_iphone = iphone_camera_calibration["PINHOLE"]["h"]
    K_iphone_pinhole = iphone_camera_calibration["PINHOLE"]["K"] # 3x3

    extracted_iphone_frames = iphone_data.get_extracted_frames()
    extracted_iphone_depths = iphone_data.get_extracted_depths()
    extracted_iphone_frames_df = pd.DataFrame({
                                            'frame_path_iphone': [str(p) for p in extracted_iphone_frames],
                                            'frame_path_depth': [str(p) for p in extracted_iphone_depths],
                                            'timestamp': [int(Path(p).stem) for p in extracted_iphone_frames],
                                            })
    extracted_iphone_frames_df = extracted_iphone_frames_df.sort_values(by='timestamp').reset_index(drop=True)
    # interpolate aria landmarks to iphone timestamps
    df_merged_iphone = interpolate_world_landmarks_to_iphone(
        df_merged_aria,
        extracted_iphone_frames_df,
        timestamp_col="timestamp",
        max_gap_ns=100e6  #ca 3x frame gap at 30fps (strict cutoff)
    )
    df_merged_iphone = df_merged_iphone.dropna().reset_index(drop=True)

    a = 2



    # df_merged_iphone = df_merged_iphone[(df_merged_iphone['right_tracking_confidence'] > conf_threshold)]
    # filter out rows, where wrist and palm are projected outside of the iphone image
    # necessary because hand tracked by aria mps might not be visible in iphone view
    extrinsic_iphone = aligned_iphone_poses_df.iloc[0][["tx_world_cam", "ty_world_cam", "tz_world_cam", "qw_world_cam", "qx_world_cam", "qy_world_cam", "qz_world_cam"]]
    T_extrinsic_iphone = np.eye(4)
    T_extrinsic_iphone[:3, 3] = extrinsic_iphone[["tx_world_cam", "ty_world_cam", "tz_world_cam"]].to_numpy()
    qw, qx, qy, qz = extrinsic_iphone[["qw_world_cam", "qx_world_cam", "qy_world_cam", "qz_world_cam"]]
    R = Rot.from_quat([qx, qy, qz, qw]).as_matrix()
    T_extrinsic_iphone[:3, :3] = R
    T_extrinsic_iphone = np.linalg.inv(T_extrinsic_iphone) # world to camera
    df_merged_iphone = filter_points_inside_image(df=df_merged_iphone, 
                                                intrinsic=K_iphone_pinhole, 
                                                extrinsic=T_extrinsic_iphone, # no extrinsics necessary as points are in world frame already
                                                frame="world", 
                                                image_width=image_width_iphone, 
                                                image_height=image_height_iphone)

    df_combined = merge_iphone_into_aria_dataframe(
        df_merged_aria=df_merged_aria,
        df_merged_iphone=df_merged_iphone,
        timestamp_col="timestamp",
        tolerance_ns=int(14e6),
    )
    df_combined = add_iphone_neighbor_paths(df_combined, extracted_iphone_frames_df)
    df_merged_total = df_combined.dropna().reset_index(drop=True)

    df_merged_total = add_right_landmark5_distance_to_iphone(
        df_merged_total,
        aligned_iphone_poses_df,
        timestamp_col="timestamp",
        max_gap_ns=int(100e6)
    )

    df_merged_total = remove_rows_with_landmark5_behind(
        df_merged_total,
        T_world_to_cam=T_extrinsic_iphone,  # world → iPhone cam
    )

    # add occlusion flag
    df_merged_total = add_landmark5_occlusion_flag(
        df_merged_total,
        aligned_iphone_poses_df,
        K=K_iphone_pinhole,
        T_world_to_cam=T_extrinsic_iphone,
        rgb_shape=(image_height_iphone, image_width_iphone),
        tau=0.03,
        ensure_distance=False,
    )


    # df_merged_total = df_merged_total.loc[df_merged_total["timestamp"] <= 1500000000000].reset_index(drop=True)
    # # 
    # a = 2

    plt.figure(figsize=(10,4))
    plt.plot(df_merged_total["timestamp"], df_merged_total["lm5_depth_expected_right"], '-', color='black', linewidth=1.5, label='Expected depth')
    plt.plot(df_merged_total["timestamp"], df_merged_total["lm5_depth_meas_right"], '.', color='steelblue', markersize=4, label='Measured depth')
    plt.xlabel("Timestamp [ns]")
    plt.ylabel("Depth [m]")
    plt.title("Landmark 5 — Measured vs Expected Depth Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()




    if visualize:
        visualize_pair_with_neighbor_iphone(
                                    df_merged_total,
                                    extracted_iphone_frames_df,
                                    K_ego=K_aria_pinhole, T_ego=T_extrinsic_aria, 
                                    K_exo=K_iphone_pinhole, T_exo=T_extrinsic_iphone,
                                    n_views=10)
        a = 2 

    # split according to distance to iphone
    dfs_split_by_distance = split_by_landmark5_distance(df_merged_total)

        
    if format == "newdays":
        ################################################################################
        # EXPORT TO NEWDAYS FORMAT
        ################################################################################
        # Unique stems for this recording
        exo_tag = f"HEJ_{rec_location}_{interaction_index}_EXO"
        ego_tag = f"HEJ_{rec_location}_{interaction_index}_EGO"

        # ---------------- EXO (iPhone) ----------------
        distance_suffixes = ["_close", "_far"]
        out_exo_suffix = ["_(-2)", "_(-1)", "", "_(+1)", "_(+2)"]


        # out_exo = data_out_root / "ground_truth" / f"hej_{rec_location}_{interaction_index}_exo"
        # out_exos = [data_out_root / "ground_truth" / f"hej_{rec_location}_{interaction_index}_exo{suffix}" for suffix in out_exo_suffix]
        # for suffix, out_exo in zip(out_exo_suffix, out_exos):
        #     for dist_suffix in distance_suffixes:
        #         dir_path = out_exo / dist_suffix
        #         dir_path.mkdir(parents=True, exist_ok=True)
        # for suffix in out_exo_suffix:
        #     for dist_suffix in distance_suffixes:
        #         dir_path = data_out_root / "ground_truth" / f"{rec_location}_{interaction_index}_exo{suffix}{dist_suffix}"
        #         dir_path.mkdir(parents=True, exist_ok=True)

        side = "right"

        yaml_file = data_out_root / "hamer/hamer/hamer" / "configs" / f"hej_{rec_location}_{interaction_index}_eval.yaml"
        for suffix in out_exo_suffix:
            for dist_suffix, df_split_by_distance in zip(distance_suffixes, dfs_split_by_distance):
                out_exo = data_out_root / "ground_truth" / f"{rec_location}_{interaction_index}_exo{suffix}{dist_suffix}"
                print(f"[EXPORT] EXO split '{suffix}{dist_suffix}': {len(df_split_by_distance)} samples")
                if len(df_split_by_distance) == 0:
                    continue

                for _, row in df_split_by_distance.iterrows():  # iPhone rows
                    payload = row_to_newdays_json(
                        row, side, image_width_iphone, image_height_iphone,
                        K=K_iphone_pinhole, T=T_extrinsic_iphone, frame="world",
                        conf_col=f"{side}_tracking_confidence", conf_thr=0.99
                    )
                    src = Path(row[f"frame_path_iphone{suffix}"])
                    dst_img = out_exo / src.name
                    dst_json = out_exo / (src.stem + ".json")
                    out_exo.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst_img)
                    with open(dst_json, "w") as f:
                        json.dump(payload, f, indent=4)

                # NPZ index for EXO split
                build_npz(out_exo, data_out_root / "hamer/hamer/hamer_evaluation_data" / f"{exo_tag}{suffix}{dist_suffix}.npz")

                # append to yaml file
                # name_data_split = f"{rec_location}_{interaction_index}_exo{suffix}{dist_suffix}"
                # type = "ImageDataset"
                # dataset_file = f"hamer_evaluation_data/{exo_tag}{suffix}{dist_suffix}.npz"
                # img_dir = data_out_root / "ground_truth" / name_data_split
                # keypoint_list = [i for i in range(21)]

                # # Add to YAML
                # with open(yaml_file, "a") as yaml_out:
                #     yaml_out.write(f"{name_data_split}:\n")
                #     yaml_out.write(f"    TYPE: {type}\n")
                #     yaml_out.write(f"    DATASET_FILE: {dataset_file}\n")
                #     yaml_out.write(f"    IMG_DIR: {img_dir}\n")
                #     yaml_out.write(f"    KEYPOINT_LIST: {keypoint_list}\n")
                append_dataset_to_yaml(
                    yaml_file=yaml_file,
                    rec_location=rec_location,
                    interaction_index=interaction_index,
                    view= "exo",
                    exo_tag=exo_tag,
                    suffix=suffix,
                    dist_suffix=dist_suffix,
                    data_out_root=data_out_root,
                    keypoint_count=21
                )
            


        # ---------------- EGO (Aria) ----------------
        out_ego = data_out_root / "ground_truth" / f"{rec_location}_{interaction_index}_ego"
        out_ego.mkdir(parents=True, exist_ok=True)

        for _, row in df_merged_total.iterrows(): 
            payload = row_to_newdays_json(
                row, side, image_width_aria, image_height_aria,
                K=K_aria_pinhole, T=T_extrinsic_aria, frame="device",
                conf_col=f"{side}_tracking_confidence", conf_thr=0.99
            )
            src = Path(row["frame_path_aria"])
            dst_img = out_ego / src.name
            dst_json = out_ego / (src.stem + ".json")
            shutil.copy2(src, dst_img)
            with open(dst_json, "w") as f:
                json.dump(payload, f, indent=4)

        # NPZ index for EGO split
        ego_npz = data_out_root / "hamer/hamer/hamer_evaluation_data" / f"{ego_tag}.npz"
        ego_npz.parent.mkdir(parents=True, exist_ok=True)
        build_npz(out_ego, ego_npz)

        append_dataset_to_yaml(
            yaml_file=yaml_file,
            rec_location=rec_location,
            interaction_index=interaction_index,
            view= "ego",
            exo_tag=ego_tag,
            suffix="",
            dist_suffix="",
            data_out_root=data_out_root,
            keypoint_count=21
        )
    

    
    a =2 
    


if __name__ == "__main__":
    ################################################3
    # recording location
    ################################################3
    rec_location = "kitchen_7"
    base_path = Path(f"/data/ikea_recordings")
    data_indexer = RecordingIndex(
        os.path.join(str(base_path), "extracted") 
    )
    visualize = True
    rec_type = "hand"
    interaction_indices = "1-3-5-7-9"

    generate_ego_exo_hand_pose_ground_truth(
        interaction_index=interaction_indices,
        rec_location=rec_location,
        base_path=base_path,
        data_out_root=Path("/data/evaluations/hand_pose_estimation/"),
        format="newdays",
        data_indexer=data_indexer,
        visualize=visualize
    )

    
