from hoi.data_tools.data_loader_aria import AriaData
from hoi.data_tools.data_loader_leica import LeicaData
from hoi.data_tools.data_loader_iphone import IPhoneData
from hoi.data_tools.data_indexer import RecordingIndex
# from hoi.data_tools.utils_mono_depth_estimation import run_map_anything_multimodal_inference, load_map_anything_model, npy_depth_to_png
import pandas as pd
from PIL import Image
from typing import Any, Dict, Tuple, List, Optional
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import json

from pathlib import Path
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage import binary_erosion
import torch
from tqdm.auto import tqdm
import math


def visualize_iphone_rgbd_projections_in_pointcloud(
    pcd: o3d.geometry.PointCloud,
    image_paths: list[str],
    depth_paths: list[str],
    w_T_wc_list: list[np.ndarray],
    K: np.ndarray,            # RGB intrinsics
    stride: int = 1,
):
    """
    Project RGBD frames into world and overlay with an existing point cloud.
    Supports different RGB/Depth resolutions assuming same optical center & extrinsics.
    K is treated as RGB intrinsics; depth intrinsics are derived by scaling K.
    """
    assert len(image_paths) == len(depth_paths) == len(w_T_wc_list), \
        "image_paths, depth_paths, and w_T_wc_list must have the same length"

    # Unpack RGB intrinsics
    fx_rgb, fy_rgb = float(K[0, 0]), float(K[1, 1])
    cx_rgb, cy_rgb = float(K[0, 2]), float(K[1, 2])

    all_pts_w = []
    all_rgb   = []

    N = len(image_paths)
    for i in range(0, N, max(1, int(stride))):
        # Load RGB & Depth
        img = Image.open(image_paths[i]).convert("RGB")
        rgb = np.asarray(img)
        H_rgb, W_rgb = rgb.shape[0], rgb.shape[1]

        depth = np.load(depth_paths[i])
        if depth.ndim == 3:
            depth = depth.squeeze()
        H_d, W_d = depth.shape

        # Derive depth intrinsics by scaling normalized params of K (assumed RGB K)
        # Keep principal point fractions and focal-per-dimension ratios.
        fxn = fx_rgb / W_rgb
        fyn = fy_rgb / H_rgb
        cxn = cx_rgb / W_rgb
        cyn = cy_rgb / H_rgb

        fx_d = fxn * W_d
        fy_d = fyn * H_d
        cx_d = cxn * W_d
        cy_d = cyn * H_d

        # Light pixel subsampling per frame (≈150k points cap)
        target_pts = 150_000
        s = max(1, int(np.ceil(np.sqrt((H_d * W_d) / target_pts))))
        vv, uu = np.mgrid[0:H_d:s, 0:W_d:s]

        d = depth[vv, uu]
        valid = np.isfinite(d) & (d > 0)
        if not np.any(valid):
            continue

        uu = uu[valid].astype(np.float64)
        vv = vv[valid].astype(np.float64)
        d  = d[valid].astype(np.float64)

        # Back-project using DEPTH intrinsics
        X = (uu - cx_d) * d / fx_d
        Y = (vv - cy_d) * d / fy_d
        Z = d
        pts_cam = np.stack([X, Y, Z], axis=1)

        # Reproject to RGB to fetch colors (handles different sizes)
        u_rgb = (fx_rgb * (X / Z)) + cx_rgb
        v_rgb = (fy_rgb * (Y / Z)) + cy_rgb

        # Keep only points that fall inside the RGB image
        u_i = np.round(u_rgb).astype(np.int64)
        v_i = np.round(v_rgb).astype(np.int64)
        in_rgb = (u_i >= 0) & (u_i < W_rgb) & (v_i >= 0) & (v_i < H_rgb)

        if not np.any(in_rgb):
            continue

        pts_cam  = pts_cam[in_rgb]
        u_i      = u_i[in_rgb]
        v_i      = v_i[in_rgb]

        cols = rgb[v_i, u_i].astype(np.float64) / 255.0

        # Transform to world with this frame pose
        T = np.asarray(w_T_wc_list[i], dtype=np.float64)
        R, t = T[:3, :3], T[:3, 3]
        pts_w = (R @ pts_cam.T).T + t

        all_pts_w.append(pts_w)
        all_rgb.append(cols)

    if not all_pts_w:
        print("[viz] no valid depth points to project.")
        o3d.visualization.draw_geometries([pcd])
        return

    pts_w = np.vstack(all_pts_w)
    cols  = np.vstack(all_rgb)

    pcd_rgbd = o3d.geometry.PointCloud()
    pcd_rgbd.points = o3d.utility.Vector3dVector(pts_w)
    pcd_rgbd.colors = o3d.utility.Vector3dVector(cols)

    o3d.visualization.draw_geometries([pcd, pcd_rgbd])


def visualize_rgbd_projections_in_pointcloud(
    pcd: o3d.geometry.PointCloud,
    image_paths: list[str],
    depth_paths: list[str],
    w_T_wc_list: list[np.ndarray],
    K: np.ndarray,
    stride: int = 1,
):
    """
    Project RGBD frames into world and overlay with an existing point cloud.

    Args
    ----
    pcd : Open3D PointCloud
        Your base/reference point cloud to compare against.
    image_paths : list[str]
        File paths to RGB images (same ordering as poses and depth).
    depth_paths : list[str]
        File paths to depth .npy (meters), same size as RGB and same ordering.
    w_T_wc_list : list[np.ndarray]
        4x4 world_T_cam poses per frame (same ordering).
    K : np.ndarray
        3x3 intrinsics (fx, 0, cx; 0, fy, cy; 0, 0, 1).
    stride : int
        Use every `stride`-th frame (frame stride).
    """
    assert len(image_paths) == len(depth_paths) == len(w_T_wc_list), \
        "image_paths, depth_paths, and w_T_wc_list must have the same length"

    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    all_pts_w = []
    all_rgb   = []

    N = len(image_paths)
    for i in range(0, N, max(1, int(stride))):
        # --- load RGB & depth
        img = Image.open(image_paths[i]).convert("RGB")
        rgb = np.asarray(img)
        H, W = rgb.shape[0], rgb.shape[1]

        depth = np.load(depth_paths[i])
        if depth.ndim == 3:
            depth = depth.squeeze()
        assert depth.shape == (H, W), f"Depth shape {depth.shape} must match RGB {(H, W)}"

        # --- light pixel subsampling per frame (cap ~150k pts/frame)
        target_pts = 150_000
        s = max(1, int(np.ceil(np.sqrt((H * W) / target_pts))))
        vv, uu = np.mgrid[0:H:s, 0:W:s]

        d = depth[vv, uu]
        valid = np.isfinite(d) & (d > 0)
        if not np.any(valid):
            continue

        uu = uu[valid].astype(np.float64)
        vv = vv[valid].astype(np.float64)
        d  = d[valid].astype(np.float64)

        # --- backproject to camera
        X = (uu - cx) * d / fx
        Y = (vv - cy) * d / fy
        Z = d
        pts_cam = np.stack([X, Y, Z], axis=1)

        # --- per-point color
        cols = rgb[vv.astype(int), uu.astype(int)].astype(np.float64) / 255.0

        # --- transform to world with this frame pose
        T = np.asarray(w_T_wc_list[i], dtype=np.float64)
        R, t = T[:3, :3], T[:3, 3]
        pts_w = (R @ pts_cam.T).T + t

        all_pts_w.append(pts_w)
        all_rgb.append(cols)

    if not all_pts_w:
        print("[viz] no valid depth points to project.")
        o3d.visualization.draw_geometries([pcd])
        return

    pts_w = np.vstack(all_pts_w)
    cols  = np.vstack(all_rgb)

    pcd_rgbd = o3d.geometry.PointCloud()
    pcd_rgbd.points = o3d.utility.Vector3dVector(pts_w)
    pcd_rgbd.colors = o3d.utility.Vector3dVector(cols)

    o3d.visualization.draw_geometries([pcd, pcd_rgbd])

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def _make_label_meshes(
    text: str,
    position: np.ndarray,
    scale: float = 0.06,
    color: np.ndarray = None,
) -> list:
    """Create three perpendicular copies of a text mesh so the label is readable
    from any viewpoint (XY flat, XZ upright front, YZ upright side).
    Returns an empty list if create_text is unavailable, printing the reason."""
    try:
        rotations = [
            np.eye(3),
            o3d.geometry.get_rotation_matrix_from_xyz((-np.pi / 2, 0.0, 0.0)),
            o3d.geometry.get_rotation_matrix_from_xyz((-np.pi / 2, np.pi / 2, 0.0)),
        ]
        meshes = []
        for R in rotations:
            tm = o3d.t.geometry.TriangleMesh.create_text(text, depth=0.002)
            tm.scale(scale, center=o3d.core.Tensor([0.0, 0.0, 0.0]))
            legacy = tm.to_legacy()
            legacy.rotate(R, center=np.zeros(3))
            legacy.translate(position)
            if color is not None:
                legacy.paint_uniform_color(color)
            legacy.compute_vertex_normals()
            meshes.append(legacy)
        return meshes
    except Exception as e:
        print(f"[label] create_text unavailable: {e}")
        return []


def visualize_articulations_in_pointcloud(
    articulations: dict,
    pcd: o3d.geometry.PointCloud,
    axis_len: float = 0.6,
    shaft_radius_scale: float = 0.03,
    cone_radius_scale: float = 0.08,
    sphere_radius: float = 0.055,
    show_origin_frame: bool = True,
    origin_frame_size: float = 0.15,
):
    """
    Visualize articulation axes (prismatic and revolute) on an Open3D point cloud,
    using colored arrows and spheres.

    Args:
        articulations: dict of articulation info with keys 'position', 'axis', 'type'
        pcd: open3d.geometry.PointCloud
        axis_len: base arrow length (scales both prismatic and revolute)
        shaft_radius_scale: relative shaft thickness
        cone_radius_scale: relative arrowhead size
    """
    geoms = [pcd]

    if show_origin_frame:
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=origin_frame_size))

    for idx, (aid, data) in enumerate(sorted(articulations.items(), key=lambda kv: int(kv[0]))):
        pos = np.array(data["position"], dtype=float)
        axis = np.array(data["axis"], dtype=float)
        axis /= np.linalg.norm(axis) + 1e-12
        typ = data.get("type", "prismatic").lower()

        color = np.array(plt.cm.tab10(idx % 10))[:3]

        if idx in [0, 6, 7]:
            pos = pos - np.array((0.0,0.0, 0.5))

        # Sphere at articulation position
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        sphere.translate(pos)
        sphere.paint_uniform_color(color)
        sphere.compute_vertex_normals()
        geoms.append(sphere)

        # Index label above the sphere (three planes so it's visible from any angle)
        geoms.extend(_make_label_meshes(
            str(aid),
            position=pos + np.array([0.0, 0.0, sphere_radius + 0.04]),
            color=np.ones(3),  # white
        ))

        # Function to create and orient an arrow along a given axis
        def make_arrow(length):
            """
            Create an Open3D arrow aligned with the global Z-axis,
            with a constant head size and variable shaft length.
            """
            cone_height = 0.1          # fixed head length
            cone_radius = 0.07          # fixed head radius
            shaft_radius = 0.03        # fixed shaft thickness
            shaft_length = max(length - cone_height, 0.01)  # ensure non-negative

            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cone_height=cone_height,
                cone_radius=cone_radius,
                cylinder_height=shaft_length,
                cylinder_radius=shaft_radius
            )

            arrow.paint_uniform_color(color)
            arrow.compute_vertex_normals()

            # Align arrow (Z → axis)
            z = np.array([0, 0, 1])
            v = np.cross(z, axis)
            c = np.dot(z, axis)
            if np.linalg.norm(v) > 1e-6:
                v /= np.linalg.norm(v)
                H = o3d.geometry.get_rotation_matrix_from_axis_angle(v * np.arccos(c))
                arrow.rotate(H, center=np.zeros(3))

            return arrow

        if typ == "revolute":
            # Single arrow for rotation axis
            arrow_1 = make_arrow(axis_len*1.2)
            arrow_2 = make_arrow(axis_len/1.5)
            arrow_2.translate(pos + axis_len * axis - 0.05 * axis)
            arrow_1.translate(pos)
            geoms.extend([arrow_1, arrow_2])

        elif typ == "prismatic":
            # Two opposing arrows to indicate translation direction
            arrow_fwd = make_arrow(axis_len)
            # arrow_bwd = make_arrow(axis_len)
            arrow_fwd.translate(pos + 0.0 * axis_len * axis)
            # arrow_bwd.translate(pos - 0.25 * axis_len * axis)
            # Flip the backward arrow
            H_flip = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * np.pi)
            # arrow_bwd.rotate(H_flip, center=pos)
            geoms.append(arrow_fwd)

    o3d.visualization.draw_geometries(geoms)
    



def visualize_depths_comparison_of_aria_pose(
    aria_data: AriaData,
    leica_data: LeicaData,
    n=1):

    # get frames
    extracted_aria_frames = aria_data.get_extracted_frames()
    extracted_aria_frames_df = pd.DataFrame({
                                            'frame_path_aria': [str(p) for p in extracted_aria_frames],
                                            'timestamp': [int(Path(p).stem) for p in extracted_aria_frames],
                                            })
    
    # get depth frames
    extracted_aria_depth_frames = aria_data.get_extracted_depth_frames()
    extracted_aria_depth_frames_df = pd.DataFrame({
                                            'frame_path_aria_depth': [str(p) for p in extracted_aria_depth_frames],
                                            'timestamp': [int(Path(p).stem) for p in extracted_aria_depth_frames],
                                            })

    # get random n timestamps
    # sampled_timestamps = extracted_aria_frames_df.sample(n=n)['timestamp'].tolist()
    # get specific timestamp at row 1455
    sampled_timestamps = extracted_aria_frames_df.iloc[2100]['timestamp'].tolist()
    sampled_timestamps = [sampled_timestamps]

    # get mesh
    mesh = leica_data.get_mesh()

    # get calibration
    aria_calibration = aria_data.get_calibration()

    T_dc = aria_calibration["PINHOLE"]["T_device_camera"]
    T_cRaw_cRect = aria_calibration["PINHOLE"]["pinhole_T_device_camera"]
    T_dcRect = T_dc @ T_cRaw_cRect

    # get pose for those timestamps
    for ts in sampled_timestamps:
        T_wd = aria_data.get_mps_pose_at_timestamp(ts, aligned=True)
        aria_cam_pose = T_wd @ T_dcRect

        depth_rendered = leica_data._render_depth(
            mesh=mesh,
            w_T_wc=np.linalg.inv(aria_cam_pose),
            K=aria_calibration['PINHOLE']["K"]
        )

        # load aria depth
        # aria_depth_frame_path = extracted_aria_depth_frames_df[extracted_aria_depth_frames_df['timestamp'] == ts]['frame_path_aria_depth'].values[0]
        # aria_depth_array = np.load(aria_depth_frame_path)
        aria_rgb_frame_path = extracted_aria_frames_df[extracted_aria_frames_df['timestamp'] == ts]['frame_path_aria'].values[0]
        # load aria frame jpg as HxWx3 numpy
        aria_rgb_array = np.asarray(Image.open(aria_rgb_frame_path).convert("RGB"))

        # resize depth_rendered to match aria_rgb_array size
        H, W = aria_rgb_array.shape[:2]
        depth_rendered = cv2.resize(
            depth_rendered,
            (W, H),
            interpolation=cv2.INTER_LINEAR
        )

        aria_depth_output = run_map_anything_multimodal_inference(
            image_path=[aria_rgb_array],
            intrinsics=[aria_calibration['PINHOLE']["K"]],
            extrinsics=[aria_cam_pose],
            depth=None,
            is_metric_scale=None,)
        aria_depth_array = aria_depth_output[0]['depth_z'].cpu().numpy()

        aria_depth_array = aria_depth_array.squeeze(axis=0)
        aria_depth_array = aria_depth_array.squeeze(axis=2)

        a = 2

        Ht, Wt = aria_depth_array.shape[:2]
        depth_rendered = cv2.resize(
            depth_rendered,
            (Wt, Ht),
            interpolation=cv2.INTER_LINEAR
        )
        
        eps=1e-6
        valid = (
                np.isfinite(aria_depth_array) &
                np.isfinite(depth_rendered) &
                (aria_depth_array > eps) &
                (depth_rendered > 0)
            )
        valid = binary_erosion(valid, iterations=1)

        ratios = (aria_depth_array[valid] /depth_rendered[valid])
        ratios = ratios[np.isfinite(ratios)]
        ratios_sorted = np.sort(ratios)
        trim = 0.1
        n = len(ratios_sorted)
        lo = int(trim * n)
        hi = int((1 - trim) * n)
        trimmed = ratios_sorted[lo:hi] if hi > lo else ratios_sorted
        scale = float(np.median(trimmed))

    # --- figure block ---
        fig = plt.figure(figsize=(15, 8))
        gs = fig.add_gridspec(2, 3, height_ratios=[3, 1])

        # compute per-pixel ratio map
        ratio_map = np.full_like(aria_depth_array, np.nan, dtype=np.float32)
        ratio_map[valid] = np.maximum(aria_depth_array[valid], eps) / depth_rendered[valid]

        # choose visualization range
        vmin, vmax = np.percentile(ratios, [1, 99]) if len(ratios) > 0 else (0.5, 1.5)

        # --- top row: 3 images ---
        ax0 = fig.add_subplot(gs[0, 0])
        im0 = ax0.imshow(aria_depth_array, cmap='plasma')
        ax0.set_title(f'Aria Depth (pred)  ts={ts}')
        ax0.axis('off')
        plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        ax1 = fig.add_subplot(gs[0, 1])
        im1 = ax1.imshow(depth_rendered, cmap='plasma')
        ax1.set_title(f'Leica Rendered Depth')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = fig.add_subplot(gs[0, 2])
        im2 = ax2.imshow(ratio_map, cmap='magma', vmin=vmin, vmax=vmax)
        ax2.set_title(f'Per-pixel Ratio (Leica / Aria)\nmedian={scale:.3f}')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # --- bottom row: histogram across all columns ---
        ax3 = fig.add_subplot(gs[1, :])
        ax3.hist(ratios, bins=200, color='royalblue', alpha=0.7)
        ax3.axvline(scale, color='red', linestyle='--', linewidth=2, label=f'median={scale:.3f}')
        ax3.axvline(np.percentile(ratios, trim*100), color='orange', linestyle=':', label=f'{trim*100:.0f}% lower trim')
        ax3.axvline(np.percentile(ratios, (1-trim)*100), color='orange', linestyle=':', label=f'{(1-trim)*100:.0f}% upper trim')
        ax3.set_xlabel("Per-pixel scale ratio (Leica / Aria)")
        ax3.set_ylabel("Pixel count")
        ax3.set_title("Distribution of scale ratios")
        ax3.legend()
        ax3.grid(alpha=0.3)
        ax3.set_xlim(0, 5)

        plt.tight_layout()
        plt.show()

        a = 2
    a =2 

import open3d as o3d
import numpy as np
from PIL import Image
import random

def visualize_random_rgb_frustum_in_pointcloud(
    pcd: o3d.geometry.PointCloud,
    image_paths: list[str],
    w_T_wc_list: list[np.ndarray],
    K: np.ndarray,
    plane_depth_m: float = 0.4,
    random_seed: int | None = None,
    idx: int | None = None,
):
    """
    Shows ONE camera frustum with the ACTUAL RGB image textured on the image plane,
    using Open3D's GUI renderer (not legacy draw_geometries).
    """
    import numpy as np
    import open3d as o3d
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering
    from PIL import Image
    from pathlib import Path
    import random

    assert len(image_paths) == len(w_T_wc_list), "image_paths and w_T_wc_list must match"

    # pick index
    rng = random.Random(random_seed) if random_seed is not None else random
    if idx is None:
        idx = rng.randrange(len(image_paths))

    # load image
    img = Image.open(image_paths[idx]).convert("RGB")
    W, H = img.size
    np_img = np.asarray(img)

    # intrinsics
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Z = float(plane_depth_m)

    def cam_point(u, v):
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        return np.array([X, Y, Z], dtype=float)

    # image plane in camera coords (OpenCV: +X right, +Y down, +Z forward)
    tl = cam_point(0, 0)
    tr = cam_point(W, 0)
    br = cam_point(W, H)
    bl = cam_point(0, H)

    # textured plane (legacy mesh is fine; GUI renderer handles textures)
    plane = o3d.geometry.TriangleMesh()
    plane.vertices  = o3d.utility.Vector3dVector(np.array([tl, tr, br, bl]))
    plane.triangles = o3d.utility.Vector3iVector(np.array([[0,1,2], [0,2,3]], dtype=np.int32))
    plane.triangle_uvs = o3d.utility.Vector2dVector(np.array([
        [0, 1], [1, 1], [1, 0],
        [0, 1], [1, 0], [0, 0],
    ], dtype=np.float64))
    plane.textures = [o3d.geometry.Image(np_img)]
    plane.compute_vertex_normals()

    # frustum lines
    origin = np.zeros(3)
    fr_pts = np.vstack([origin, tl, tr, br, bl])
    fr_lines = np.array([[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]], dtype=np.int32)
    frustum = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(fr_pts),
        lines=o3d.utility.Vector2iVector(fr_lines),
    )
    frustum.colors = o3d.utility.Vector3dVector([[1,0,0]] * len(fr_lines))

    # transform both to world
    w_T_wc = np.asarray(w_T_wc_list[idx], dtype=np.float64)
    plane.transform(w_T_wc)
    frustum.transform(w_T_wc)
    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    cam_frame.transform(w_T_wc)

    # ==== GUI renderer (blocking window that shows textures) ====
    gui.Application.instance.initialize()
    win = gui.Application.instance.create_window(
        f"Camera {idx} | {Path(image_paths[idx]).name}", 1600, 900
    )
    scene_widget = gui.SceneWidget()
    win.add_child(scene_widget)
    scene = rendering.Open3DScene(win.renderer)
    scene_widget.scene = scene
    scene.set_background([1, 1, 1, 1])
    texture_image = o3d.geometry.Image(np_img)

    # materials
    mat_img = rendering.MaterialRecord(); mat_img.shader = "defaultUnlit"
    mat_img.albedo_img = texture_image
    mat_line = rendering.MaterialRecord(); mat_line.shader = "unlitLine"
    mat_pc = rendering.MaterialRecord();   mat_pc.shader = "defaultUnlit"


    # add geometries
    scene.add_geometry("pcd", pcd, mat_pc)
    scene.add_geometry("plane", plane, mat_img)
    scene.add_geometry("frustum", frustum, mat_line)
    scene.add_geometry("cam_frame", cam_frame, mat_img)

    # camera fit
    bbox = plane.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    scene_widget.setup_camera(60.0, bbox, center)

    # escape to close
    def on_key(e):
        if e.key == gui.KeyName.ESCAPE and e.type == gui.KeyEvent.DOWN:
            gui.Application.instance.post_to_main_thread(win, win.close)
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED
    win.set_on_key(on_key)

    gui.Application.instance.run()  # blocks until window closed

def visualize_articulations_and_random_rgb_frustum_in_pointcloud(
    articulations: dict,
    pcd: o3d.geometry.PointCloud,
    image_paths: list[str],
    depth_paths: list[str],
    w_T_wc_list: list[np.ndarray],
    K: np.ndarray,
    axis_len: float = 0.35,
    sphere_radius: float = 0.02,
    show_origin_frame: bool = True,
    origin_frame_size: float = 0.15,
    plane_depth_m: float = 0.4,
    random_seed: int | None = None,
    idx: int | None = None,
):
    """
    Combined visualization with the same inputs/notation as the two originals:
      - point cloud
      - articulation axes (prismatic: line, revolute: arrow) + position spheres
      - ONE random (or chosen) camera frustum textured with its RGB image
      - camera axes and optional world origin frame
      - trajectory automatically drawn from w_T_wc_list (no extra inputs)
      - OPTIONAL: if a global `metric_scale_depth_files` (list[str]) is present,
        the corresponding depth is projected into 3D and shown as a colored point cloud.
    """
    import numpy as np
    import open3d as o3d
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering
    from PIL import Image
    from pathlib import Path
    import random

    assert len(image_paths) == len(w_T_wc_list), "image_paths and w_T_wc_list must match"

    geoms = []

    # World origin frame (optional)
    if show_origin_frame:
        geoms.append(("origin_frame",
                      o3d.geometry.TriangleMesh.create_coordinate_frame(size=origin_frame_size)))

    # Base point cloud
    geoms.append(("pcd", pcd))

    # -------- Articulations (kept notation) --------
    def _key(k):
        try:
            return int(k)
        except Exception:
            return k

    for idx_a, (aid, data) in enumerate(sorted(articulations.items(), key=lambda kv: _key(kv[0]))):
        pos = np.array(data["position"], dtype=float)
        axis = np.array(data["axis"], dtype=float)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        typ = data.get("type", "prismatic")

        rng = random.Random(idx_a + 12345)
        color = np.array([rng.random(), rng.random(), rng.random()], dtype=float)

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        sphere.translate(pos)
        sphere.paint_uniform_color(color)
        sphere.compute_vertex_normals()
        geoms.append((f"sphere_{aid}", sphere))

        if typ == "prismatic":
            p0 = pos - 0.5 * axis_len * axis
            p1 = pos + 0.5 * axis_len * axis
            line = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector([p0, p1]),
                lines=o3d.utility.Vector2iVector([[0, 1]])
            )
            line.colors = o3d.utility.Vector3dVector([color])
            geoms.append((f"pris_{aid}", line))

        elif typ == "revolute":
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cone_height=0.1 * axis_len,
                cone_radius=0.04 * axis_len,
                cylinder_height=0.9 * axis_len,
                cylinder_radius=0.015 * axis_len
            )
            arrow.paint_uniform_color(color)
            arrow.compute_vertex_normals()
            z = np.array([0.0, 0.0, 1.0])
            v = np.cross(z, axis)
            c = float(np.dot(z, axis))
            if np.linalg.norm(v) > 1e-6:
                v = v / np.linalg.norm(v)
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(v * np.arccos(np.clip(c, -1.0, 1.0)))
                arrow.rotate(R, center=np.zeros(3))
            arrow.translate(pos)
            geoms.append((f"rev_{aid}", arrow))

    # -------- Trajectory from w_T_wc_list (no new inputs) --------
    if len(w_T_wc_list) >= 2:
        traj_xyz = np.array([np.asarray(T, dtype=float)[:3, 3] for T in w_T_wc_list], dtype=float)
        lines = np.stack([np.arange(len(traj_xyz)-1), np.arange(1, len(traj_xyz))], axis=1).astype(np.int32)
        traj = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(traj_xyz),
            lines=o3d.utility.Vector2iVector(lines)
        )
        traj.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.4, 0.4, 0.4]]), (len(lines), 1)))
        geoms.append(("trajectory", traj))

    # -------- One camera frustum + textured plane (kept notation) --------
    rng = random.Random(random_seed) if random_seed is not None else random
    if idx is None:
        idx = rng.randrange(len(image_paths))

    img = Image.open(image_paths[idx]).convert("RGB")
    W, H = img.size
    np_img = np.asarray(img)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Zp = float(plane_depth_m)

    def cam_point(u, v):
        X = (u - cx) * Zp / fx
        Y = (v - cy) * Zp / fy
        return np.array([X, Y, Zp], dtype=float)

    tl = cam_point(0, 0)
    tr = cam_point(W, 0)
    br = cam_point(W, H)
    bl = cam_point(0, H)

    plane = o3d.geometry.TriangleMesh()
    plane.vertices  = o3d.utility.Vector3dVector(np.array([tl, tr, br, bl]))
    plane.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32))
    plane.triangle_uvs = o3d.utility.Vector2dVector(np.array([
        [0, 1], [1, 1], [1, 0],
        [0, 1], [1, 0], [0, 0],
    ], dtype=np.float64))
    plane.textures = [o3d.geometry.Image(np_img)]
    plane.compute_vertex_normals()

    origin = np.zeros(3)
    fr_pts = np.vstack([origin, tl, tr, br, bl])
    fr_lines = np.array([[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]], dtype=np.int32)
    frustum = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(fr_pts),
        lines=o3d.utility.Vector2iVector(fr_lines),
    )
    frustum.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(fr_lines))

    w_T_wc = np.asarray(w_T_wc_list[idx], dtype=np.float64)
    plane.transform(w_T_wc)
    frustum.transform(w_T_wc)
    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    cam_frame.transform(w_T_wc)

    geoms.extend([
        ("plane", plane),
        ("frustum", frustum),
        ("cam_frame", cam_frame),
    ])

    # -------- OPTIONAL DEPTH -> colored 3D points (uses global metric_scale_depth_files) --------
    try:
        depth_path_list = depth_paths
        if isinstance(depth_path_list, (list, tuple)) and len(depth_path_list) == len(image_paths):
            import numpy as _np
            depth = _np.load(depth_path_list[idx])  # meters, same size as RGB
            if depth.ndim == 3:
                depth = depth.squeeze()
            assert depth.shape == (H, W), "Depth shape must match RGB (H, W)."

            # Choose a stride to cap the number of points for speed (≈150k max)
            target_pts = 150_000
            s = max(1, int(_np.ceil(_np.sqrt((H * W) / target_pts))))

            vv, uu = _np.mgrid[0:H:s, 0:W:s]
            d = depth[vv, uu]
            valid = _np.isfinite(d) & (d > 0)

            uu = uu[valid].astype(_np.float64)
            vv = vv[valid].astype(_np.float64)
            d  = d[valid].astype(_np.float64)

            X = (uu - cx) * d / fx
            Y = (vv - cy) * d / fy
            Z = d
            pts_cam = _np.stack([X, Y, Z], axis=1)

            # color from RGB
            rgb = np_img[vv.astype(int), uu.astype(int)].astype(_np.float64) / 255.0

            # transform to world
            R = w_T_wc[:3, :3]
            t = w_T_wc[:3, 3]
            pts_w = (R @ pts_cam.T).T + t

            pcd_depth = o3d.geometry.PointCloud()
            pcd_depth.points  = o3d.utility.Vector3dVector(pts_w)
            pcd_depth.colors  = o3d.utility.Vector3dVector(rgb)
            geoms.append(("depth_points", pcd_depth))
    except NameError:
        # metric_scale_depth_files not defined -> silently skip
        pass

    # -------- GUI renderer (supports textures) --------
    gui.Application.instance.initialize()
    win = gui.Application.instance.create_window(
        f"Merged viz | Cam {idx} | {Path(image_paths[idx]).name}", 1600, 900
    )
    scene_widget = gui.SceneWidget()
    win.add_child(scene_widget)
    scene = rendering.Open3DScene(win.renderer)
    scene_widget.scene = scene
    scene.set_background([1, 1, 1, 1])

    mat_pc  = rendering.MaterialRecord(); mat_pc.shader  = "defaultUnlit"
    mat_img = rendering.MaterialRecord(); mat_img.shader = "defaultUnlit"; mat_img.albedo_img = o3d.geometry.Image(np_img)
    mat_line = rendering.MaterialRecord(); mat_line.shader = "unlitLine"

    for name, g in geoms:
        if isinstance(g, o3d.geometry.PointCloud):
            scene.add_geometry(name, g, mat_pc)
        elif isinstance(g, o3d.geometry.LineSet):
            scene.add_geometry(name, g, mat_line)
        elif isinstance(g, o3d.geometry.TriangleMesh):
            scene.add_geometry(name, g, mat_img if name == "plane" else mat_pc)
        else:
            scene.add_geometry(name, g, mat_pc)

    # Fit camera to union AABB (CUDA-safe)
    aabbs = []
    for _, g in geoms:
        try:
            aabbs.append(g.get_axis_aligned_bounding_box())
        except Exception:
            pass
    if aabbs:
        mins = np.vstack([bb.get_min_bound() for bb in aabbs]).min(axis=0)
        maxs = np.vstack([bb.get_max_bound() for bb in aabbs]).max(axis=0)
        A = o3d.geometry.AxisAlignedBoundingBox(mins, maxs)
        scene_widget.setup_camera(60.0, A, A.get_center())

    def on_key(e):
        if e.key == gui.KeyName.ESCAPE and e.type == gui.KeyEvent.DOWN:
            gui.Application.instance.post_to_main_thread(win, win.close)
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED
    win.set_on_key(on_key)

    gui.Application.instance.run()


def visualize_depth(
    depth_estimate: np.ndarray,
    depth_rendered: np.ndarray,
    depth_metric_scale: np.ndarray,
    scaled_ratios: np.ndarray,
    orig_ratios: np.ndarray,
    *,
    robust_percentiles=(1, 99),
):
    """
    Show: 3 depth maps with shared color scale, 1 scale map, + histogram of the ORIGINAL per-pixel scale.
    - Depth colorbars labeled 'Depth [m]'
    - Scaled-ratios colorbar labeled 'Scale [-]'
    - Histogram uses orig_ratios (depth_estimate / depth_rendered)
    """

    # Shared depth scale (robust) across the 3 depth images
    all_depth = np.concatenate([
        depth_estimate[np.isfinite(depth_estimate)],
        depth_rendered[np.isfinite(depth_rendered)],
        depth_metric_scale[np.isfinite(depth_metric_scale)],
    ])
    if all_depth.size:
        dmin, dmax = np.nanpercentile(all_depth, robust_percentiles)
        if dmin == dmax:  # fallback if flat
            dmin, dmax = float(np.nanmin(all_depth)), float(np.nanmax(all_depth))
    else:
        dmin, dmax = 0.0, 1.0

    # Scales (unitless), robust color range for scaled_ratios
    finite_scaled = scaled_ratios[np.isfinite(scaled_ratios)]
    if finite_scaled.size:
        smin, smax = np.nanpercentile(finite_scaled, robust_percentiles)
        if smin == smax:
            smin, smax = float(np.nanmin(finite_scaled)), float(np.nanmax(finite_scaled))
    else:
        smin, smax = 0.5, 1.5

    # Layout: 1x5 (4 images + 1 histogram)
    fig, axs = plt.subplots(1, 5, figsize=(26, 5))

    # Depth Estimate
    im0 = axs[0].imshow(depth_estimate, cmap='plasma', vmin=dmin, vmax=dmax)
    axs[0].set_title('Depth Estimate')
    axs[0].set_xlabel('X [px]'); axs[0].set_ylabel('Y [px]')
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04).ax.set_ylabel('Depth [m]', rotation=270, labelpad=15)

    # Depth Rendered
    im1 = axs[1].imshow(depth_rendered, cmap='plasma', vmin=dmin, vmax=dmax)
    axs[1].set_title('Depth Rendered')
    axs[1].set_xlabel('X [px]'); axs[1].set_ylabel('Y [px]')
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04).ax.set_ylabel('Depth [m]', rotation=270, labelpad=15)

    # Depth Metric-Scaled
    im2 = axs[2].imshow(depth_metric_scale, cmap='plasma', vmin=dmin, vmax=dmax)
    axs[2].set_title('Depth Metric Scale')
    axs[2].set_xlabel('X [px]'); axs[2].set_ylabel('Y [px]')
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04).ax.set_ylabel('Depth [m]', rotation=270, labelpad=15)

    # Scaled Ratios (unitless)
    im3 = axs[3].imshow(scaled_ratios, cmap='magma', vmin=smin, vmax=smax)
    axs[3].set_title('Scaled Ratios (metric/Leica)')
    axs[3].set_xlabel('X [px]'); axs[3].set_ylabel('Y [px]')
    plt.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04).ax.set_ylabel('Scale [-]', rotation=270, labelpad=15)

    # Histogram of ORIGINAL per-pixel scale (before scaling)
    orig = orig_ratios[np.isfinite(orig_ratios)]
    axs[4].hist(orig, bins=200, alpha=0.8)
    axs[4].set_title('Original Per-Pixel Scale')
    axs[4].set_xlabel('Scale [-]'); axs[4].set_ylabel('Pixel count')
    axs[4].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def _to_cpu(obj):
    """Move any torch tensors (nested in dict/list/tuple) to CPU (no grad)."""
    if torch.is_tensor(obj):
        return obj.detach().to("cpu")
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [_to_cpu(v) for v in obj]
        return type(obj)(t) if isinstance(obj, tuple) else t
    return obj

def batched_map_anything_inference(
    image_paths: List[str],
    intrinsics_list: List[np.ndarray],
    batch_size: int = 4,
) -> Dict[int, Dict[str, Any]]:
    """
    Runs MapAnything inference in batches and returns a dict {timestamp -> output_dict}.
    Assumes run_map_anything_multimodal_inference returns a list of outputs aligned with inputs.
    """

    if len(image_paths) != len(intrinsics_list):
        raise ValueError(f"image_paths ({len(image_paths)}) and intrinsics_list ({len(intrinsics_list)}) must match.")

    # Derive timestamps and keep paired to preserve association
    try:
        items: List[Tuple[int, str, np.ndarray]] = [
            (int(Path(p).stem), p, K) for p, K in zip(image_paths, intrinsics_list)
        ]
    except ValueError as e:
        raise ValueError(
            "Failed to parse integer timestamps from image filenames. "
            "Ensure Path(p).stem is an integer (e.g., '1729600000000')."
        ) from e

    # Optional: sort by timestamp to enforce deterministic order
    items.sort(key=lambda x: x[0])  # (ts, path, K)

    outputs_by_ts: Dict[int, Dict[str, Any]] = {}

    # Load model once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_map_anything_model(model_name="facebook/map-anything", device=device)

    for i in tqdm(
        range(0, len(items), batch_size),
        total=math.ceil(len(items) / batch_size),
        desc="MapAnything batches",
        unit="batch",
        dynamic_ncols=True,
    ):
        batch = items[i:i + batch_size]
        batch_ts   = [ts for ts, _, _ in batch]
        batch_imgs = [p  for _, p, _ in batch]
        batch_K    = [K  for _, _, K in batch]

        with torch.inference_mode():
            batch_outputs = run_map_anything_multimodal_inference(
                model=model,
                image_path=batch_imgs,
                intrinsics=batch_K,
                extrinsics=None,
                depth=None,
                is_metric_scale=None,
            )

        # Associate outputs to timestamps (alignment guaranteed by input order)
        for ts, out in zip(batch_ts, batch_outputs):
            # Optionally attach the timestamp into the dict:
            out_cpu = _to_cpu(out)
            out_cpu["timestamp"] = ts
            outputs_by_ts[ts] = out_cpu

        # Optional: free cache between batches to reduce OOM risk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return outputs_by_ts

def get_depth_estimates_for_image_list(image_paths: List[str], 
                                        intrinsics: List[np.ndarray],
                                        data_root: Path,
                                        batch_size: int) -> Dict[int, Dict[str, Any]]:
    
    # if depth npys exist in estimated_depth_dir, return paths
    # else run batched inference and save to estimated_depth_dir and return paths
    data_root = Path(data_root)
    estimated_depth_dir = data_root / "estimated_depth"
    estimated_depth_dir.mkdir(parents=True, exist_ok=True)

    if all((estimated_depth_dir / f"{int(Path(p).stem)}.npy").exists() for p in image_paths):
        # return all paths as list
        depth_estimates = [str(estimated_depth_dir / f"{int(Path(p).stem)}.npy") for p in image_paths]
        return depth_estimates
    
    else:
        depth_outputs = batched_map_anything_inference(
            image_paths=image_paths,
            intrinsics_list=intrinsics,
            batch_size=batch_size,
        )

        # save depth estimates to estimated_depth_dir
        for ts, out in tqdm(
            depth_outputs.items(),
            desc="Saving estimated depths",
            unit="frame",
            total=len(depth_outputs),
        ):
            depth_array = out['depth_z'].cpu().numpy()
            depth_array = depth_array.squeeze(axis=0)
            # handle possible singleton channel dim
            if depth_array.ndim == 3 and depth_array.shape[2] == 1:
                depth_array = depth_array.squeeze(axis=2)

            depth_save_path = estimated_depth_dir / f"{ts}.npy"
            np.save(depth_save_path, depth_array)

        # return all paths as list
        depth_estimates = [str(estimated_depth_dir / f"{int(Path(p).stem)}.npy") for p in image_paths]
        return depth_estimates

def get_depth_rendered_for_image_list(
    image_paths: List[str],
    aria_data: AriaData,
    leica_data: LeicaData,
    data_root: Path,
) -> Dict[int, Path]:
    """
    For each RGB image (timestamped filename), render Leica depth in the ARIA rectified camera
    frame and save as .npy in rendered_depth_dir. Skips frames that already exist.
    Returns: list of paths to rendered depth .npy files.
    """

    rendered_depth_dir = Path(data_root) / "rendered_depth"
    rendered_depth_dir.mkdir(parents=True, exist_ok=True)

    if all((rendered_depth_dir / f"{int(Path(p).stem)}.npy").exists() for p in image_paths):
        depth_rendered = [str(rendered_depth_dir / f"{int(Path(p).stem)}.npy") for p in image_paths]
        return depth_rendered

    w_T_wc_list, w_T_cw_list = get_extrinsics_for_image_list(
        image_paths=image_paths,
        aria_data=aria_data,
        data_root=data_root
    )

    # Parse timestamps
    timestamps = [int(Path(p).stem) for p in image_paths]

    aria_calib = aria_data.get_calibration()
    K = aria_calib['PINHOLE']["K"]
    T_device_cam_raw     = aria_calib["PINHOLE"]["T_device_camera"]
    T_pinhole_cam_rect   = aria_calib["PINHOLE"]["pinhole_T_device_camera"]
    T_device_cam_rect    = T_device_cam_raw @ T_pinhole_cam_rect  # device -> rectified cam

    # Load mesh once
    mesh = leica_data.get_mesh()

    # Batch render all depths
    depths = leica_data.render_depth_batched(
        mesh=mesh,
        K=K,
        w_T_cw_list=w_T_cw_list,
        clip_max_dist=3.0,
        clip_min_dist=0.25,
    )

    # Save depths
    for ts, depth in zip(timestamps, depths):
        depth_save_path = rendered_depth_dir / f"{ts}.npy"
        np.save(depth_save_path, depth)


    # return all paths as list
    out = [str(rendered_depth_dir / f"{ts}.npy") for ts in timestamps]
    return out

def get_extrinsics_for_image_list(
    image_paths: List[str],
    aria_data: AriaData,
    data_root: str | Path
):

    extrinic_dir = Path(data_root) / "extrinsics"
    extrinic_dir.mkdir(parents=True, exist_ok=True)

    # if csv exists, load and return
    if (extrinic_dir / "odom.csv").exists():
        df_quat = pd.read_csv(extrinic_dir / "odom.csv")
        w_T_wc_list = []
        w_T_cw_list = []
        for idx, row in df_quat.iterrows():
            tx, ty, tz, qx, qy, qz, qw = row['tx'], row['ty'], row['tz'], row['qx'], row['qy'], row['qz'], row['qw']
            R_world_cam = R.from_quat([qx, qy, qz, qw]).as_matrix()
            t_world_cam = np.array([tx, ty, tz])
            T_world_cam = np.eye(4)
            T_world_cam[0:3, 0:3] = R_world_cam
            T_world_cam[0:3, 3] = t_world_cam
            w_T_wc_list.append(T_world_cam)
            w_T_cw_list.append(np.linalg.inv(T_world_cam))
        return w_T_wc_list, w_T_cw_list

    # Parse timestamps
    timestamps = [int(Path(p).stem) for p in image_paths]

    aria_calib = aria_data.get_calibration()
    T_device_cam_raw     = aria_calib["PINHOLE"]["T_device_camera"]
    T_pinhole_cam_rect   = aria_calib["PINHOLE"]["pinhole_T_device_camera"]
    T_device_cam_rect    = T_device_cam_raw @ T_pinhole_cam_rect  # device -> rectified cam

    # get cam poses as list
    w_T_cw_list = []
    w_T_wc_list = []
    w_quat_wc_list = []
    for ts in tqdm(
        timestamps,
        desc="Getting extrinsics",
        unit="frame",
        total=len(timestamps),
        dynamic_ncols=True,
    ):

        T_world_device = aria_data.get_mps_pose_at_timestamp(ts, aligned=True)  # world_T_device
        T_world_cam    = T_world_device @ T_device_cam_rect                      # world_T_cam(rect)
        w_T_cw_list.append(np.linalg.inv(T_world_cam))
        w_T_wc_list.append(T_world_cam)

        # tx ty tz x y z w quat
        #TODO FIX
        # T_world_cam = np.linalg.inv(T_world_cam)

        R_world_cam = T_world_cam[0:3, 0:3]
        t_world_cam = T_world_cam[0:3, 3]
        quat_world_cam = R.from_matrix(R_world_cam).as_quat()
        w_quat_wc_list.append([t_world_cam[0], t_world_cam[1], t_world_cam[2], quat_world_cam[0], quat_world_cam[1], quat_world_cam[2], quat_world_cam[3]])

    # to csv quat
    df_quat = pd.DataFrame(w_quat_wc_list, columns=["tx", "ty", "tz", "qx", "qy", "qz", "qw"])
    df_quat.to_csv(extrinic_dir / "odom.csv", index=False)

    return w_T_wc_list, w_T_cw_list

def get_extrinsics_for_iphone_image_list(
    image_paths: List[str],
    iphone_data: IPhoneData):

    # get cam poses as list
    cam_poses_df = iphone_data.get_trajectory_aligned()

    w_T_wc_list = []
    w_T_cw_list = []
    for p in tqdm(
        image_paths,
        desc="Getting extrinsics for iPhone images",
        unit="frame",
        total=len(image_paths),
        dynamic_ncols=True,
    ):
        ts = int(Path(p).stem)
        T_world_cam = iphone_data.get_pose_aligned_at_timestamp(ts)
        w_T_wc_list.append(T_world_cam)
        w_T_cw_list.append(np.linalg.inv(T_world_cam))

    return w_T_wc_list, w_T_cw_list


def get_metric_scale_depth_from_aria_and_leica(
    aria_data: AriaData,
    leica_data: LeicaData,
    data_root: str | Path,
    stride: int = 2,
    visualize: bool = True,
):

    data_root = Path(data_root)
    metric_scale_depth_dir = data_root / "metric_scale_depth"
    metric_scale_depth_dir.mkdir(parents=True, exist_ok=True)

    # get frames
    extracted_aria_frames = aria_data.get_extracted_frames()
    extracted_aria_frames_df = pd.DataFrame({
                                            'frame_path_aria': [str(p) for p in extracted_aria_frames],
                                            'timestamp': [int(Path(p).stem) for p in extracted_aria_frames],
                                            })
    # get first aria pose timestamp and discrad earlier frames
    first_aria_pose_timestamp = aria_data.get_closed_loop_trajectory_aligned().iloc[0]['timestamp']
    extracted_aria_frames_df = extracted_aria_frames_df[extracted_aria_frames_df['timestamp'] >= first_aria_pose_timestamp]
    # extracted_aria_frames_df = extracted_aria_frames_df.iloc[::stride].reset_index(drop=True)

    # get last aria pose timestamp and discrad later frames
    last_aria_pose_timestamp = aria_data.get_closed_loop_trajectory_aligned().iloc[-1]['timestamp']
    extracted_aria_frames_df = extracted_aria_frames_df[extracted_aria_frames_df['timestamp'] <= last_aria_pose_timestamp]
    extracted_aria_frames_df = extracted_aria_frames_df.reset_index(drop=True)

    # for debug, get frames between 331 and 497 index
    # extracted_aria_frames_df = extracted_aria_frames_df.iloc[780:810].reset_index(drop=True)
    relevant_indices = get_relevant_frame_indices_from_fine_grained_interaction_window(
        data_root=data_root,
        rgb_frames_list_df=extracted_aria_frames_df,
        stride=1,
    )

    extracted_aria_frames_df = extracted_aria_frames_df.iloc[relevant_indices].reset_index(drop=True)
    # stride
    extracted_aria_frames_df = extracted_aria_frames_df.iloc[::stride].reset_index(drop=True)

    # get calibration
    aria_calibration = aria_data.get_calibration()
    aria_H = aria_calibration['PINHOLE']['h']
    aria_W = aria_calibration['PINHOLE']['w']

    # run depth model on image list to get near metric scale depth
    image_paths_list = extracted_aria_frames_df["frame_path_aria"].to_list()
    intrinsics = aria_calibration['PINHOLE']["K"]
    intrinsics_list = [intrinsics] * len(image_paths_list)

    w_T_wc_list, w_T_cw_list = get_extrinsics_for_image_list(
        image_paths=image_paths_list,
        aria_data=aria_data,
        data_root=data_root
    )

    # return early if all metric scale depths exist
    if all((metric_scale_depth_dir / f"{int(Path(p).stem)}.npy").exists() for p in image_paths_list):
        # return all paths as list
        depth_files = [str(metric_scale_depth_dir / f"{int(Path(p).stem)}.npy") for p in image_paths_list]
        return depth_files, image_paths_list, w_T_wc_list
    
    # batched inference (rgb + intrinsics)
    batch_size = 32
    depth_estimates = get_depth_estimates_for_image_list(
        image_paths=image_paths_list,
        intrinsics=intrinsics_list,
        data_root=data_root,
        batch_size=batch_size,
    )

    depth_rendered = get_depth_rendered_for_image_list(
        image_paths=image_paths_list,
        aria_data=aria_data,
        leica_data=leica_data,
        data_root=data_root,
    )

    # make depth estimates metric using leica depth as reference
    # load depth pairs and compute scale
    pairs = list(zip(depth_estimates, depth_rendered))
    for idx, (depth_estimate_path, depth_rendered_path) in enumerate(
        tqdm(
            pairs,
            desc="Making metric scale depths",
            unit="frame",
            total=len(pairs),
            dynamic_ncols=True,
        )
    ):

        depth_estimate = np.load(depth_estimate_path)
        depth_rendered = np.load(depth_rendered_path)

        # resize rendered depth to match estimate
        depth_rendered = cv2.resize(
            depth_rendered,
            (aria_W, aria_H),
            interpolation=cv2.INTER_LINEAR
        )

        depth_estimate = cv2.resize(
            depth_estimate,
            (aria_W, aria_H),
            interpolation=cv2.INTER_LINEAR
        )

        eps=0.25
        valid = (
                np.isfinite(depth_estimate) &
                np.isfinite(depth_rendered) &
                (depth_estimate > eps) &
                (depth_rendered > eps)
            )
        valid = binary_erosion(valid, iterations=10)
        valid_no_bleed = (
            (depth_estimate > eps) 
        )
        valid_no_bleed = binary_erosion(valid_no_bleed, iterations=5)

        ratios = (depth_estimate[valid] / depth_rendered[valid])
        ratios = ratios[np.isfinite(ratios)]
        ratios_sorted = np.sort(ratios)
        trim = 0.1
        n = len(ratios_sorted)
        lo = int(trim * n)
        hi = int((1 - trim) * n)
        trimmed = ratios_sorted[lo:hi] if hi > lo else ratios_sorted
        scale = float(np.median(trimmed))

        # derive scale as the histogram mode (argmax bin center) instead of median
        ratios_finite = ratios[np.isfinite(ratios)]
        if ratios_finite.size == 0:
            scale = 1.0
        else:
            # use robust range to avoid extreme outliers dominating binning
            lo_p, hi_p = 1.0, 99.0
            vmin = float(np.percentile(ratios_finite, lo_p))
            vmax = float(np.percentile(ratios_finite, hi_p))
            if vmin == vmax:
                # fallback to full range
                vmin, vmax = float(np.min(ratios_finite)), float(np.max(ratios_finite))
            if vmin == vmax:
                scale = float(np.median(ratios_finite))
            else:
                hist, edges = np.histogram(ratios_finite, bins=200, range=(vmin, vmax))
                centers = 0.5 * (edges[:-1] + edges[1:])
                scale = float(centers[np.argmax(hist)])

        # scale depth estimate
        depth_metric_scale = depth_estimate / scale
        depth_metric_scale[~valid_no_bleed] = 0.0

        # compute scale deviation map after scaling
        scaled_ratios = np.full_like(depth_estimate, np.nan, dtype=np.float32)
        scaled_ratios[valid] = (depth_metric_scale[valid] / depth_rendered[valid])

        if visualize and idx % 5 == 0:
            visualize_depth(
                depth_estimate=depth_estimate,
                depth_rendered=depth_rendered,
                depth_metric_scale=depth_metric_scale,
                scaled_ratios=scaled_ratios,
                orig_ratios=ratios,
            )
            a = 2

        # save scaled depth
        ts = extracted_aria_frames_df.iloc[idx]['timestamp']
        depth_save_path = metric_scale_depth_dir / f"{ts}.npy"
        np.save(depth_save_path, depth_metric_scale)

    # return all paths as list
    depth_files = [str(metric_scale_depth_dir / f"{int(Path(p).stem)}.npy") for p in image_paths_list]
    return depth_files, image_paths_list, w_T_wc_list

def get_iphone_depth_iphone(calibration: dict,
                            output_path: Path,
                            frame_id: str = "rgb_camera_link") -> Path:
    
    """Write a ROS-style CameraInfo text file using the provided PINHOLE calibration.
    Args:
        calibration: dict containing the "PINHOLE" calibration entry.
        output_path: path to save the text file (e.g., Path("camera_info.txt")).
        frame_id: optional frame name (default: "rgb_camera_link").
    Returns:
        Path to the written text file.
    """
    pinhole = calibration["PINHOLE"]

    width  = pinhole["dw"]
    height = pinhole["dh"]
    K = pinhole["K"].flatten().tolist()
    D = pinhole.get("distortion", np.zeros(8)).flatten().tolist()
    R = np.eye(3, dtype=float).flatten().tolist()
    P = [K[0], 0.0, K[2], 0.0,
         0.0, K[4], K[5], 0.0,
         0.0, 0.0, 1.0, 0.0]

    # Use current timestamp
    from datetime import datetime
    now = datetime.now()
    secs = int(now.timestamp())
    nsecs = int((now.timestamp() - secs) * 1e9)

    # --- format the file exactly like your example ---
    lines = [
        f"width: {width}",
        f"height: {height}",
        f"distortion_model: rational_polynomial",
        f"D: ({', '.join(map(str, D))})",
        f"K: ({', '.join(map(str, K))})",
        f"R: ({', '.join(map(str, R))})",
        f"P: ({', '.join(map(str, P))})",
        "binning_x: 0",
        "binning_y: 0",
        "roi: x_offset: 0",
        "y_offset: 0",
        "height: 0",
        "width: 0",
        "do_rectify: False",
        f"header: seq: 0",
        "stamp:",
        f"  secs: {secs}",
        f"  nsecs: {nsecs}",
        f'frame_id: "{frame_id}"',
        ""
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    return output_path

def get_camera_info(calibration: dict, output_path: Path, seq: int = 0,
                    frame_id: str = "rgb_camera_link") -> Path:
    """
    Write a ROS-style CameraInfo text file using the provided PINHOLE calibration.
    Args:
        calibration: dict containing the "PINHOLE" calibration entry.
        output_path: path to save the text file (e.g., Path("camera_info.txt")).
        seq: optional sequence number for the header.
        frame_id: optional frame name (default: "rgb_camera_link").
    Returns:
        Path to the written text file.
    """
    pinhole = calibration["PINHOLE"]

    width  = pinhole["w"]
    height = pinhole["h"]
    K = pinhole["K"].flatten().tolist()
    D = pinhole.get("distortion", np.zeros(8)).flatten().tolist()
    R = np.eye(3, dtype=float).flatten().tolist()
    P = [K[0], 0.0, K[2], 0.0,
         0.0, K[4], K[5], 0.0,
         0.0, 0.0, 1.0, 0.0]

    # Use current timestamp
    from datetime import datetime
    now = datetime.now()
    secs = int(now.timestamp())
    nsecs = int((now.timestamp() - secs) * 1e9)

    # --- format the file exactly like your example ---
    lines = [
        f"width: {width}",
        f"height: {height}",
        f"distortion_model: rational_polynomial",
        f"D: ({', '.join(map(str, D))})",
        f"K: ({', '.join(map(str, K))})",
        f"R: ({', '.join(map(str, R))})",
        f"P: ({', '.join(map(str, P))})",
        "binning_x: 0",
        "binning_y: 0",
        "roi: x_offset: 0",
        "y_offset: 0",
        "height: 0",
        "width: 0",
        "do_rectify: False",
        f"header: seq: {seq}",
        "stamp:",
        f"  secs: {secs}",
        f"  nsecs: {nsecs}",
        f'frame_id: "{frame_id}"',
        ""
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    return output_path


def load_ground_truth_articulations_from_json(json_path: str) -> dict:
    """Loads ground truth articulation data from a JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def load_interaction_time_window_from_json(json_path: str) -> Dict:
    """Loads interaction time window data from a JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def load_fine_grained_interaction_window_from_json(json_path: str) -> Dict:
    """Loads fine-grained interaction window data from a JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def get_relevant_frame_indices_from_fine_grained_interaction_window(
    data_root: str | Path,
    rgb_frames_list_df: pd.DataFrame,
    stride: int
) -> List[int]:
    
    data_root = Path(data_root)
    json_path = data_root / "_windows.json"

    fine_grained_windows = load_fine_grained_interaction_window_from_json(json_path)

    #get indices of frames within fine grained windows
    relevant_frame_indices = []
    for key, val in fine_grained_windows.items():
        start_ns = val['start_ns']
        end_ns   = val['end_ns']

        # get frame indices within this window
        mask = (rgb_frames_list_df['timestamp'] >= start_ns) & (rgb_frames_list_df['timestamp'] <= end_ns)
        frame_indices = rgb_frames_list_df[mask].index.tolist()
        relevant_frame_indices.extend(frame_indices)

    relevant_frame_indices = sorted(list(set(relevant_frame_indices))) 
    return relevant_frame_indices

def make_matched_cues_from_fine_grained_window(
    data_root: str | Path,
    rgb_frames_list: List[str],
    stride: int
) -> List[Dict]:
    
    data_root = Path(data_root)
    json_path = data_root / "_windows.json"
    # pass

    fine_grained_windows = load_fine_grained_interaction_window_from_json(json_path)

    # get df of rgb frames with timestamp column
    df_rgb_frames = pd.DataFrame({
        'frame_path': rgb_frames_list,
        'timestamp': [int(Path(p).stem) for p in rgb_frames_list],
    })

    # make csv with columns: AXIS_NAME CUE_START CUE_END VERIFICATION
    rows = []
    for key, val in fine_grained_windows.items():
        axis_name = int(key)
        cue_start = val['start_ns']
        cue_end   = val['end_ns']
        verification = 'VERIFIED'

        # get frame index instead of timestamp
        # cue_start
        # use closest timestamp in df_rgb_frames
        cue_start_idx = int(((df_rgb_frames['timestamp'] - cue_start).abs()).values.argmin())
        cue_end_idx   = int(((df_rgb_frames['timestamp'] - cue_end).abs()).values.argmin())

        # apply stride
        cue_start_idx = int(cue_start_idx / stride)
        cue_end_idx = int(cue_end_idx / stride)

        rows.append({
            'AXIS_NAME': axis_name,
            'CUE_START': cue_start_idx,
            'CUE_END': cue_end_idx,
            'VERIFICATION': verification,
        })

    df_cues = pd.DataFrame(rows)
    output_csv_path = data_root / f"matched_cues.csv"
    df_cues.to_csv(output_csv_path, index=False)

    a = 2


def make_matched_cues(json_path: str,
                      rgb_frames_list: List[str],
                      stride: int) -> List[Dict]:
    pass

    interaction_windows = load_interaction_time_window_from_json(json_path)

    # get df of rgb frames with timestamp column
    df_rgb_frames = pd.DataFrame({
        'frame_path': rgb_frames_list,
        'timestamp': [int(Path(p).stem) for p in rgb_frames_list],
    })

    # make csv with columns: AXIS_NAME CUE_START CUE_END VERIFICATION
    rows = []
    for key, val in interaction_windows.items():
        axis_name = val['index']
        cue_start = val['start_ns']
        cue_end   = val['end_ns']
        verification = 'VERIFIED'

        # get frame index instead of timestamp
        # cue_start
        # use closest timestamp in df_rgb_frames
        cue_start_idx = int(((df_rgb_frames['timestamp'] - cue_start).abs()).values.argmin())
        cue_end_idx   = int(((df_rgb_frames['timestamp'] - cue_end).abs()).values.argmin())

        # apply stride
        cue_start_idx = int(cue_start_idx / stride)
        cue_end_idx = int(cue_end_idx / stride)

        rows.append({
            'AXIS_NAME': axis_name,
            'CUE_START': cue_start_idx,
            'CUE_END': cue_end_idx,
            'VERIFICATION': verification,
        })

    df_cues = pd.DataFrame(rows)
    output_csv_path = Path(json_path).parent / f"matched_cues.csv"
    df_cues.to_csv(output_csv_path, index=False)

    a = 2

def generate_artipoint_iphone_evaluation_data(
    iphone_data: IPhoneData,
    leica_data: LeicaData,
    data_root: str | Path,
    rec_location: str,
    interaction_indices: str,
):

    """Generates evaluation data for articulated object estimation using iPhone and Leica data.
    Args:
        iphone_data (IPhoneData): The iPhone data loader instance.
        leica_data (LeicaData): The Leica data loader instance.


    Returns:
        None

    """

    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    # rgb frames from iphone
    rgb_frames = iphone_data.get_extracted_frames()
    rgb_frames_df = pd.DataFrame({
        'frame_path_iphone': [str(p) for p in rgb_frames],
        'timestamp': [int(Path(p).stem) for p in rgb_frames],
    })

    # depth frames from iphone
    depth_frames = iphone_data.get_extracted_depths()
    depth_frames_df = pd.DataFrame({
        'depth_frame_path_iphone': [str(p) for p in depth_frames],
        'timestamp': [int(Path(p).stem) for p in depth_frames],
    })

    relevant_frame_indices = get_relevant_frame_indices_from_fine_grained_interaction_window(
        data_root=data_root,
        rgb_frames_list_df=rgb_frames_df,
        stride=1,
    )

    rgb_frames_df = rgb_frames_df.iloc[relevant_frame_indices].reset_index(drop=True)
    depth_frames_df = depth_frames_df.iloc[relevant_frame_indices].reset_index(drop=True)


    # copy rgb frames to data_root/rgb
    rgb_dir = data_root / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)

    if not all((rgb_dir / f"{row['timestamp']}.jpg").exists() for idx, row in rgb_frames_df.iterrows()):
    
        for idx, row in rgb_frames_df.iterrows():
            frame_path_iphone = row['frame_path_iphone']
            ts = row['timestamp']
            rgb_dest_path = rgb_dir / f"{ts}.jpg"
            if not rgb_dest_path.exists():
                import shutil
                shutil.copy2(frame_path_iphone, str(rgb_dest_path))


    # copy depth frames to data_root/depth as png
    depth_dir = data_root / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)

    if not all((depth_dir / f"{row['timestamp']}.png").exists() for idx, row in depth_frames_df.iterrows()):
    
        for idx, row in depth_frames_df.iterrows():
            depth_frame_path_iphone = row['depth_frame_path_iphone']
            ts = row['timestamp']
            depth_dest_path = depth_dir / f"{ts}.png"
            if not depth_dest_path.exists():
                npy_depth_to_png(
                    npy_path=depth_frame_path_iphone,
                    png_path=depth_dest_path,
    )
                
    # get poses for rgb frames
    w_T_wc_list, w_T_cw_list = get_extrinsics_for_iphone_image_list(
        image_paths=rgb_frames_df['frame_path_iphone'].to_list(),
        iphone_data=iphone_data,
    )

    # save odom
    odom_dir = data_root / "odom"
    odom_dir.mkdir(parents=True, exist_ok=True)
    odom_src_path = odom_dir / "odom.csv"
    if not odom_src_path.exists():
        # to csv
        w_quat_wc_list = []
        for T_world_cam in w_T_wc_list:
            R_world_cam = T_world_cam[0:3, 0:3]
            t_world_cam = T_world_cam[0:3, 3]
            quat_world_cam = R.from_matrix(R_world_cam).as_quat()
            w_quat_wc_list.append([t_world_cam[0], t_world_cam[1], t_world_cam[2], quat_world_cam[0], quat_world_cam[1], quat_world_cam[2], quat_world_cam[3]])

        df_quat = pd.DataFrame(w_quat_wc_list, columns=["tx", "ty", "tz", "qx", "qy", "qz", "qw"])
        df_quat.to_csv(odom_src_path, index=False)

    # save camera info to rgb
    calibration = iphone_data.calibration
    camera_info_path = rgb_dir / "camera_info.txt"
    if not camera_info_path.exists():
        get_camera_info(
            calibration=calibration,
            output_path=camera_info_path,
            seq=0,
            frame_id="rgb_camera_link",
        )

    # save depth camera info to depth
    camera_info_path = depth_dir / "camera_info.txt"
    if not camera_info_path.exists():
        get_iphone_depth_iphone(
            calibration=calibration,
            output_path=camera_info_path,
            frame_id="depth_camera_link",
        )

    # save mesh to data_root
    mesh = leica_data.get_mesh().to_legacy()
    mesh_path = data_root / "compressed_mesh.ply"
    if not mesh_path.exists():
        o3d.io.write_triangle_mesh(str(mesh_path), mesh, write_ascii=False, compressed=True)

    # save downsampled point cloud to data_root
    pcd = leica_data.get_downsampled_points(voxel=0.01)
    pcd_path = data_root / "downsampled_point_cloud.ply"
    if not pcd_path.exists():
        o3d.io.write_point_cloud(str(pcd_path), pcd)

    # make matched cues from fine grained window
    make_matched_cues_from_fine_grained_window(
        json_path=str(data_root / f"_windows.json"),
        rgb_frames_list=rgb_frames_df['frame_path_iphone'].to_list(),
        stride=1,
    )

    visualize_iphone_rgbd_projections_in_pointcloud(
        pcd=leica_data.get_downsampled_points(voxel=0.01),
        image_paths=rgb_frames_df['frame_path_iphone'].to_list(),
        depth_paths=depth_frames_df['depth_frame_path_iphone'].to_list(),
        w_T_wc_list=w_T_wc_list,
        K=iphone_data.calibration['PINHOLE']["K"],
        stride=1
    )


    a = 2



def generate_artipoint_evaluation_data(
    aria_data: AriaData,
    leica_data: LeicaData,
    data_root: str | Path,
    rec_location: str,
    interaction_indices: str,
):

    """Generates evaluation data for articulated object estimation using Aria and Leica data.
    Args:
        aria_data (AriaData): The Aria data loader instance.
        leica_data (LeicaData): The Leica data loader instance.
    Returns:
        None

    1. Load aria rgb and DepthAnythingV2 depth data and aligned poses.
    2. Downsample to 15 fps (which artipoint uses).
    3. Load leica mesh
    4. Render leica depth from aria poses and scale the DepthAnythingV2 depth to match leica depth.

    """

    # make data directories
    data_root = Path(data_root)
    tmp_dir = data_root / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    stride = 2  # downsample to 15 fps
    metric_scale_depth_files, image_paths_list, w_T_wc_list = get_metric_scale_depth_from_aria_and_leica(
        aria_data=aria_data,
        leica_data=leica_data,
        data_root=tmp_dir,
        stride=stride,  # downsample to 15 fps
        visualize=False
    )/home/cvg-robotics/bathroom_2.json

    # make subdirs
    for subdir in ["depth", "odom", "rgb"]:
        (data_root / subdir).mkdir(parents=True, exist_ok=True)

    # copy files to subdirs
    for idx, image_path in enumerate(image_paths_list):
        ts = int(Path(image_path).stem)

        # copy rgb
        rgb_dest_path = data_root / "rgb" / f"{ts}.jpg"
        if not rgb_dest_path.exists():
            os.system(f"cp {image_path} {rgb_dest_path}")

        # convert npy → png
        depth_src_path = metric_scale_depth_files[idx]
        depth_dest_path = data_root / "depth" / f"{ts}.png"   # direct .png target

        if not depth_dest_path.exists():
            npy_depth_to_png(
                npy_path=depth_src_path,
                png_path=depth_dest_path,
            )

    # save odom
    odom_src_path = tmp_dir / "extrinsics" / "odom.csv"
    odom_dest_path = data_root / "odom" / "odom.csv"
    if not odom_dest_path.exists():
        os.system(f"cp {odom_src_path} {odom_dest_path}")
    

    # save camera info to rgb
    calibration = aria_data.get_calibration()
    camera_info_path = data_root / "rgb" / "camera_info.txt"
    if not camera_info_path.exists():
        get_camera_info(
            calibration=calibration,
            output_path=camera_info_path,
            seq=0,
            frame_id="rgb_camera_link",
        )

    # save camera info to depth
    camera_info_path = data_root / "depth" / "camera_info.txt"
    if not camera_info_path.exists():
        get_camera_info(
            calibration=calibration,
            output_path=camera_info_path,
            seq=0,
            frame_id="depth_camera_link",
        )

    # save mesh to data_root
    mesh = leica_data.get_mesh().to_legacy()
    mesh_path = data_root / "compressed_mesh.ply"
    if not mesh_path.exists():
        o3d.io.write_triangle_mesh(str(mesh_path), mesh, write_ascii=False, compressed=True)

    # save downsampled point cloud to data_root
    pcd = leica_data.get_downsampled_points(voxel=0.01)
    pcd_full = leica_data.get_full_points()
    pcd_path = data_root / "downsampled_point_cloud.ply"
    if not pcd_path.exists():
        o3d.io.write_point_cloud(str(pcd_path), pcd)

    if False:
        # visualize one random frustum in point cloud
        K = calibration['PINHOLE']["K"]
        visualize_random_rgb_frustum_in_pointcloud(
            pcd=pcd,
            image_paths=image_paths_list,
            w_T_wc_list=w_T_wc_list,
            K=K,
            plane_depth_m=0.3,
        )

    articulations = load_ground_truth_articulations_from_json(
        json_path=str(data_root / f"{rec_location}.json")
    )


    make_matched_cues_from_fine_grained_window(
        data_root=str(data_root),
        rgb_frames_list=image_paths_list,
        stride=1,
    )

    visualize_articulations_in_pointcloud(
        pcd=pcd_full,
        articulations=articulations,
    )

    visualize_rgbd_projections_in_pointcloud(
        pcd=pcd_full,
        image_paths=image_paths_list,
        depth_paths=metric_scale_depth_files,
        w_T_wc_list=w_T_wc_list,
        K=calibration['PINHOLE']["K"],
        stride=1
    )

    visualize_articulations_and_random_rgb_frustum_in_pointcloud(
        pcd=pcd,
        articulations=articulations,
        image_paths=image_paths_list,
        depth_paths=metric_scale_depth_files,
        w_T_wc_list=w_T_wc_list,
        K=calibration['PINHOLE']["K"],
        plane_depth_m=0.3,
        random_seed=42,
    )


    a = 2



    a = 2

if __name__ == "__main__":
    rec_location = "bedroom_1"
    base_path = Path(f"/data/ikea_recordings")
    data_indexer = RecordingIndex(
        os.path.join(str(base_path), "extracted") 
    )

    rec_type = "hand"

    # interaction indices "666" for mocap tests
    interaction_indices = "1-8"

    aria_human_data = AriaData(base_path=base_path, 
                        rec_loc=rec_location, 
                        rec_type=rec_type, 
                        rec_module="aria_human", 
                        interaction_indices=interaction_indices,
                        data_indexer=data_indexer)

    leica_data = LeicaData(base_path=base_path, 
                           rec_loc=rec_location, 
                           initial_setup="001")
    
    # iphone_data = IPhoneData(base_path=base_path, 
    #                     rec_loc=rec_location, 
    #                     rec_type=rec_type, 
    #                     rec_module="iphone_1 (darkblue)", 
    #                     interaction_indices=interaction_indices,
    #                     data_indexer=data_indexer)
    
    # generate_artipoint_iphone_evaluation_data(
    #     iphone_data=iphone_data,
    #     leica_data=leica_data,
    #     data_root=data_root,
    #     rec_location=rec_location,
    #     interaction_indices=interaction_indices,
    #     )
    
    
    articulations = load_ground_truth_articulations_from_json(
        json_path=str(base_path / "extracted/bedroom_1/leica" / f"{rec_location}.json")
    )
    
    visualize_articulations_in_pointcloud(
        pcd=leica_data.get_downsampled_points(),
        articulations=articulations,
    )
        

    data_root = f"/data/evaluations/articulated_object_estimation/ground_truth/hej/raw/ikea/{rec_location}_{interaction_indices}"


    generate_artipoint_evaluation_data(
        aria_data=aria_human_data,
        leica_data=leica_data,
        data_root=data_root,
        rec_location=rec_location,
        interaction_indices=interaction_indices,
        )