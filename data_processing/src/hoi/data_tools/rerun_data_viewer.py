"""
Synchronized Rerun visualizer for gripper recordings.

Edit RECORDING_CONFIG below to match your extracted recording, then run:
    python -m hoi.data_tools.rerun_data_viewer
"""

from pathlib import Path
import threading

import numpy as np
import pandas as pd
import rerun as rr
import rerun.blueprint as rrb

from hoi.data_tools.data_loader_aria import AriaData
from hoi.data_tools.data_loader_gripper import GripperData
from hoi.data_tools.data_loader_iphone import IPhoneData
from hoi.data_tools.data_loader_leica import LeicaData


# ---------------------------------------------------------------------------
# Viewer-local subclasses that bypass the parent __init__ entirely.
# The parent constructors require VRS files and calibration YAMLs that are
# only needed for data extraction — not for reading already-extracted frames.
# We set exactly the attributes the viewer's getter methods actually use
# (confirmed from reading each method's source).
# ---------------------------------------------------------------------------

class _AriaViewer(AriaData):
    """Minimal AriaData init — sets only what get_extracted_frames() and
    get_closed_loop_trajectory_aligned() need."""

    def __init__(self, base_path: Path, rec_loc: str, rec_type: str,
                 rec_module: str, interaction_indices: str, **_ignored):
        self.base_path          = Path(base_path)
        self.rec_loc            = rec_loc
        self.rec_type           = rec_type
        self.rec_module         = rec_module
        self.interaction_indices = interaction_indices
        self.extraction_path    = (
            self.base_path / "extracted" / rec_loc / rec_type / rec_module
            / f"{rec_loc}_{interaction_indices}_{rec_type}_vrs"
        )
        self.label_rgb          = "/camera_rgb"
        self.label_slam         = "/slam"
        self.label_clt_aligned  = "/slam/closed_loop_trajectory_aligned"
        self.rgb_extension      = ".jpg"
        self.logging_tag        = f"{rec_loc}_{rec_type}_{rec_module}".upper()
        self.calibration        = {}

    def get_calibration(self):
        return {}


class _GripperViewer(GripperData):
    """Minimal GripperData init — sets only what get_frames_digit() and
    get_force_torque_measurements() need, plus loader_aria_gripper."""

    def __init__(self, base_path: Path, rec_loc: str, rec_type: str,
                 rec_module: str, interaction_indices: str,
                 color: str = "yellow", **_ignored):
        self.base_path          = Path(base_path)
        self.rec_loc            = rec_loc
        self.rec_type           = rec_type
        self.rec_module         = rec_module
        self.interaction_indices = interaction_indices
        self.color              = color
        self.extraction_path    = (
            self.base_path / "extracted" / rec_loc / rec_type / rec_module
            / f"{rec_loc}_{interaction_indices}_{rec_type}_bag"
        )
        self.rgb_extension      = ".jpg"
        self.logging_tag        = f"{rec_loc}_{rec_type}_{rec_module}".upper()
        self.calibration        = {}
        self.loader_aria_gripper = _AriaViewer(
            base_path=base_path, rec_loc=rec_loc, rec_type=rec_type,
            rec_module=f"aria_{rec_type}", interaction_indices=interaction_indices,
        )

    def get_calibration(self):
        return {}


class _LeicaViewer(LeicaData):
    """Minimal LeicaData init — no E57 conversion, no downsampling.
    Sets only what get_downsampled_points() needs to read the cached PLY."""

    def __init__(self, base_path: Path, rec_loc: str,
                 initial_setup: str = "001", voxel: float = 0.05, **_ignored):
        self.base_path       = Path(base_path)
        self.rec_loc         = rec_loc
        self.initial_setup   = initial_setup
        self.voxel           = voxel
        self.extraction_path = self.base_path / "extracted" / rec_loc / "leica"
        self.label_downsampled = "points_downsampled"
        self.logging_tag     = f"{rec_loc}_LEICA".upper()
        # Auto-detect available setups on disk; fall back to configured one.
        if self.extraction_path.exists():
            self.setups = sorted(
                d.name for d in self.extraction_path.iterdir() if d.is_dir()
            ) or [initial_setup]
        else:
            self.setups = [initial_setup]


class _IPhoneViewer(IPhoneData):
    """Minimal IPhoneData init — sets only what get_extracted_frames() needs."""

    def __init__(self, base_path: Path, rec_loc: str, rec_type: str,
                 rec_module: str, interaction_indices: str, **_ignored):
        self.base_path           = Path(base_path)
        self.rec_loc             = rec_loc
        self.rec_type            = rec_type
        self.rec_module          = rec_module
        self.interaction_indices = interaction_indices
        self.extraction_path     = (
            self.base_path / "extracted" / rec_loc / rec_type / rec_module
            / f"{rec_loc}_{interaction_indices}_{rec_type}"
        )
        self.label_rgb           = "/camera_rgb"
        self.label_poses_aligned = "/poses_aligned"
        self.rgb_extension       = ".jpg"
        self.logging_tag         = f"{rec_loc}_{rec_type}_{rec_module}".upper()
        self.extracted_rgbd      = (self.extraction_path / "camera_rgb").exists()
        self.calibration         = {}


# ---------------------------------------------------------------------------
# Edit these values to point at your extracted recording
# ---------------------------------------------------------------------------
RECORDING_CONFIG = {
    "base_path":          Path("/data/ikea_recordings"),  # path to parent of "extracted/"
    "rec_location":       "office_1",
    "rec_type":           "gripper",
    "interaction_index":  "8-14",
    "color":              "yellow",        # gripper color for calibration lookup
    "aria_human_module":  "aria_human",
    "iphone_module":      "iphone_1 (babyblue)",   # office_1: "iphone_1 (babyblue)"
    "iphone_module_2":    "iphone_2 (rosa)",       # livingroom_1: "iphone_2 (green)" (None to disable)
    "leica_setup":        "001",       # which Leica scan setup to overlay in the 3D view
    "leica_voxel":        0.05,        # must match an existing points_voxel_X.XXX.ply
    "show_instances":     True,        # overlay 3D bounding boxes from instances.json
    # Subsampling — raise to make the viewer load faster (1 = no skip)
    "image_stride":       3,    # log every Nth image frame. Cameras are ~30 Hz → 10 Hz playback.
                                # Image data dominates viewer RAM, so this is the biggest knob.
    "traj_stride":        200,  # log every Nth SLAM pose. SLAM is ~1 kHz so 200 → 5 Hz trail.
                                # Drop only if you have a beefy GPU — trail = 1 point-cloud per
                                # pose, and rendering many of them at end-of-timeline is slow.
    # Viewer memory cap — when reached, the viewer drops oldest data (FIFO). Avoids OOM.
    "viewer_memory_limit": "8GB",   # e.g. "4GB", "16GB", or "75%" of system RAM
}


# ---------------------------------------------------------------------------
# Rerun logging helpers
# ---------------------------------------------------------------------------

def _log_image_stream(entity_path: str, frame_paths: list, stride: int = 1) -> None:
    """Log image files keyed by their aligned nanosecond timestamp (file stem).
    Set stride > 1 to skip frames (e.g. stride=5 logs every 5th frame)."""
    if stride > 1:
        frame_paths = frame_paths[::stride]
    for p in frame_paths:
        rr.set_time_nanos("recording_time", int(p.stem))
        rr.log(entity_path, rr.EncodedImage(path=p))


def _log_trajectory(entity_path: str, traj_df: pd.DataFrame,
                    tx: str, ty: str, tz: str, stride: int = 1) -> None:
    """
    Log timestamped 3D pose points. With the 3D view's VisibleTimeRange set
    to [-inf, cursor], the points accumulate as a growing trail of past poses.
    """
    positions_full = traj_df[[tx, ty, tz]].to_numpy(dtype=float)
    timestamps_full = traj_df["timestamp"].to_numpy(dtype=np.int64)

    if stride > 1:
        positions  = positions_full[::stride]
        timestamps = timestamps_full[::stride]
    else:
        positions, timestamps = positions_full, timestamps_full

    rr.send_columns(
        entity_path,
        times=[rr.TimeNanosColumn("recording_time", timestamps)],
        components=[
            rr.Points3D.indicator(),
            rr.components.Position3DBatch(positions),
        ],
    )


_FT_SERIES_STYLE = {
    "force_x":  ([220,  50,  50], "Fx [N]"),
    "force_y":  ([ 50, 200,  50], "Fy [N]"),
    "force_z":  ([ 50, 100, 220], "Fz [N]"),
    "torque_x": ([220, 140,  50], "Tx [Nm]"),
    "torque_y": ([180, 220,  50], "Ty [Nm]"),
    "torque_z": ([ 50, 210, 210], "Tz [Nm]"),
}


def _log_scalar_series(entity_prefix: str, df: pd.DataFrame,
                       col_map: dict) -> None:
    """Log one scalar time-series per channel using send_columns — one batch call per axis."""
    ts = df["timestamp"].to_numpy(dtype=np.int64)

    for suffix, col in col_map.items():
        if col not in df.columns:
            continue
        values = df[col].to_numpy(dtype=float)
        color, name = _FT_SERIES_STYLE.get(suffix, ([200, 200, 200], suffix))
        rr.log(f"{entity_prefix}/{suffix}",
               rr.SeriesLine(color=color, name=name), static=True)
        rr.send_columns(
            f"{entity_prefix}/{suffix}",
            times=[rr.TimeNanosColumn("recording_time", ts)],
            components=[rr.components.ScalarBatch(values)],
        )


# ---------------------------------------------------------------------------
# Main viewer
# ---------------------------------------------------------------------------

def view_recording(config: dict) -> None:
    base_path    = Path(config["base_path"])
    rec_loc      = config["rec_location"]
    rec_type     = config["rec_type"]
    iidx         = config["interaction_index"]
    color        = config.get("color", "yellow")
    human_mod    = config.get("aria_human_module", "aria_human")
    iphone_mod   = config.get("iphone_module", "iphone_left")
    iphone_mod_2 = config.get("iphone_module_2", None)
    image_stride = int(config.get("image_stride", 1))
    traj_stride  = int(config.get("traj_stride", 1))
    mem_limit    = config.get("viewer_memory_limit", "75%")
    leica_setup  = config.get("leica_setup", "001")
    leica_voxel  = float(config.get("leica_voxel", 0.05))

    # ------------------------------------------------------------------
    # Instantiate data loaders (each wrapped so one failure doesn't abort)
    # ------------------------------------------------------------------
    gripper = aria_human = iphone = iphone_2 = leica = None

    print("Loading GripperData …")
    try:
        gripper = _GripperViewer(
            base_path=base_path, rec_loc=rec_loc, rec_type=rec_type,
            rec_module="gripper", interaction_indices=iidx, color=color,
        )
    except Exception as e:
        print(f"[WARN] Could not create GripperData: {e}")

    print("Loading AriaData (human) …")
    try:
        aria_human = _AriaViewer(
            base_path=base_path, rec_loc=rec_loc, rec_type=rec_type,
            rec_module=human_mod, interaction_indices=iidx,
        )
    except Exception as e:
        print(f"[WARN] Could not create AriaData (human): {e}")

    print("Loading IPhoneData …")
    try:
        iphone = _IPhoneViewer(
            base_path=base_path, rec_loc=rec_loc, rec_type=rec_type,
            rec_module=iphone_mod, interaction_indices=iidx,
        )
    except Exception as e:
        print(f"[WARN] Could not create IPhoneData: {e}")

    if iphone_mod_2:
        print("Loading IPhoneData (2) …")
        try:
            iphone_2 = _IPhoneViewer(
                base_path=base_path, rec_loc=rec_loc, rec_type=rec_type,
                rec_module=iphone_mod_2, interaction_indices=iidx,
            )
        except Exception as e:
            print(f"[WARN] Could not create IPhoneData (2): {e}")

    print("Loading LeicaData …")
    try:
        leica = _LeicaViewer(
            base_path=base_path, rec_loc=rec_loc,
            initial_setup=leica_setup, voxel=leica_voxel,
        )
    except Exception as e:
        print(f"[WARN] Could not create LeicaData: {e}")

    # ------------------------------------------------------------------
    # Rerun init + blueprint
    # ------------------------------------------------------------------
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(name="Aria Human RGB",   origin="aria_human/rgb"),
                rrb.Spatial2DView(name="Aria Gripper RGB", origin="aria_gripper/rgb"),
                rrb.Spatial2DView(name="iPhone 1 RGB",     origin="iphone/rgb"),
            ),
            rrb.Horizontal(
                rrb.Spatial2DView(name="Tactile Left",      origin="gripper/tactile/left"),
                rrb.Spatial2DView(name="iPhone 2 RGB",       origin="iphone_2/rgb"),
                rrb.Spatial3DView(
                    name="SLAM Trajectories + Leica",
                    origin="/",
                    # Accumulate: trail grows from start up to current cursor.
                    # The static Leica pcd ignores time ranges, so it stays as backdrop.
                    time_ranges=[
                        rrb.VisibleTimeRange(
                            timeline="recording_time",
                            start=rrb.TimeRangeBoundary.infinite(),
                            end=rrb.TimeRangeBoundary.cursor_relative(),
                        )
                    ],
                    contents=[
                        "aria_human/trajectory",
                        "aria_gripper/trajectory",
                        "leica/pointcloud",
                        "leica/instances/**",
                        "viewpoint_camera",
                    ],
                ),
            ),
            rrb.TimeSeriesView(
                name="Force-Torque",
                origin="gripper/ft",
                # Accumulate: show samples from -inf up to (and including) the cursor
                time_ranges=[
                    rrb.VisibleTimeRange(
                        timeline="recording_time",
                        start=rrb.TimeRangeBoundary.infinite(),
                        end=rrb.TimeRangeBoundary.cursor_relative(),
                    )
                ],
            ),
            row_shares=[3, 3, 2],
        ),
        collapse_panels=True,
    )
    rr.init("gripper_recording_viewer")
    # Spawn with explicit memory cap. When the viewer hits the limit it drops
    # the oldest data (FIFO) instead of OOM-killing the process.
    rr.spawn(memory_limit=mem_limit)
    rr.send_blueprint(blueprint)

    # ------------------------------------------------------------------
    # Log streams
    # ------------------------------------------------------------------

    # ---- Trajectories and scalars first (batch calls, near-instant) ----

    # Aria human — SLAM trajectory (aligned)
    if aria_human is not None:
        try:
            traj = aria_human.get_closed_loop_trajectory_aligned()
            print(f"  Aria human SLAM traj:  {len(traj)} poses (stride={traj_stride})")
            _log_trajectory("aria_human/trajectory", traj,
                            "tx_world_device", "ty_world_device", "tz_world_device",
                            stride=traj_stride)
        except Exception as e:
            print(f"[WARN] Skipping Aria human trajectory: {e}")

    # Aria gripper — SLAM trajectory (aligned)
    if gripper is not None:
        try:
            traj = gripper.loader_aria_gripper.get_closed_loop_trajectory_aligned()
            print(f"  Aria gripper SLAM traj:{len(traj)} poses (stride={traj_stride})")
            _log_trajectory("aria_gripper/trajectory", traj,
                            "tx_world_device", "ty_world_device", "tz_world_device",
                            stride=traj_stride)
        except Exception as e:
            print(f"[WARN] Skipping Aria gripper trajectory: {e}")

    # Leica scene point cloud — static, always visible in the 3D view
    if leica is not None:
        try:
            pcd = leica.get_downsampled_points(setup=leica_setup, voxel=leica_voxel)
            positions = np.asarray(pcd.points)
            colors = (
                (np.asarray(pcd.colors) * 255).astype(np.uint8)
                if pcd.has_colors() else None
            )
            print(f"  Leica pointcloud:      {len(positions)} points "
                  f"(setup={leica_setup}, voxel={leica_voxel})")
            rr.log("leica/pointcloud",
                   rr.Points3D(positions, colors=colors, radii=0.01),
                   static=True)
        except Exception as e:
            print(f"[WARN] Skipping Leica pointcloud: {e}")

    # Leica 3D instance annotations — load each instance's lifted PLY and show
    # the actual points (coloured by class). A short "#idx <abbr>" label is
    # placed at the median of inlier points; no bounding boxes.
    if leica is not None and config.get("show_instances", True):
        try:
            import json, colorsys, hashlib, open3d as o3d
            inst_path = (Path(base_path) / "extracted" / rec_loc / "leica" /
                         leica_setup / "instance_annotations_3d" / "instances.json")
            with open(inst_path) as f:
                items = json.load(f)["items"]
            if not items:
                raise ValueError("instances.json has no items")

            def _class_color(cls: str) -> tuple:
                h = int(hashlib.md5(cls.encode()).hexdigest()[:6], 16) / 0xFFFFFF
                r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
                return (int(r * 255), int(g * 255), int(b * 255))

            _CLASS_ABBR = {"revolute": "rev", "prismatic": "pris", "fixed": "fix"}

            all_pts, all_cols = [], []
            label_pts, label_cols, label_text = [], [], []

            for it in items:
                ply_path = Path(it["output_ply"])
                if not ply_path.exists():
                    print(f"[WARN] missing instance PLY #{it['index']}: {ply_path}")
                    continue
                pcd = o3d.io.read_point_cloud(str(ply_path))
                pts = np.asarray(pcd.points)
                if len(pts) == 0:
                    continue

                color = _class_color(it["class"])
                all_pts.append(pts)
                all_cols.append(np.tile(color, (len(pts), 1)))

                # Label marker at median of inlier points (5–95 percentile)
                lo, hi   = np.percentile(pts, [5, 95], axis=0)
                centroid = (lo + hi) / 2.0
                label_pts.append(centroid)
                label_cols.append(color)
                abbr = _CLASS_ABBR.get(it["class"], it["class"][:4])
                label_text.append(f"#{it['index']} {abbr}")

            if not all_pts:
                raise ValueError("no instance PLYs could be loaded")

            all_pts  = np.concatenate(all_pts).astype(np.float32)
            all_cols = np.concatenate(all_cols).astype(np.uint8)

            rr.log("leica/instances/points",
                   rr.Points3D(all_pts, colors=all_cols, radii=0.008),
                   static=True)
            rr.log("leica/instances/labels",
                   rr.Points3D(np.array(label_pts),
                               colors=np.array(label_cols, dtype=np.uint8),
                               labels=label_text,
                               radii=0.03),
                   static=True)

            # Suggested viewpoint: pinhole at origin looking at items[0] centroid
            target = np.array(label_pts[0])
            eye    = np.array([0.0, 0.0, 0.0])
            up     = np.array([0.0, 0.0, 1.0])     # Z-up world
            fwd    = target - eye
            fwd   /= np.linalg.norm(fwd)
            right  = np.cross(fwd, up); right /= np.linalg.norm(right)
            down   = np.cross(fwd, right)
            # rerun Pinhole convention: cam +X=right, +Y=down, +Z=forward
            R = np.column_stack([right, down, fwd])
            rr.log("viewpoint_camera",
                   rr.Pinhole(focal_length=600, width=800, height=600),
                   static=True)
            rr.log("viewpoint_camera",
                   rr.Transform3D(translation=eye, mat3x3=R),
                   static=True)

            print(f"  Leica instances:       {len(label_pts)} loaded "
                  f"({len(all_pts):,} total points)")
            print(f"  → select 'viewpoint_camera' in rerun and use the "
                  f"'Look at this camera' action to jump to that view.")
        except Exception as e:
            print(f"[WARN] Skipping Leica instance annotations: {e}")

    # Gripper — force-torque (prefer gravity-compensated/filtered columns)
    if gripper is not None:
        try:
            ft_df = gripper.get_force_torque_measurements()
            has_comp = any(
                "wrench_ext" in c and "_filt" in c for c in ft_df.columns
            )
            if has_comp:
                col_map = {
                    "force_x":  "wrench_ext.force.x_filt",
                    "force_y":  "wrench_ext.force.y_filt",
                    "force_z":  "wrench_ext.force.z_filt",
                    "torque_x": "wrench_ext.torque.x_filt",
                    "torque_y": "wrench_ext.torque.y_filt",
                    "torque_z": "wrench_ext.torque.z_filt",
                }
            else:
                col_map = {
                    "force_x":  "wrench.force.x",
                    "force_y":  "wrench.force.y",
                    "force_z":  "wrench.force.z",
                    "torque_x": "wrench.torque.x",
                    "torque_y": "wrench.torque.y",
                    "torque_z": "wrench.torque.z",
                }
            print(f"  Force-torque:          {len(ft_df)} samples "
                  f"({'compensated' if has_comp else 'raw'})")
            _log_scalar_series("gripper/ft", ft_df, col_map)
        except Exception as e:
            print(f"[WARN] Skipping force-torque: {e}")

    # ---- Image streams — each runs in its own thread so they log concurrently ----

    def _run(name, get_frames, entity_path):
        try:
            frames = get_frames()
            kept = (len(frames) + image_stride - 1) // image_stride
            print(f"  {name:24s} {len(frames)} frames "
                  f"(stride={image_stride}, logging {kept})")
            _log_image_stream(entity_path, frames, stride=image_stride)
        except Exception as e:
            print(f"[WARN] Skipping {name}: {e}")

    image_tasks = []
    if aria_human is not None:
        image_tasks.append(("Aria human RGB",
                            aria_human.get_extracted_frames,
                            "aria_human/rgb"))
    if gripper is not None:
        image_tasks.append(("Aria gripper RGB",
                            gripper.loader_aria_gripper.get_extracted_frames,
                            "aria_gripper/rgb"))
    if iphone is not None:
        image_tasks.append(("iPhone 1 RGB",
                            iphone.get_extracted_frames,
                            "iphone/rgb"))
    if iphone_2 is not None:
        image_tasks.append(("iPhone 2 RGB",
                            iphone_2.get_extracted_frames,
                            "iphone_2/rgb"))
    if gripper is not None:
        image_tasks.append(("DIGIT left",
                            lambda: gripper.get_frames_digit("left"),
                            "gripper/tactile/left"))

    threads = [threading.Thread(target=_run, args=task, daemon=True)
               for task in image_tasks]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("Done — Rerun viewer is running.")


if __name__ == "__main__":
    view_recording(RECORDING_CONFIG)
