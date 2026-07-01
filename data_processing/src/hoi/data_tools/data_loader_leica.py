from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Sequence, Tuple
import colorsys

import numpy as np
import open3d as o3d
import pye57
from tqdm import tqdm
import math
import json
import re
import cv2
import OpenEXR, Imath
from scipy.spatial.transform import Rotation as R
from hoi.data_tools.data_indexer import RecordingIndex
import os
import scipy.io as sio
import shutil

os.environ["EGL_PLATFORM"] = "surfaceless"

from hoi.data_tools.utils_articulation_gt import get_pcd_diff, visualize_pcds

class LeicaData:
    def __init__(self, base_path: Path, rec_loc: str, initial_setup: str, voxel: float = 0.05) -> None:
    
        self.base_path = base_path
        self.rec_loc = rec_loc

        self.extraction_path = base_path / "extracted" / rec_loc / "leica"

        self.leica_path_raw = base_path / "raw" / rec_loc / "leica"

        self.initial_setup = initial_setup
        self.setups = []

        self.label_points = "points"
        self.label_downsampled = "points_downsampled"
        self.label_renderings = "renderings"
        self.label_pano_tiles = "pano_tiles"
        self.label_images = "images"
        self.label_mesh = "mesh"

        self.voxel = voxel

        self.extract_all_setups()

        
    def extract_all_setups(self) -> Tuple[Path, Path]:

        """ Extracts setups from the raw Leica data directory.
        This method scans the raw directory for files containing "Setup ..." in their names,
        and creates a directory for each setup in the extraction path.
        """

        if self._extracted():
            print(f"[Leica] Data already extracted for {self.rec_loc}.")
            self.setups = sorted([d.name for d in self.extraction_path.iterdir() if d.is_dir()])
            return

        # List files in raw directory
        raw_files = list(self.leica_path_raw.glob("*"))

        # Find files containing "Setup ..." in their names and get all setups
        setup_files = [f for f in raw_files if "Setup " in f.stem]
        setups = [f.stem.split("Setup ")[-1] for f in setup_files]
        setups = [s[0:3] for s in setups]  # take only the first 3 characters
        self.setups = sorted(set(setups))  # unique setups

        # Check for existing mesh files and copy the one with the lowest number
        mesh_files = [f for f in raw_files if "mesh" in f.stem and f.suffix == ".ply"]
        if mesh_files:
            # Sort setup directories by name and pick the one with the lowest name
            lowest_setup_dir = self.extraction_path / sorted(self.setups)[0]
            target_mesh_dir = lowest_setup_dir / self.label_mesh
            target_mesh_dir.mkdir(parents=True, exist_ok=True)
            target_mesh_file = target_mesh_dir / "mesh.ply"
            
            # Sort mesh files by their numeric part and pick the one with the lowest number
            mesh_file = mesh_files[0]
            
            if not target_mesh_file.exists():
                shutil.copy(mesh_file, target_mesh_file)
            print(f"[Leica] Copied {mesh_file.name} to {target_mesh_file}")

        for setup in self.setups:
            pattern = f"Setup {setup}"
            # find all files in dir that contrain the pattern
            matching_files = [f for f in raw_files if pattern in f.stem]
            # create setup dir in extration path
            setup_dir = self.extraction_path / f"{setup}"
            setup_dir.mkdir(parents=True, exist_ok=True)
            # copy files to setup dir
            for file in matching_files:
                if file.suffix == ".e57":
                    target_file = setup_dir / self.label_points / f"points.ply"
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    if not target_file.exists():
                        self._points_e57_to_ply(e57_path=file, ply_path=target_file)
                        print(f"[Leica] Converted {file.name} to PLY and saved to {target_file}")
                    # downsample
                    self._make_downsampled(setup=setup)
                    self._make_downsampled(setup=setup, voxel=0.01)
                else:
                    target_file = setup_dir / self.label_images / file.name
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    if not target_file.exists():
                        shutil.copy(file, target_file)
                        print(f"[Leica] Copied {file.name} to {target_file}")

        print(f"[Leica] Extracted {len(self.setups)} setups from {self.rec_loc}.")

    
    def get_downsampled_points(self, setup: str | None = None, voxel: float | None = None) -> o3d.geometry.PointCloud:
        """
        Returns downsampled_cloud.  Caches both to disk.
        """

        if setup is None:
            setup = self.setups[0]

        if voxel is None:
            voxel = self.voxel

        if setup not in self.setups:
            raise ValueError(f"Setup {setup} not found in {self.setups}")
        
        down_path = self.extraction_path / setup / self.label_downsampled / f"points_voxel_{self.voxel:.3f}.ply"

        return o3d.io.read_point_cloud(str(down_path))
    
    def get_full_points(self, setup: str | None = None) -> o3d.geometry.PointCloud:
        """
        Returns the full point cloud for the given setup.
        Caches the PLY file if it does not exist.
        """
        if setup is None:
            setup = self.setups[0]

        if setup not in self.setups:
            raise ValueError(f"Setup {setup} not found in {self.setups}")

        ply_path = self.extraction_path / setup / self.label_points / f"points.ply"

        return o3d.io.read_point_cloud(str(ply_path))
    
    def get_mesh(self, setup: str | None = None) -> o3d.t.geometry.TriangleMesh:
        """
        Returns the mesh for the given setup.
        Caches the mesh if it does not exist.
        """
        if setup is None:
            setup = self.setups[0]

        if setup not in self.setups:
            raise ValueError(f"Setup {setup} not found in {self.setups}")

        mesh_path = self.extraction_path / setup / self.label_mesh / f"mesh.ply"

        if not mesh_path.exists():
            print(f"[Leica] Mesh not found at {mesh_path}. Exists only for the first setup of the recording.")

        return o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(str(mesh_path)))
    
    def get_panos(self, setup: str | None = None) -> dict:
        """
        Returns a dictionary containing the paths to the panorama images and their associated metadata. 
        The dictionary contains the following keys:
        - "depth": Path to the depth image (EXR file).
        - "rgb": Path to the RGB image (PNG file).
        - "pose": Parsed pose information from the TXT file.
        """
        if setup is None:
            setup = self.setups[0]

        if setup not in self.setups:
            raise ValueError(f"Setup {setup} not found in {self.setups}")

        panos_dir = self.extraction_path / setup / self.label_images
    
        # Check if the panos directory contains 1 JPG, 1 EXR, 2 PNGs, and 1 TXT
        if not panos_dir.exists():
            raise FileNotFoundError(f"Panos directory not found: {panos_dir}")

        jpg_files = list(panos_dir.glob("*.jpg"))
        exr_files = list(panos_dir.glob("*.exr"))
        png_files = list(panos_dir.glob("*.png"))
        txt_files = list(panos_dir.glob("*.txt"))

        if len(jpg_files) != 1:
            raise ValueError(f"Expected 1 JPG file in {panos_dir}, found {len(jpg_files)}")
        if len(exr_files) != 1:
            raise ValueError(f"Expected 1 EXR file in {panos_dir}, found {len(exr_files)}")
        if len(png_files) != 2:
            raise ValueError(f"Expected 2 PNG files in {panos_dir}, found {len(png_files)}")
        if len(txt_files) != 1:
            raise ValueError(f"Expected 1 TXT file in {panos_dir}, found {len(txt_files)}")

        # get files
        pose_info_file = list(panos_dir.glob(f"*{setup}.txt"))[0]
        depth_pano_file = list(panos_dir.glob(f"*{setup}.exr"))[0]
        rgb_pano_file = list(panos_dir.glob(f"*{setup}.png"))[0]

        depth_pano = self._read_exr(str(depth_pano_file))
        
        rgb_pano = cv2.imread(rgb_pano_file, cv2.IMREAD_UNCHANGED)

        panos = {}


        panos["rgb"] = rgb_pano
        panos["pose"] = self._parse_pano_pose(pose_info_file)
        
        return panos

    def _points_e57_to_ply(self, e57_path: str | Path, ply_path: str | Path) -> None:

        e57 = pye57.E57(str(e57_path))
        data = e57.read_scan(0, colors=True, ignore_missing_fields=True)

        xyz = np.column_stack(
            (data["cartesianX"], data["cartesianY"], data["cartesianZ"])
        ).astype(np.float32)

        # Filter invalids ------------------------------------------------------
        mask = ~np.isnan(xyz).any(axis=1)
        xyz = xyz[mask]

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))

        if {"colorRed", "colorGreen", "colorBlue"}.issubset(data.keys()):
            rgb = np.stack(
                (data["colorRed"], data["colorGreen"], data["colorBlue"]),
                axis=1
            ).astype(np.float32) / 255.0
            rgb = rgb[mask]
            pcd.colors = o3d.utility.Vector3dVector(rgb)

        if {"normalX", "normalY", "normalZ"}.issubset(data.keys()):
            normals = np.column_stack(
                (data["normalX"], data["normalY"], data["normalZ"])
            ).astype(np.float32)
            normals = normals[mask]
            pcd.normals = o3d.utility.Vector3dVector(normals)

        with tqdm(total=1, desc="Saving PLY", unit="file") as pbar:
            o3d.io.write_point_cloud(str(ply_path), pcd, write_ascii=False)
            pbar.update(1)        
        
        print(f"[Leica] Saved full-resolution PLY at {ply_path}")

    def make_360_views_from_render(self, setup: str | None = None) -> None:
        
        if setup is None:
            setup = self.setups[0]  # default to first setup

        if setup not in self.setups:
            raise ValueError(f"Setup {setup} not found in {self.setups}")

        out_dir = self.extraction_path / setup / self.label_renderings

        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"[Leica] Renderings already exist for setup {setup} at {out_dir}.")
            return

        # ---------- Load point cloud and downsample ------------------------------
        pcd = self.get_full_points(setup=setup)#.voxel_down_sample(0.01)

        # ---------- Camera intrinsics --------------------------------------------
        N, fov, W, H = 20, 80, 640, 480
        fx = fy = (W / 2) / math.tan(math.radians(fov / 2))
        cx, cy = W / 2, H / 2
        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

        # ---------- Setup renderer ------------------------------------------------
        renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit" if pcd.has_colors() else "defaultUnlit"
        renderer.scene.add_geometry("leica", pcd, mat)
        renderer.scene.set_background([1, 1, 1, 1])

        (out_dir / "rgb").mkdir(parents=True, exist_ok=True)
        (out_dir / "poses").mkdir(parents=True, exist_ok=True)

        center = np.zeros(3)  # Leica scanner at origin

        # ---------- Render loop ---------------------------------------------------
        from tqdm import tqdm
        for i in tqdm(range(N), desc=f"Rendering {setup}"):

            r = 1.5  # distance from the center of the scanner
            if i < N / 2:
                yaw = 2 * math.pi * i / (N / 2)  
                translation_low = np.array([r * math.cos(yaw), -0.3, r * math.sin(yaw)])
                translation_high = np.array([r * math.cos(yaw), 0.3, r * math.sin(yaw)])
            else:
                yaw = 2 * math.pi * (-(i - N)) / (N / 2) - math.pi
                translation_low = np.array([r * math.cos(yaw + math.pi), -0.3, r * math.sin(yaw + math.pi)])
                translation_high = np.array([r * math.cos(yaw + math.pi), 0.3, r * math.sin(yaw + math.pi)])

            translation_no_radius = np.array([0, 0, 0])

            # Convert from Z-up to Y-up coordinate system
            R = np.array([
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0]
            ])

            R_1 = np.array([
                [math.cos(yaw), 0, math.sin(yaw)],
                [0, 1, 0],
                [-math.sin(yaw), 0, math.cos(yaw)]
            ])

            ext = np.eye(4, dtype=float)
            ext[:3, :3] = R_1 @ R

            for idx, translation in enumerate([translation_low, translation_high, translation_no_radius]):
                ext[:3, 3] = translation + center
                renderer.setup_camera(intrinsic, ext)
                img = renderer.render_to_image()
                name = idx + i * 3
                o3d.io.write_image(str(out_dir / "rgb" / f"{name:04d}.png"), img, 9)

                meta = dict(
                width=W,
                height=H,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                extrinsic=ext.tolist())


                with open(out_dir / "poses" / f"{name:04d}.json", "w") as f:
                    json.dump(meta, f, indent=2)

        print(f"[Leica] Rendered {N* 2} rotating tripod views → {out_dir/'images'}")

    def make_360_views_from_pano(self, setup: str | None = None, manual_xyz: Tuple[float] = None) -> None:
        """
        Slice a full-equirect pano into 45degx45deg pinhole crops (32 total),
        writing both RGB + depth tiles + 3D points and a per-tile JSON with intrinsics+pose.
        """
        if setup is None:
            setup = self.setups[0]

        pano_data = self.get_panos(setup=setup)
        rgb_pano  = pano_data["rgb"]

        meta      = pano_data["pose"] ["HDR"]
           # {"t": [...], "q": [...]}

        # load once
        equi_rgb   = rgb_pano

        if manual_xyz is not None:
            meta["position"] = list(manual_xyz)

        # output dirs
        out_base = self.extraction_path / setup / "pano_tiles"

        if out_base.exists() and any(out_base.iterdir()):
            print(f"[LEICA] Pano tiles already exist for setup {setup} at {out_base}.")
            return
        
        rgb_out  = out_base / "rgb"
        depth_out= out_base / "depth"
        depth_vis_out = out_base / "depth_vis"
        pose_out = out_base / "poses"
        xyz_out = out_base / "xyz"
        for d in (rgb_out, depth_out, pose_out, xyz_out, depth_vis_out):

            d.mkdir(parents=True, exist_ok=True)

        # tiling params
        hfov = 90.0
        vfov = 120.0
        W, H = 1024, 1364
        step = hfov * 0.1

        # get initial camera pose and rots around world axes
        euler0 = R.from_quat(meta["orientation"], scalar_first=True).as_euler("xyz", degrees=True)
        rot_initial_around_world_z = euler0[2]
        rot_initial_around_world_y = euler0[1]  
        rot_initial_around_world_x = euler0[0]  

        # define default camera pose for tiel cropping and rendering
        # no rotation, translation from metadata
        t0 = np.array(meta["position"])
        R_wc_initial = R.from_quat([1.0, 0.0, 0.0, 0.0], scalar_first=True).as_matrix()

        # create a grid of yaw angles to cover the full 360 degrees
        yaws = np.arange(0.0, 360.0, step) 

        T_o3d_leica = np.array([
                [1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ])
                
        idx = 0
        for yaw_deg in yaws:
            yaw   = math.radians(yaw_deg)

            # compute target camera pose
            # yaw the camera around its own y-axis for horizontal rotation in camera frame
            R_yaw = R.from_euler("y", yaw, degrees=False).as_matrix()
            R_wc_target = R_yaw # can prolly remove R_wc_initial, since we use identny quat for initial pose
            t_wc_target = t0 

            # compute new camera pose in world coordinates after rotation
            T_leica_cam = np.eye(4, dtype=float)
            T_leica_cam[:3, :3] = R_wc_target  # convert from Leica to Open3D coordinate system
            T_leica_cam[:3, 3] = T_o3d_leica[:3,:3] @ t_wc_target # changed this
            T_cam_leica = np.linalg.inv(T_leica_cam)

            T_cw_o3d = T_cam_leica @ T_o3d_leica  # convert from Leica to Open3D coordinate system  # convert from Open3D to Leica coordinate system
            T_wc_save = np.linalg.inv(T_o3d_leica) @ T_leica_cam  # save in Open3D coordinate system

            #RGB pinhole tile in the equirectangular image
            # adjust yaw to match the initial rotation for crops
            # this is needed to align the crops with the original panorama
            yaw_adjusted = yaw_deg + rot_initial_around_world_z  
            R_yaw = R.from_euler("y", yaw_adjusted, degrees=True).as_matrix()
            tile_rgb = self._equirect_to_pinhole(
                equi_rgb, R_yaw, hfov, vfov, W, H
            )

            # write RGB tile
            fn = f"{idx:03d}.jpg"
            cv2.imwrite(str(rgb_out/fn), tile_rgb)

            # compute intrinsics
            fx = (W/2) / math.tan(math.radians(hfov/2))
            fy = (H/2) / math.tan(math.radians(vfov/2))
            K  = np.array([[fx,0,W/2],
                [0, fy,H/2],
                [0,  0,  1]])

            # render depth from mesh
            mesh = self.get_mesh(setup=setup)
            tile_depth = self._render_depth(mesh, K, T_cw_o3d)

            # 2) write depth tile
            depth_fn = fn.replace(".jpg", ".exr")
            self._write_exr(str(depth_out/depth_fn), tile_depth)

            # 3) write depth visualization
            depth_vis_fn = fn.replace(".jpg", "_vis.png")
            self._write_depth_vis(str(depth_vis_out/depth_vis_fn), tile_depth)

            # 4) write xyz tile to mat
            xyz_tile = self._depth_to_world_xyz(tile_depth, K,  T_wc_save) #ext_save
            xyz_fn = fn + ".mat"
            self._save_mat(str(xyz_out/xyz_fn), rgb_image=tile_rgb, xyz_array=xyz_tile)

            q = R.from_matrix(T_wc_save[:3, :3]).as_quat(scalar_first=False) 
             # convert to quaternion

            # 5) Dump JSON
            pose = {
                "w_T_wc": T_wc_save.tolist(),
                "K": K.tolist(),
                "h": H,
                "w": W}
            
            with open(pose_out/fn.replace(".jpg",".json"), "w") as f:
                json.dump(pose, f, indent=2)

            idx += 1

        print(f"[LEICA] Exported {idx} tiles → {out_base}")

    def lift_pano_instance_annotations_to_3d(
        self,
        setup: str | None = None,
        annotation_json_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        dedup_voxel: float = 0.005,
        handle_radius_px: int = 5,
    ) -> Path:
        """
        Lift manual panorama instance annotations into Leica world-frame 3D points.

        This method intentionally reuses the same panorama rectification logic used by
        `make_360_views_from_pano()`: the panorama annotations are projected into the
        existing pinhole tiles with the same yaw schedule and the same
        `_equirect_to_pinhole()` transform, and the corresponding world points are read
        from the existing `pano_tiles/xyz/*.mat` files used downstream by hloc/InLoc.

        Args:
            setup: Leica setup identifier, defaults to the first available setup.
            annotation_json_path: Optional explicit path to the panorama annotation JSON.
            output_dir: Optional output directory for 3D instance outputs.
            dedup_voxel: Optional world-space voxel size for de-duplicating overlapping
                points coming from neighboring pano tiles. Set <= 0 to disable.
            handle_radius_px: Radius in panorama pixels used when only a handle point is
                available and no handle mask/polygon exists.

        Returns:
            Path to the written manifest JSON file.
        """

        if setup is None:
            setup = self.setups[0]

        if setup not in self.setups:
            raise ValueError(f"Setup {setup} not found in {self.setups}")

        annotation_json = self._resolve_instance_annotation_json(
            setup=setup,
            annotation_json_path=annotation_json_path,
        )
        if annotation_json is None:
            raise FileNotFoundError(
                f"No instance annotation JSON found for Leica setup {setup} in {self.extraction_path}"
            )

        pano_tiles_dir = self.extraction_path / setup / self.label_pano_tiles
        pose_dir = pano_tiles_dir / "poses"
        xyz_dir = pano_tiles_dir / "xyz"

        if not pose_dir.exists() or not xyz_dir.exists():
            print(
                f"[Leica] Missing pano_tiles for setup {setup}; generating them with "
                f"make_360_views_from_pano() before lifting annotations."
            )
            self.make_360_views_from_pano(setup=setup)

        if output_dir is None:
            output_dir = self.extraction_path / setup / "instance_annotations_3d"
        else:
            output_dir = Path(output_dir)

        points_dir = output_dir / "points"
        points_dir.mkdir(parents=True, exist_ok=True)

        annotation_data = json.loads(annotation_json.read_text())
        items = annotation_data.get("items", [])
        if not items:
            raise ValueError(f"No annotated items found in {annotation_json}")

        pano_data = self.get_panos(setup=setup)
        rgb_pano = pano_data["rgb"]
        pose_meta = pano_data["pose"]["HDR"]

        pano_h, pano_w = rgb_pano.shape[:2]
        if "image_size" in annotation_data and len(annotation_data["image_size"]) == 2:
            anno_w = int(annotation_data["image_size"][0])
            anno_h = int(annotation_data["image_size"][1])
            if (anno_w, anno_h) != (pano_w, pano_h):
                print(
                    f"[Leica] Warning: annotation image size {(anno_w, anno_h)} does not match "
                    f"pano image size {(pano_w, pano_h)} for setup {setup}. "
                    "Using the annotation size for mask rasterization and resizing masks if needed."
                )
        else:
            anno_w, anno_h = pano_w, pano_h

        tile_defs = self._build_pano_tile_projection_definitions(
            setup=setup,
            pano_pose_meta=pose_meta,
            xyz_dir=xyz_dir,
        )
        if not tile_defs:
            raise FileNotFoundError(
                f"No pano tile XYZ files found for setup {setup} under {xyz_dir}"
            )

        summary_items: List[Dict[str, Any]] = []

        for item in tqdm(items, desc=f"[Leica] Lift instance annotations ({setup})"):
            item_index = int(item["index"])
            object_mask, object_mask_source = self._resolve_instance_object_mask(
                item=item,
                annotation_json_path=annotation_json,
                setup=setup,
                image_width=anno_w,
                image_height=anno_h,
            )
            handle_mask, handle_mask_source = self._resolve_instance_handle_mask(
                item=item,
                annotation_json_path=annotation_json,
                setup=setup,
                image_width=anno_w,
                image_height=anno_h,
                radius_px=handle_radius_px,
            )

            object_mask_pixels = int(np.count_nonzero(object_mask))

            object_points_per_tile: List[np.ndarray] = []
            handle_points_per_tile: List[np.ndarray] = []
            contributing_tiles: List[str] = []

            for tile_def in tile_defs:
                tile_mask = self._equirect_to_pinhole(
                    object_mask,
                    tile_def["rot_mat"],
                    tile_def["hfov_deg"],
                    tile_def["vfov_deg"],
                    tile_def["tile_width"],
                    tile_def["tile_height"],
                )
                tile_mask = tile_mask > 127
                if not np.any(tile_mask):
                    continue

                xyz_tile = self._load_xyz_tile(tile_def["xyz_path"])
                valid_xyz = np.isfinite(xyz_tile).all(axis=2) & np.any(xyz_tile != 0.0, axis=2)

                object_selection = tile_mask & valid_xyz
                if not np.any(object_selection):
                    continue

                object_points_per_tile.append(xyz_tile[object_selection])
                contributing_tiles.append(tile_def["stem"])

                if handle_mask is not None:
                    tile_handle_mask = self._equirect_to_pinhole(
                        handle_mask,
                        tile_def["rot_mat"],
                        tile_def["hfov_deg"],
                        tile_def["vfov_deg"],
                        tile_def["tile_width"],
                        tile_def["tile_height"],
                    )
                    tile_handle_mask = tile_handle_mask > 127
                    handle_selection = tile_handle_mask & valid_xyz
                    if np.any(handle_selection):
                        handle_points_per_tile.append(xyz_tile[handle_selection])

            if object_points_per_tile:
                object_points = np.concatenate(object_points_per_tile, axis=0).astype(np.float32)
                object_points = self._deduplicate_world_points(object_points, voxel=dedup_voxel)
            else:
                object_points = np.empty((0, 3), dtype=np.float32)

            if handle_points_per_tile:
                handle_points = np.concatenate(handle_points_per_tile, axis=0).astype(np.float32)
                handle_points = self._deduplicate_world_points(handle_points, voxel=dedup_voxel)
                handle_point_3d = handle_points.mean(axis=0).astype(np.float32)
                handle_support_points = int(handle_points.shape[0])
            else:
                handle_points = np.empty((0, 3), dtype=np.float32)
                handle_point_3d = None
                handle_support_points = 0

            ply_path = points_dir / f"{item_index:03d}.ply"
            if object_points.shape[0] > 0:
                self._write_points_as_ply(object_points, ply_path)

            item_summary: Dict[str, Any] = {
                "index": item_index,
                "class": item.get("class"),
                "description": item.get("description"),
                "source_mask_path": object_mask_source,
                "source_handle_mask_path": handle_mask_source,
                "polygon": item.get("polygon"),
                "handle": item.get("handle"),
                "mask_pixel_count": object_mask_pixels,
                "tile_ids": contributing_tiles,
                "num_tiles": len(contributing_tiles),
                "num_points_3d": int(object_points.shape[0]),
                "output_ply": str(ply_path) if object_points.shape[0] > 0 else None,
                "bbox_min": object_points.min(axis=0).tolist() if object_points.shape[0] > 0 else None,
                "bbox_max": object_points.max(axis=0).tolist() if object_points.shape[0] > 0 else None,
                "handle_point_3d": handle_point_3d.tolist() if handle_point_3d is not None else None,
                "handle_support_points": handle_support_points,
            }
            summary_items.append(item_summary)

            print(
                f"[Leica] Instance {item_index:03d}: {item_summary['num_points_3d']} 3D points "
                f"from {item_summary['num_tiles']} tile(s)"
            )

        manifest = {
            "rec_loc": self.rec_loc,
            "setup": setup,
            "source_annotation_json": str(annotation_json),
            "pano_tiles_dir": str(pano_tiles_dir),
            "output_dir": str(output_dir),
            "dedup_voxel": dedup_voxel,
            "num_instances": len(summary_items),
            "items": summary_items,
        }

        manifest_path = output_dir / "instances.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(
            f"[Leica] Lifted {len(summary_items)} instance annotation(s) for setup {setup}. "
            f"Manifest saved to {manifest_path}"
        )
        return manifest_path

    def _build_pano_tile_projection_definitions(
        self,
        setup: str,
        pano_pose_meta: Dict[str, Any],
        xyz_dir: Path,
    ) -> List[Dict[str, Any]]:
        """
        Build the exact pano-tile projection schedule used by make_360_views_from_pano().
        """

        hfov = 90.0
        vfov = 120.0
        tile_width = 1024
        tile_height = 1364
        step = hfov * 0.1

        euler0 = R.from_quat(
            pano_pose_meta["orientation"],
            scalar_first=True,
        ).as_euler("xyz", degrees=True)
        rot_initial_around_world_z = euler0[2]

        yaws = np.arange(0.0, 360.0, step)
        tile_defs: List[Dict[str, Any]] = []

        for idx, yaw_deg in enumerate(yaws):
            stem = f"{idx:03d}"
            xyz_path = self._resolve_xyz_tile_path(xyz_dir=xyz_dir, stem=stem)
            if xyz_path is None or not xyz_path.exists():
                print(
                    f"[Leica] Warning: missing XYZ tile for setup {setup}, tile {stem}; "
                    "skipping this tile during annotation lifting."
                )
                continue

            yaw_adjusted = yaw_deg + rot_initial_around_world_z
            rot_mat = R.from_euler("y", yaw_adjusted, degrees=True).as_matrix()

            tile_defs.append(
                {
                    "stem": stem,
                    "yaw_deg": yaw_deg,
                    "rot_mat": rot_mat,
                    "xyz_path": xyz_path,
                    "hfov_deg": hfov,
                    "vfov_deg": vfov,
                    "tile_width": tile_width,
                    "tile_height": tile_height,
                }
            )

        return tile_defs

    def _resolve_instance_annotation_json(
        self,
        setup: str,
        annotation_json_path: str | Path | None = None,
    ) -> Path | None:
        """
        Resolve the panorama instance annotation JSON for a Leica setup.
        """

        if annotation_json_path is not None:
            annotation_json = Path(annotation_json_path)
            if not annotation_json.exists():
                raise FileNotFoundError(f"Annotation JSON not found: {annotation_json}")
            return annotation_json

        candidates: List[Path] = []
        setup_pattern = f"*Setup {setup}.json"
        candidate_dirs = [
            self.extraction_path / "instance_annotations",
            self.extraction_path / setup / self.label_images / "instance_annotations",
            self.extraction_path / setup / "instance_annotations",
        ]
        for candidate_dir in candidate_dirs:
            if candidate_dir.exists():
                candidates.extend(sorted(candidate_dir.glob(setup_pattern)))

        if candidates:
            return candidates[0]

        return None

    def _resolve_instance_object_mask(
        self,
        item: Dict[str, Any],
        annotation_json_path: Path,
        setup: str,
        image_width: int,
        image_height: int,
    ) -> Tuple[np.ndarray, str | None]:
        """
        Resolve a binary panorama-sized mask for an annotated object instance.
        """

        object_index = int(item["index"])
        mask_path = self._resolve_instance_mask_path(
            mask_path=item.get("mask_path"),
            annotation_json_path=annotation_json_path,
            setup=setup,
            filename=f"{object_index}.png",
        )
        if mask_path is not None:
            mask = self._load_binary_mask(mask_path, image_width=image_width, image_height=image_height)
            mask = self._fill_binary_mask(mask)
            polygon = item.get("polygon")
            if polygon:
                polygon_mask = self._polygon_to_mask(
                    polygon=polygon,
                    image_width=image_width,
                    image_height=image_height,
                )
                mask = np.maximum(mask, polygon_mask)
            return mask, str(mask_path)

        polygon = item.get("polygon")
        if polygon:
            mask = self._polygon_to_mask(
                polygon=polygon,
                image_width=image_width,
                image_height=image_height,
            )
            return mask, None

        raise FileNotFoundError(
            f"Could not resolve an object mask or polygon for instance {object_index} "
            f"in {annotation_json_path}"
        )

    def _resolve_instance_handle_mask(
        self,
        item: Dict[str, Any],
        annotation_json_path: Path,
        setup: str,
        image_width: int,
        image_height: int,
        radius_px: int,
    ) -> Tuple[np.ndarray | None, str | None]:
        """
        Resolve an optional panorama-sized handle mask for an annotated instance.
        """

        handle = item.get("handle") or {}
        object_index = int(item["index"])

        handle_mask_path = self._resolve_instance_mask_path(
            mask_path=handle.get("mask_path"),
            annotation_json_path=annotation_json_path,
            setup=setup,
            filename=f"{object_index}_handle.png",
        )
        if handle_mask_path is not None:
            mask = self._load_binary_mask(
                handle_mask_path,
                image_width=image_width,
                image_height=image_height,
            )
            return mask, str(handle_mask_path)

        handle_polygon = handle.get("polygon")
        if handle_polygon:
            mask = self._polygon_to_mask(
                polygon=handle_polygon,
                image_width=image_width,
                image_height=image_height,
            )
            return mask, None

        handle_point = handle.get("point")
        if handle_point is not None:
            mask = self._point_to_mask(
                point=handle_point,
                image_width=image_width,
                image_height=image_height,
                radius_px=radius_px,
            )
            return mask, None

        return None, None

    def _resolve_instance_mask_path(
        self,
        mask_path: str | None,
        annotation_json_path: Path,
        setup: str,
        filename: str,
    ) -> Path | None:
        """
        Resolve an annotation mask path robustly, accounting for stale absolute paths.
        """

        candidate_paths: List[Path] = []
        if mask_path:
            candidate_paths.append(Path(mask_path))

        candidate_paths.extend(
            [
                annotation_json_path.parent / filename,
                self.extraction_path / "instance_annotations" / filename,
                self.extraction_path / setup / self.label_images / "instance_annotations" / filename,
                self.extraction_path / setup / "instance_annotations" / filename,
            ]
        )

        for candidate_path in candidate_paths:
            if candidate_path.exists():
                return candidate_path

        return None

    def _load_binary_mask(
        self,
        mask_path: str | Path,
        image_width: int,
        image_height: int,
    ) -> np.ndarray:
        """
        Load a binary annotation mask and ensure it matches the panorama size.
        """

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read annotation mask: {mask_path}")

        if mask.shape != (image_height, image_width):
            print(
                f"[Leica] Warning: resizing mask {mask_path} from {mask.shape[::-1]} "
                f"to {(image_width, image_height)} to match the panorama."
            )
            mask = cv2.resize(
                mask,
                (image_width, image_height),
                interpolation=cv2.INTER_NEAREST,
            )

        return np.where(mask > 0, 255, 0).astype(np.uint8)

    def _fill_binary_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Turn a binary edge/outline mask into a filled region mask.

        This is safe for already-filled masks and helps when manual annotations
        contain only instance boundaries.
        """

        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        if not np.any(binary):
            return binary

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return binary

        filled = np.zeros_like(binary)
        cv2.drawContours(filled, contours, contourIdx=-1, color=255, thickness=cv2.FILLED)
        return filled

    def _polygon_to_mask(
        self,
        polygon: List[List[float]],
        image_width: int,
        image_height: int,
    ) -> np.ndarray:
        """
        Rasterize a polygon into a binary panorama-sized mask.
        """

        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        if not polygon:
            return mask

        pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] = np.clip(pts[:, 0], 0, image_width - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, image_height - 1)
        pts = np.round(pts).astype(np.int32)

        if pts.shape[0] >= 3:
            cv2.fillPoly(mask, [pts], color=255)

        return mask

    def _point_to_mask(
        self,
        point: List[float],
        image_width: int,
        image_height: int,
        radius_px: int = 5,
    ) -> np.ndarray:
        """
        Turn a single panorama point into a small binary disk mask.
        """

        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        if point is None or len(point) != 2:
            return mask

        x = int(round(np.clip(point[0], 0, image_width - 1)))
        y = int(round(np.clip(point[1], 0, image_height - 1)))
        cv2.circle(mask, center=(x, y), radius=max(1, int(radius_px)), color=255, thickness=-1)
        return mask

    def _resolve_xyz_tile_path(self, xyz_dir: Path, stem: str) -> Path | None:
        """
        Resolve the XYZ mat file corresponding to a pano tile stem.
        """

        candidates = [
            xyz_dir / f"{stem}.jpg.mat",
            xyz_dir / f"{stem}.mat",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _load_xyz_tile(self, xyz_path: str | Path) -> np.ndarray:
        """
        Load only the XYZcut array from one Leica pano-tile mat file.
        """

        mat_data = sio.loadmat(str(xyz_path), variable_names=["XYZcut"])
        xyz = mat_data["XYZcut"].astype(np.float32)
        return xyz

    def _deduplicate_world_points(self, points: np.ndarray, voxel: float = 0.005) -> np.ndarray:
        """
        De-duplicate world points from overlapping pano tiles with voxel quantization.
        """

        if points.size == 0 or voxel is None or voxel <= 0:
            return points

        quantized = np.round(points / voxel).astype(np.int64)
        _, unique_indices = np.unique(quantized, axis=0, return_index=True)
        unique_indices = np.sort(unique_indices)
        return points[unique_indices]

    def _write_points_as_ply(self, points: np.ndarray, ply_path: str | Path) -> None:
        """
        Save a set of XYZ world points as a PLY point cloud.
        """

        ply_path = Path(ply_path)
        ply_path.parent.mkdir(parents=True, exist_ok=True)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        o3d.io.write_point_cloud(str(ply_path), pcd, write_ascii=False)

    def visualize_instance_annotations_3d(
        self,
        setup: str | None = None,
        manifest_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        instance_indices: Sequence[int] | None = None,
        background_color: Tuple[float, float, float] = (0.55, 0.55, 0.55),
        show: bool = True,
    ) -> Dict[str, Any]:
        """
        Visualize lifted Leica instance annotations as colored point clouds over the
        downsampled Leica map point cloud.

        Args:
            setup: Leica setup identifier, defaults to the first setup.
            manifest_path: Optional explicit path to the instance manifest JSON.
            output_dir: Optional directory containing `instances.json` and `points/*.ply`.
            instance_indices: Optional subset of instance indices to visualize.
            background_color: Uniform color used for the downsampled Leica map.
            show: If True, launch the Open3D visualizer. If False, only prepare and
                return the visualization data, which is useful for headless validation.

        Returns:
            A dictionary containing the manifest path, selected items, colors, and
            prepared Open3D geometries.
        """

        if setup is None:
            setup = self.setups[0]

        manifest_path = self._resolve_instance_annotation_manifest_path(
            setup=setup,
            manifest_path=manifest_path,
            output_dir=output_dir,
        )

        manifest = json.loads(Path(manifest_path).read_text())
        items = manifest.get("items", [])
        if not items:
            raise ValueError(f"No lifted instance items found in manifest {manifest_path}")

        if instance_indices is not None:
            selected_indices = {int(idx) for idx in instance_indices}
            items = [item for item in items if int(item["index"]) in selected_indices]

        items = [item for item in items if item.get("output_ply")]
        if not items:
            raise ValueError(f"No visualizable instance point clouds found in {manifest_path}")

        base_pcd = self.get_downsampled_points(setup=setup)
        base_pcd_vis = o3d.geometry.PointCloud(base_pcd)
        base_pcd_vis.paint_uniform_color(list(background_color))

        geometries: List[o3d.geometry.Geometry] = [base_pcd_vis]
        legend: List[Dict[str, Any]] = []

        total_items = len(items)
        for vis_idx, item in enumerate(items):
            ply_path = Path(item["output_ply"])
            if not ply_path.exists():
                print(f"[Leica] Warning: instance point cloud missing at {ply_path}; skipping.")
                continue

            pcd = o3d.io.read_point_cloud(str(ply_path))
            color = self._distinct_color(vis_idx, total_items)
            pcd.paint_uniform_color(color.tolist())
            geometries.append(pcd)

            legend_entry = {
                "index": int(item["index"]),
                "class": item.get("class"),
                "description": item.get("description"),
                "color_rgb": color.tolist(),
                "num_points_3d": int(item.get("num_points_3d", 0)),
                "ply_path": str(ply_path),
            }
            legend.append(legend_entry)
            print(
                f"[Leica] Visualize instance {legend_entry['index']:03d} "
                f"({legend_entry['class']}): {legend_entry['num_points_3d']} pts "
                f"color={legend_entry['color_rgb']}"
            )

        result = {
            "manifest_path": str(manifest_path),
            "setup": setup,
            "legend": legend,
            "geometries": geometries,
        }

        if show:
            o3d.visualization.draw_geometries(
                geometries,
                window_name=f"Leica instance annotations 3D - {self.rec_loc}/{setup}",
            )

        return result

    def _resolve_instance_annotation_manifest_path(
        self,
        setup: str,
        manifest_path: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> Path:
        """
        Resolve the manifest path for lifted Leica instance annotations.
        """

        if manifest_path is not None:
            manifest = Path(manifest_path)
            if not manifest.exists():
                raise FileNotFoundError(f"Instance annotation manifest not found: {manifest}")
            return manifest

        candidate_dirs: List[Path] = []
        if output_dir is not None:
            candidate_dirs.append(Path(output_dir))
        candidate_dirs.append(self.extraction_path / setup / "instance_annotations_3d")

        for candidate_dir in candidate_dirs:
            candidate_manifest = candidate_dir / "instances.json"
            if candidate_manifest.exists():
                return candidate_manifest

        raise FileNotFoundError(
            f"Could not find an instance annotation manifest for setup {setup}. "
            "Run lift_pano_instance_annotations_to_3d() first or pass --manifest-path."
        )

    def _distinct_color(self, index: int, total: int) -> np.ndarray:
        """
        Generate a stable vivid RGB color for an instance point cloud.
        """

        hue = 0.0 if total <= 1 else index / float(total)
        rgb = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
        return np.asarray(rgb, dtype=np.float64)

    def make_articulation_groundtruth(self) -> None:
        """
        Creates a ground truth file for articulation from the Leica data.
        Currently a placeholder function.
        """
        print("[Leica] Creating articulation ground truth... (not implemented)")

        # get all setups
        if self.setups == []:
            self.extract_all_setups()

        pcds = []
        for setup in self.setups:
            print(f"[Leica] Processing setup {setup}...")
            # load downsampled point cloud
            pcd = self.get_full_points(setup=setup)
            pcds.append(pcd)

        # debug 
        pcd_base = pcds[0]
        pcd_query = pcds[2]
        diff_pcd = get_pcd_diff(pcd_query, pcd_base, threshold=0.02)
        visualize_pcds([diff_pcd], [np.array([0,0,1])])


        a = 2


    def _make_downsampled(self, setup: str, voxel: float = None) -> None:

        if voxel is not None:
            voxel = voxel
        else:
            voxel = self.voxel

        down_path = self.extraction_path / setup / self.label_downsampled / f"points_voxel_{voxel:.3f}.ply"

        if down_path.exists():
            print(f"[Leica] Downsampled cloud already exist at {down_path}")
            return
        
        # ensure dir exists
        down_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[Leica] Loading full cloud for setupn {setup} ...")
        full = self.get_full_points(setup=setup)  # ensures .ply exists

        with tqdm(total=4, desc="[Leica] Downsample", unit="step") as pbar:
            print(f"[Leica] Down-sampling at voxel={voxel:.3f}")
            down = full.voxel_down_sample(voxel_size=voxel)
            pbar.update(1)

            print("[Leica] Estimating normals...")
            down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=voxel * 2.0, max_nn=30
                )
            )
            pbar.update(1)

            print("[Leica] Saving downsampled cloud ...") # stored in PLY comment
            o3d.io.write_point_cloud(str(down_path), down, write_ascii=False)
            pbar.update(1)

        print(
            f"[Leica] Cached downsampled cloud → {down_path.name}"
        )

    def _extracted(self) -> bool:
        """
        Check if the data for the given label has been extracted.
        """
        label_path = self.extraction_path 
        return label_path.exists() and any(label_path.iterdir())

    def _parse_pano_pose(self, txt_path: str | Path):
        """
        Parses a Leica panorama TXT file containing lines like:
        position = [x, y, z];
        orientation = [qx, qy, qz, qw];
        Returns a dict with keys "LDR" and "HDR", each a dict with "position" and "orientation" lists.
        """
        data = {"LDR": {}, "HDR": {}}
        section = None

        # regex to capture numbers inside the brackets
        array_re = re.compile(r"\[([^\]]+)\]")
        
        if isinstance(txt_path, str):
            txt_path = Path(txt_path)

        for line in txt_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("Ldr Image"):
                section = "LDR"
            elif line.startswith("Hdr Image"):
                section = "HDR"
            elif section and line.startswith("position"):
                m = array_re.search(line)
                if m:
                    nums = [float(x) for x in m.group(1).split(",")]
                    data[section]["position"] = nums
            elif section and line.startswith("orientation"):
                m = array_re.search(line)
                if m:
                    nums = [float(x) for x in m.group(1).split(",")]
                    data[section]["orientation"] = nums
        return data
    

    def _equirect_to_pinhole(self,
                            equi_img: np.ndarray,
                            rot_mat: np.ndarray,
                            hfov_deg: float,
                            vfov_deg: float,
                            out_w: int,
                            out_h: int) -> np.ndarray:
        """
        Turn an equirectangular image (H_e x W_e) into a pinhole view
        at (yaw, pitch) with horizontal FOV=hfov_deg and vertical FOV=vfov_deg.
        Returns an out_h x out_w x C BGR image.
        """
        H_e, W_e = equi_img.shape[:2]

        # compute the tangent extents for each axis
        tan_h = math.tan(math.radians(hfov_deg / 2))
        tan_v = math.tan(math.radians(vfov_deg / 2))

        # screen coords in camera space
        xs = np.linspace(-tan_h, +tan_h, out_w)
        ys = np.linspace(-tan_v, +tan_v, out_h)
        xv, yv = np.meshgrid(xs, -ys)       # note the -ys to flip vertically
        zv = np.ones_like(xv)

        dirs = (rot_mat @ np.stack([xv, yv, zv], -1).reshape(-1,3).T).T

        # convert to spherical coords
        lon = np.arctan2(dirs[:,0], dirs[:,2])   # range [-π, π]
        lat = np.arcsin(dirs[:,1] / np.linalg.norm(dirs, axis=1))  # [-π/2, π/2]

        # map to equirectangular pixel coords
        uf = (lon / (2 * math.pi) + 0.5) * W_e
        vf = (0.5 - lat / math.pi) * H_e

        map_x = uf.reshape(out_h, out_w).astype(np.float32)
        map_y = vf.reshape(out_h, out_w).astype(np.float32)

        # sample with wrap‑around horizontally
        return cv2.remap(
            equi_img, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP
        )

    def _depth_to_world_xyz(self, depth: np.ndarray, K: np.ndarray, w_T_wc: np.ndarray) -> np.ndarray:
        
        H, W = depth.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))  # pixel coordinates

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Reconstruct 3D camera coordinates
        x_cam = ((u - cx) * depth) / fx
        y_cam = ((v - cy) * depth) / fy
        z_cam = depth

        pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (H, W, 3)
        pts_cam_flat = pts_cam.reshape(-1, 3)  # (H*W, 3)

        # Convert camera-to-world rotation
        R_wc = w_T_wc[:3, :3]  # 3×3 rotation matrix
        t_wc = w_T_wc[:3, 3]   # 3×1
        pts_w = (R_wc @ pts_cam_flat.T + t_wc[:, None]).T  # 3×N

        return pts_w.reshape(H, W, 3).astype(np.float32)


    def _render_depth(self,
                    mesh: o3d.t.geometry.TriangleMesh,
                    K: np.ndarray,
                    w_T_wc: np.ndarray,
                    clip_max_dist: float = 10.0,
                    ):
        """
        Renders a depth map for the current pin‑hole view.

        Returns
        -------
        depth : (H, W) float32            — always
        xyz   : (H, W, 3) float32 or None — only if `return_xyz`
        """

        K = np.asarray(K, dtype=np.float64)
        width  = int(round(K[0, 2] * 2))
        height = int(round(K[1, 2] * 2))

        # transform from Leica world to Open3D world
        # to o3d coord system
        # Convert from Z-up to Y-up coordinate system
        # R_o3d_leica = np.array([
        #     [1, 0, 0],
        #     [0, 0, -1],
        #     [0, 1, 0]
        # ])

        # T_o3d_leica = np.eye(4, dtype=float)
        # T_o3d_leica[:3, :3] = R_o3d_leica

        # mesh_o3d = mesh.clone().transform(T_o3d_leica)

        # --- set‑up off‑screen renderer --------------------------------
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        renderer.scene.set_background([0, 0, 0, 0])

        mat = o3d.visualization.rendering.MaterialRecord()
        # mat.shader = "defaultLit"
        mat.shader = "defaultUnlit"
        renderer.scene.add_geometry("mesh", mesh.to_legacy(), mat)

        renderer.setup_camera(K, w_T_wc, width, height)

        # --- render -----------------------------------------------------
        depth_img = renderer.render_to_depth_image(z_in_view_space=True)
        depth = np.asarray(depth_img, dtype=np.float32)  # (H, W)

        # clip depth values
        depth[depth > clip_max_dist] = 0.0  # mark as invalid
        depth[depth < 0.01] = 0.0           # mark as

        return depth
    
    def render_depth_batched(
        self,
        mesh: o3d.t.geometry.TriangleMesh,
        K: np.ndarray,
        w_T_cw_list: List[np.ndarray],
        clip_max_dist: float = 10.0,
        clip_min_dist: float = 0.25) -> List[np.ndarray]:
        """
        Efficiently render depth maps for many camera poses (shared intrinsics & resolution).

        Args:
            mesh: Open3D Tensor mesh (Leica world coordinates).
            K: 3x3 camera intrinsics (shared for all frames).
            w_T_wc_list: list of 4x4 OpenCV cam2world extrinsics.
            clip_max_dist: maximum depth distance (meters).

        Returns:
            List of (H, W) float32 depth maps.
        """
        # --- shared resolution ---
        K = np.asarray(K, dtype=np.float64)
        W = int(round(K[0, 2] * 2))
        H = int(round(K[1, 2] * 2))

        # # --- Leica -> Open3D coordinate transform ---
        # R_o3d_leica = np.array([[1, 0, 0],
        #                         [0, 0, -1],
        #                         [0, 1, 0]], dtype=float)
        # T_o3d_leica = np.eye(4, dtype=float)
        # T_o3d_leica[:3, :3] = R_o3d_leica
        # mesh_o3d = mesh.clone().transform(T_o3d_leica)

        # --- one renderer, one scene ---
        renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
        renderer.scene.set_background([0, 0, 0, 0])
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"  # minimal shading, faster
        renderer.scene.add_geometry("mesh", mesh.to_legacy(), mat)

        # --- render each frame ---
        depths = []
        for w_T_cw in w_T_cw_list:
            w_T_cw = np.asarray(w_T_cw, dtype=np.float64)
            renderer.setup_camera(K, w_T_cw, W, H)
            depth_img = renderer.render_to_depth_image(z_in_view_space=True)
            depth = np.asarray(depth_img, dtype=np.float32)
            depth[(depth > clip_max_dist) | (depth < clip_min_dist)] = 0.0
            depths.append(depth)
        return depths


    def _pcd_to_mesh(self,
                pcd: o3d.geometry.PointCloud,
                depth: int = 8,         # Poisson reconstruction depth
                scale: float = 1.1) -> o3d.t.geometry.TriangleMesh:
        """
        Turns a point‑cloud into a watertight triangle mesh (Poisson) 
        Returns an *Open3D‑Tensor* mesh (ready for cuda or cpu scene).
        """
        pcd = pcd.voxel_down_sample(0.01)     # optional light decimation
        pcd.estimate_normals()

        mesh_legacy, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, scale=scale)
        mesh_legacy.remove_duplicated_vertices()
        mesh_legacy.remove_degenerate_triangles()
        mesh_legacy.remove_non_manifold_edges()

        # convert to Tensor mesh on the default device (CPU or CUDA)
        return o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)

    def _write_exr(self, path: str, img: np.ndarray):
        """
        Write a single‐channel or 3‐channel float32 NumPy image to an EXR.
        The img must be H×W (single‐channel) or H×W×3 (RGB), dtype=float32.
        """
        assert img.dtype == np.float32, "convert to float32 first"
        H, W = img.shape[:2]

        # 1) Create an empty header with the correct size
        header = OpenEXR.Header(W, H)

        # 2) Define your channels in that header
        #    Here we assume RGB; if single‐channel, just define 'R'
        if img.ndim == 2:
            header['channels'] = {'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
        else:
            header['channels'] = {
                'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
            }

        # 3) Open the file for writing
        exr = OpenEXR.OutputFile(path, header)

        # 4) Prepare raw byte strings per channel
        if img.ndim == 2:
            # single‐channel
            r_chan = img
            channel_data = {'R': r_chan.tobytes()}
        else:
            # RGB, split into three planes
            r_chan, g_chan, b_chan = cv2.split(img)
            channel_data = {
                'R': r_chan.tobytes(),
                'G': g_chan.tobytes(),
                'B': b_chan.tobytes()
            }

        # 5) Write out the pixels
        exr.writePixels(channel_data)
        exr.close()

    def _read_exr(self, path: str) -> np.ndarray:
        """
        Read a single‐channel EXR file into a NumPy array.
        Assumes the EXR has a channel named "R" with float32/float16/uint32 data.
        Returns an H×W NumPy array with the pixel values.
        """
        exr = OpenEXR.InputFile(path)
        header = exr.header()
        dw = header['dataWindow']
        W = dw.max.x - dw.min.x + 1
        H = dw.max.y - dw.min.y + 1

        # determine dtype from the channel’s pixel type
        chan = header['channels']['R'].type
        if   chan == Imath.PixelType(Imath.PixelType.FLOAT): dtype = np.float32
        elif chan == Imath.PixelType(Imath.PixelType.HALF):  dtype = np.float16
        elif chan == Imath.PixelType(Imath.PixelType.UINT):  dtype = np.uint32
        else:
            raise ValueError(f"Unsupported EXR channel type: {chan}")

        # read raw bytes and convert
        raw = exr.channel('R', chan)
        arr = np.frombuffer(raw, dtype=dtype)

        # reshape into H×W
        arr = arr.reshape(H, W)

        return arr
    
    def _save_mat(self, mat_path: str | Path, rgb_image: np.ndarray, xyz_array) -> None:

        sio.savemat(str(mat_path),
                    {"RGBcut": rgb_image, "XYZcut": xyz_array},
                    do_compression=False)
        
    def _load_mat_as_ply(self, mat_path: str | Path) -> o3d.geometry.PointCloud:
        """
        Load a .mat file containing RGB and XYZ data and convert it to an Open3D PointCloud.
        The .mat file should contain 'RGBcut' and 'XYZcut' keys.
        """
        mat_data = sio.loadmat(str(mat_path))
        rgb = mat_data['RGBcut'].astype(np.float32) / 255.0
        xyz = mat_data['XYZcut'].astype(np.float32)

        # filter out zero coords
        valid_mask = np.all(xyz != 0, axis=2)
        xyz = xyz[valid_mask]
        rgb = rgb[valid_mask]

        if rgb.ndim == 2:  # single channel
            rgb = np.repeat(rgb[:, :, np.newaxis], 3, axis=-1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.reshape(-1, 3))
        pcd.colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3)) 
        return pcd
        
    def _write_depth_vis(self, path: str, depth: np.ndarray) -> None:
        # Mask out invalid (zero or NaN) values
        valid_mask = (depth > 0) & np.isfinite(depth)
        if not np.any(valid_mask):
            print(f"Warning: No valid depth values to visualize in {path}")
            return

        # Compute per-image min/max for normalization
        min_depth = np.min(depth[valid_mask])
        max_depth = np.max(depth[valid_mask])
        
        # Normalize to [0, 255] and convert to uint8
        depth_vis = np.clip(depth, min_depth, max_depth)
        depth_vis = ((depth_vis - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)

        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        # Save
        cv2.imwrite(str(path), depth_colored)

    def show_crop_frustums(
            self,
            setup: str | None = None,
            pose_dir: Path | None = None,
            mesh: o3d.t.geometry.TriangleMesh | None = None,
            frustum_depth: float = 0.5,       # metres from pin‑hole to image‑plane
            max_tiles: int | None = None,     # None = show all
            color: tuple[float, float, float] = (1, 0, 0),
        ) -> None:
        """
        Visualise the mesh + camera frusta for every 45°×45° crop tile.

        *Requirements*: the JSON files produced by `make_360_views_from_pano()`
        must live in   .../pano_tiles/poses/###.json.

        Args
        ----
        setup          : Leica setup name (defaults to first in `self.setups`)
        pose_dir       : override path to the pose JSONs
        mesh           : pass your own mesh if you already have it in memory
        frustum_depth  : distance from camera centre to image plane (m)
        max_tiles      : display only the N first tiles (speed / clarity)
        color          : RGB line colour for all frusta  (0‑1 floats)
        """
        if setup is None:
            setup = self.setups[0]

        if mesh is None:
            mesh = self.get_mesh(setup)                        # t.geometry
        mesh_legacy = mesh.to_legacy()

        pcd = self.get_downsampled_points(setup=setup)  # t.geometry

        R_o3d_leica = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])

        T_o3d_leica = np.eye(4, dtype=float)
        T_o3d_leica[:3, :3] = R_o3d_leica
        # mesh_legacy.transform(T_o3d_leica)  # convert to Open3D world coords

        # ---------------------------------------------------------------------
        # 1) Gather pose JSON files
        if pose_dir is None:
            pose_dir = self.extraction_path / setup / "pano_tiles" / "poses"
        pose_files = sorted(pose_dir.glob("*.json"))
        if max_tiles is not None:
            pose_files = pose_files[:max_tiles]

        # get mat files for xyz tiles
        xyz_files = sorted((pose_dir.parent / "xyz").glob("*.mat"))
        if max_tiles is not None:
            xyz_files = xyz_files[:max_tiles]

        # ---------------------------------------------------------------------
        # 2) Build one big LineSet with all frusta
        all_pts: list[np.ndarray]  = []
        all_lines: list[tuple[int, int]] = []
        all_cols: list[tuple[float, float, float]] = []
        all_mats: list[o3d.geometry.PointCloud] = []

        for idx, jf in enumerate(pose_files):
            with open(jf) as f:
                meta = json.load(f)

            # open the corresponding XYZ mat file
            if idx < 5:
                xyz_file = xyz_files[idx]
                xyz_pcd = self._load_mat_as_ply(xyz_file)
                # add the XYZ points from the mat file
                all_mats.append(xyz_pcd)

            K   = np.asarray(meta["K"], dtype=np.float64)
            w_T = np.asarray(meta["w_T_wc"], dtype=np.float64)
            W   = int(meta["w"])
            H   = int(meta["h"])

            # ----- camera‑space corner pixels at z = frustum_depth ----------
            corners_px = np.array([[0,   0,   1],
                                [W,   0,   1],
                                [W,   H,   1],
                                [0,   H,   1]], dtype=np.float64).T  # 3×4
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

            # un‑project (pinhole) → camera coords
            z     = np.full((1, 4), frustum_depth)
            x_cam = (corners_px[0] - cx) / fx * z
            y_cam = (corners_px[1] - cy) / fy * z
            cam_pts = np.vstack([x_cam, y_cam, z])              # 3×4

            # include optical centre at origin
            cam_pts = np.hstack([np.zeros((3, 1)), cam_pts])    # 3×5

            # ----- transform to world ---------------------------------------
            R = w_T[:3, :3]  # 3×3 rotation matrix
            t = w_T[:3, 3:4]
            world_pts = (R @ cam_pts) + t                       # 3×5

            base = len(all_pts)
            all_pts.extend(world_pts.T)                         # add 5 points

            # pyramid edges (indices relative to this frustum’s base)
            edges = [(0, 1), (0, 2), (0, 3), (0, 4),
                    (1, 2), (2, 3), (3, 4), (4, 1)]
            all_lines.extend([(base + a, base + b) for a, b in edges])
            all_cols.extend([color] * len(edges))


        # ---------------------------------------------------------------------
        # 3) Create a single LineSet
        ls = o3d.geometry.LineSet()
        ls.points  = o3d.utility.Vector3dVector(np.array(all_pts))
        ls.lines   = o3d.utility.Vector2iVector(np.array(all_lines, dtype=int))
        ls.colors  = o3d.utility.Vector3dVector(np.array(all_cols))

        # ---------------------------------------------------------------------
        # 4) Show everything
        o3d.visualization.draw_geometries([mesh_legacy, ls, pcd]+all_mats,
                                        zoom=0.6,
                                        window_name=f"Frusta for '{setup}'")

if __name__ == "__main__":
    from pathlib import Path


    base_path = Path(f"/data/ikea_recordings")
    rec_location = "bedroom_6"

    leica_data = LeicaData(base_path, rec_location, initial_setup="001")
    leica_data.extract_all_setups()
    # leica_data.make_360_views_from_pano()
    # leica_data.show_crop_frustums(max_tiles=20)
    leica_data.make_articulation_groundtruth()
