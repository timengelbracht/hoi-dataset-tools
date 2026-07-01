from pathlib import Path
import open3d as o3d
import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional, DefaultDict, Dict
import copy
import time
import os
import pickle
from scipy.spatial.transform import Rotation as R
import pandas as pd
import plotly.graph_objects as go
import shutil
from PIL import Image
import json
from scipy.spatial.transform import Rotation
from .data_loader_aria import AriaData
from .data_loader_leica import LeicaData
from .data_loader_iphone import IPhoneData
from .data_loader_gripper import GripperData
from .data_loader_umi import UmiData
from .data_indexer import RecordingIndex
from .utils_trajectory_optimization import TrajectoryOptimization
from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_retrieval,
    triangulation
)
from hloc.localize_sfm import main as localize_sfm_main
from hloc import localize_inloc
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d
import pycolmap
import torch
import cv2
from contextlib import contextmanager
from typing import Dict, List
from scipy.spatial.transform import Rotation as R, Slerp
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Mokey patches for pycolmap compatibility and missing features
if not hasattr(pycolmap, "absolute_pose_estimation"):
    def absolute_pose_estimation(points2D, points3D, camera,
                                 estimation_options=None,
                                 refinement_options=None):
        return pycolmap.estimate_and_refine_absolute_pose(
            points2D, points3D, camera,
            estimation_options or {},
            refinement_options or {},
        )
    
    pycolmap.absolute_pose_estimation = absolute_pose_estimation

if not hasattr(pycolmap.Rigid3d, "essential_matrix"):

    def _skew(t):
        tx, ty, tz = t
        return np.array([[ 0, -tz,  ty],
                         [ tz,   0, -tx],
                         [-ty,  tx,   0]])

    def essential_matrix(self: pycolmap.Rigid3d) -> np.ndarray:
        """
        Return E = [t]_x R  (world→left, right pose = self)
        Equivalent to the old C++ helper that existed in pycolmap < 0.7.
        """
        R = self.rotation.matrix()      
        t = self.translation            
        return _skew(t) @ R            

    pycolmap.Rigid3d.essential_matrix = essential_matrix

# Monkey patch for hloc localize_inloc to accept additional focal length parameter
def set_intrinsics(*, fx: float, fy: float):
    global FX, FY
    FX, FY = float(fx), float(fy)

def set_camera_model(config: Dict):
    global CFG
    CFG = config

def pose_from_cluster_patched(
        dataset_dir, q, retrieved,
        feature_file, match_file,
        skip=None
    ):
    """Drop‑in replacement for hloc.localize_inloc.pose_from_cluster
    that takes focal_length as an optional argument."""

    all_mkpq, all_mkpr, all_mkp3d, all_indices = [], [], [], []
    kpq = feature_file[q]["keypoints"].__array__()
    num_matches = 0

    for i, r in enumerate(retrieved):
        kpr = feature_file[r]["keypoints"].__array__()
        pair = localize_inloc.names_to_pair(q, r)
        m = match_file[pair]["matches0"].__array__()
        v = m > -1

        if skip and (np.count_nonzero(v) < skip):
            continue

        mkpq, mkpr = kpq[v], kpr[m[v]]
        num_matches += len(mkpq)

        scan_r = localize_inloc.loadmat(Path(dataset_dir, r + ".mat"))["XYZcut"]
        mkp3d, valid = localize_inloc.interpolate_scan(scan_r, mkpr)
        Tr = localize_inloc.get_scan_pose(dataset_dir, r)
        mkp3d = (Tr[:3, :3] @ mkp3d.T + Tr[:3, -1:]).T

        all_mkpq.append(mkpq[valid])
        all_mkpr.append(mkpr[valid])
        all_mkp3d.append(mkp3d[valid])
        all_indices.append(np.full(np.count_nonzero(valid), i))

    # Guard: nothing survived matching/skip -> return safe failure with identity pose
    if len(all_mkpq) == 0:
        ret = {
            "success": False,
            "error": "no_correspondences",
            "message": "No valid 2D-3D correspondences after matching/interpolation.",
            "cfg": CFG,
            "cam_from_world": pycolmap.Rigid3d()
        }
        empty_2d = np.empty((0, 2), dtype=np.float32)
        empty_3d = np.empty((0, 3), dtype=np.float32)
        empty_i  = np.empty((0,), dtype=int)
        return ret, empty_2d, empty_2d, empty_3d, empty_i, num_matches

    all_mkpq  = np.concatenate(all_mkpq,  0)
    all_mkpr  = np.concatenate(all_mkpr,  0)
    all_mkp3d = np.concatenate(all_mkp3d, 0)
    all_indices = np.concatenate(all_indices, 0)

    cfg = CFG

    opts = pycolmap.AbsolutePoseEstimationOptions()
    opts.ransac.max_error = 48
    ret = pycolmap.estimate_and_refine_absolute_pose(
        all_mkpq, all_mkp3d, cfg, opts
    )

    # Guard: nothing survived matching/skip -> return safe failure with identity pose
    if ret == None:
        ret = {
            "success": False,
            "error": "no_correspondences",
            "message": "No valid 2D-3D correspondences after matching/interpolation.",
            "cfg": CFG,
            "cam_from_world": pycolmap.Rigid3d()
        }
        empty_2d = np.empty((0, 2), dtype=np.float32)
        empty_3d = np.empty((0, 3), dtype=np.float32)
        empty_i  = np.empty((0,), dtype=int)
        return ret, empty_2d, empty_2d, empty_3d, empty_i, num_matches
    
    ret["cfg"] = cfg
    return ret, all_mkpq, all_mkpr, all_mkp3d, all_indices, num_matches
# Patch the original function
localize_inloc.pose_from_cluster = pose_from_cluster_patched

def interpolate_scan_patched(scan, kp):
    h, w, c = scan.shape
    kp = kp / np.array([[w - 1, h - 1]]) * 2 - 1
    kp = kp.astype(np.float32)
    assert np.all(kp > -1) and np.all(kp < 1)
    scan = torch.from_numpy(scan).permute(2, 0, 1)[None]
    kp = torch.from_numpy(kp)[None, None]
    grid_sample = torch.nn.functional.grid_sample

    # To maximize the number of points that have depth:
    # do bilinear interpolation first and then nearest for the remaining points
    interp_lin = grid_sample(scan, kp, align_corners=True, mode="bilinear")[0, :, 0]
    interp_nn = torch.nn.functional.grid_sample(
        scan, kp, align_corners=True, mode="nearest"
    )[0, :, 0]
    interp = torch.where(torch.isnan(interp_lin), interp_nn, interp_lin)
    valid = ~torch.any(torch.isnan(interp), 0)

    kp3d = interp.T.numpy()
    valid = valid.numpy()
    return kp3d, valid
# Patch the original function
localize_inloc.interpolate_scan = interpolate_scan_patched

@contextmanager
def pass_focal_length(focal_length: float):                  # focal is a scalar, px
    """
    Temporarily monkey‑patch localize_inloc.pose_from_cluster so it uses
    the given focal length. 
    """
    original = localize_inloc.pose_from_cluster      # keep reference

    def patched(dataset_dir, q, retrieved,
                feature_file, match_file,
                skip=None, *, focal_length=focal):
        # call the original but override the kwarg
        return original(dataset_dir, q, retrieved,
                        feature_file, match_file,
                        skip=skip, focal_length=focal_length)

    localize_inloc.pose_from_cluster = patched       # <‑‑ patch
    try:
        yield
    finally:                                         # <‑‑ restore
        localize_inloc.pose_from_cluster = original

class SpatialRegistrator:
    """
    Class to handle spatial registration of sensor modules to Leica scans.
    """

    def __init__(self, loader_map: object, loader_query: object):
        
        if not isinstance(loader_map, LeicaData):
            raise TypeError("loader_map must be an instance of LeicaData.")
        
        if not isinstance(loader_query, (AriaData, IPhoneData, UmiData)):
            raise TypeError("loader_query must be an instance of AriaData, IPhoneData, or UmiData.")

        self.loader_map = loader_map
        self.loader_query = loader_query

        self.image_path_map = self.loader_map.extraction_path / self.loader_map.initial_setup / "pano_tiles" / "rgb"
        self.pose_path_map = self.loader_map.extraction_path / self.loader_map.initial_setup / "pano_tiles" / "poses"
        self.depth_path_map = self.loader_map.extraction_path / self.loader_map.initial_setup / "pano_tiles" / "depth"
        self.xyz_path_map = self.loader_map.extraction_path / self.loader_map.initial_setup / "pano_tiles" / "xyz"

        self.image_path_query = Path(self.loader_query.extraction_path / self.loader_query.label_keyframes.strip("/"))

        # HLoc configuration
        self.retrieval_conf = extract_features.confs["netvlad"]
        self.feature_conf = extract_features.confs["superpoint_inloc"]
        self.matcher_conf = match_features.confs["superpoint+lightglue"]

        self.visual_registration_output_path = self.loader_query.visual_registration_output_path  
        self.visual_registration_output_path.mkdir(parents=True, exist_ok=True)

        self.sfm_pairs = self.visual_registration_output_path / "outputs" / "pairs-netvlad.txt"
        self.loc_pairs = self.visual_registration_output_path / "outputs" / "pairs-loc.txt"
        self.sfm_dir = self.visual_registration_output_path / "outputs" / "sfm"
        self.initial_sfm_dir = self.visual_registration_output_path / "outputs" / "initial_sfm"
        self.results_dir = self.visual_registration_output_path / "outputs" / "results.txt"
        self.matches = self.visual_registration_output_path / "outputs" / f"{self.matcher_conf['output']}.h5"
        self.features = self.visual_registration_output_path / "outputs" / f"{self.feature_conf['output']}.h5"
        self.features_retrieval = self.visual_registration_output_path / "outputs" / f"{self.retrieval_conf['output']}.h5"


        self.images = self.visual_registration_output_path / "inloc" 
        self._set_up_visual_registration_inloc()

        self.references = [p.relative_to(self.images).as_posix() for p in (self.images / "database" / "cutouts").glob("*.jpg")]
        self.query = [p.relative_to(self.images).as_posix() for p in (self.images / "query" / "iphone7").glob("*.png")]
        self.T_world_query = None

    def _set_up_visual_registration_sfm(self):
        """
        Set up the visual registration HLoc directory structure and copy images.
        """

        out_dir = self.visual_registration_output_path / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)

        image_dir = self.visual_registration_output_path / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        image_dir_mapping = image_dir / "mapping"
        image_dir_mapping.mkdir(parents=True, exist_ok=True)

        image_dir_query = image_dir / "query"
        image_dir_query.mkdir(parents=True, exist_ok=True)

        # Copy images to the mapping and query directories
        for image in tqdm(self.image_path_map.glob("*"), desc="Copying images to mapping", total=len(list(self.image_path_map.glob("*")))):
            if image.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                dest_path = image_dir_mapping / f"{image.stem}.png"
                try:
                    img = Image.open(image)
                    img.convert("RGB").save(dest_path, "PNG")
                except Exception as e:
                    print(f"Failed to convert {image}: {e}")

        for image in tqdm(self.image_path_query.glob("*"), desc="Copying images to query", total=len(list(self.image_path_query.glob("*.png")))):
            if image.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                dest_path = image_dir_query / f"{image.stem}.png"
                try:
                    img = Image.open(image)
                    img.convert("RGB").save(dest_path, "PNG")
                except Exception as e:
                    print(f"Failed to convert {image}: {e}")

        print(f"[REGISTRATION] Visual registration set up at {self.visual_registration_output_path}")

    def _set_up_visual_registration_inloc(self):
        """
        Set up the visual registration HLoc directory structure and copy images for InLoc.
        """

        out_dir = self.visual_registration_output_path / "outputs"
        image_dir = self.visual_registration_output_path / "inloc" 
        image_dir_mapping = image_dir / "database" / "cutouts"
        image_dir_query = image_dir / "query" / "iphone7"
        transform_dir = image_dir / "database" / "alignments" / "database" / "transformations"

        if self.images.exists():
            print(f"[REGISTRATION] Outputs directory already exists: {out_dir}. Use force=True to overwrite.")
            return

        out_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        image_dir_mapping.mkdir(parents=True, exist_ok=True)
        image_dir_query.mkdir(parents=True, exist_ok=True)
        transform_dir.mkdir(parents=True, exist_ok=True)

        # Copy images to the mapping and query directories
        for image in tqdm(self.image_path_map.glob("*"), desc="Copying images to mapping", total=len(list(self.image_path_map.glob("*")))):
            if image.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                dest_path = image_dir_mapping / f"000_database_cutouts_{image.stem}.jpg"
            try:
                img = Image.open(image)
                img.convert("RGB").save(dest_path, "JPEG")
            except Exception as e:
                print(f"Failed to convert {image}: {e}")

        # copy xyz to mapping directory
        for xyz in tqdm(self.xyz_path_map.glob("*"), desc="Copying xyz to mapping", total=len(list(self.xyz_path_map.glob("*")))):
            if xyz.suffix.lower() in [".mat"]:
                dest_path = image_dir_mapping / f"000_database_cutouts_{xyz.stem}.mat"
                dest_path_transform = transform_dir / f"000_trans_cutouts.txt"
                try:
                    shutil.copy(xyz, dest_path)
                    self._generate_dummy_transformations_files(dest_path_transform)
                except Exception as e:
                    print(f"Failed to copy {xyz}: {e}")


        for image in tqdm(self.image_path_query.glob("*"), desc="Copying images to query", total=len(list(self.image_path_query.glob("*.png")))):
            if image.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                dest_path = image_dir_query / f"000_database_cutouts_{image.stem}.png"
                try:
                    img = Image.open(image)
                    img.convert("RGB").save(dest_path, "PNG")
                except Exception as e:
                    print(f"Failed to convert {image}: {e}")

        print(f"[REGISTRATION] Visual registration set up at {self.visual_registration_output_path}")

    def visual_registration_inloc(self, force: bool = False):
        """
        End-to-end HLoc pipeline for InLoc:
            1  Extract features for mapping and query images
            2  Match features using SuperGlue
            3  Localize query images in the mapping using InLoc
        """

        if force and any(Path(self.visual_registration_output_path / "outputs").iterdir()):
            shutil.rmtree(self.visual_registration_output_path / "outputs")
            print(f"[REGISTRATION] Force removing previous outputs.")
        if not force and Path(self.visual_registration_output_path / "outputs").exists() and any(Path(self.visual_registration_output_path / "outputs").iterdir()):
            print(f"[REGISTRATION] Outputs already exist. Use force=True to overwrite.")
            return

        print("[REG] Starting visual registration…")

        extract_features.main(
            conf=self.feature_conf,
            image_dir=self.images,
            image_list=self.references,
            feature_path=self.features,
            overwrite=True)
        
        extract_features.main(
            conf=self.feature_conf, 
            image_dir=self.images,
            image_list=self.query,
            feature_path=self.features,
            overwrite=False)

        extract_features.main(                       
            conf=self.retrieval_conf, 
            image_dir=self.images,
            image_list=self.references,
            feature_path=self.features_retrieval, 
            overwrite=True)

        extract_features.main(                       
            conf=self.retrieval_conf, 
            image_dir = self.images,
            image_list=self.query,
            feature_path=self.features_retrieval, 
            overwrite=False)
        
        pairs_from_retrieval.main(
            descriptors = self.features_retrieval,
            output      = self.loc_pairs,
            num_matched = 20,
            query_list  = self.query,        
            db_list     = self.references,  
        )

        match_features.main(
            conf      = self.matcher_conf,
            pairs     = self.loc_pairs,
            features  = self.features,
            matches   = self.matches,        
            overwrite = True
        )

        # use monkey patch to inject focal length into localize_inloc
        # fx = self.loader_query.calibration["K"][0, 0]
        # fy = self.loader_query.calibration["K"][1, 1]
        # set_intrinsics(fx=fx, fy=fy)
        pycolmap_camera_config = self.loader_query.calibration["PINHOLE"]["colmap_camera_cfg"]
        set_camera_model(config=pycolmap_camera_config)

        localize_inloc.main(
            dataset_dir=self.images,
            retrieval=self.loc_pairs,
            features=self.features,
            matches=self.matches,
            results=self.results_dir,
            skip_matches=5)

        print(f"[REGISTRATION] Visual registration completed. Results saved to {self.results_dir}")

    def visual_registration_sfm(self, from_gt: bool = True, force: bool = False):
        """
        End-to-end HLoc pipeline:
            1  NetVLAD for map + query
            2  SuperPoint + SuperGlue on map -> SfM model
            3  NetVLAD query→map pairs  (single call)
            4  SuperGlue matching on those pairs
            5  localize_sfm with intrinsics
        """

        print(f"[REGISTRATION] Starting visual registration...")

        if force and Path(self.visual_registration_output_path / "outputs").exists():
            shutil.rmtree(self.visual_registration_output_path / "outputs")
            print(f"[REGISTRATION] Force removing previous outputs.")
        if not force and Path(self.visual_registration_output_path / "outputs").exists():
            print(f"[REGISTRATION] Outputs already exist. Use force=True to overwrite.")
            return
        
        extract_features.main(
            conf=self.feature_conf,
            image_dir=self.images,
            image_list=self.references,
            feature_path=self.features,
            overwrite=True)
        
        extract_features.main(
            conf=self.feature_conf, 
            image_dir=self.images,
            image_list=self.query,
            feature_path=self.features,
            overwrite=False)

        extract_features.main(                       
            conf=self.retrieval_conf, 
            image_dir=self.images,
            image_list=self.references,
            feature_path=self.features_retrieval, 
            overwrite=True)

        extract_features.main(                       
            conf=self.retrieval_conf, 
            image_dir = self.images,
            image_list=self.query,
            feature_path=self.features_retrieval, 
            overwrite=False)

        pairs_from_retrieval.main(                  
            descriptors=self.features_retrieval,
            output=self.sfm_pairs, 
            num_matched=5,
            query_list=self.references,
            db_list=self.references)

        match_features.main(      
            conf=self.matcher_conf,
            pairs=self.sfm_pairs,
            features=self.features,
            matches=self.matches)

        if from_gt:
            self.create_reconstruction_from_gt_poses()
            triangulation.main(
                sfm_dir=self.sfm_dir,
                reference_model=self.initial_sfm_dir,
                image_dir=self.images,
                pairs=self.sfm_pairs,
                features=self.features,
                matches=self.matches,
                mapper_options=dict(
                    ba_refine_extra_params = False,
                    ba_refine_focal_length = False,
                    ba_refine_principal_point = False,
                    fix_existing_images = True),
                estimate_two_view_geometries = False,)
            model = pycolmap.Reconstruction(self.initial_sfm_dir)
        else:
            model = reconstruction.main(            
                sfm_dir=self.sfm_dir, 
                image_dir=self.images,
                image_list=self.references,
                pairs=self.sfm_pairs, 
                features=self.features, 
                matches=self.matches)

        pairs_from_retrieval.main(
            descriptors=self.features_retrieval,
            output=self.loc_pairs,
            num_matched=20,
            query_list=self.query,  
            db_list=self.references)

        match_features.main(
            conf=self.matcher_conf, 
            pairs=self.loc_pairs,
            features=self.features,
            matches=self.matches, 
            overwrite=True)
        
        good_refs = {j for _,j in map(str.split, open(self.loc_pairs))}
        self.references = good_refs
        
        query_list_path = self.visual_registration_output_path / "outputs" / "query_list.txt"
        with open(query_list_path, "w") as f:
            for name in self.query:
                K_pinhole = self.loader_query.calibration["PINHOLE"]["K"]
                f_x = K_pinhole[0, 0]
                f_y = K_pinhole[1, 1]
                c_x = K_pinhole[0, 2]
                c_y = K_pinhole[1, 2]
                h = self.loader_query.calibration["PINHOLE"]["h"]
                w = self.loader_query.calibration["PINHOLE"]["w"]
                
                f.write(f"{name} PINHOLE {w} {h} {f_x} {f_y} {c_x} {c_y}\n")


        localize_sfm_main(self.sfm_dir, query_list_path, self.loc_pairs,
                        self.features, self.matches, self.results_dir)
        
        print(f"[REGISTRATION] Visual registration completed. Results saved to {self.results_dir}")

    def pcd_to_pcd_registration(self, vis: bool = False, force: bool = False) -> Tuple[np.ndarray, np.ndarray]:

        """
        Perform point cloud to point cloud registration based on the visual registration results.
        Hloc Query poses and therem corresponding odoemtry poses are used to
        compute the transformation between the map and the query. ICP refinement
        is performed to align the point clouds.
        Args:
            vis (bool): Whether to visualize the registration process.
        Returns:
            Tuple[np.ndarray, np.ndarray]: The transformation matrix and the aligned point cloud.
        """

        if force and Path(self.visual_registration_output_path / "T_wq.json").exists():
            os.remove(self.visual_registration_output_path / "T_wq.json")
            print(f"[REGISTRATION] Force removing previous transformation matrix.")
        if not force and Path(self.visual_registration_output_path / "T_wq.json").exists():
            print(f"[REGISTRATION] Transformation matrix already exists. Use force=True to overwrite.")
            return

        print(f"[REGISTRATION] Starting point cloud registration...")
        
        # get the point clouds from the map (leica) and query
        pcd_map_gt = self.loader_map.get_downsampled_points()

        # if isinstance(self.loader_query, AriaData):
        #     pcd_query = self.loader_query.get_semidense_points_pcd()
        # elif isinstance(self.loader_query, IPhoneData):
        #     pass
        #     pcds_query = []
        #     # pcd_query is extracted below per time stamp of the query
        # elif isinstance(self.loader_query, GripperData):
        #     # TODO - implement extraction for GripperData
        #     pass

        # get image timestamp and pose of query in Leica world frame
        pose_query = self.get_poses_query() 
        name = list(pose_query.keys())[0]

        # get the poses of the query images and compute rigid body transformation
        # between the query and the map
        T_was = []
        for name in pose_query.keys():
            T_wc = pose_query[name]["w_T_wc"]
            timestamp = int(name)

            frames = []
            if isinstance(self.loader_query, AriaData):
                T_ad = self.loader_query.get_mps_pose_at_timestamp(timestamp)
                if T_ad is None:
                    print(f"[!] No pose found for timestamp {timestamp}. Skipping.")
                    continue
                T_dc = self.loader_query.calibration["PINHOLE"]["T_device_camera"]
                T_cRaw_cRect = self.loader_query.calibration["PINHOLE"]["pinhole_T_device_camera"]
                T_ca = np.linalg.inv(T_dc @ T_cRaw_cRect) @ np.linalg.inv(T_ad)
                T_wa = T_wc @ T_ca
                T_was.append(T_wa)

                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
                frame.transform(T_wc)
                frames.append(frame)

            elif isinstance(self.loader_query, IPhoneData):
                # pcd_query = self.loader_query.get_cloud_at_timestamp(timestamp, voxel=None)
                # pcds_query.append(pcd_query)
                T_qc = self.loader_query.get_pose_at_timestamp(timestamp)
                T_arkit_to_o3d = np.diag([1, -1, -1, 1]) 
                T_wa = T_wc @ np.linalg.inv(T_qc @ T_arkit_to_o3d)
                T_was.append(T_wa)

                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
                frame.transform(T_wc)
                frames.append(frame)

            elif isinstance(self.loader_query, GripperData):
                raise NotImplementedError("Gripper data not implemented yet.")

        # if isinstance(self.loader_query, IPhoneData):
        #     pcd_query = pcds_query[0]

        # get average transformation and filter outliers
        T_wa = self.mean_transformation(T_was)

        # pcd_query_aligned = copy.deepcopy(pcd_query)
        # pcd_query_aligned.transform(T_wa)

        # icp alignment
        # threshold = 0.02
        # reg_icp = o3d.pipelines.registration.registration_icp(
        #     pcd_query_aligned, pcd_map_gt, threshold,
        #     np.eye(4),
        #     o3d.pipelines.registration.TransformationEstimationPointToPoint())

        # w_pcd_query_aligned_icp = copy.deepcopy(pcd_query_aligned)
        # w_pcd_query_aligned_icp.transform(reg_icp.transformation)
        # w_pcd_query_aligned_icp.paint_uniform_color([0, 1, 0]) 

        T_wa_final = T_wa # reg_icp.transformation @ T_wa
        self.T_world_query = T_wa_final

        if vis:
            # visualize the trajectory of the query module
            stride = 200
            trajectory_query = self.get_trajectory_query()
            trajectory_query = trajectory_query.iloc[::stride, :].reset_index(drop=True)
            
            frames_query = []
            # T_dc = self.loader_query.calibration["T_device_camera"]
            # T_cRaw_cRect = self.loader_query.calibration["pinhole_T_device_camera"]
            for i in range(len(trajectory_query)):
                qw = trajectory_query["qw"].iloc[i]
                qx = trajectory_query["qx"].iloc[i]
                qy = trajectory_query["qy"].iloc[i]
                qz = trajectory_query["qz"].iloc[i]
                tx = trajectory_query["tx"].iloc[i]
                ty = trajectory_query["ty"].iloc[i]
                tz = trajectory_query["tz"].iloc[i]

                T_ad = np.eye(4)
                T_ad[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
                T_ad[:3, 3] = np.array([tx, ty, tz])

                # T_ca = np.linalg.inv(T_dc @ T_cRaw_cRect) @ np.linalg.inv(T_ad)
                # T_wa = T_wc @ T_ca

                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                frame.transform(T_ad)
                frame.transform(T_wa_final)
                frames_query.append(copy.deepcopy(frame))

            # Visualize both point clouds
            o3d.visualization.draw_geometries(
                [pcd_map_gt] + frames + frames_query,
                point_show_normal=False
            )

        # save the transformation matrix as a json file
        T_wa_final_dict = {
            "T_wq": T_wa_final.tolist()
        }
        with open(self.visual_registration_output_path / "T_wq.json", "w") as f:
            json.dump(T_wa_final_dict, f, indent=4)

        print(f"[REGISTRATION] Transformation matrix saved to {self.visual_registration_output_path / 'T_wq.json'}")
        print(f"[REGISTRATION] Transformation matrix: {T_wa_final}")

        return T_wa_final
    
    def compute_transform_world_aria(self):
        """ Compute the transformation between the world frame and the query frame for AriaData.
        This function computes the transformation matrix based on the poses of the query images
        and saves it as a JSON file.
        Returns:
            np.ndarray: The transformation matrix from world to query frame.
        """

        outfile = self.visual_registration_output_path / "T_wq.json"

        if outfile.exists():
            print(f"[REGISTRATION] Transformation matrix already exists at {outfile}. Use force=True to overwrite.")
            return
        
        if not isinstance(self.loader_query, AriaData):
            raise TypeError("loader_query must be an instance of AriaData for this method.")
        
        print(f"[REGISTRATION] Starting transformation computation between query and map...")

        # get image timestamp and pose of query in Leica world frame
        pose_query = self.get_poses_query(min_inliers=400) 
        
        if len(pose_query) == 0:
            return None
        
        name = list(pose_query.keys())[0]

        # get the poses of the query images and compute rigid body transformation
        # between the query and the map
        T_was = []
        for name in pose_query.keys():
            T_wc = pose_query[name]["w_T_wc"]
            timestamp = int(name)

            frames = []

            # device to aria world transformation from mps
            T_ad = self.loader_query.get_mps_pose_at_timestamp(timestamp)
            if T_ad is None:
                print(f"[!] No pose found for timestamp {timestamp}. Skipping.")
                continue
            
            # camera to device transformation
            T_dc = self.loader_query.calibration["PINHOLE"]["T_device_camera"]
            # camera to raw camera rectification transformation
            T_cRaw_cRect = self.loader_query.calibration["PINHOLE"]["pinhole_T_device_camera"]
            # compute the transformation from device to camera
            T_ca = np.linalg.inv(T_dc @ T_cRaw_cRect) @ np.linalg.inv(T_ad)
            # compute the transformation from device to world
            T_wa = T_wc @ T_ca
            T_was.append(T_wa)
            # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            # frame.transform(T_wc)
            # frames.append(frame)
        # elif isinstance(self.loader_query, IPhoneData):
        #     #camera to arkit transformation
        #     T_qc = self.loader_query.get_pose_at_timestamp(timestamp)
        #     # o3d to arkit transformation
        #     T_arkit_to_o3d = np.diag([1, -1, -1, 1]) 
        #     # compute the transformation from arkit to world
        #     T_wa = T_wc @ np.linalg.inv(T_qc @ T_arkit_to_o3d)
        #     T_was.append(T_wa)

        #     frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        #     frame.transform(T_wc)
        #     frames.append(frame)

        # get average transformation and filter outliers to get the final transformation
        # between the query and the map
        if not T_was:
            raise ValueError("No valid transformations found. Check the poses in the query loader.")
        self.T_world_query = self.mean_transformation(T_was)

        # save the transformation matrix as a json file
        T_wa_final_dict = {
            "T_wq": self.T_world_query.tolist()
        }
        with open(outfile, "w") as f:
            json.dump(T_wa_final_dict, f, indent=4)

        print(f"[REGISTRATION] Transformation matrix saved to {self.visual_registration_output_path / 'T_wq.json'}")
        print(f"[REGISTRATION] Transformation matrix: {self.T_world_query}")

        return self.T_world_query
    
    def compute_transform_world_iphone(self):
        """ Compute the transformation between the world frame and the query frame for AriaData.
        This function computes the transformation matrix based on the poses of the query images
        and saves it as a JSON file.
        Returns:
            np.ndarray: The transformation matrix from world to query frame.
        """
        outfile = self.visual_registration_output_path / "T_wq.json"
        if outfile.exists():
            print(f"[REGISTRATION] Transformation matrix already exists at {outfile}. Use force=True to overwrite.")
            return
        
        if not isinstance(self.loader_query, IPhoneData):
            raise TypeError("loader_query must be an instance of IphoneData for this method.")
        
        print(f"[REGISTRATION] Starting transformation computation between query and map...")

        # get image timestamp and pose of query in Leica world frame
        pose_query = self.get_poses_query(min_inliers=400) 
        
        if len(pose_query) == 0:
            return None
        
        name = list(pose_query.keys())[0]
        T_was = []
        for name in pose_query.keys():
            T_wc = pose_query[name]["w_T_wc"]
            timestamp = int(name)

            #camera to arkit transformation
            T_qc = self.loader_query.get_pose_at_timestamp(timestamp)
            # o3d to arkit transformation
            # T_arkit_to_o3d = np.diag([1, -1, -1, 1]) 
            # compute the transformation from arkit to world
            # T_wa = T_wc @ np.linalg.inv(T_qc @ T_arkit_to_o3d)
            # T_wa = T_wc @ np.linalg.inv(T_qc)
            T_was.append(T_wc)


        # get average transformation and filter outliers to get the final transformation
        # between the query and the map
        if not T_was:
            raise ValueError("No valid transformations found. Check the poses in the query loader.")
        self.T_world_query = self.mean_transformation(T_was)

        # save the transformation matrix as a json file
        T_wa_final_dict = {
            "T_wq": self.T_world_query.tolist()
        }
        with open(outfile, "w") as f:
            json.dump(T_wa_final_dict, f, indent=4)

        print(f"[REGISTRATION] Transformation matrix saved to {self.visual_registration_output_path / 'T_wq.json'}")
        print(f"[REGISTRATION] Transformation matrix: {self.T_world_query}")

        return self.T_world_query
    
    def apply_transform_world_aria(self):

        # ensure the transform has been computed
        if not (self.visual_registration_output_path / "T_wq.json").exists():
            raise FileNotFoundError(f"Transformation matrix not found: {self.visual_registration_output_path / 'T_wq.json'}. Run compute_transform_world_aria() first.")

        # load the transformation matrix
        with open(self.visual_registration_output_path / "T_wq.json", "r") as f:
            T_wa_final_dict = json.load(f)
        T_wa = np.array(T_wa_final_dict["T_wq"])
        R_wa = T_wa[:3, :3]
        t_wa = T_wa[:3, 3] 

        # ensure the dirs for pointcloud_aligned, slam_aligned, handtracking_aligned exist
        file_closed_loop_trajectory_aligned = self.loader_query.extraction_path / self.loader_query.label_clt_aligned.strip("/") / "data.csv"

        if file_closed_loop_trajectory_aligned.exists():
            print(f"[REGISTRATION] Aligned closed loop trajectory already exists at {file_closed_loop_trajectory_aligned}. Skipping alignment.")
            return

        if not file_closed_loop_trajectory_aligned.parent.exists():
            file_closed_loop_trajectory_aligned.parent.mkdir(parents=True, exist_ok=True)

        file_semidense_points_aligned = self.loader_query.extraction_path / self.loader_query.label_sdp_aligned.strip("/") / "data.csv"
        if not file_semidense_points_aligned.parent.exists():
            file_semidense_points_aligned.parent.mkdir(parents=True, exist_ok=True)

        file_hand_tracking_aligned = self.loader_query.extraction_path / self.loader_query.label_hand_tracking_aligned.strip("/") / "data.csv"
        if not file_hand_tracking_aligned.parent.exists():
            file_hand_tracking_aligned.parent.mkdir(parents=True, exist_ok=True)

        # file_palm_tracking_aligned = self.loader_query.extraction_path / self.loader_query.label_palm_and_wrist_tracking_aligned.strip("/") / "data.csv"
        # if not file_palm_tracking_aligned.parent.exists():
        #     file_palm_tracking_aligned.parent.mkdir(parents=True, exist_ok=True)

        file_eye_gaze_aligned = self.loader_query.extraction_path / self.loader_query.label_eye_gaze_aligned.strip("/") / "data.csv"
        if not file_eye_gaze_aligned.parent.exists():
            file_eye_gaze_aligned.parent.mkdir(parents=True, exist_ok=True)

        # apply the transformation to the closed loop trajectory
        closed_loop_trajectory = self.loader_query.get_closed_loop_trajectory()
        closed_loop_trajectory_aligned = copy.deepcopy(closed_loop_trajectory)

        pos_cols = ["tx_world_device", "ty_world_device", "tz_world_device"]
        pos_q = closed_loop_trajectory[pos_cols].to_numpy(float)
        pos_w = (pos_q @ R_wa.T) + t_wa        
        closed_loop_trajectory_aligned[pos_cols] = pos_w

        quat_cols = ["qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device"]
        quat_q = closed_loop_trajectory[quat_cols].to_numpy(float)                     # Nx4, [x,y,z,w]
        rot_q = R.from_quat(quat_q)
        rot_wq = R.from_matrix(R_wa)
        rot_w = rot_wq * rot_q                                      # compose: R_w = R_wq ∘ R_q
        quat_w = rot_w.as_quat()                                    # Nx4, [x,y,z,w]
        closed_loop_trajectory_aligned[quat_cols] = quat_w

        grav_cols = ["gravity_x_world", "gravity_y_world", "gravity_z_world"]
        if all(c in closed_loop_trajectory.columns for c in grav_cols):
            g_q = closed_loop_trajectory[grav_cols].to_numpy(float)                   # Nx3
            g_w = g_q @ R_wa.T
            closed_loop_trajectory_aligned[grav_cols] = g_w

        closed_loop_trajectory_aligned.to_csv(file_closed_loop_trajectory_aligned, index=False)
        
        print(f"[REGISTRATION] Closed loop trajectory aligned and saved to {file_closed_loop_trajectory_aligned}")

        # apply the transformation to the palm tracking points
        # palm poses are in aria device frame so we need to go from aria device to aria world to world
        closed_loop_trajectory = self.loader_query.get_closed_loop_trajectory()
        # palm_tracking = self.loader_query.get_palm_and_wrist_tracking()
        # palm_tracking_aligned = copy.deepcopy(palm_tracking)
        hand_tracking = self.loader_query.get_hand_tracking()
        hand_tracking_aligned = copy.deepcopy(hand_tracking)

        # interpolate
        t0_ns = int(closed_loop_trajectory['timestamp'].iloc[0])
        traj_t = (closed_loop_trajectory['timestamp'].to_numpy(np.int64) - t0_ns) * 1e-9
        # palm_t = (palm_tracking['timestamp'].to_numpy(np.int64) - t0_ns) * 1e-9

        # --- Interpolate rotation (SLERP) ---
        quat_cols = ['qx_world_device', 'qy_world_device', 'qz_world_device', 'qw_world_device']
        quat_traj = closed_loop_trajectory[quat_cols].to_numpy(float)
        rot_traj = R.from_quat(quat_traj)
        slerp = Slerp(traj_t, rot_traj)
        # palm_t_clamped = np.clip(palm_t, traj_t[0], traj_t[-1])
        # rot_palm = slerp(palm_t_clamped)
        # R_wd_all = rot_palm.as_matrix()  # shape (N, 3, 3)

        # --- Interpolate translation (linear) ---
        # tx = np.interp(palm_t, traj_t, closed_loop_trajectory['tx_world_device'], left=np.nan, right=np.nan)
        # ty = np.interp(palm_t, traj_t, closed_loop_trajectory['ty_world_device'], left=np.nan, right=np.nan)
        # tz = np.interp(palm_t, traj_t, closed_loop_trajectory['tz_world_device'], left=np.nan, right=np.nan)

        # tx = np.where(np.isnan(tx), np.interp(palm_t_clamped, traj_t, closed_loop_trajectory['tx_world_device']), tx)
        # ty = np.where(np.isnan(ty), np.interp(palm_t_clamped, traj_t, closed_loop_trajectory['ty_world_device']), ty)
        # tz = np.where(np.isnan(tz), np.interp(palm_t_clamped, traj_t, closed_loop_trajectory['tz_world_device']), tz)
        # t_wd_all = np.stack([tx, ty, tz], axis=1)

        # ---- Apply transforms with validity masking ----
        eps = 1e-6  # near-zero threshold

        # for side in ['left', 'right']:
        #     conf = palm_tracking_aligned[f'{side}_tracking_confidence'].to_numpy(float)
        #     valid_conf = conf > 0.7

        #     for part in ['palm', 'wrist']:
        #         # --- positions ---
        #         pos_cols = [f'tx_{side}_{part}_device',
        #                     f'ty_{side}_{part}_device',
        #                     f'tz_{side}_{part}_device']
        #         pos_dev = palm_tracking_aligned[pos_cols].to_numpy(float)            # (N,3)
        #         valid_mag = np.linalg.norm(pos_dev, axis=1) > eps
        #         valid = valid_conf & valid_mag

        #         # transform only valid rows
        #         pos_w = (R_wd_all @ pos_dev[..., None]).squeeze(-1) + t_wd_all       # Aria world
        #         pos_world = (pos_w @ R_wa.T) + t_wa                                   # Your world

        #         # write with NaNs where invalid
        #         outP = np.full_like(pos_world, np.nan)
        #         outP[valid] = pos_world[valid]
        #         palm_tracking_aligned[f'tx_{side}_{part}_world'] = outP[:, 0]
        #         palm_tracking_aligned[f'ty_{side}_{part}_world'] = outP[:, 1]
        #         palm_tracking_aligned[f'tz_{side}_{part}_world'] = outP[:, 2]

        #         # --- normals ---
        #         n_cols = [f'nx_{side}_{part}_device',
        #                 f'ny_{side}_{part}_device',
        #                 f'nz_{side}_{part}_device']
        #         n_dev = palm_tracking_aligned[n_cols].to_numpy(float)
        #         valid_nmag = np.linalg.norm(n_dev, axis=1) > eps
        #         valid_n = valid_conf & valid_nmag

        #         n_w = (R_wd_all @ n_dev[..., None]).squeeze(-1)                       # Aria world (rotation only)
        #         n_world = (n_w @ R_wa.T)                                              # Your world
        #         # renormalize safely
        #         n_norm = np.linalg.norm(n_world, axis=1, keepdims=True)
        #         n_world = np.divide(n_world, np.maximum(n_norm, 1e-12))

        #         outN = np.full_like(n_world, np.nan)
        #         outN[valid_n] = n_world[valid_n]
        #         palm_tracking_aligned[f'nx_{side}_{part}_world'] = outN[:, 0]
        #         palm_tracking_aligned[f'ny_{side}_{part}_world'] = outN[:, 1]
        #         palm_tracking_aligned[f'nz_{side}_{part}_world'] = outN[:, 2]

        # palm_tracking_aligned.to_csv(file_palm_tracking_aligned, index=False)
        # print(f"[REGISTRATION] Palm tracking aligned and saved to {file_palm_tracking_aligned}")

        hand_t = (hand_tracking['timestamp'].to_numpy(np.int64) - t0_ns) * 1e-9
        hand_t_clamped = np.clip(hand_t, traj_t[0], traj_t[-1])

        # interpolate device->Aria world pose at hand timestamps
        rot_hand = slerp(hand_t_clamped)
        R_wd_hand = rot_hand.as_matrix()  # if your traj quats are world->device, use rot_hand.inv().as_matrix()

        tx_h = np.interp(hand_t, traj_t, closed_loop_trajectory['tx_world_device'], left=np.nan, right=np.nan)
        ty_h = np.interp(hand_t, traj_t, closed_loop_trajectory['ty_world_device'], left=np.nan, right=np.nan)
        tz_h = np.interp(hand_t, traj_t, closed_loop_trajectory['tz_world_device'], left=np.nan, right=np.nan)
        tx_h = np.where(np.isnan(tx_h), np.interp(hand_t_clamped, traj_t, closed_loop_trajectory['tx_world_device']), tx_h)
        ty_h = np.where(np.isnan(ty_h), np.interp(hand_t_clamped, traj_t, closed_loop_trajectory['ty_world_device']), ty_h)
        tz_h = np.where(np.isnan(tz_h), np.interp(hand_t_clamped, traj_t, closed_loop_trajectory['tz_world_device']), tz_h)
        t_wd_hand = np.stack([tx_h, ty_h, tz_h], axis=1)

        # validity
        eps = 1e-6
        for side in ['left', 'right']:
            conf_col = f'{side}_tracking_confidence'
            if conf_col in hand_tracking_aligned.columns:
                conf_h = hand_tracking_aligned[conf_col].to_numpy(float)
                valid_conf_h = conf_h > 0.7
            else:
                valid_conf_h = np.ones(len(hand_tracking_aligned), dtype=bool)

            # landmarks 0..20
            for i in range(21):
                cols = [f'tx_{side}_landmark_{i}_device',
                        f'ty_{side}_landmark_{i}_device',
                        f'tz_{side}_landmark_{i}_device']
                if not all(c in hand_tracking_aligned.columns for c in cols):
                    continue
                pts_dev = hand_tracking_aligned[cols].to_numpy(float)          # (N,3)
                valid_mag = np.linalg.norm(pts_dev, axis=1) > eps
                valid = valid_conf_h & valid_mag

                pts_w = (R_wd_hand @ pts_dev[..., None]).squeeze(-1) + t_wd_hand   # Aria world
                pts_world = (pts_w @ R_wa.T) + t_wa                                 # Your world

                out = np.full_like(pts_world, np.nan)
                out[valid] = pts_world[valid]
                hand_tracking_aligned[f'tx_{side}_landmark_{i}_world'] = out[:, 0]
                hand_tracking_aligned[f'ty_{side}_landmark_{i}_world'] = out[:, 1]
                hand_tracking_aligned[f'tz_{side}_landmark_{i}_world'] = out[:, 2]

            # wrist frame pose (device<-wrist) -> Your-world<-wrist (if present)
            t_cols = [f'tx_{side}_device_wrist', f'ty_{side}_device_wrist', f'tz_{side}_device_wrist']
            q_cols = [f'qx_{side}_device_wrist', f'qy_{side}_device_wrist', f'qz_{side}_device_wrist', f'qw_{side}_device_wrist']
            if all(c in hand_tracking_aligned.columns for c in (t_cols + q_cols)):
                t_d = hand_tracking_aligned[t_cols].to_numpy(float)                 # device<-wrist translation
                q_d = hand_tracking_aligned[q_cols].to_numpy(float)                 # device<-wrist quaternion [x,y,z,w]
                valid_pose = valid_conf_h & (np.linalg.norm(t_d, axis=1) > eps)

                # world<-wrist rotation: R_w = R_wd_hand * R_dh
                R_dh = R.from_quat(q_d).as_matrix()                                 # device<-wrist
                R_w  = np.einsum('nij,njk->nik', R_wd_hand, R_dh)                   # world<-wrist
                t_w  = (R_wd_hand @ t_d[..., None]).squeeze(-1) + t_wd_hand         # world origin of wrist

                # Your-world<-wrist
                R_out = np.einsum('ij,njk->nik', R_wa, R_w)
                t_out = (t_w @ R_wa.T) + t_wa
                q_out = R.from_matrix(R_out).as_quat()

                t_buf = np.full_like(t_out, np.nan); t_buf[valid_pose] = t_out[valid_pose]
                q_buf = np.full_like(q_out, np.nan); q_buf[valid_pose] = q_out[valid_pose]

                hand_tracking_aligned[f'tx_{side}_wrist_world'] = t_buf[:, 0]
                hand_tracking_aligned[f'ty_{side}_wrist_world'] = t_buf[:, 1]
                hand_tracking_aligned[f'tz_{side}_wrist_world'] = t_buf[:, 2]
                hand_tracking_aligned[f'qx_{side}_wrist_world'] = q_buf[:, 0]
                hand_tracking_aligned[f'qy_{side}_wrist_world'] = q_buf[:, 1]
                hand_tracking_aligned[f'qz_{side}_wrist_world'] = q_buf[:, 2]
                hand_tracking_aligned[f'qw_{side}_wrist_world'] = q_buf[:, 3]

            # normals in hand_tracking (if present)
            for part in ['palm', 'wrist']:
                n_cols = [f'nx_{side}_{part}_device',
                         f'ny_{side}_{part}_device',
                         f'nz_{side}_{part}_device']
                if not all(c in hand_tracking_aligned.columns for c in n_cols):
                    continue
                n_dev = hand_tracking_aligned[n_cols].to_numpy(float)
                valid_n = valid_conf_h & (np.linalg.norm(n_dev, axis=1) > eps)

                n_w = (R_wd_hand @ n_dev[..., None]).squeeze(-1)                    # Aria world
                n_world = (n_w @ R_wa.T)                                            # Your world
                n_norm = np.linalg.norm(n_world, axis=1, keepdims=True)
                n_world = np.divide(n_world, np.maximum(n_norm, 1e-12))

                outN = np.full_like(n_world, np.nan)
                outN[valid_n] = n_world[valid_n]
                hand_tracking_aligned[f'nx_{side}_{part}_world'] = outN[:, 0]
                hand_tracking_aligned[f'ny_{side}_{part}_world'] = outN[:, 1]
                hand_tracking_aligned[f'nz_{side}_{part}_world'] = outN[:, 2]


        hand_tracking_aligned.to_csv(file_hand_tracking_aligned, index=False)
        print(f"[REGISTRATION] Hand tracking aligned and saved to {file_hand_tracking_aligned}")

        # TODO: apply the transformation to the semidense points

    def apply_transform_world_iphone_(self):
        """
        since the iphones are stationary, we just build a dataframe with the
        timestamps and the world to device transformation
        """

        # ensure the transform has been computed
        if not (self.visual_registration_output_path / "T_wq.json").exists():
            raise FileNotFoundError(
                f"Transformation matrix not found: "
                f"{self.visual_registration_output_path / 'T_wq.json'}. "
                "Run compute_transform_world_iphone() first."
            )

        # ensurenot to overwrite existing aligned trajectory
        file_closed_loop_trajectory_aligned = (
            self.loader_query.extraction_path
            / self.loader_query.label_poses_aligned.strip("/")
            / "data.csv"
        )
        if file_closed_loop_trajectory_aligned.exists():
            print(
                f"[REGISTRATION] Aligned trajectory already exists at "
                f"{file_closed_loop_trajectory_aligned}. Use force=True to overwrite."
            )
            return

        # load the transformation matrix
        with open(self.visual_registration_output_path / "T_wq.json", "r") as f:
            T_wa_final_dict = json.load(f)
        T_wa = np.array(T_wa_final_dict["T_wq"])
        R_wa = T_wa[:3, :3]
        t_wa = T_wa[:3, 3]

        # load the iPhone trajectory (in ARKit/q frame)
        traj = self.loader_query.get_trajectory()

        # extract quaternion and translation arrays
        quat_cols = ["qx", "qy", "qz", "qw"]
        trans_cols = ["tx", "ty", "tz"]

        R_qc = R.from_quat(traj[quat_cols].to_numpy(float)).as_matrix()  # (N,3,3)
        t_qc = traj[trans_cols].to_numpy(float)                           # (N,3)

        # apply the rigid transform: world<-cam
        R_wc = np.einsum("ij,njk->nik", R_wa, R_qc)                      # (N,3,3)
        t_wc = (t_qc @ R_wa.T) + t_wa                                    # (N,3)
        q_wc = R.from_matrix(R_wc).as_quat()                              # (N,4)

        # build aligned dataframe
        closed_loop_trajectory_aligned = pd.DataFrame({
            "timestamp": traj["timestamp"].astype(np.int64),
            "tx_world_cam": t_wc[:, 0],
            "ty_world_cam": t_wc[:, 1],
            "tz_world_cam": t_wc[:, 2],
            "qx_world_cam": q_wc[:, 0],
            "qy_world_cam": q_wc[:, 1],
            "qz_world_cam": q_wc[:, 2],
            "qw_world_cam": q_wc[:, 3],
        }).sort_values("timestamp")

        # ensure output dir exists
        if not file_closed_loop_trajectory_aligned.parent.exists():
            file_closed_loop_trajectory_aligned.parent.mkdir(parents=True, exist_ok=True)

        # write aligned trajectory
        closed_loop_trajectory_aligned.to_csv(file_closed_loop_trajectory_aligned, index=False)
        print(f"[REGISTRATION] Camera poses saved to {file_closed_loop_trajectory_aligned}")

    def apply_transform_world_iphone(self):
        """
        since the iphones are stationary, we just build a dataframe with the
        timestamps and the world to device transformation
        """

        # ensure the transform has been computed
        if not (self.visual_registration_output_path / "T_wq.json").exists():
            raise FileNotFoundError(f"Transformation matrix not found: {self.visual_registration_output_path / 'T_wq.json'}. Run compute_transform_world_iphone() first.")

        # ensurenot to overwrite existing aligned trajectory
        file_closed_loop_trajectory_aligned = self.loader_query.extraction_path / self.loader_query.label_poses_aligned.strip("/") / "data.csv"
        if file_closed_loop_trajectory_aligned.exists():
            print(f"[REGISTRATION] Aligned trajectory already exists at {file_closed_loop_trajectory_aligned}. Use force=True to overwrite.")
            return

        # load the transformation matrix
        with open(self.visual_registration_output_path / "T_wq.json", "r") as f:
            T_wa_final_dict = json.load(f)
        T_wa = np.array(T_wa_final_dict["T_wq"])
        R_wa = T_wa[:3, :3]
        t_wa = T_wa[:3, 3] 

        # get image timestamps
        frame_files = self.loader_query.get_extracted_frames()
        timestamps = [int(f.stem) for f in frame_files]
        timestamps.sort()

        closed_loop_trajectory_aligned = pd.DataFrame({
            "timestamp": timestamps,
            "tx_world_cam": [t_wa[0]] * len(timestamps),
            "ty_world_cam": [t_wa[1]] * len(timestamps),
            "tz_world_cam": [t_wa[2]] * len(timestamps),
            "qx_world_cam": [R.from_matrix(R_wa).as_quat()[0]] * len(timestamps),
            "qy_world_cam": [R.from_matrix(R_wa).as_quat()[1]] * len(timestamps),
            "qz_world_cam": [R.from_matrix(R_wa).as_quat()[2]] * len(timestamps),
            "qw_world_cam": [R.from_matrix(R_wa).as_quat()[3]] * len(timestamps),
        })
        

        # ensure the dirs for pointcloud_aligned, slam_aligned, handtracking_aligned exist
        file_closed_loop_trajectory_aligned = self.loader_query.extraction_path / self.loader_query.label_poses_aligned.strip("/") / "data.csv"
        if not file_closed_loop_trajectory_aligned.parent.exists():
            file_closed_loop_trajectory_aligned.parent.mkdir(parents=True, exist_ok=True)

        # write the trajectory
        closed_loop_trajectory_aligned.to_csv(file_closed_loop_trajectory_aligned, index=False)
        print(f"[REGISTRATION] Camera poses saved to {file_closed_loop_trajectory_aligned}")

    def test(self):

        timestamp = 271339268936
        T_arkit_to_o3d = np.diag([1, -1, -1, 1]) 
        pose_query = self.get_poses_query() 
        T_wc = pose_query[str(timestamp)]["w_T_wc"]

        pcd_query = self.loader_query.get_cloud_at_timestamp(timestamp)
        T_qc = self.loader_query.get_pose_at_timestamp(timestamp)
        T_wa = T_wc @ np.linalg.inv(T_qc @ T_arkit_to_o3d)

        pcd = copy.deepcopy(pcd_query)
        pcd_world = pcd.transform(T_wa)

        # Create coordinate frame (frustum-like pose)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        frame.transform(T_qc)
        frame.transform(T_wa)

        frame_hloc = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        frame_hloc.transform(T_wc)
        

        pcd_map_gt = self.loader_map.get_downsampled(scan="post")

        # Optionally also render camera position from query (T_wc)
        # camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        # camera_frame.transform(T_wc)

        # Visualize
        o3d.visualization.draw_geometries([pcd_world, pcd_map_gt, frame_hloc],)

        
        a = 2

    def get_sfm_model(self) -> pycolmap.Reconstruction:
        """
        Get the SFM model from the visual registration output.
        Returns:
            pycolmap.Reconstruction: The SFM model.
        """
        if not self.sfm_dir.exists():
            raise FileNotFoundError(f"SFM directory not found: {self.sfm_dir}, re-run visual registration.")

        model = pycolmap.Reconstruction(self.sfm_dir)

        if model.num_images == 0:
            raise ValueError(f"No images found in SFM directory: {self.sfm_dir}")

        return model


    def get_poses_query(self, as_dataframe: bool = False, min_matches: int = 400, min_inliers: int= 400) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Get the poses of the query images.
        Returns:
            dict: {
                "w_T_wc": world to camera transformation matrix (4x4)
                "K": camera intrinsics matrix (3x3)
                "h": image height
                "w": image width
            }
        """

        if not Path(str(self.results_dir) + "_logs.pkl").exists():
            raise FileNotFoundError(f"Results log file not found: {self.results_dir}, re-run visual registration.")

        with open(str(self.results_dir) + "_logs.pkl", "rb") as f:
            logs = pickle.load(f)

        K = self.loader_query.calibration["PINHOLE"]["K"]
        h = self.loader_query.calibration["PINHOLE"]["h"]
        w = self.loader_query.calibration["PINHOLE"]["w"]

        poses = {}

        inliers = []
        matches = []
        for name, log in logs['loc'].items():
            
                if "success" in log["PnP_ret"] and log["PnP_ret"]["success"] == False:
                    print(f"[!] Not enough matches or inliers for {name}. Skipping.")
                    continue 
                
                # Check if the log has enough matches and inliers
                if log["PnP_ret"]['num_inliers'] < min_inliers or log['num_matches'] < min_matches:
                    print(f"[!] Not enough matches or inliers for {name}. Skipping.")
                    print(log["PnP_ret"]['num_inliers'], log['num_matches'])
                    continue
                
                # TODO not happy with solution, should look into renaming the files again
                if "000_database_cutouts_" in Path(name).stem:
                    name = Path(name).stem.replace("000_database_cutouts_", "")
                else:
                    name = Path(name).stem


                a = 2
                # TODO check quat layout
                R_cw = log['PnP_ret']['cam_from_world'].rotation.matrix()
                c_t_cw = log['PnP_ret']['cam_from_world'].translation

                # world to cam in world system
                R_wc = R_cw.T
                # invert quaternion
                q_wc = R.from_matrix(R_wc).inv().as_quat()
                w_t_wc = -R_wc @ c_t_cw

                # to Transformation matrix 4x4
                w_T_wc = np.eye(4)
                w_T_wc[:3, :3] = R_wc
                w_T_wc[:3, 3] = w_t_wc

                poses[Path(name).stem] = {
                    "w_T_wc": w_T_wc,
                    "K": K,
                    "h": h,
                    "w": w,
                    "q_wc": q_wc,
                    "R_wc": R_wc,
                    "w_t_wc": w_t_wc,
                    "c_t_cw": c_t_cw
                }
        
        if as_dataframe:
            rows = []
            for ts, p in poses.items():
                t = p["w_t_wc"]  # [tx, ty, tz]
                q = p["q_wc"]  # [qx, qy, qz, qw]
                rows.append({
                    "timestamp": int(ts),
                    "tx":        t[0],
                    "ty":        t[1],
                    "tz":        t[2],
                    "qx":        q[0],
                    "qy":        q[1],
                    "qz":        q[2],
                    "qw":        q[3],
                })

            return pd.DataFrame(rows)

        return poses                

    def get_poses_gt(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Get the gt poses of the rendered map images.
        Returns:
            dict: {
                "w_T_wc": world to camera transformation matrix (4x4)
                "K": camera intrinsics matrix (3x3)
                "h": image height
                "w": image width
            }
        """

        poses = {}
        json_files = sorted(self.pose_path_map.glob("*.json"))

        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)

            fx, fy, cx, cy = data['fx'], data['fy'], data['cx'], data['cy']
            w = data['width']
            h = data['height']

            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0,  0,  1]
            ])

            T = np.array(data['extrinsic'])

            poses[json_file.stem] = {
                "w_T_wc": T,
                "K": K,
                "w": w,
                "h": h
            }

        return poses
    
    def get_trajectory_query(self) -> pd.DataFrame:
        """
        Get trajectory full trajectry of the query module (not just the keyframes used for alignemnt)
        Trajectory is given in query module frame.
        NOT IN WORLD FRAME!
        Returns:
            pd.DataFrame: Trajectory of the query module.
        """

        if isinstance(self.loader_query, AriaData):
            trajectory = self.loader_query.get_closed_loop_trajectory()
            # get timestamp, qw_world, qx_world, qy_world, qz_world, tx_world, ty_world, tz_world
            trajectory = trajectory[["timestamp", "tx_world_device", "ty_world_device", "tz_world_device", "qw_world_device", "qx_world_device", "qy_world_device", "qz_world_device"]]
            #rename columns
            trajectory.columns = ["timestamp", "tx", "ty", "tz", "qw", "qx", "qy", "qz"]
        elif isinstance(self.loader_query, IPhoneData):
            trajectory = self.loader_query.get_trajectory()
        elif isinstance(self.loader_query, GripperData):
            # TODO - implement extraction for GripperData
            raise NotImplementedError("GripperData trajectory extraction not implemented yet.")
        elif isinstance(self.loader_query, UmiData):
            trajectory = self.loader_query.get_odometry_orbslam()

        return trajectory
    
    def align_and_optimize_orbslam_poses(self):

        """
        Align and optimize the ORB-SLAM poses of the query module to the map.
        Meant for UMIData, as we dont have a pre-optimized trajectory (unlike AriaData).
        We first need to align the drifty ORB-SLAM poses to the map poses, and then optimize them.
        """

        if not isinstance(self.loader_query, UmiData):
            raise TypeError("align_and_optimize_orbslam_poses is only implemented for UmiData.")
        
        # save the optimized poses to a csv file
        optimized_poses_path = self.loader_query.extraction_path / self.loader_query.label_poses / "data.csv"

        if optimized_poses_path.exists():
            print(f"[REGISTRATION] Optimized poses already exist at {optimized_poses_path}. Skipping optimization.")
            return

        print(f"[REGISTRATION] Aligning and optimizing ORB-SLAM poses for UmiData...")

        poses_query_orbslam = self.get_trajectory_query()
        poses_query_hloc = self.get_poses_query(as_dataframe=True, min_matches=300, min_inliers=150)

        optimizer = TrajectoryOptimization(poses_traj= poses_query_orbslam,
                                           poses_anchor=poses_query_hloc)
        poses_query_optimized = optimizer.hierarchical_optimization()

        # save the optimized poses to a csv file
        optimized_poses_path.parent.mkdir(parents=True, exist_ok=True)
        poses_query_optimized.to_csv(optimized_poses_path, index=False)
        print(f"[REGISTRATION] Optimized poses saved to {optimized_poses_path}")

        return poses_query_optimized

    
    def mean_transformation(self, T_list: list) -> np.ndarray:

        """
        Compute the mean transformation matrix from a list of transformation matrices, filtering out outliers.
        Args:
            T_list (list): List of transformation matrices (4x4).
        Returns:
            np.ndarray: Mean transformation matrix (4x4).
        """

        N = len(T_list)
        if N == 0:
            raise ValueError("Empty pose list.")
           
        translations = np.array([T[:3, 3] for T in T_list])
        rotations = R.from_matrix([T[:3, :3] for T in T_list]) 

        t_norm = np.linalg.norm(translations, axis=1)
        t_norm_median = np.median(t_norm)

        dist = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                r1 = rotations[i].as_matrix()
                r2 = rotations[j].as_matrix()

                frob_dist = np.sqrt(np.trace((r1 - r2).T @ (r1 - r2)))
                dist[i, j] = frob_dist
                
        # compute average distance to others
        avg_dist = np.mean(dist, axis=1)
        median_dist = np.median(avg_dist)

        # filter outlier if distance is greater than 1.5 * median distance or less than 0.5 * median distance
        outlier_mask_rot = np.logical_and(avg_dist <= 1.25 * median_dist, avg_dist >= 0.75 * median_dist)
        outlier_mask_trans = np.logical_and(t_norm <= 1.25 * t_norm_median, t_norm >= 0.75 * t_norm_median)
        outlier_mask = np.logical_and(outlier_mask_rot, outlier_mask_trans)

        # compute mean transformation
        T_mean = np.eye(4)
        T_mean[:3, :3] = np.mean(rotations[outlier_mask].as_matrix(), axis=0)
        T_mean[:3, 3] = np.mean(translations[outlier_mask], axis=0)

        return T_mean

    
    def create_reconstruction_from_gt_poses(self):

        poses_gt = self.get_poses_gt()

        rec = pycolmap.Reconstruction()

        for name, pose in poses_gt.items():

            f_x = pose["K"][0, 0]
            f_y = pose["K"][1, 1]
            c_x = pose["K"][0, 2]
            c_y = pose["K"][1, 2]
            w = pose["w"]
            h = pose["h"]

            tvec = pose["w_T_wc"][:3, 3]
            R = pose["w_T_wc"][:3, :3]
            qvec = Rotation.from_matrix(R).as_quat()

            cam_from_world = pycolmap.Rigid3d(pose["w_T_wc"][:3, :4])
            image_name = f"mapping/{name}.png"

            # Create a camera object
            camera = pycolmap.Camera(
                model="PINHOLE",
                width=w,
                height=h,
                params=np.array([f_x, f_y, c_x, c_y], dtype=np.float32),
                camera_id=int(name)
            )

            # Add the camera to the reconstruction
            rec.add_camera(camera)

            # Create an image object
            image = pycolmap.Image(
                name=image_name,
                camera_id=camera.camera_id,
                image_id=camera.camera_id,
                cam_from_world=cam_from_world
            )

            rec.add_image(image)

        # Save the reconstruction
        self.initial_sfm_dir.mkdir(parents=True, exist_ok=True)
        rec.write(self.initial_sfm_dir)
        print(f"[REGISTRATION] Reconstruction created from gt poses. Saved to {self.initial_sfm_dir}")

    def visual_registration_viz(self, show_trajectory: bool = True):
        """
        Visualise the map (points + map cameras) together with all
        localised query cameras inside an interactive Plotly view.
        """

        if not self.sfm_dir.exists():
            raise FileNotFoundError(f"SFM directory not found: {self.sfm_dir}, re-run visual registration.")

        model = pycolmap.Reconstruction(self.sfm_dir)

        fig = viz_3d.init_figure()
        viz_3d.plot_reconstruction(
            fig, model,
            color="rgba(255,0,0,0.5)",
            name="mapping",
            points_rgb=True)

        K_pinhole = self.loader_query.calibration["PINHOLE"]["K"]

        query_poses = self.get_poses_query()
        for name, pose in query_poses.items():
            w_T_wc = pose["w_T_wc"]
            R_wc = w_T_wc[:3, :3]
            C_wc = w_T_wc[:3, 3]

            viz_3d.plot_camera(fig,
                R_wc,
                C_wc,  
                K=K_pinhole,
                name=name,
                text=name,
                color="rgba(0,255,0,0.5)")

        fig.update_layout(height=800)
        fig.show()
        a = 2

    def visualize_umi_trajectory_aligned(self, stride: int = 30, color: List = [0,1,0], mode: str = "point", pcd_sampling: str = "downsampled"):

        if not isinstance(self.loader_query, UmiData):
            raise TypeError("visualize_umi_trajectory_aligned is only implemented for UmiData.")
        
        if mode not in ["point", "device_frame", "camera_frame"]:
            raise ValueError("mode must be one of ['point', 'camera_frame']")
        
        if pcd_sampling not in ["downsampled", "raw"]:
            raise ValueError("pcd_sampling must be one of ['downsampled', 'raw']")

        if pcd_sampling == "downsampled":
            pcd_map = self.loader_map.get_downsampled_points()
        elif pcd_sampling == "raw":
            pcd_map = self.loader_map.get_full_points()

        trajectory_query = self.loader_query.get_closed_loop_trajectory_aligned()
        trajectory_query = trajectory_query.iloc[::stride, :].reset_index(drop=True)

        markers = []
        for i in range(len(trajectory_query)):
            qw = trajectory_query["qw"].iloc[i]
            qx = trajectory_query["qx"].iloc[i]
            qy = trajectory_query["qy"].iloc[i]
            qz = trajectory_query["qz"].iloc[i]
            tx = trajectory_query["tx"].iloc[i]
            ty = trajectory_query["ty"].iloc[i]
            tz = trajectory_query["tz"].iloc[i]

            T_wd = np.eye(4)
            T_wd[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
            T_wd[:3, 3] = np.array([tx, ty, tz])

            if mode == "point":
                # show point in world frame
                T = T_wd
                marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                marker.paint_uniform_color(color)  # red color
                marker.transform(T)
                markers.append(copy.deepcopy(marker))
            if mode == "camera_frame":
                # show camera frame in world frame
                # T_ca = np.linalg.inv(T_dc @ T_cRaw_cRect) @ np.linalg.inv(T_wd)
                T = T_wd
                marker = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                marker.transform(T)
                markers.append(copy.deepcopy(marker))

        o3d.visualization.draw_geometries(
            [pcd_map] + markers,
            point_show_normal=False
        )

    def visualize_aria_trajectory_aligned(self, stride: int = 200, traj: List[str] = ["aria", "palm", "hand"], mode: str = "point", pcd_sampling: str = "downsampled"):
        """
        Visualize the trajectory of the AriaData query module aligned to the map.
        Args:
            stride (int): Step size for downsampling the trajectory points.
            mode (str): Visualization mode, can be "point", "device_frame", or "camera_frame".
            pcd_sampling (str): Point cloud sampling method, can be "downsampled" or "raw".
        """

        if not isinstance(self.loader_query, AriaData):
            raise TypeError(f"visualize_aria_trajectory_aligned is only implemented for AriaData. Got {type(self.loader_query).__name__} instead.")
        
        if mode not in ["point", "device_frame", "camera_frame"]:
            raise ValueError("mode must be one of ['point', 'device_frame', 'camera_frame']")
        
        if pcd_sampling not in ["downsampled", "raw"]:
            raise ValueError("pcd_sampling must be one of ['downsampled', 'raw']")
        


        if pcd_sampling == "downsampled":
            pcd_map = self.loader_map.get_downsampled_points()
        elif pcd_sampling == "raw":
            pcd_map = self.loader_map.get_full_points()
        
        markers_palm = []
        markers = []

        if "aria" in traj:
            trajectory_query = self.loader_query.get_closed_loop_trajectory_aligned()
            trajectory_query = trajectory_query[["timestamp", "tx_world_device", "ty_world_device", "tz_world_device", "qw_world_device", "qx_world_device", "qy_world_device", "qz_world_device"]]
            trajectory_query.columns = ["timestamp", "tx", "ty", "tz", "qw", "qx", "qy", "qz"]
            trajectory_query = trajectory_query.iloc[::stride, :].reset_index(drop=True)

            cmap = cm.get_cmap("inferno")  # 'viridis', 'plasma', 'cool', 'inferno'
            norm = mcolors.Normalize(vmin=0, vmax=len(trajectory_query)-1)

            T_dc = self.loader_query.calibration["PINHOLE"]["T_device_camera"]
            T_cRaw_cRect = self.loader_query.calibration["PINHOLE"]["pinhole_T_device_camera"]
            for i in range(len(trajectory_query)):
                qw = trajectory_query["qw"].iloc[i]
                qx = trajectory_query["qx"].iloc[i]
                qy = trajectory_query["qy"].iloc[i]
                qz = trajectory_query["qz"].iloc[i]
                tx = trajectory_query["tx"].iloc[i]
                ty = trajectory_query["ty"].iloc[i]
                tz = trajectory_query["tz"].iloc[i]

                T_wd = np.eye(4)
                T_wd[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
                T_wd[:3, 3] = np.array([tx, ty, tz])

                color = cmap(norm(i))[:3]

                if mode == "point":
                    # show point in world frame
                    T = T_wd
                    marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                    marker.paint_uniform_color(color)  # red color
                    marker.transform(T)
                    markers.append(copy.deepcopy(marker))
                elif mode == "device_frame":
                    # show device frame in world frame
                    T = T_wd
                    marker = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                    marker.transform(T)
                    markers.append(copy.deepcopy(marker))
                if mode == "camera_frame":
                    # show camera frame in world frame
                    # T_ca = np.linalg.inv(T_dc @ T_cRaw_cRect) @ np.linalg.inv(T_wd)
                    T_dcRect = T_dc @ T_cRaw_cRect
                    T = T_wd @ T_dcRect
                    marker = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                    marker.transform(T)
                    markers.append(copy.deepcopy(marker))

        if "palm" in traj:
            palms = self.loader_query.get_palm_and_wrist_tracking_aligned()
            palms_right = palms[["timestamp", "tx_right_palm_world", "ty_right_palm_world", "tz_right_palm_world", "nx_right_palm_world", "ny_right_palm_world", "nz_right_palm_world"]]
            palms_right.columns = ["timestamp", "tx", "ty", "tz", "nx", "ny", "nz"]
            palms_right = palms_right.iloc[::2, :].reset_index(drop=True)
            
            for i in range(len(palms_right)):
                nx = palms_right["nx"].iloc[i]
                ny = palms_right["ny"].iloc[i]
                nz = palms_right["nz"].iloc[i]
                tx = palms_right["tx"].iloc[i]
                ty = palms_right["ty"].iloc[i]
                tz = palms_right["tz"].iloc[i]

                T_wd = np.eye(4)
                # T_wd[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
                T_wd[:3, 3] = np.array([tx, ty, tz])

                # show point in world frame
                T = T_wd
                marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                marker.paint_uniform_color([1,0,0])  # red color
                marker.transform(T)
                markers.append(copy.deepcopy(marker))

        if "wrist" in traj: 
            wrists = self.loader_query.get_palm_and_wrist_tracking_aligned()
            wrists_right = palms[["timestamp", "tx_right_wrist_world", "ty_right_wrist_world", "tz_right_wrist_world", "nx_right_wrist_world", "ny_right_wrist_world", "nz_right_wrist_world"]]
            wrists_right.columns = ["timestamp", "tx", "ty", "tz", "nx", "ny", "nz"]
            wrists_right = wrists_right.iloc[::5, :].reset_index(drop=True)
            
            for i in range(len(wrists_right)):
                nx = wrists_right["nx"].iloc[i]
                ny = wrists_right["ny"].iloc[i]
                nz = wrists_right["nz"].iloc[i]
                tx = wrists_right["tx"].iloc[i]
                ty = wrists_right["ty"].iloc[i]
                tz = wrists_right["tz"].iloc[i]

                T_wd = np.eye(4)
                # T_wd[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
                T_wd[:3, 3] = np.array([tx, ty, tz])

                # show point in world frame
                T = T_wd
                marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                marker.paint_uniform_color([0,0.5,0.5])  # red color
                marker.transform(T)
                markers.append(copy.deepcopy(marker))

        o3d.visualization.draw_geometries(
            [pcd_map] + markers + markers_palm,
            point_show_normal=False
        )

    def visualize_iphone_trajectory_aligned(self, stride: int = 200, mode: str = "point", pcd_sampling: str = "downsampled"):
        """
        Visualize the trajectory of the iphoneData query module aligned to the map.
        Args:
            stride (int): Step size for downsampling the trajectory points.
            mode (str): Visualization mode, can be "point" or "camera_frame".
            pcd_sampling (str): Point cloud sampling method, can be "downsampled" or "raw".
        """

        if not isinstance(self.loader_query, IPhoneData):
            raise TypeError("visualize_aria_trajectory_aligned is only implemented for ihpeonData.")
        
        if mode not in ["point", "camera_frame"]:
            raise ValueError("mode must be one of ['point', 'device_frame', 'camera_frame']")
        
        if pcd_sampling not in ["downsampled", "raw"]:
            raise ValueError("pcd_sampling must be one of ['downsampled', 'raw']")
        
        if self.loader_query.get_trajectory_aligned() is None:
            print("NOT ALIGNED YET")
            return

        if pcd_sampling == "downsampled":
            pcd_map = self.loader_map.get_downsampled_points()
        elif pcd_sampling == "raw":
            pcd_map = self.loader_map.get_full_points()
        
        markers_palm = []
        markers = []

        trajectory_query = self.loader_query.get_trajectory_aligned()
        trajectory_query = trajectory_query[["timestamp", "tx_world_cam", "ty_world_cam", "tz_world_cam", "qw_world_cam", "qx_world_cam", "qy_world_cam", "qz_world_cam"]]
        trajectory_query.columns = ["timestamp", "tx", "ty", "tz", "qw", "qx", "qy", "qz"]
        trajectory_query = trajectory_query.iloc[::stride, :].reset_index(drop=True)

        cmap = cm.get_cmap("inferno")  # 'viridis', 'plasma', 'cool', 'inferno'
        norm = mcolors.Normalize(vmin=0, vmax=len(trajectory_query)-1)

        for i in range(len(trajectory_query)):
            qw = trajectory_query["qw"].iloc[i]
            qx = trajectory_query["qx"].iloc[i]
            qy = trajectory_query["qy"].iloc[i]
            qz = trajectory_query["qz"].iloc[i]
            tx = trajectory_query["tx"].iloc[i]
            ty = trajectory_query["ty"].iloc[i]
            tz = trajectory_query["tz"].iloc[i]

            T_wd = np.eye(4)
            T_wd[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
            T_wd[:3, 3] = np.array([tx, ty, tz])

            color = cmap(norm(i))[:3]

            if mode == "point":
                # show point in world frame
                T = T_wd
                marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                marker.paint_uniform_color(color)  # red color
                marker.transform(T)
                markers.append(copy.deepcopy(marker))
            if mode == "camera_frame":
                # show camera frame in world frame
                # T_ca = np.linalg.inv(T_dc @ T_cRaw_cRect) @ np.linalg.inv(T_wd)
                T = T_wd
                marker = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                marker.transform(T)
                markers.append(copy.deepcopy(marker))

        o3d.visualization.draw_geometries(
            [pcd_map] + markers,
        point_show_normal=False
    )

    def viz_2d(self):

        from hloc.visualization import visualize_loc
        model = pycolmap.Reconstruction(self.sfm_dir)
        # visualization.visualize_sfm_2d(model,self.images, color_by="visibility", n=2)
        from hloc.localize_sfm import QueryLocalizer, pose_from_cluster

        camera = pycolmap.infer_camera_from_image(self.images / self.query[0])
        camera.model = "PINHOLE"
        camera.width = self.loader_query.calibration["PINHOLE"]["w"]
        camera.height = self.loader_query.calibration["PINHOLE"]["h"]
        camera.params[0] = self.loader_query.calibration["PINHOLE"]["K"][0, 0]
        camera.params[1] = self.loader_query.calibration["PINHOLE"]["K"][1, 1]
        camera.params[2] = self.loader_query.calibration["PINHOLE"]["K"][0, 2]
        camera.params[3] = self.loader_query.calibration["PINHOLE"]["K"][1, 2]

        good_refs = {j for _,j in map(str.split, open(self.loc_pairs))}
        self.references = good_refs

        ref_ids = [model.find_image_with_name(r).image_id for r in self.references]
        conf = {
            "estimation": {"ransac": {"max_error": 12}},
            "refinement": {"refine_focal_length": True, "refine_extra_params": True},
        }
        localizer = QueryLocalizer(model, conf)
        ret, log = pose_from_cluster(localizer, self.query[0], camera, ref_ids, self.features, self.matches)

        mask = log["PnP_ret"]["inlier_mask"]
        log["PnP_ret"]["inliers"] = np.where(mask)[0]

        # print(f'found {ret["num_inliers"]}/{len(ret["inliers"])} inlier correspondences.')
        visualization.visualize_loc_from_log(self.images, self.query[0], log, model)
        a = 2
 
        fig = viz_3d.init_figure()
        pose = pycolmap.Image(cam_from_world=ret["cam_from_world"])
        viz_3d.plot_camera_colmap(
            fig, pose, camera, color="rgba(0,255,0,0.5)", name=self.query[0], fill=True
        )
        # visualize 2D-3D correspodences
        inl_3d = np.array(
            [model.points3D[pid].xyz for pid in np.array(log["points3D_ids"])[ret["inliers"]]]
        )
        viz_3d.plot_points(fig, inl_3d, color="lime", ps=1, name=self.query[0])
        fig.show()

        # visualize_loc(results=self.results_dir,image_dir=self.images,reconstruction=self.sfm_dir,db_image_dir=None)

    def viz_gtsam_poses(self, df_poses_optimized: pd.DataFrame,):
        """
        Display a PLY point cloud + mapping/query cameras in an interactive
        Plotly 3‑D viewer.

        Args
        ----
        ply_path : Path or str
            Full path to the point‑cloud *.ply* you exported from Leica / COLMAP.
        show_trajectory : bool
            If True, draw a line through the localised query cameras.
        """

        # ---------- 1. load the point cloud ---------------------------------
        pcd = self.loader_map.get_downsampled_points()
        xyz = np.asarray(pcd.points)
        if pcd.has_colors():
            rgb = np.asarray(pcd.colors)
        else:                                # default light‑grey
            rgb = np.full_like(xyz, 0.7)
            
        stride = 10
        trajectory_query = df_poses_optimized.iloc[::stride, :].reset_index(drop=True)
        
        frames_query = []
        for i in range(len(trajectory_query)):
            qw = trajectory_query["qw"].iloc[i]
            qx = trajectory_query["qx"].iloc[i]
            qy = trajectory_query["qy"].iloc[i]
            qz = trajectory_query["qz"].iloc[i]
            tx = trajectory_query["tx"].iloc[i]
            ty = trajectory_query["ty"].iloc[i]
            tz = trajectory_query["tz"].iloc[i]

            T_ad = np.eye(4)
            T_ad[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
            T_ad[:3, 3] = np.array([tx, ty, tz])

            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            frame.transform(T_ad)
            frames_query.append(copy.deepcopy(frame))

        o3d.visualization.draw_geometries(
            [pcd] + frames_query,
            point_show_normal=False
        )

        a = 2



    def visual_registration_viz_ply(self,
                                show_trajectory: bool = False):
        """
        Display a PLY point cloud + mapping/query cameras in an interactive
        Plotly 3‑D viewer.

        Args
        ----
        ply_path : Path or str
            Full path to the point‑cloud *.ply* you exported from Leica / COLMAP.
        show_trajectory : bool
            If True, draw a line through the localised query cameras.
        """

        # ---------- 1. load the point cloud ---------------------------------
        pcd = self.loader_map.get_downsampled_points()
        xyz = np.asarray(pcd.points)
        if pcd.has_colors():
            rgb = np.asarray(pcd.colors)
        else:                                # default light‑grey
            rgb = np.full_like(xyz, 0.7)

        # ---------- 2. init Plotly figure -----------------------------------
        fig = viz_3d.init_figure()

        # Scatter‑3D for the raw points
        fig.add_trace(go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
            mode="markers",
            marker=dict(size=1.5,
                        color=(rgb * 255).astype(np.uint8),
                        opacity=0.7),
            name="map points"
        ))

        # ---------- 3. plot mapping cameras ---------------------------------
        K_pinhole = self.loader_query.calibration["PINHOLE"]["K"]

        for name, pose in self.get_poses_query().items():
            w_T_wc = pose["w_T_wc"]
            R_wc   = w_T_wc[:3, :3]
            C_wc   = w_T_wc[:3, 3]

            if np.linalg.norm(C_wc) > 20:
                print(f"[!] Camera {name} is too far from the origin, skipping.")
                continue

            viz_3d.plot_camera(fig,
                R_wc, C_wc,
                K=K_pinhole,
                name=name,
                text=name,
                color="rgba(0,255,0,0.6)")

        # ---------- 5. layout + show ----------------------------------------
        fig.update_layout(scene_aspectmode="data",
                        height=800,
                        title=f"PLY map + localised queries)")
        fig.show()

    def _generate_dummy_transformations_files(self, out_path: str) -> None:
        
        transform = np.eye(4, dtype=np.float32)  # 4x4 identity matrix
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for _ in range(7):
                f.write("dummy\n")
            for row in transform:
                f.write(" ".join(f"{x:.6f}" for x in row) + "\n")

    def vis_2d_inloc(self, top_n_frames: int = 5):
        visualization.visualize_loc(results=self.results_dir, image_dir=self.images, n=top_n_frames, top_k_db=1, seed=2)
        import matplotlib.pyplot as plt
        plt.show()

    


if __name__ == "__main__":

    # Example usage
    base_path = Path(f"/data/ikea_recordings")
    rec_location = "office_2"

    leica_data = LeicaData(base_path,rec_loc=rec_location, initial_setup="001")

    data_indexer = RecordingIndex(
        os.path.join(str(base_path), "raw") 
    )

    # queries_at_loc = data_indexer.query(
    #     location=rec_location, 
    #     interaction="gripper", 
    #     recorder="aria_gripper*"
    # )

    queries_at_loc = data_indexer.query(
        location=rec_location, 
        interaction="gripper", 
        recorder="gripper*"
    )

    for loc, inter, rec, ii, path in queries_at_loc:
        print(f"Found recorder: {rec} at {path}")

        rec_type = inter
        rec_module = rec
        interaction_indices = ii

        data = GripperData(base_path, rec_location, rec_type, rec_module, interaction_indices, data_indexer)


    spatial_registrator = SpatialRegistrator(loader_map=leica_data, loader_query=data)
    # spatial_registrator.visual_registration_inloc(force=False)
    # spatial_registrator.vis_2d_inloc()
    # spatial_registrator.visual_registration_viz_ply()
    # spatial_registrator.pcd_to_pcd_registration(vis=True, force=True)
    # spatial_registrator.align_and_optimize_orbslam_poses()
    spatial_registrator.visualize_gripper_trajectory_aligned(traj=["gripper"], mode="point", pcd_sampling="downsampled")



