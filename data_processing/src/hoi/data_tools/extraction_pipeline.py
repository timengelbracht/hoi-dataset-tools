from hoi.data_tools.data_loader_aria import AriaData
from hoi.data_tools.data_loader_umi import UmiData
from hoi.data_tools.data_loader_gripper import GripperData
from hoi.data_tools.data_loader_iphone import IPhoneData
from hoi.data_tools.data_loader_leica import LeicaData
from hoi.data_tools.data_indexer import RecordingIndex
from hoi.data_tools.time_align_extracted_single_recording import Datasyncer
from hoi.data_tools.spatial_registrator import SpatialRegistrator
from hoi.data_tools.mps_request import MPSClient
from pathlib import Path
from typing import Dict, List, Sequence

from pathlib import Path
import os
import argparse
import yaml

def run_pipeline_umi_recording(interaction_index: str, 
                               rec_location: str, 
                               base_path: Path, 
                               path_docker_root_odometry: Path,
                               leica_data: LeicaData,
                               color: str,
                               data_indexer: RecordingIndex,
                               visualize: bool = False):

    ################################################3
    # UMI Recording Data Extraction for single location and interaction index
    ################################################3

    rec_type = "umi"

    queries_at_loc = data_indexer.query(
        location=rec_location, 
        interaction=rec_type, 
        recorder=None,
        interaction_index=interaction_index
    )
    # extract all recording module for the give UMI recording
    for loc, inter, rec, ii, path in queries_at_loc:
        print(f"Found recorder: {rec} at {path}")

        rec_type = inter
        rec_module = rec
        interaction_indices = ii

        if rec == "umi_gripper":
            umi_data = UmiData(base_path, 
                               rec_loc=rec_location, 
                               rec_type=rec_type, 
                               rec_module=rec_module, 
                               interaction_indices=interaction_indices, 
                               color=color,
                               data_indexer=data_indexer)            
            umi_data.extract_mp4()
        elif "aria" in rec:
            aria_data = AriaData(base_path, rec_location, rec_type, rec_module, interaction_indices, data_indexer)
            aria_data.request_mps(force=False)
            # aria_data.request_mps_all_devices(force=False)
            aria_data.extract_vrs(undistort=True)
            aria_data.extract_mps()
            #aria_data.extract_mps_multi(force=False)
        elif "iphone" in rec:
            iphone_data = IPhoneData(base_path, rec_location, rec_type, rec_module, interaction_indices, data_indexer)
            iphone_data.extract_rgbd()
            iphone_data.extract_poses()

    # time alignment of all extracted data
    data_syncer = Datasyncer(
        base_path=base_path,
        rec_location=rec_location,
        rec_type=rec_type,
        interaction_indices=interaction_index,
        data_indexer=data_indexer
    )
    data_syncer.register_all_data_loaders()
    data_syncer.apply_time_deltas_to_all_data_streams()
    data_syncer.apply_time_window_cropping_to_all_data_streams() 

    # post time alignment 
    # extract keyframes and run orbslam
    # spatial registration
    for loc, inter, rec, ii, path in queries_at_loc:

        rec_type = inter
        rec_module = rec
        interaction_indices = ii

        if rec == "umi_gripper":
            umi_data = UmiData(base_path, 
                               rec_loc=rec_location, 
                               rec_type=rec_type, 
                               rec_module=rec_module, 
                               interaction_indices=interaction_indices,
                               color=color,
                               data_indexer=data_indexer)
            umi_data.extract_keyframes(stride=1, n_keyframes=600)
            umi_data.extract_euroc_format_for_orbslam(mask=True, image_scale=2, stride=1)
            umi_data.run_orbslam(docker_root_path=path_docker_root_odometry)
            spatial_registrator = SpatialRegistrator(loader_map=leica_data, loader_query=umi_data)
            spatial_registrator.visual_registration_inloc(force=False)
            # spatial_registrator.visual_registration_viz_ply()
            spatial_registrator.align_and_optimize_orbslam_poses()  
            # spatial_registrator.visualize_umi_trajectory_aligned(stride=10, mode="point", pcd_sampling="downsampled", color=[1,0,0]) 
            umi_data.anonymize_rgb()
        elif "aria" in rec:
            aria_data = AriaData(base_path, 
                                rec_loc=rec_location, 
                                rec_type=rec_type, 
                                rec_module=rec_module, 
                                interaction_indices=interaction_indices, 
                                data_indexer=data_indexer)
            aria_data.extract_mono_depth(force=False)
            aria_data.extract_keyframes(stride=2, n_keyframes=20)
            spatial_registrator = SpatialRegistrator(loader_map=leica_data, loader_query=aria_data)
            spatial_registrator.visual_registration_inloc(force=False)
            # spatial_registrator.visual_registration_viz_ply()
            ret = spatial_registrator.compute_transform_world_aria()
            if ret is not None:
                spatial_registrator.apply_transform_world_aria()
            # spatial_registrator.visualize_aria_trajectory_aligned(stride=100, mode="point", traj=["aria"], pcd_sampling="downsampled")
            aria_data.anonymize_extracted_frames()
        elif "iphone" in rec:
            iphone_data = IPhoneData(base_path, 
                                    rec_loc=rec_location, 
                                    rec_type=rec_type, 
                                    rec_module=rec_module, 
                                    interaction_indices=interaction_indices, 
                                    data_indexer=data_indexer)
            iphone_data.extract_keyframes()
            spatial_registrator = SpatialRegistrator(loader_map=leica_data, loader_query=iphone_data)
            spatial_registrator.visual_registration_inloc(force=False)
            # spatial_registrator.visual_registration_viz_ply()
            ret = spatial_registrator.compute_transform_world_iphone()
            if ret is not None:
                spatial_registrator.apply_transform_world_iphone()
            iphone_data.anonymize_extracted_frames()

        # post registration
        # split intearctions
    data_syncer.get_interaction_time_windows_from_qr_codes()

    # vis
    if visualize:
        for loc, inter, rec, ii, path in queries_at_loc:

            rec_type = inter
            rec_module = rec
            interaction_indices = ii

            if rec == "umi_gripper":
                umi_data = UmiData(base_path, 
                                rec_loc=rec_location, 
                                rec_type=rec_type, 
                                rec_module=rec_module, 
                                interaction_indices=interaction_indices,
                                color=color,
                                data_indexer=data_indexer)
                spatial_registrator = SpatialRegistrator(loader_map=leica_data, loader_query=umi_data)
                spatial_registrator.visual_registration_viz_ply()
                spatial_registrator.visualize_umi_trajectory_aligned(stride=10, mode="point", pcd_sampling="downsampled", color=[1,0,0])  
            elif "aria" in rec:
                aria_data = AriaData(base_path, 
                                    rec_loc=rec_location, 
                                    rec_type=rec_type, 
                                    rec_module=rec_module, 
                                    interaction_indices=interaction_indices, 
                                    data_indexer=data_indexer)
                spatial_registrator = SpatialRegistrator(loader_map=leica_data, loader_query=aria_data)
                # spatial_registrator.visual_registration_viz_ply()
                spatial_registrator.visualize_aria_trajectory_aligned(stride=100, mode="point", traj=["aria"], pcd_sampling="downsampled")
            elif "iphone" in rec:
                iphone_data = IPhoneData(base_path, 
                                        rec_loc=rec_location, 
                                        rec_type=rec_type, 
                                        rec_module=rec_module, 
                                        interaction_indices=interaction_indices, 
                                        data_indexer=data_indexer)
                spatial_registrator = SpatialRegistrator(loader_map=leica_data, loader_query=iphone_data)
                # spatial_registrator.visual_registration_viz_ply()

    a = 2

def run_pipeline_gripper_recording(interaction_index: str, 
                               rec_location: str, 
                               base_path: Path, 
                               leica_data: LeicaData,
                               data_indexer: RecordingIndex,
                               color: str,
                               visualize: bool = False,
                               rec_type: str = "gripper"):

    ################################################3
    # UMI Recording Data Extraction for single location and interaction index
    ################################################3

    # rec_type = "gripper"

    queries_at_loc = data_indexer.query(
        location=rec_location, 
        interaction=rec_type, 
        recorder=None,
        interaction_index=interaction_index
    )
    # extract all recording module for the give UMI recording
    for loc, inter, rec, ii, path in queries_at_loc:
        print(f"Found recorder: {rec} at {path}")

        rec_type = inter
        rec_module = rec
        interaction_indices = ii

        if rec == "gripper":
            gripper_data = GripperData(base_path, 
                               rec_loc=rec_location, 
                               rec_type=rec_type, 
                               rec_module=rec_module, 
                               color=color,
                               interaction_indices=interaction_indices, 
                               data_indexer=data_indexer)            
            gripper_data.extract_bag_full()
        elif "aria" in rec:
            aria_data = AriaData(base_path, rec_location, rec_type, rec_module, interaction_indices, data_indexer)
            aria_data.request_mps(force=False)
            # aria_data.request_mps_all_devices(force=False)
            aria_data.extract_vrs(undistort=True)
            aria_data.extract_mps()
            # aria_data.extract_mps_multi(force=False)
        elif "iphone" in rec:
            iphone_data = IPhoneData(base_path, rec_location, rec_type, rec_module, interaction_indices, data_indexer)
            iphone_data.extract_rgbd()
            iphone_data.extract_poses()

    # time alignment of all extracted data
    data_syncer = Datasyncer(
        base_path=base_path,
        rec_location=rec_location,
        rec_type=rec_type,
        interaction_indices=interaction_index,
        data_indexer=data_indexer
    )
    data_syncer.register_all_data_loaders()
    data_syncer.apply_time_deltas_to_all_data_streams()
    data_syncer.apply_time_window_cropping_to_all_data_streams() 

    # post time alignment 
    for loc, inter, rec, ii, path in queries_at_loc:

        rec_type = inter
        rec_module = rec
        interaction_indices = ii

        if rec == "gripper":
            gripper_data = GripperData(base_path, 
                               rec_loc=rec_location, 
                               rec_type=rec_type, 
                               rec_module=rec_module, 
                               interaction_indices=interaction_indices,
                               data_indexer=data_indexer)
            gripper_data.apply_force_torque_gravity_compensation(visualize=visualize)
            gripper_data.visualize_forces_in_pointcloud(leica_data, visualize=visualize)
            gripper_data.anonymize_all_zed_rgb()
        elif "aria" in rec:
            aria_data = AriaData(base_path, 
                                rec_loc=rec_location, 
                                rec_type=rec_type, 
                                rec_module=rec_module, 
                                interaction_indices=interaction_indices, 
                                data_indexer=data_indexer)
            aria_data.extract_mono_depth(force=False)
            aria_data.extract_keyframes(stride=2, n_keyframes=20)
            spatial_registrator = SpatialRegistrator(loader_map=leica_data, loader_query=aria_data)
            spatial_registrator.visual_registration_inloc(force=False)
            # spatial_registrator.vis_2d_inloc()
            # spatial_registrator.visual_registration_viz_ply()
            ret = spatial_registrator.compute_transform_world_aria()
            if ret is not None:
                spatial_registrator.apply_transform_world_aria()
            # spatial_registrator.visualize_aria_trajectory_aligned(stride=100, mode="point", traj=["aria"], pcd_sampling="downsampled")
            aria_data.anonymize_extracted_frames()
        elif "iphone" in rec:
            iphone_data = IPhoneData(base_path, 
                                    rec_loc=rec_location, 
                                    rec_type=rec_type, 
                                    rec_module=rec_module, 
                                    interaction_indices=interaction_indices, 
                                    data_indexer=data_indexer)
            iphone_data.extract_keyframes()
            spatial_registrator = SpatialRegistrator(loader_map=leica_data, loader_query=iphone_data)
            spatial_registrator.visual_registration_inloc(force=False)
            # spatial_registrator.vis_2d_inloc()
            # spatial_registrator.visual_registration_viz_ply()
            # TODO apply trafo to all poses/csv
            ret = spatial_registrator.compute_transform_world_iphone()
            if ret is not None:
                spatial_registrator.apply_transform_world_iphone()
            a = 2
            iphone_data.anonymize_extracted_frames()

    data_syncer.get_interaction_time_windows_from_qr_codes()
    


    # vis
    if visualize:
        for loc, inter, rec, ii, path in queries_at_loc:

            rec_type = inter
            rec_module = rec
            interaction_indices = ii
            if "aria" in rec:
                aria_data = AriaData(base_path, 
                                    rec_loc=rec_location, 
                                    rec_type=rec_type, 
                                    rec_module=rec_module, 
                                    interaction_indices=interaction_indices, 
                                    data_indexer=data_indexer)
                spatial_registrator = SpatialRegistrator(loader_map=leica_data, loader_query=aria_data)
                # spatial_registrator.visual_registration_viz_ply()
                spatial_registrator.visualize_aria_trajectory_aligned(stride=100, mode="point", traj=["aria"], pcd_sampling="downsampled")
            elif "iphone" in rec:
                iphone_data = IPhoneData(base_path, 
                                        rec_loc=rec_location, 
                                        rec_type=rec_type, 
                                        rec_module=rec_module, 
                                        interaction_indices=interaction_indices, 
                                        data_indexer=data_indexer)
                spatial_registrator = SpatialRegistrator(loader_map=leica_data, loader_query=iphone_data)
                # spatial_registrator.visual_registration_viz_ply()

def run_pipeline_wrist_recording(interaction_index: str, 
                               rec_location: str, 
                               base_path: Path, 
                               leica_data: LeicaData,
                               data_indexer: RecordingIndex,
                               visualize: bool = False):

    ################################################3
    # UMI Recording Data Extraction for single location and interaction index
    ################################################3

    rec_type = "wrist"

    queries_at_loc = data_indexer.query(
        location=rec_location, 
        interaction=rec_type, 
        recorder=None,
        interaction_index=interaction_index
    )
    # extract all recording module for the give UMI recording
    for loc, inter, rec, ii, path in queries_at_loc:
        print(f"Found recorder: {rec} at {path}")

        rec_type = inter
        rec_module = rec
        interaction_indices = ii
        if "aria" in rec:
            aria_data = AriaData(base_path, rec_location, rec_type, rec_module, interaction_indices, data_indexer)
            aria_data.request_mps(force=False)
            # aria_data.request_mps_all_devices(force=False)
            aria_data.extract_vrs(undistort=True)
            aria_data.extract_mps()
            #aria_data.extract_mps_multi(force=False)
        elif "iphone" in rec:
            iphone_data = IPhoneData(base_path, rec_location, rec_type, rec_module, interaction_indices, data_indexer)
            iphone_data.extract_rgbd()
            iphone_data.extract_poses()

    # time alignment of all extracted data
    data_syncer = Datasyncer(
        base_path=base_path,
        rec_location=rec_location,
        rec_type=rec_type,
        interaction_indices=interaction_index,
        data_indexer=data_indexer
    )
    data_syncer.register_all_data_loaders()
    data_syncer.apply_time_deltas_to_all_data_streams()
    data_syncer.apply_time_window_cropping_to_all_data_streams() 

    # post time alignment 
    for loc, inter, rec, ii, path in queries_at_loc:

        rec_type = inter
        rec_module = rec
        interaction_indices = ii

        if "aria" in rec:
            aria_data = AriaData(base_path, 
                                rec_loc=rec_location, 
                                rec_type=rec_type, 
                                rec_module=rec_module, 
                                interaction_indices=interaction_indices, 
                                data_indexer=data_indexer)
            aria_data.extract_mono_depth(force=False)
            aria_data.extract_keyframes(stride=2, n_keyframes=50)
            spatial_registrator = SpatialRegistrator(loader_map=leica_data, loader_query=aria_data)
            spatial_registrator.visual_registration_inloc(force=False)
            # spatial_registrator.visual_registration_viz_ply()
            ret = spatial_registrator.compute_transform_world_aria()
            if ret is not None:
                spatial_registrator.apply_transform_world_aria()
            #spatial_registrator.visualize_aria_trajectory_aligned(stride=100, mode="point", traj=["aria"], pcd_sampling="downsampled")
            aria_data.anonymize_extracted_frames()
        elif "iphone" in rec:
            iphone_data = IPhoneData(base_path, 
                                    rec_loc=rec_location, 
                                    rec_type=rec_type, 
                                    rec_module=rec_module, 
                                    interaction_indices=interaction_indices, 
                                    data_indexer=data_indexer)
            iphone_data.extract_keyframes()
            spatial_registrator = SpatialRegistrator(loader_map=leica_data, loader_query=iphone_data)
            spatial_registrator.visual_registration_inloc(force=False)
            # spatial_registrator.visual_registration_viz_ply()

            ret = spatial_registrator.compute_transform_world_iphone()
            if ret is not None:
                spatial_registrator.apply_transform_world_iphone()

            a = 2
            iphone_data.anonymize_extracted_frames()

    data_syncer.get_interaction_time_windows_from_qr_codes()

    # vis
    if visualize:
        for loc, inter, rec, ii, path in queries_at_loc:
            rec_type = inter
            rec_module = rec
            interaction_indices = ii
            if "aria" in rec:
                aria_data = AriaData(base_path, 
                                    rec_loc=rec_location, 
                                    rec_type=rec_type, 
                                    rec_module=rec_module, 
                                    interaction_indices=interaction_indices, 
                                    data_indexer=data_indexer)
                spatial_registrator = SpatialRegistrator(loader_map=leica_data, loader_query=aria_data)
                # spatial_registrator.visual_registration_viz_ply()
                spatial_registrator.visualize_aria_trajectory_aligned(stride=100, mode="point", traj=["aria"], pcd_sampling="downsampled")
            elif "iphone" in rec:
                iphone_data = IPhoneData(base_path, 
                                        rec_loc=rec_location, 
                                        rec_type=rec_type, 
                                        rec_module=rec_module, 
                                        interaction_indices=interaction_indices, 
                                        data_indexer=data_indexer)
                spatial_registrator = SpatialRegistrator(loader_map=leica_data, loader_query=iphone_data)
                # spatial_registrator.visual_registration_viz_ply()


def run_pipeline_hand_recording(interaction_index: str, 
                               rec_location: str, 
                               base_path: Path, 
                               leica_data: LeicaData,
                               data_indexer: RecordingIndex,
                               visualize: bool = False):

    ################################################3
    # UMI Recording Data Extraction for single location and interaction index
    ################################################3

    rec_type = "hand"

    queries_at_loc = data_indexer.query(
        location=rec_location, 
        interaction=rec_type, 
        recorder=None,
        interaction_index=interaction_index
    )
    # extract all recording module for the give UMI recording
    for loc, inter, rec, ii, path in queries_at_loc:
        print(f"Found recorder: {rec} at {path}")

        rec_type = inter
        rec_module = rec
        interaction_indices = ii
        if "aria" in rec:
            aria_data = AriaData(base_path, rec_location, rec_type, rec_module, interaction_indices, data_indexer)
            aria_data.request_mps(force=False)
            # aria_data.request_mps_all_devices(force=False)
            aria_data.extract_vrs(undistort=True)
            aria_data.extract_mps()
            # aria_data.extract_mps_multi(force=False)
        elif "iphone" in rec:
            iphone_data = IPhoneData(base_path, rec_location, rec_type, rec_module, interaction_indices, data_indexer)
            iphone_data.extract_rgbd()
            iphone_data.extract_poses()

    # time alignment of all extracted data
    data_syncer = Datasyncer(
        base_path=base_path,
        rec_location=rec_location,
        rec_type=rec_type,
        interaction_indices=interaction_index,
        data_indexer=data_indexer
    )
    data_syncer.register_all_data_loaders()
    data_syncer.apply_time_deltas_to_all_data_streams()
    data_syncer.apply_time_window_cropping_to_all_data_streams() 

    # post time alignment 
    for loc, inter, rec, ii, path in queries_at_loc:

        rec_type = inter
        rec_module = rec
        interaction_indices = ii

        if "aria" in rec:
            aria_data = AriaData(base_path, 
                                rec_loc=rec_location, 
                                rec_type=rec_type, 
                                rec_module=rec_module, 
                                interaction_indices=interaction_indices, 
                                data_indexer=data_indexer)
            aria_data.extract_mono_depth(force=False)
            aria_data.extract_keyframes(stride=2, n_keyframes=50)
            spatial_registrator = SpatialRegistrator(loader_map=leica_data, loader_query=aria_data)
            spatial_registrator.visual_registration_inloc(force=False)
            # spatial_registrator.visual_registration_viz_ply()
            ret = spatial_registrator.compute_transform_world_aria()
            if ret is not None:
                spatial_registrator.apply_transform_world_aria()
            # spatial_registrator.visualize_aria_trajectory_aligned(stride=100, mode="point", traj=["palm"], pcd_sampling="downsampled")
            aria_data.anonymize_extracted_frames()
        elif "iphone" in rec:
            iphone_data = IPhoneData(base_path, 
                                    rec_loc=rec_location, 
                                    rec_type=rec_type, 
                                    rec_module=rec_module, 
                                    interaction_indices=interaction_indices, 
                                    data_indexer=data_indexer)
            iphone_data.extract_keyframes()
            spatial_registrator = SpatialRegistrator(loader_map=leica_data, loader_query=iphone_data)
            spatial_registrator.visual_registration_inloc(force=False)
            # spatial_registrator.vis_2d_inloc()
            ret = spatial_registrator.compute_transform_world_iphone()
            if ret is not None:
                spatial_registrator.apply_transform_world_iphone()
            # spatial_registrator.visual_registration_viz_ply()
            # TODO apply trafo to all poses/csv
            iphone_data.anonymize_extracted_frames()
            a = 2
        
    data_syncer.get_interaction_time_windows_from_qr_codes()

    # vis
    if visualize:
        for loc, inter, rec, ii, path in queries_at_loc:

            rec_type = inter
            rec_module = rec
            interaction_indices = ii
            if "aria" in rec:
                aria_data = AriaData(base_path, 
                                    rec_loc=rec_location, 
                                    rec_type=rec_type, 
                                    rec_module=rec_module, 
                                    interaction_indices=interaction_indices, 
                                    data_indexer=data_indexer)
                spatial_registrator = SpatialRegistrator(loader_map=leica_data, loader_query=aria_data)
                # spatial_registrator.visual_registration_viz_ply()
                spatial_registrator.visualize_aria_trajectory_aligned(stride=100, mode="point", traj=["aria"], pcd_sampling="downsampled")
            elif "iphone" in rec:
                iphone_data = IPhoneData(base_path, 
                                        rec_loc=rec_location, 
                                        rec_type=rec_type, 
                                        rec_module=rec_module, 
                                        interaction_indices=interaction_indices, 
                                        data_indexer=data_indexer)
                spatial_registrator = SpatialRegistrator(loader_map=leica_data, loader_query=iphone_data)
                # spatial_registrator.visual_registration_viz_ply()
                spatial_registrator.visualize_iphone_trajectory_aligned(stride=100, mode="camera_frame", pcd_sampling="downsampled")

def request_mps_for_aria_recordings_fors_single_location(rec_location: str, 
                                                         data_indexer: RecordingIndex,
                                                         base_path: Path,
                                                         no_ui: bool = True
                                                         ):
    ################################################3
    # Request all mps for all aria recordings at single location
    ################################################3
    
    recorder = "aria*"
    interaction=None

  

    all_vrs_files = data_indexer.vrs_files(
        location=rec_location, 
        interaction=interaction, 
        recorder=recorder,
        interaction_index=None
    )

    # all_vrs_files = [all_vrs_files[2]]

    mps_client = MPSClient()
    # try:
    #     mps_client.request_multi(
    #         input_paths=all_vrs_files,
    #         output_dir=Path(base_path) / rec_location / "mps_all_devices",
    #         force=False,
    #         no_ui=True
    #     )
    # except Exception as e:
    #     print(f"[ Failed to request MPS data: {e}")

    try: # will prompt for credentials
        mps_client.request_single(
            input_path=all_vrs_files,
            features=["SLAM", "HAND_TRACKING", "EYE_GAZE"],
            no_ui=no_ui, 
            force=False,
            retry_failed=True

        )
    except Exception as e:
        print(f"[ Failed to request MPS data: {e}")

    a = 0


# ---------------------------------------------------------------------------
# Config-driven CLI entry point
# ---------------------------------------------------------------------------

STAGES_ALL = ["mps", "leica", "hand", "gripper", "wrist", "umi"]


def _resolve_docker_odometry(value: str) -> Path:
    """Resolve the odometry docker dir; relative paths are taken from the repo root."""
    p = Path(value)
    if p.is_absolute():
        return p
    # extraction_pipeline.py: data_tools -> hoi -> src -> data_processing -> <repo root>
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / p


def load_config(path: Path) -> dict:
    """Load and validate a YAML run config."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} is not a YAML mapping.")
    missing = [k for k in ("base_path", "location", "interactions") if k not in cfg]
    if missing:
        raise ValueError(f"Config {path} missing required keys: {missing}")
    return cfg


def run_extraction(cfg: dict) -> None:
    """Run the extraction pipeline for a single location from a config dict."""
    base_path = Path(cfg["base_path"])
    if not base_path.exists():
        raise FileNotFoundError(f"base_path does not exist: {base_path}")

    location = cfg["location"]
    interactions = cfg["interactions"]
    if isinstance(interactions, (str, int)):
        interactions = [interactions]
    interactions = [str(i) for i in interactions]

    color = cfg.get("color", "blue")
    if color not in ("blue", "yellow"):
        raise ValueError(f"color must be 'blue' or 'yellow', got {color!r}")
    visualize = bool(cfg.get("visualize", False))
    mps_no_ui = bool(cfg.get("mps_no_ui", True))

    stages = cfg.get("stages", STAGES_ALL)
    unknown = [s for s in stages if s not in STAGES_ALL]
    if unknown:
        raise ValueError(f"Unknown stages {unknown}; valid stages: {STAGES_ALL}")

    docker_odometry = _resolve_docker_odometry(
        cfg.get("docker_odometry", "data_processing/docker/odometry")
    )

    # Which subtree to index for recording discovery. Normally "raw" (the
    # recordings to extract). Use "extracted" when the raw data is gone and you
    # only want to (re)run downstream steps on already-extracted data.
    index_from = cfg.get("index_from", "raw")
    if index_from not in ("raw", "extracted"):
        raise ValueError(f"index_from must be 'raw' or 'extracted', got {index_from!r}")
    data_indexer = RecordingIndex(os.path.join(str(base_path), index_from))

    print(f"[extraction] location={location} interactions={interactions} "
          f"color={color} stages={stages} index_from={index_from}")

    # Aria MPS request for the whole location (once, before the interaction loop)
    if "mps" in stages:
        request_mps_for_aria_recordings_fors_single_location(
            rec_location=location,
            data_indexer=data_indexer,
            base_path=base_path,
            no_ui=mps_no_ui,
        )

    for interaction_index in interactions:
        # Leica map is needed by every recorder pipeline for spatial registration,
        # so the loader is always constructed; only its extraction is stage-gated.
        leica_data = LeicaData(base_path, location, initial_setup="001")
        if "leica" in stages:
            leica_data.extract_all_setups()
            leica_data.make_360_views_from_pano(manual_xyz=None)

        if "hand" in stages:
            run_pipeline_hand_recording(
                interaction_index=interaction_index,
                rec_location=location,
                base_path=base_path,
                leica_data=leica_data,
                data_indexer=data_indexer,
                visualize=visualize,
            )

        if "gripper" in stages:
            run_pipeline_gripper_recording(
                interaction_index=interaction_index,
                rec_location=location,
                base_path=base_path,
                leica_data=leica_data,
                data_indexer=data_indexer,
                color=color,
                visualize=visualize,
            )

        if "wrist" in stages:
            run_pipeline_wrist_recording(
                interaction_index=interaction_index,
                rec_location=location,
                base_path=base_path,
                leica_data=leica_data,
                data_indexer=data_indexer,
                visualize=visualize,
            )

        if "umi" in stages:
            run_pipeline_umi_recording(
                interaction_index=interaction_index,
                rec_location=location,
                base_path=base_path,
                path_docker_root_odometry=docker_odometry,
                leica_data=leica_data,
                color=color,
                data_indexer=data_indexer,
                visualize=visualize,
            )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Hoi! extraction pipeline for a single recording location.",
    )
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to a YAML run config (see data_processing/configs/).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run_extraction(load_config(args.config))
