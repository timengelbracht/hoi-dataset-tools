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
from PIL import Image

from pathlib import Path
import os

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
            aria_data.request_mps_all_devices(force=False)
            aria_data.extract_vrs(undistort=True)
            aria_data.extract_mps()
            aria_data.extract_mps_multi(force=False)
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
            spatial_registrator.compute_transform_world_aria()
            spatial_registrator.apply_transform_world_aria()
            # spatial_registrator.visualize_aria_trajectory_aligned(stride=100, mode="point", traj=["aria"], pcd_sampling="downsampled")
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
            # TODO apply trafo to all poses/csv
            a = 2

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
                spatial_registrator.visual_registration_viz_ply()
                spatial_registrator.visualize_aria_trajectory_aligned(stride=100, mode="point", traj=["aria"], pcd_sampling="downsampled")
            elif "iphone" in rec:
                iphone_data = IPhoneData(base_path, 
                                        rec_loc=rec_location, 
                                        rec_type=rec_type, 
                                        rec_module=rec_module, 
                                        interaction_indices=interaction_indices, 
                                        data_indexer=data_indexer)
                spatial_registrator = SpatialRegistrator(loader_map=leica_data, loader_query=iphone_data)
                spatial_registrator.visual_registration_viz_ply()

    a = 2

def run_pipeline_gripper_recording(interaction_index: str, 
                               rec_location: str, 
                               base_path: Path, 
                               leica_data: LeicaData,
                               data_indexer: RecordingIndex,
                               color: str,
                               visualize: bool = False):

    ################################################3
    # UMI Recording Data Extraction for single location and interaction index
    ################################################3

    rec_type = "gripper"

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
            aria_data.request_mps_all_devices(force=False)
            aria_data.extract_vrs(undistort=True)
            aria_data.extract_mps()
            aria_data.extract_mps_multi(force=False)
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
            spatial_registrator.compute_transform_world_aria()
            spatial_registrator.apply_transform_world_aria()
            # spatial_registrator.visualize_aria_trajectory_aligned(stride=100, mode="point", traj=["aria"], pcd_sampling="downsampled")
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
            # TODO apply trafo to all poses/csv
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
                spatial_registrator.visual_registration_viz_ply()
                spatial_registrator.visualize_aria_trajectory_aligned(stride=100, mode="point", traj=["aria"], pcd_sampling="downsampled")
            elif "iphone" in rec:
                iphone_data = IPhoneData(base_path, 
                                        rec_loc=rec_location, 
                                        rec_type=rec_type, 
                                        rec_module=rec_module, 
                                        interaction_indices=interaction_indices, 
                                        data_indexer=data_indexer)
                spatial_registrator = SpatialRegistrator(loader_map=leica_data, loader_query=iphone_data)
                spatial_registrator.visual_registration_viz_ply()

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
            aria_data.request_mps_all_devices(force=False)
            aria_data.extract_vrs(undistort=True)
            aria_data.extract_mps()
            aria_data.extract_mps_multi(force=False)
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
            spatial_registrator.compute_transform_world_aria()
            spatial_registrator.apply_transform_world_aria()
            #spatial_registrator.visualize_aria_trajectory_aligned(stride=100, mode="point", traj=["aria"], pcd_sampling="downsampled")
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
            # TODO apply trafo to all poses/csv
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
                spatial_registrator.visual_registration_viz_ply()
                spatial_registrator.visualize_aria_trajectory_aligned(stride=100, mode="point", traj=["aria"], pcd_sampling="downsampled")
            elif "iphone" in rec:
                iphone_data = IPhoneData(base_path, 
                                        rec_loc=rec_location, 
                                        rec_type=rec_type, 
                                        rec_module=rec_module, 
                                        interaction_indices=interaction_indices, 
                                        data_indexer=data_indexer)
                spatial_registrator = SpatialRegistrator(loader_map=leica_data, loader_query=iphone_data)
                spatial_registrator.visual_registration_viz_ply()


def run_pipeline_hand_recording(interaction_index: str, 
                               rec_location: str, 
                               base_path: Path, 
                               data_indexer: RecordingIndex):

    ################################################3
    # UMI Recording Data Extraction for single location and interaction index
    ################################################3

    rec_type = "hand"

    queries_at_loc = data_indexer.query(
        location=rec_location, 
        interaction=rec_type, 
        recorder="aria*",
        interaction_index=interaction_index
    )

    # post time alignment 
    for loc, inter, rec, ii, path in queries_at_loc:

        rec_type = inter
        rec_module = rec
        interaction_indices = ii

        aria_data = AriaData(base_path, 
                            rec_loc=rec_location, 
                            rec_type=rec_type, 
                            rec_module=rec_module, 
                            interaction_indices=interaction_indices, 
                            data_indexer=data_indexer)
        dir = aria_data.extraction_path / aria_data.label_rgb.strip("/")
        convert_png_to_jpg_inplace(str(dir), quality=95)





def convert_png_to_jpg_inplace(directory: str, quality: int = 95):
    """
    Converts all .png images in the given directory (recursively) to .jpg,
    overwriting in place (i.e., removes the .png after successful conversion).

    Args:
        directory (str): Path to the directory to process.
        quality (int): JPEG quality (default: 95, high quality, low loss).
    """
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"'{directory}' is not a valid directory")

    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".png"):
                png_path = os.path.join(root, file)
                jpg_path = os.path.splitext(png_path)[0] + ".jpg"
                try:
                    with Image.open(png_path) as img:
                        rgb_img = img.convert("RGB")  # drop alpha safely
                        rgb_img.save(jpg_path, "JPEG", quality=quality)
                    os.remove(png_path)
                    count += 1
                    print(f"Converted: {png_path} → {jpg_path}")
                except Exception as e:
                    print(f"⚠️ Failed to convert {png_path}: {e}")
    print(f"✅ Done — converted {count} PNGs in '{directory}'.")


if __name__ == "__main__":
    ################################################3
    # recording location
    ################################################3
    rec_location = "bedroom_6"
    base_path = Path(f"/data/ikea_recordings")
    data_indexer = RecordingIndex(
        os.path.join(str(base_path), "extracted") 
    )
    path_docker_root_odometry = Path("/exchange/hoi-dataset-tools/data_processing/docker/odometry")
    # interaction_index = "1-6"
    color = "yellow"
    visualize = False


    for interaction_index in ["1-5"]:

        ################################################3
        # Wrist Recording Data Extraction for single location and interaction index
        ################################################3
        # run_pipeline_wrist_recording(
        #     interaction_index=interaction_index, 
        #     rec_location=rec_location,
        #     base_path=base_path,
        #     leica_data=leica_data,
        #     data_indexer=data_indexer,
        #     visualize=visualize
        # )


        ################################################3
        # UMI Recording Data Extraction for single location and interaction index
        ################################################3
        # run_pipeline_umi_recording(
        #     interaction_index=interaction_index, 
        #     rec_location=rec_location,
        #     base_path=base_path,
        #     path_docker_root_odometry=path_docker_root_odometry,
        #     leica_data=leica_data,
        #     color=color,
        #     data_indexer=data_indexer,
        #     visualize=visualize
        # )


        ################################################3
        # hand Recording Data Extraction for single location and interaction index
        ################################################3
        run_pipeline_hand_recording(
           interaction_index=interaction_index, 
           rec_location=rec_location,
           base_path=base_path,
           data_indexer=data_indexer
           
        )

        ###############################################3
        #Gripper Recording Data Extraction for single location and interaction index
        ###############################################3
        # run_pipeline_gripper_recording(
        #     interaction_index=interaction_index, 
        #     rec_location=rec_location,
        #     base_path=base_path,
        #     leica_data=leica_data,
        #     data_indexer=data_indexer,
        #     color=color,
        #     visualize=visualize
        # )