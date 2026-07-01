from pathlib import Path
from hoi.data_tools.data_loader_aria import AriaData
from hoi.data_tools.data_loader_iphone import IPhoneData
from hoi.data_tools.data_loader_gripper import GripperData
from hoi.data_tools.data_indexer import RecordingIndex
from hoi.data_tools.time_align_extracted_single_recording import Datasyncer

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

import matplotlib
# graphixal backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_pipeline_gripper_recording(interaction_index: str, 
                               rec_location: str, 
                               base_path: Path, 
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
            aria_data.extract_vrs(undistort=True)
            aria_data.extract_mps()

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

            aria_data.anonymize_extracted_frames()

        
def evaluate_forces_gripper_recording(gripper_data: GripperData, gt_forces: dict):

    # -------------------
    # Load force/torque
    # -------------------
    ft_data = gripper_data.get_force_torque_measurements()
    ft_data = ft_data[[
        "timestamp",
        "wrench_ext.torque.x_filt",
        "wrench_ext.torque.y_filt",
        "wrench_ext.torque.z_filt",
        "wrench_ext.force.x_filt",
        "wrench_ext.force.y_filt",
        "wrench_ext.force.z_filt",
        "wrench.force.x",
        "wrench.force.y",
        "wrench.force.z",
    ]].copy()



    # get force magnitude overall
    ft_data["absmag"] = np.linalg.norm(
        ft_data[[
        "wrench_ext.force.x_filt",
        "wrench_ext.force.y_filt",
        "wrench_ext.force.z_filt",
        ]].values,
        axis=1
    ) 


    #ft_data["absmag"] = np.abs(ft_data["wrench_ext.force.z_filt"].values)

    # get minimal force error window of 1 sec

    gt_force = gt_forces["m_milk_bottle"] * 9.81  # N
    error = np.abs(ft_data["absmag"] - gt_force)

    timestamps = ft_data["timestamp"].values
    window_ns = int(1.0*1e9)  # 1 second

    min_error = np.inf
    best_start = None
    best_end = None

    j = 0
    N = len(timestamps)

    for i in range(N):
        start_t = timestamps[i]
        while j < N and timestamps[j] < start_t + window_ns:
            j += 1

        if j <= i + 1:
            continue  # skip tiny windows

        window_error = np.nanmean(error[i:j])

        if window_error < min_error:
            min_error = window_error
            best_start = start_t
            best_end = timestamps[j - 1]

    best_window_data = ft_data[
        (timestamps >= best_start) & (timestamps <= best_end)
    ]

    # compute stats in best window
    mean_force = np.nanmean(best_window_data["absmag"].values)
    std_force = np.nanstd(best_window_data["absmag"].values)




    #ft_data_world = ft_data["absmag"]
    #ft_data_world = error

    # plot manually
    fig = plt.figure(figsize=(8,4))
    plt.plot(ft_data["timestamp"], ft_data["absmag"], label="Estimated force magnitude")
    plt.xlabel("Time [ns]")
    plt.ylabel("Force magnitude [N]")
    plt.title("Estimated force magnitude in world frame")
    plt.grid(True, alpha=0.3)

    # add gt force line
    plt.axhline(y=gt_forces['m_milk_bottle']*9.81, color='r', linestyle='--', label="Ground truth force")
    plt.axhline(y=gt_forces['m_wine_bottle']*9.81, color='g', linestyle='--', label="GT force wine bottle")  
    plt.axhline(y=gt_forces['m_mug']*9.81, color='b', linestyle='--', label="GT force mug")

    plt.legend()
    out_path = gripper_data.extraction_path / "force_magnitude_world.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()











    
    a = 2
    

def plot_world_forces_with_norm(ft_data, title="World-frame forces",
                                out_path: Path = None):
    t = ft_data["timestamp"].values
    t = (t - t[0]) * 1e-9

    F = ft_data[[
        "fx_world",
        "fy_world",
        "fz_world"
    ]].values

    F_norm = np.linalg.norm(F, axis=1)

    fig, ax1 = plt.subplots(figsize=(10, 4))

    ax1.plot(t, F[:, 0], label="Fx")
    ax1.plot(t, F[:, 1], label="Fy")
    ax1.plot(t, F[:, 2], label="Fz")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Force components [N]")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(t, F_norm, linestyle="--", label="|F|")
    ax2.set_ylabel("Force magnitude [N]")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title(title)
    plt.tight_layout()

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    

if __name__ == "__main__":
    import os
    ################################################3
    # recording location
    ################################################3
    rec_location = "rebuttal_1"
    base_path = Path(f"/data/ikea_recordings")
    data_indexer = RecordingIndex(
        os.path.join(str(base_path), "raw") 
    )
    path_docker_root_odometry = Path("/exchange/hoi-dataset-tools/data_processing/docker/odometry")
    # interaction_index = "1-6"
    color = "blue"
    visualize = True

    #for interaction_index in ["66"]:
    #    run_pipeline_gripper_recording(
     #       interaction_index=interaction_index,
     #       rec_location=rec_location,
     #       base_path=base_path,
     #       data_indexer=data_indexer,
     #       color=color,
     #       visualize=visualize,
     #       rec_type="gripper"
      #  )

    gt_forces = {
        "m_wine_bottle": 1.187,
        "m_milk_bottle": 0.729,
        "m_mug": 0.344
    }

    gripper_data = GripperData(base_path, 
                               rec_loc=rec_location, 
                               rec_type='gripper', 
                               rec_module='gripper', 
                               color=color,
                               interaction_indices='666', 
                               data_indexer=data_indexer) 
    
    evaluate_forces_gripper_recording(
        gripper_data=gripper_data,
        gt_forces=gt_forces
    )