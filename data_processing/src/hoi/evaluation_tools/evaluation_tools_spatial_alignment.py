from hoi.data_tools.data_loader_gripper import GripperData
from hoi.data_tools.data_loader_aria import AriaData
from hoi.data_tools.data_indexer import RecordingIndex
from hoi.data_tools.utils_calibration import load_hand_eye_calibration_asl

from math import ceil
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg") 

# trajectory utils imports
from trajectory_utils.trajectory import Trajectory
from trajectory_utils.eval import compare_trajectories
import numpy as np
import torch
import roma

import os
from pathlib import Path
import pandas as pd

def export_aria_traj_to_csv_for_asl_hand_eye_calibration(df_aria: pd.DataFrame, out_file: str | Path):

    df_ad = df_aria[['timestamp_aria','aria_position_x','aria_position_y','aria_position_z',
                   'aria_orientation_x','aria_orientation_y','aria_orientation_z','aria_orientation_w']].copy()
    df_ad.columns = ['t','x','y','z','q_x','q_y','q_z','q_w']
    df_ad['t'] = df_ad['t'] / 1e9  # if in ns
    df_ad.to_csv(out_file, index=False, header=False)

def export_mocap_traj_to_csv_for_asl_hand_eye_calibration(df_mocap: pd.DataFrame, out_file: str | Path):
    df_mb = df_mocap[['timestamp_mocap','mocap_position_x','mocap_position_y','mocap_position_z',
                    'mocap_orientation_x','mocap_orientation_y','mocap_orientation_z','mocap_orientation_w']].copy()
    df_mb.columns = ['t','x','y','z','q_x','q_y','q_z','q_w']
    df_mb['t'] = df_mb['t'] / 1e9  # if in ns
    df_mb.to_csv(out_file, index=False, header=False)

def trajectory_to_asl_df(traj, ts_scale=1.0):
    """
    Convert a Trajectory to ASL hand–eye CSV format.
    Columns: ['t','x','y','z','q_x','q_y','q_z','q_w']

    Args:
        traj: Trajectory (positions Nx3, orientations Nx4 as [qx,qy,qz,qw], timesteps float64)
        ts_scale: multiply timesteps by this to get **seconds**
                  e.g. if timesteps are in ns, use ts_scale=1e-9

    Returns:
        pandas.DataFrame with ASL headers.
    """
    # tensors -> numpy
    p = traj.positions.detach().cpu().numpy()
    q = traj.orientations.detach().cpu().numpy()  # assumed [qx,qy,qz,qw]
    t = traj.timesteps.detach().cpu().numpy().astype(np.float64) * ts_scale

    df = pd.DataFrame({
        "t":   t,
        "x":   p[:, 0],
        "y":   p[:, 1],
        "z":   p[:, 2],
        "q_x": q[:, 0],
        "q_y": q[:, 1],
        "q_z": q[:, 2],
        "q_w": q[:, 3],
    })
    return df

def trajectory_from_dataframe(
    df,
    child_frame: str,
    parent_frame: str,
    timestamp_col: str,
    position_cols=None,          # e.g. ["x","y","z"] (defaults set below)
    orientation_cols=None,       # e.g. ["qx","qy","qz","qw"]
    rotmat_cols=None,            # optional flattened rotation matrix cols (9 or 12 values). If 12, last 3 are translation.
    timestamp_unit: float = 1e9  # ns -> s by default
) -> Trajectory:
    """
    Build a Trajectory from a DataFrame.
    Supported inputs (pick ONE style):
      1) position_cols + orientation_cols (xyz + qx qy qz qw)
      2) position_cols only (orientations set to identity; warning printed)
      3) rotmat_cols (9 or 12 floats). If 12, last 3 are translation; if 9, you must also pass position_cols.
    """

    # Sensible defaults matching your earlier rename
    if position_cols is None and orientation_cols is None and rotmat_cols is None:
        position_cols = ["aria_position_x", "aria_position_y", "aria_position_z"]
        orientation_cols = ["aria_orientation_x", "aria_orientation_y", "aria_orientation_z", "aria_orientation_w"]

    # Timestamps (float64, seconds)
    stamps = torch.as_tensor(df[timestamp_col].to_numpy(), dtype=torch.float64) / timestamp_unit

    # Prepare outputs
    N = len(df)
    positions = torch.zeros((N, 3), dtype=torch.float32)
    orientations = torch.zeros((N, 4), dtype=torch.float32)
    orientations[:, -1] = 1.0  # identity by default

    # Case 3: rotation matrix provided
    if rotmat_cols is not None:
        rotvals = torch.as_tensor(df[rotmat_cols].to_numpy(), dtype=torch.float32)  # (N, 9 or 12)
        if rotvals.shape[1] == 12:
            R = rotvals[:, :9].reshape(-1, 3, 3)
            t = rotvals[:, 9:12]
            positions = t
        elif rotvals.shape[1] == 9:
            if not position_cols:
                raise ValueError("rotmat_cols has 9 values; please also provide position_cols.")
            R = rotvals.reshape(-1, 3, 3)
            positions = torch.as_tensor(df[position_cols].to_numpy(), dtype=torch.float32)
        else:
            raise ValueError(f"rotmat_cols must have 9 or 12 columns, got {rotvals.shape[1]}.")

        orientations = roma.rotmat_to_unitquat(R)

    # Case 1/2: position (+/- orientation) provided
    else:
        if position_cols:
            positions = torch.as_tensor(df[position_cols].to_numpy(), dtype=torch.float32)
        if orientation_cols:
            quats = torch.as_tensor(df[orientation_cols].to_numpy(), dtype=torch.float32)
            if quats.shape[1] != 4:
                raise ValueError(f"orientation_cols must have 4 columns (qx,qy,qz,qw), got {quats.shape[1]}.")
            orientations = quats
        elif position_cols and orientation_cols is None:
            print(f"[WARNING] No orientation provided, using identity quaternion for {parent_frame}->{child_frame}")

    return Trajectory(positions, orientations, stamps, parent_frame, child_frame)

def show_trajectories_side_by_side(
    trajectories,
    titles=None,
    ncols=2,
    colors=None,                 # e.g. ["#2C4A7A","#4B6EA8","#E6B450","#1C1C1C"]
    time_as_color=False,
    show_frames=False,
    frame_scale=0.05,
    figure_size=(500, 400),      # (width_per_col, height_per_row)
    show=True
):
    n = len(trajectories)
    rows = ceil(n / ncols)
    specs = [[{"type": "scene"} for _ in range(ncols)] for _ in range(rows)]
    fig = make_subplots(rows=rows, cols=ncols, specs=specs,
                        subplot_titles=(titles or [
                            f"{t._parent_frame} → {t._child_frame}" for t in trajectories
                        ]))

    # muted IKEA-inspired defaults if no colors passed
    default_palette = ["#2C4A7A", "#4B6EA8", "#E6B450", "#1C1C1C", "#D6D8DC"]
    palette = colors or default_palette

    for i, traj in enumerate(trajectories):
        r = i // ncols + 1
        c = i % ncols + 1
        line_color = palette[i % len(palette)]
        traj.show(
            fig=fig,
            show=False,
            line_color=line_color,
            show_frames=show_frames,
            frame_scale=frame_scale,
            time_as_color=time_as_color,
            trace_kwargs={"row": r, "col": c},
        )

    # equal aspect for each 3D scene
    scene_updates = {("scene" if idx == 0 else f"scene{idx+1}"): {"aspectmode": "data"}
                     for idx in range(rows * ncols)}
    fig.update_layout(**scene_updates,
                      width=figure_size[0] * ncols,
                      height=figure_size[1] * rows,
                      margin=dict(l=0, r=0, t=40, b=0))

    if show:
        fig.show()
    return fig


def evaluate_spatial_alignment_for_single_aria(
    aria_data: AriaData,
    gripper_data: GripperData,):
    
    # get aria trajectory
    aria_trajectory = aria_data.get_closed_loop_trajectory_aligned()
    # get mocap trajectory
    mocap_trajectory = gripper_data.get_mocap_trajectory(aria_target=aria_data.rec_module)

    # make traj names consistent and drop unnecessary columns
    aria_trajectory = aria_trajectory.rename(columns={
        "tx_world_device": "aria_position_x",
        "ty_world_device": "aria_position_y",
        "tz_world_device": "aria_position_z",
        "qx_world_device": "aria_orientation_x",
        "qy_world_device": "aria_orientation_y",
        "qz_world_device": "aria_orientation_z",
        "qw_world_device": "aria_orientation_w",
        "timestamp": "timestamp_aria"
    })[[
        "timestamp_aria",
        "aria_position_x", "aria_position_y", "aria_position_z",
        "aria_orientation_x", "aria_orientation_y", "aria_orientation_z", "aria_orientation_w"
    ]]

    mocap_trajectory = mocap_trajectory.rename(columns={
        "pose.position.x": "mocap_position_x",
        "pose.position.y": "mocap_position_y",
        "pose.position.z": "mocap_position_z",
        "pose.orientation.x": "mocap_orientation_x",
        "pose.orientation.y": "mocap_orientation_y",
        "pose.orientation.z": "mocap_orientation_z",
        "pose.orientation.w": "mocap_orientation_w",
        "timestamp": "timestamp_mocap"
    })[[
        "timestamp_mocap",
        "mocap_position_x", "mocap_position_y", "mocap_position_z",
        "mocap_orientation_x", "mocap_orientation_y", "mocap_orientation_z", "mocap_orientation_w"
    ]]


    out_dir = Path("/data/ikea_recordings/raw/calib/hand_eye")
    out_dir.mkdir(parents=True, exist_ok=True)

    # get extrinsic calibration
    t_b_d, quat_b_d = load_hand_eye_calibration_asl(
        calib_path=out_dir / f"{aria_data.rec_module}_calib.json"
    )

    # to pytorch tensors (3,) and (4,)
    t_b_d = torch.as_tensor(t_b_d, dtype=torch.float32)
    quat_b_d = torch.as_tensor(quat_b_d, dtype=torch.float32)
    
    aria_traj = trajectory_from_dataframe(
        aria_trajectory,
        child_frame="device",
        parent_frame="world",
        timestamp_col="timestamp_aria",
        position_cols=["aria_position_x", "aria_position_y", "aria_position_z"],
        orientation_cols=["aria_orientation_x", "aria_orientation_y", "aria_orientation_z", "aria_orientation_w"]
    )

    mocap_traj = trajectory_from_dataframe(
        mocap_trajectory,
        child_frame="device",
        parent_frame="world",
        timestamp_col="timestamp_mocap",
        position_cols=["mocap_position_x", "mocap_position_y", "mocap_position_z"],
        orientation_cols=["mocap_orientation_x", "mocap_orientation_y", "mocap_orientation_z", "mocap_orientation_w"]
    )

    # mocap_trajectory_aligned, aria_trajectory_aligned, infos = mocap_traj.clone().temporal_align(aria_traj, return_infos=True)
    aria_start_ts = aria_traj._timesteps[0].item()
    mocap_start_ts = mocap_traj._timesteps[0].item()
    aria_duration = aria_traj.duration
    mocap_duration = mocap_traj.duration
    start_ts = max(aria_start_ts, mocap_start_ts)
    end_ts = min(aria_start_ts+aria_duration, mocap_start_ts+mocap_duration)
    traj_mocap_body = mocap_traj.resample(start_time = start_ts, end_time = end_ts, frequency = 30)
    traj_aria_device = aria_traj.resample(start_time = start_ts, end_time = end_ts, frequency = 30)

    out_dir = Path("/data/ikea_recordings/raw/calib/hand_eye")
    # out_dir.mkdir(parents=True, exist_ok=True)
    trajectory_to_asl_df(aria_traj, ts_scale=1.0).to_csv(out_dir / f"{aria_data.rec_module}_traj_time_aligned.csv", index=False, header=False)
    trajectory_to_asl_df(mocap_traj, ts_scale=1.0).to_csv(out_dir / f"mocap_{aria_data.rec_module}_traj_time_aligned.csv", index=False, header=False)

    traj_device_aria = traj_aria_device.clone().inverse()
    traj_body_aria = traj_device_aria.clone().transform(t_b_d, quat_b_d)
    traj_aria_body = traj_body_aria.clone().inverse()

    traj_aria_body = traj_aria_body.spatial_align(traj_mocap_body)

    compare_trajectories(traj_mocap_body, traj_aria_body, headless=True)





    a = 2

if __name__ == "__main__":


    rec_location = "mlhall_1"
    base_path = Path(f"/data/ikea_recordings")
    data_indexer = RecordingIndex(
        os.path.join(str(base_path), "extracted") 
    )

    # always "gripper", since the gripper module is always used for mocap stuff.
    # gripper is always recorded for mocap as the mocap run via the gripper module ros core
    rec_type = "wrist"

    # interaction indices "666" for mocap tests
    interaction_indices = "8-13"
    

    aria_gripper_data = AriaData(base_path=base_path, 
                        rec_loc=rec_location, 
                        rec_type=rec_type, 
                        rec_module="aria_wrist", 
                        interaction_indices=interaction_indices,
                        data_indexer=data_indexer)
    
    aria_human_data = AriaData(base_path=base_path, 
                        rec_loc=rec_location, 
                        rec_type=rec_type, 
                        rec_module="aria_human", 
                        interaction_indices=interaction_indices,
                        data_indexer=data_indexer)
    
    gripper_data = GripperData(base_path, 
                    rec_loc=rec_location, 
                    rec_type=rec_type, 
                    rec_module="gripper", 
                    interaction_indices=interaction_indices,
                    data_indexer=data_indexer,
                    color="blue")
    
    evaluate_spatial_alignment_for_single_aria(
        aria_data=aria_gripper_data,
        gripper_data=gripper_data)
    

    
    a = 2