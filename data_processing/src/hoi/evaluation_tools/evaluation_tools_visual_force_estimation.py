
from hoi.data_tools.data_loader_gripper import GripperData
from hoi.data_tools.data_loader_leica import LeicaData
from hoi.data_tools.data_indexer import RecordingIndex
from hoi.data_tools.utils_gripper_model import GripperModel
from pathlib import Path
import os
import json
import pandas as pd
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import open3d as o3d

def compute_linear_tool_velocity(df: pd.DataFrame, device_t_tool_device: np.ndarray,
                                 device_lin_vel_cols: list[str],
                                 device_ang_vel_cols: list[str]) -> pd.DataFrame:
    """Compute linear velocity of the gripper tool in tool frame.
    Args:
        df: DataFrame with device linear and angular velocities.
        device_t_tool_device: translation vector from device to tool frame EXPRESSED in device frame.
        device_lin_vel_cols: List of column names for device linear velocities [vx, vy, vz].
        device_ang_vel_cols: List of column names for device angular velocities [wx, wy, wz].
    Returns:
        df: DataFrame with added tool linear velocity columns [vx_tool, vy_tool, vz_tool].
    """

    for idx, row in df.iterrows():
        v_device = np.array([row[device_lin_vel_cols[0]],
                             row[device_lin_vel_cols[1]],
                             row[device_lin_vel_cols[2]]], dtype=np.float32)
        w_device = np.array([row[device_ang_vel_cols[0]],
                             row[device_ang_vel_cols[1]],
                             row[device_ang_vel_cols[2]]], dtype=np.float32)

        # v_tool = v_device + w_device x device_t_tool_device
        v_tool = v_device + np.cross(w_device, device_t_tool_device)

        df.at[idx, "tool_linear_velocity_x_device"] = v_tool[0]
        df.at[idx, "tool_linear_velocity_y_device"] = v_tool[1]
        df.at[idx, "tool_linear_velocity_z_device"] = v_tool[2]

    return df

def write_forcesight_episode(df_merged, dst_root: Path, prompt: str):
    if len(df_merged) == 0:
        print("[WARN] No frames to write for this episode.")
        return
    dst_root = Path(dst_root)
    for sub in ["rgb", "depth", "ft", "state", "fingertips"]:
        (dst_root / sub).mkdir(parents=True, exist_ok=True)

    # 1) prompt + ft_calibration (zeros since already compensated)
    (dst_root / "prompt.txt").write_text(prompt)
    np.save(dst_root / "ft_calibration.npy", np.zeros(6, dtype=np.float32))

    indices = []
    for _, row in df_merged.iterrows():
        ts = int(row["timestamp"])  # filename stem

        # ---- RGB (copy/write from source path) ----
        rgb_path = row["frame_path_iphone"]
        img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        cv2.imwrite(str(dst_root / "rgb" / f"{ts}.png"), img)

        # ---- Depth ----
        depth_path = row["frame_path_depth"]
        depth = np.load(depth_path)  

        if depth is None:
            continue
        _write_depth_png(depth_path, dst_root / "depth" / f"{ts}.png")

        # ---- FT in camera frame (6D) ----
        F_cam = np.array([row["f_ext_x_cam0"], row["f_ext_y_cam0"], row["f_ext_z_cam0"]], dtype=np.float32)
        T_cam = np.array([row["torque_ext_x_cam0"], row["torque_ext_y_cam0"], row["torque_ext_z_cam0"]], dtype=np.float32)
        ft6 = np.concatenate([F_cam, T_cam]).astype(np.float32)
        np.save(dst_root / "ft" / f"{ts}.npy", ft6)

        # ---- Fingertips (camera frame) ----
        left  = np.array([row["tip_left_x_cam0"],  row["tip_left_y_cam0"],  row["tip_left_z_cam0"]],  dtype=np.float32)
        right = np.array([row["tip_right_x_cam0"], row["tip_right_y_cam0"], row["tip_right_z_cam0"]], dtype=np.float32)
        np.savez(dst_root / "fingertips" / f"{ts}.npz", left=left, right=right)

        # ---- State (minimal; used for optional grip-force head) ----
        # If you don’t have matching signals, 0.0 is fine.
        cam_pose = row['cam_pose_world_6d']
        state_dict = {
            "timestamp": ts,
            "gripper": float(row.get("gap", 0.0)),           # or position.0 if preferred
            "gripper_effort": float(row.get("effort.0", 0.0)),
            "cam_pose_world_6d": tuple(map(float, cam_pose))
        }
        (dst_root / "state" / f"{ts}.txt").write_text(str(state_dict))


        (dst_root / "grip_force").mkdir(parents=True, exist_ok=True)
        if "Fc.total" in df_merged.columns:
            gf = np.float32(row["Fc.total"])
            np.save(dst_root / "grip_force" / f"{ts}.npy", gf)
        indices.append(ts)

    if len(indices) < 2:
        raise RuntimeError("Need >= 2 frames to form (initial→final) pairs.")

    indices = np.array(sorted(indices), dtype=int)
    steps = np.zeros_like(indices, dtype=int)
    steps[-1] = 1
    # every 5th frame
    # steps[::5] = 1
    # 2) Adjacent pairing (non-bipartite): i -> i+1
    np.savez(dst_root / "keyframe_list.npz", keyframe_index_list=indices, keyframe_step_list=steps)


    print(f"[OK] Wrote {len(indices)} frames to {dst_root}")

def _write_depth_png(src_path: str, dst_path: Path):
    d = np.load(src_path)  # shape HxW, float meters or int millimeters
    if d.dtype.kind == 'f':
        # assume meters → convert to millimeters for 16-bit PNG
        d_mm = np.clip(np.round(d * 1000.0), 0, 65535).astype(np.uint16)
    elif d.dtype == np.uint16:
        d_mm = d  # already mm
    elif d.dtype == np.int32:
        d_mm = np.clip(d, 0, 65535).astype(np.uint16)
    else:
        # fallback: try to scale floats
        d_mm = np.clip(np.round(d.astype(np.float32) * 1000.0), 0, 65535).astype(np.uint16)

    cv2.imwrite(str(dst_path), d_mm)

def pose_to_mat_from_quat(pose):
    """Convert a 7D pose (tx, ty, tz, qx, qy, qz, qw) to a 4x4 transformation matrix."""
    import scipy.spatial.transform

    t = np.array(pose[0:3])
    q = np.array(pose[3:7])  # x, y, z, w

    R = scipy.spatial.transform.Rotation.from_quat(q).as_matrix()

    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t

    return T

def mat_to_pose(T):
    """Convert a 4x4 transformation matrix to a 6D pose [x, y, z, roll, pitch, yaw]."""
    import scipy.spatial.transform

    t = T[0:3, 3]
    R = T[0:3, 0:3]

    r = scipy.spatial.transform.Rotation.from_matrix(R)
    roll, pitch, yaw = r.as_euler('xyz', degrees=False)

    pose = [t[0], t[1], t[2], roll, pitch, yaw]
    return pose

def visualize_trajectory_in_pointcloud(
    pcd: o3d.geometry.PointCloud,
    df_traj: pd.DataFrame,
    T_device_tool: np.ndarray
):
    """
    Generates geometries for visualizing the tool's trajectory and
    device velocity arrows inside the point cloud.
    
    Args:
        pcd: The Open3D point cloud.
        df_traj: Trajectory dataframe with device poses.
        T_device_tool: 4x4 rigid transform from tool to device.

    Returns:
        A list of Open3D geometry objects (pcd, arrows, trajectory line)
    """
    
    tool_poses_world = []
    geometries_to_visualize = [pcd] # Start with the main point cloud
    stride = 100

    # 1. Calculate the T_world_tool path and create arrows
    for i, row in df_traj.iterrows():
        T_world_device_pose = [
            row["tx_world_device"], row["ty_world_device"], row["tz_world_device"],
            row["qx_world_device"], row["qy_world_device"], row["qz_world_device"], row["qw_world_device"]
        ]
        # Assuming pose_to_mat_from_quat is available
        T_world_device = pose_to_mat_from_quat(T_world_device_pose) 
        T_world_tool = T_world_device @ T_device_tool
        
        tool_poses_world.append(T_world_tool[:3, 3]) # Append the (x,y,z) position

        # Get tool velocity in its local frame
        tool_velocity_device = np.array([
            row["tool_linear_velocity_x_device"],
            row["tool_linear_velocity_y_device"],
            row["tool_linear_velocity_z_device"]
        ])

        if i % stride == 0 and np.linalg.norm(tool_velocity_device) > 1e-4:
            start_point = T_world_tool[:3, 3]
            R_world_device = T_world_device[:3, :3]
            tool_velocity_world = R_world_device @ tool_velocity_device
            scale_factor = 0.1 # Scale for visualization
            end_point = start_point + scale_factor * tool_velocity_world
            arrow_length = np.linalg.norm(end_point - start_point)
            if arrow_length < 1e-6: # Avoid creating zero-length arrows
                continue
                
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=0.005,
                cone_radius=0.01,
                cylinder_height=arrow_length * 0.8, # Adjust proportions
                cone_height=arrow_length * 0.2
            )
            arrow.paint_uniform_color([0, 0, 1])  # blue color
            
            # Get the rotation matrix to align Z-axis (0,0,1) with our world velocity vector
            direction_vector = tool_velocity_world / np.linalg.norm(tool_velocity_world)
            z_axis = np.array([0, 0, 1])
            rot = R.align_vectors(direction_vector[np.newaxis, :], z_axis[np.newaxis, :])
            rotation_matrix = rot[0].as_matrix()

            arrow.rotate(rotation_matrix, center=[0, 0, 0]) # Rotate first
            arrow.translate(start_point)                     # Then translate
            
            geometries_to_visualize.append(arrow)

    # 2. (Optional) Add the tool trajectory as a red line
    if tool_poses_world:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(tool_poses_world)
        lines = [[i, i + 1] for i in range(len(tool_poses_world) - 1)]
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color([1, 0, 0]) # Red trajectory line
        geometries_to_visualize.append(line_set)

    o3d.visualization.draw_geometries(geometries_to_visualize)

def visualize_forces_in_pointcloud(
    pcd: o3d.geometry.PointCloud,
    df_window: pd.DataFrame,
    T_device_tool: np.ndarray,
    T_device_ft: np.ndarray,
    force_scale: float = 0.01  # Adjust this to make forces bigger/smaller
):
    """
    Generates geometries for visualizing forces (total and projected)
    as arrows from the tool's trajectory.

    Args:
        pcd: The Open3D point cloud.
        df_window: The MERGED DataFrame for a specific window, containing
                   poses, velocities, and calculated forces.
        T_device_tool: 4x4 rigid transform from tool to device.
        force_scale: A multiplier to scale force arrows for visibility.
                     (e.g., 1 Newton = force_scale * 1 meter arrow length)
    """
    
    geometries_to_visualize = [pcd]
    stride = 1 # Show more arrows for the shorter window duration
    
    # We need a helper function to create scaled+oriented arrows
    def create_force_arrow(start_point, force_vector_world, color, scale):
        if np.linalg.norm(force_vector_world) < 1e-4:
            return None # Skip zero-length vectors

        # Scale the force vector for visualization
        scaled_force_vec = force_vector_world * scale
        end_point = start_point + scaled_force_vec
        arrow_length = np.linalg.norm(scaled_force_vec)

        if arrow_length < 1e-6:
            return None
            
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.003 * arrow_length / 0.05, # Scale radius with length
            cone_radius=0.006 * arrow_length / 0.05,
            cylinder_height=arrow_length * 0.8,
            cone_height=arrow_length * 0.2
        )
        arrow.paint_uniform_color(color)
        
        # Get rotation to align Z-axis (0,0,1) with our force vector
        direction_vector = scaled_force_vec / arrow_length
        z_axis = np.array([0, 0, 1])
        
        # Use scipy for robust rotation
        rot, _ = R.align_vectors(direction_vector[np.newaxis, :], z_axis[np.newaxis, :])
        rotation_matrix = rot.as_matrix()

        arrow.rotate(rotation_matrix, center=[0, 0, 0])
        arrow.translate(start_point)
        
        return arrow

    # Iterate through the merged dataframe window
    for i, row in df_window.iterrows():
        if i % stride != 0:
            continue

        # 1. Get world pose of the tool (where the force originates)
        T_world_device_pose = [
            row["tx_world_device"], row["ty_world_device"], row["tz_world_device"],
            row["qx_world_device"], row["qy_world_device"], row["qz_world_device"], row["qw_world_device"]
        ]
        T_world_device = pose_to_mat_from_quat(T_world_device_pose) 
        T_world_tool = T_world_device @ T_device_tool
        
        start_point = T_world_tool[:3, 3] # Position of the tool in world
        R_world_device = T_world_device[:3, :3]
        
        # 2. Get total force and projected force (already in 'device' frame)
        total_force_ft = np.array([
            row["wrench_ext.force.x_filt"],
            row["wrench_ext.force.y_filt"],
            row["wrench_ext.force.z_filt"]
        ])
        
        projected_force_device = np.array([
            row["proj_force_x_device"], # This is the one you just calculated
            row["proj_force_y_device"],
            row["proj_force_z_device"]
        ])
        
        # 3. Rotate forces into world frame for visualization
        total_force_world = R_world_device @ (T_device_ft[:3, :3] @ total_force_ft)
        projected_force_world = R_world_device @ projected_force_device
        
        # 4. Create and add arrows
        
        # Total force arrow (RED)
        arrow_total = create_force_arrow(
            start_point, total_force_world, [1, 0, 0], force_scale
        )
        if arrow_total:
            geometries_to_visualize.append(arrow_total)

        # Projected force arrow (GREEN)
        arrow_proj = create_force_arrow(
            start_point, projected_force_world, [0, 1, 0], force_scale
        )
        if arrow_proj:
            geometries_to_visualize.append(arrow_proj)

    # 5. (Optional) Add the tool trajectory as a blue line
    tool_poses_world = []
    for i, row in df_window.iterrows():
        T_world_device = pose_to_mat_from_quat(row[["tx_world_device", "ty_world_device", "tz_world_device", "qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device"]].values)
        T_world_tool = T_world_device @ T_device_tool
        tool_poses_world.append(T_world_tool[:3, 3])

    if tool_poses_world:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(tool_poses_world)
        lines = [[i, i + 1] for i in range(len(tool_poses_world) - 1)]
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color([0, 0, 1]) # Blue trajectory line
        geometries_to_visualize.append(line_set)

    # 6. Visualize
    o3d.visualization.draw_geometries(geometries_to_visualize)

def project_wrenches_onto_motion(
    df_merged: pd.DataFrame,
    R_device_ft: np.ndarray
) -> pd.DataFrame:
    """
    Projects the measured 3D force/torque vectors onto the 3D tool
    linear/angular velocity vectors.

    Args:
        df_merged: The merged DataFrame.
        R_device_ft: The 3x3 rotation matrix to transform vectors from the
                     F/T sensor frame to the 'device' frame.

    Returns:
        The original DataFrame with new columns added for projected
        force and projected torque.
    """

    # ==================================================
    # 1. Linear Force / Linear Velocity Projection
    # ==================================================

    # Get force vectors in the F/T sensor's local frame
    force_vectors_local = df_merged[[
        "wrench_ext.force.x_filt",
        "wrench_ext.force.y_filt",
        "wrench_ext.force.z_filt"
    ]].values

    # Get tool linear velocity vectors (already in 'device' frame)
    lin_vel_vectors_device = df_merged[[
        "tool_linear_velocity_x_device",
        "tool_linear_velocity_y_device",
        "tool_linear_velocity_z_device"
    ]].values

    # Rotate force vectors to 'device' frame
    force_vectors_device = (R_device_ft @ force_vectors_local.T).T
    
    # Store total rotated force
    df_merged['total_force_x_device'] = force_vectors_device[:, 0]
    df_merged['total_force_y_device'] = force_vectors_device[:, 1]
    df_merged['total_force_z_device'] = force_vectors_device[:, 2]

    # --- Linear Projection Calculation ---
    epsilon = 1e-12
    lin_vel_norm = np.linalg.norm(lin_vel_vectors_device, axis=1)
    lin_dot_product = np.einsum('ij,ij->i', force_vectors_device, lin_vel_vectors_device)

    df_merged['force_in_motion_direction'] = np.divide(
        lin_dot_product,
        lin_vel_norm,
        out=np.zeros_like(lin_dot_product),
        where=(lin_vel_norm > epsilon)
    )

    # --- Store projected linear force vector ---
    lin_vel_direction = np.divide(
        lin_vel_vectors_device,
        lin_vel_norm[:, np.newaxis],
        out=np.zeros_like(lin_vel_vectors_device),
        where=(lin_vel_norm[:, np.newaxis] > epsilon)
    )
    force_vec_in_motion_dir = lin_vel_direction * (df_merged['force_in_motion_direction'].values[:, np.newaxis])
    df_merged['proj_force_x_device'] = force_vec_in_motion_dir[:, 0]
    df_merged['proj_force_y_device'] = force_vec_in_motion_dir[:, 1]
    df_merged['proj_force_z_device'] = force_vec_in_motion_dir[:, 2]


    # ==================================================
    # 2. Rotational Torque / Angular Velocity Projection
    # ==================================================

    # Get torque vectors in the F/T sensor's local frame
    torque_vectors_local = df_merged[[
        "wrench_ext.torque.x_filt",
        "wrench_ext.torque.y_filt",
        "wrench_ext.torque.z_filt"
    ]].values

    # Get tool angular velocity vectors (already in 'device' frame)
    # NOTE: Your columns are "angular_velocity_..._device"
    ang_vel_vectors_device = df_merged[[
        "angular_velocity_x_device",
        "angular_velocity_y_device",
        "angular_velocity_z_device"
    ]].values

    # Rotate torque vectors to 'device' frame
    torque_vectors_device = (R_device_ft @ torque_vectors_local.T).T
    
    # Store total rotated torque
    df_merged['total_torque_x_device'] = torque_vectors_device[:, 0]
    df_merged['total_torque_y_device'] = torque_vectors_device[:, 1]
    df_merged['total_torque_z_device'] = torque_vectors_device[:, 2]

    # --- Rotational Projection Calculation ---
    ang_vel_norm = np.linalg.norm(ang_vel_vectors_device, axis=1)
    rot_dot_product = np.einsum('ij,ij->i', torque_vectors_device, ang_vel_vectors_device)

    df_merged['torque_in_motion_direction'] = np.divide(
        rot_dot_product,
        ang_vel_norm,
        out=np.zeros_like(rot_dot_product),
        where=(ang_vel_norm > epsilon)
    )
    
    # --- Store projected torque vector ---
    ang_vel_direction = np.divide(
        ang_vel_vectors_device,
        ang_vel_norm[:, np.newaxis],
        out=np.zeros_like(ang_vel_vectors_device),
        where=(ang_vel_norm[:, np.newaxis] > epsilon)
    )
    torque_vec_in_motion_dir = ang_vel_direction * (df_merged['torque_in_motion_direction'].values[:, np.newaxis])
    df_merged['proj_torque_x_device'] = torque_vec_in_motion_dir[:, 0]
    df_merged['proj_torque_y_device'] = torque_vec_in_motion_dir[:, 1]
    df_merged['proj_torque_z_device'] = torque_vec_in_motion_dir[:, 2]

    return df_merged

def generate_forcesight_ground_truth(gripper_data: GripperData,
                                     leica_data: LeicaData,
                                     data_root: str | Path,
                                     projected: bool):
    

    # get force torque measurements
    df_ft = gripper_data.get_force_torque_measurements()
    df_ft = df_ft[["timestamp", "wrench_ext.force.x_filt",
                        "wrench_ext.force.y_filt",
                        "wrench_ext.force.z_filt",
                        "wrench_ext.torque.x_filt",
                        "wrench_ext.torque.y_filt",
                        "wrench_ext.torque.z_filt"]]  
    
    # get rgb and depth frames
    rgb_frames = gripper_data.get_frames_rgb(side="left")
    df_rgb_frames = pd.DataFrame({
        'frame_path_iphone': [str(p) for p in rgb_frames],
        'timestamp': [int(Path(p).stem) for p in rgb_frames],
    })

    depth_frames = gripper_data.get_frames_depth()
    df_depth_frames = pd.DataFrame({
        'frame_path_depth': [str(p) for p in depth_frames],
        'timestamp': [int(Path(p).stem) for p in depth_frames],
    })


    # get time windows from annotation
    windows_file = gripper_data.loader_aria_gripper.extraction_path / gripper_data.loader_aria_gripper.label_rgb.strip("/") / "forcesight_windows.json"
    if not windows_file.exists():
        print(f"No forcesight windows annotation found at {windows_file}")
        return
    with open(windows_file, "r") as f:
        windows = json.load(f)

    # get gripper calibrations
    gripper_calibration = gripper_data.get_calibration()
    T_cam0_imu = gripper_calibration["cam0"]["T_cam_imu"]
    T_imu_ft = gripper_calibration["imu0"]["T_imu_sensor"]
    T_imu_tool = gripper_calibration["imu0"]["T_imu_tool"]
    T_cam0_ft = T_cam0_imu @ T_imu_ft
    R_cam0_ft = T_cam0_imu[:3, :3] @ T_imu_ft[:3, :3]

    # get traj
    df_traj_aria = gripper_data.loader_aria_gripper.get_closed_loop_trajectory_aligned()
    df_traj_aria = df_traj_aria[["timestamp", "tx_world_device", "ty_world_device", "tz_world_device",
                                 "qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device",
                                 "angular_velocity_x_device", "angular_velocity_y_device", "angular_velocity_z_device",
                                 "device_linear_velocity_x_device", "device_linear_velocity_y_device", "device_linear_velocity_z_device"]]
    

    # get aria calibration
    # caRaw IS cam2 in the gripper rig (wide angle)
    aria_calibration = gripper_data.loader_aria_gripper.get_calibration()
    T_device_camRaw     = aria_calibration["PINHOLE"]["T_device_camera"]
    T_cam2_cam1      = gripper_calibration["cam2"]["T_cn_cnm1"]
    T_cam1_cam0      = gripper_calibration["cam1"]["T_cn_cnm1"]
    T_camRaw_cam0 = T_cam2_cam1 @ T_cam1_cam0
    T_device_cam0 = T_device_camRaw @ T_camRaw_cam0
    R_device_ft = T_device_cam0[:3, :3] @ R_cam0_ft
    T_device_ft = T_device_cam0 @ T_cam0_ft

    # device to tool
    tool_T_tool_device = np.linalg.inv(T_device_cam0 @ T_cam0_imu @ T_imu_tool)
    device_T_device_tool = np.linalg.inv(tool_T_tool_device)

    # get only translation part of this transform and express in device trame
    tool_t_tool_device = tool_T_tool_device[0:3, 3]
    R_tool_device = tool_T_tool_device[0:3, 0:3]
    R_device_tool = R_tool_device.T
    device_t_tool_device = R_device_tool @ tool_t_tool_device


    df_traj_aria = compute_linear_tool_velocity(
        df_traj_aria,
        device_t_tool_device=device_t_tool_device,
        device_lin_vel_cols=["device_linear_velocity_x_device", "device_linear_velocity_y_device", "device_linear_velocity_z_device"],
        device_ang_vel_cols=["angular_velocity_x_device", "angular_velocity_y_device", "angular_velocity_z_device"],
    )

    # visualize
    # visualize_trajectory_in_pointcloud(
    #     pcd=leica_data.get_downsampled_points(voxel=0.01),
    #     df_traj=df_traj_aria,
    #     T_device_tool=device_T_device_tool,
    # )


    # gripper geometry constants
    df_ms = gripper_data.get_motor_states() # ca 60 Hz
    df_ms = df_ms[["timestamp", "position.0", "effort.0", "velocity.0"]] 
    gripper_model = GripperModel()


    # add other columns
    df_ms["alpha.rad"] = df_ms["position.0"] + np.deg2rad(180 - gripper_model.TAU)  # in degrees, convert from rad and offset
    df_ms["x.single"] = gripper_model.x_of_alpha(df_ms["alpha.rad"].to_numpy())  # x per side
    df_ms["gap"] = 2.0 * df_ms["x.single"]
    #jacobian to map torques to forces
    df_ms["dg_dalpha"] = gripper_model.dg_dalpha(df_ms["alpha.rad"].to_numpy())  # m/rad
    df_ms["torque.0"] = gripper_model.current_to_torque(df_ms["effort.0"]/1000.0)  # Nm, motor torque estimate

    # get binned eta values to calibrate motor current to clamp force
    eta_values = gripper_model.eta_of_current(df_ms["effort.0"].to_numpy())
    df_ms["Fc.per_finger"] = np.abs(eta_values * df_ms["torque.0"]) / np.maximum(np.abs(df_ms["dg_dalpha"]), gripper_model.eps)
    df_ms["Fc.total"] = 2.0 * df_ms["Fc.per_finger"]


    num_episodes = len(windows)
    test_window_key = int(np.random.choice(list(windows.keys()), size=1)[0])
    for key, window in windows.items():
        # Get the start and end timestamps
        t0 = int(window["start_ns"])
        t1 = int(window["end_ns"])

        if int(key) % 2 == 1:
            prompt = "close"
        
        else:
            prompt = "open"

        # Get the data for the current window
        df_rgb_win = df_rgb_frames[(df_rgb_frames["timestamp"] >= t0) & (df_rgb_frames["timestamp"] <= t1)]
        df_depth_win = df_depth_frames[(df_depth_frames["timestamp"] >= t0) & (df_depth_frames["timestamp"] <= t1)]
        df_ft_win = df_ft[(df_ft["timestamp"] >= t0) & (df_ft["timestamp"] <= t1)]

        # merge dataframes on timestamp
        df_merged = pd.merge_asof(df_rgb_win.sort_values("timestamp"), 
                                df_depth_win.sort_values("timestamp"), 
                                on="timestamp", 
                                direction="nearest", 
                                tolerance=5000000)  # 5ms tolerance
        df_merged = pd.merge_asof(df_merged.sort_values("timestamp"), 
                                df_ms.sort_values("timestamp"), 
                                on="timestamp", 
                                direction="nearest", 
                                tolerance=10000000)  # 10ms tolerance
        df_merged = pd.merge_asof(df_merged.sort_values("timestamp"), 
                                df_ft_win.sort_values("timestamp"), 
                                on="timestamp", 
                                direction="nearest", 
                                tolerance=5000000)  # 5ms tolerance
        df_merged = df_merged.dropna().reset_index(drop=True)
        df_merged = pd.merge_asof(df_merged.sort_values("timestamp"),
                                df_traj_aria.sort_values("timestamp"),
                                on="timestamp",
                                direction="nearest",
                                tolerance=5000000)  # 5ms tolerance
        df_merged = df_merged.dropna().reset_index(drop=True)


        df_merged = project_wrenches_onto_motion(
            df_merged,
            R_device_ft=R_device_ft,
        )

        # visualize_forces_in_pointcloud(
        #     pcd=leica_data.get_downsampled_points(voxel=0.01),
        #     df_window=df_merged,
        #     T_device_tool=device_T_device_tool,
        #     T_device_ft=T_device_ft,
        #     force_scale=0.01,
        # )

        # compute fingertip coords in cam0 frame (vectorized) and add to dataframe
        x_single = df_merged["x.single"].to_numpy()
        n = x_single.size
        ones = np.ones(n)
        zeros = np.zeros(n)
        p_left_tool = np.vstack([-x_single, zeros, zeros, ones])   # 4 x N
        p_right_tool = np.vstack([ x_single, zeros, zeros, ones])  # 4 x N
        p_left_cam = (T_cam0_imu @ T_imu_tool @ p_left_tool)       # 4 x N
        p_right_cam = (T_cam0_imu @ T_imu_tool @ p_right_tool)     # 4 x N
        p_left_cam = (p_left_cam[:3, :] / p_left_cam[3, :]).T      # N x 3
        p_right_cam = (p_right_cam[:3, :] / p_right_cam[3, :]).T   # N x 3
        df_merged[["tip_left_x_cam0", "tip_left_y_cam0", "tip_left_z_cam0"]] = p_left_cam
        df_merged[["tip_right_x_cam0", "tip_right_y_cam0", "tip_right_z_cam0"]] = p_right_cam

        if projected:
            mode = "projected"
            proj_f_ext = df_merged[["proj_force_x_device",
                                "proj_force_y_device",
                                "proj_force_z_device"]].to_numpy()
            proj_f_ext = (R_cam0_ft @ proj_f_ext.T).T  # N x 3
            df_merged[["f_ext_x_cam0", "f_ext_y_cam0", "f_ext_z_cam0"]] = proj_f_ext
            proj_torque_ext = df_merged[["proj_torque_x_device",
                                        "proj_torque_y_device",
                                        "proj_torque_z_device"]].to_numpy()
            proj_torque_ext = (R_cam0_ft @ proj_torque_ext.T).T  # N x 3
            df_merged[["torque_ext_x_cam0", "torque_ext_y_cam0", "torque_ext_z_cam0"]] = proj_torque_ext
        else:
            mode = "raw"
        # transform force torques to cam0 frame (vectorized) and add to dataframe
            torque_ext = df_merged[["wrench_ext.torque.x_filt",
                                    "wrench_ext.torque.y_filt",
                                    "wrench_ext.torque.z_filt"]].to_numpy()
            torque_ext = (R_cam0_ft @ torque_ext.T).T  # N x 3
            df_merged[["torque_ext_x_cam0", "torque_ext_y_cam0", "torque_ext_z_cam0"]] = torque_ext
            f_ext = df_merged[["wrench_ext.force.x_filt",
                                "wrench_ext.force.y_filt",
                                "wrench_ext.force.z_filt"]].to_numpy()
            f_ext = (R_cam0_ft @ f_ext.T).T  # N x 3
            df_merged[["f_ext_x_cam0", "f_ext_y_cam0", "f_ext_z_cam0"]] = f_ext

        cam_poses_world_6d = []
        for _, row in df_merged.iterrows():
            # Get T_world_device from the row
            T_world_device_pose = [
                row["tx_world_device"], row["ty_world_device"], row["tz_world_device"],
                row["qx_world_device"], row["qy_world_device"], row["qz_world_device"], row["qw_world_device"]
                # or use a 7D pose_to_mat function
            ]
            # Convert 7D pose (tx, ty, tz, qx, qy, qz, qw) to 4x4 matrix
            T_world_device = pose_to_mat_from_quat(T_world_device_pose) # You'll need this helper
            T_world_cam0 = T_world_device @ T_device_cam0
            pose_world_cam0 = mat_to_pose(T_world_cam0) # You'll need this helper
            cam_poses_world_6d.append(pose_world_cam0)

        df_merged['cam_pose_world_6d'] = cam_poses_world_6d

        # write to train or test folder
        # one random episode of location to test

        # if int(key) == test_window_key:
            # write forcesight episode
        write_forcesight_episode(df_merged,
                                    dst_root=Path(data_root) / f"test_{mode}" / f"{gripper_data.rec_loc}" / f"{gripper_data.rec_loc}_{gripper_data.interaction_indices}_{int(key):03d}",
                                    prompt=prompt)
        # else:
        #     # write forcesight episode
        #     write_forcesight_episode(df_merged,
        #                             dst_root=Path(data_root) / "train" / f"{gripper_data.rec_loc}_{gripper_data.interaction_indices}_{int(key):03d}",
        #                             prompt=prompt)

    a = 2
        




    a = 1


if __name__ == "__main__":
    ################################################3
    # recording location
    ################################################3

    rec_location = "livingroom_1"
    base_path = Path(f"/data/ikea_recordings")
    data_indexer = RecordingIndex(
        os.path.join(str(base_path), "extracted") 
    )
    visualize = True
    rec_type = "gripper"
    rec_module = "gripper"
    interaction_indices = "8-14"
    color = "blue"

    for projected in [False, True]:

        gripper_data = GripperData(base_path, 
                                rec_loc=rec_location, 
                                rec_type=rec_type, 
                                rec_module=rec_module, 
                                interaction_indices=interaction_indices,
                                data_indexer=data_indexer,
                                color=color,)
        
        leica_data = LeicaData(base_path=base_path, 
                            rec_loc=rec_location, 
                            initial_setup="001")
        
        generate_forcesight_ground_truth(gripper_data,
                                        leica_data=leica_data,
                                        data_root="/data/evaluations/visual_force_estimation/forcesight/data/hej",
                                        projected=projected)