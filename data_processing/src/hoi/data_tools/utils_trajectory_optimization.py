import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as Rsp
import matplotlib.pyplot as plt
import gtsam
from typing import List, Dict
import copy
import open3d as o3d
from pathlib import Path

class TrajectoryOptimization:
    
    def __init__(self, poses_traj: pd.DataFrame, poses_anchor: pd.DataFrame):
        """
        Initialize the TrajectoryOptimization class with trajectory and anchor poses.
        
        :param poses_traj: DataFrame containing trajectory poses [timestamp, tx, ty, tz, qx, qy, qz, qw].
        :param poses_anchor: DataFrame containing anchor poses. [timestamp, tx, ty, tz, qx, qy, qz, qw].

        """
        self.poses_traj = poses_traj
        self.poses_traj.sort_values(by="timestamp", inplace=True)
        self.poses_anchor = poses_anchor
        self.poses_anchor.sort_values(by="timestamp", inplace=True)

        self.poses_traj_scaled = None
        self.poses_traj_trimmed = None
        self.poses_optimized = None


    def trim_trajectory(self, start_time: float, end_time: float) -> pd.DataFrame:
        """
        Trim the trajectory poses to the specified time window.
        
        :param start_time: Start timestamp for trimming.
        :param end_time: End timestamp for trimming.
        :return: DataFrame with trimmed trajectory poses.
        """
        self.poses_traj_trimmed = self.poses_traj[
            (self.poses_traj["timestamp"] >= start_time) & 
            (self.poses_traj["timestamp"] <= end_time)
        ].reset_index(drop=True)
        
        return self.poses_traj_trimmed
    
    def compute_transformation_between_poses(self, pose_anchor: pd.DataFrame, pose_traj: pd.DataFrame) -> np.array:
        """
        Compute the initial transformation matrix between the trajectory pose and the anchor pose.
        """

        translation_traj = pose_traj[["tx", "ty", "tz"]].to_numpy()
        rotation_traj = pose_traj[["qx", "qy", "qz", "qw"]].to_numpy()

        translation_anchor = pose_anchor[["tx", "ty", "tz"]].to_numpy()
        rotation_anchor = pose_anchor[["qx", "qy", "qz", "qw"]].to_numpy()

        # Convert quaternion to rotation matrix
        rot_traj = Rsp.from_quat(rotation_traj).as_matrix()
        rot_anchor = Rsp.from_quat(rotation_anchor).as_matrix()
        # Compute the initial transformation matrix
        transformation = np.eye(4)
        transformation[:3, :3] = rot_anchor @ rot_traj.T
        transformation[:3, 3] = translation_anchor - transformation[:3, :3] @ translation_traj      

        return transformation
    
    def transform_trajectory(self, trajectory: pd.DataFrame, transformation: np.array) -> pd.DataFrame:
        transformed_poses = []

        rotation_matrix = transformation[:3, :3]
        translation_vector = transformation[:3, 3]


        for index, row in trajectory.iterrows():
            # 1. Transform Position (Translation)
            # Extract original position vector
            original_position = row[["tx", "ty", "tz"]].to_numpy()
            
            # Apply the transformation: p_new = R @ p_orig + t
            new_position = rotation_matrix @ original_position + translation_vector

            # 2. Transform Orientation (Rotation)
            # Extract original rotation quaternion
            original_quat = row[["qx", "qy", "qz", "qw"]].to_numpy()
            
            # Create Rotation objects for easy multiplication
            rot_orig = Rsp.from_quat(original_quat)
            rot_transform = Rsp.from_matrix(rotation_matrix)
            
            # Apply the transformation: Q_new = Q_transform * Q_orig
            new_rot = rot_transform * rot_orig
            new_quat = new_rot.as_quat()

            # 3. Store the Transformed Pose
            transformed_poses.append({
                "timestamp": row["timestamp"],
                "tx": new_position[0],
                "ty": new_position[1],
                "tz": new_position[2],
                "qx": new_quat[0],
                "qy": new_quat[1],
                "qz": new_quat[2],
                "qw": new_quat[3],
            })

        # Convert the list of dictionaries back to a DataFrame
        return pd.DataFrame(transformed_poses)


    def _create_gtsam_pose3(self, row):
        t = gtsam.Point3(row["tx"], row["ty"], row["tz"])
        q = gtsam.Rot3(row["qw"], row["qx"], row["qy"], row["qz"])
        return gtsam.Pose3(q, t)
    
    def _robust_noise_model(self, sigmas, k=1.345):
        base = gtsam.noiseModel.Diagonal.Sigmas(np.array(sigmas))
        return gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber(k), base)
    
    def hierarchical_optimization(self, moving_average: bool = True) -> pd.DataFrame:

        # compute the segments between anchor poses
        segments = self.get_segments_between_hloc_poses()

        scales = []
        optimized_poses = []

        for idx, segment in enumerate(segments):

            if len(segment) < 2:
                continue

            poses_traj_segment = self.poses_traj[self.poses_traj["timestamp"].isin(segment)]
            poses_anchor_segment = self.poses_anchor[self.poses_anchor["timestamp"].isin(segment)]

            # compute the initial transformation between the first anchor pose and the first trajectory pose
            initial_transformation = self.compute_transformation_between_poses(
                pose_anchor=poses_anchor_segment.iloc[0],
                pose_traj=poses_traj_segment.iloc[0]
            )

            # apply the transformation to the trajectory poses
            poses_traj_segment_transformed = self.transform_trajectory(
                trajectory=poses_traj_segment,
                transformation=initial_transformation
            )

            # compute the scale factor between the trajectory segment and the anchor segment
            # scale = self.compute_segment_scale(
            #     poses_traj_segment= poses_traj_segment_transformed,
            #     poses_anchor_segment=poses_anchor_segment
            # )
            scale = 1.0

            # poses_traj_segment_transformed_scaled = self.scale_trajectory_from_first_pose(
            #     poses_traj_segment= poses_traj_segment_transformed,
            #     scale=scale
            # )

            # optimize the trajectory segment
            optimized_segment = self.optimize_trajectory(
                poses_traj=poses_traj_segment_transformed,
                poses_anchor=poses_anchor_segment
            )

            optimized_poses.append(optimized_segment)
            scales.append(scale)

        # concatenate all optimized segments into a single DataFrame
        full_optimzed_traj = pd.concat(optimized_poses, ignore_index=True)

        # find dupliate timestamps and keep the first occurrence
        full_optimzed_traj.drop_duplicates(subset=["timestamp"], keep="first", inplace=True)
        full_optimzed_traj.sort_values(by="timestamp", inplace=True)

        # self.viz_poses_in_ply(
        #     poses_traj=full_optimzed_traj,
        #     poses_anchor=poses_anchor_segment,
        #     path_pcd=Path("/data/ikea_recordings/extracted/bedroom_2/leica/001/points_downsampled/points_voxel_0.050.ply"),
        # )

        # global optimization
        full_optimzed_traj = self.optimize_trajectory(
            poses_traj=full_optimzed_traj,
            poses_anchor=self.poses_anchor
        )

        # self.viz_poses_in_ply(
        #     poses_traj=full_optimzed_traj,
        #     poses_anchor=poses_anchor_segment,
        #     path_pcd=Path("/data/ikea_recordings/extracted/bedroom_2/leica/001/points_downsampled/points_voxel_0.050.ply"),
        # )

        # calculate speed of the optimized trajectory between consecutive poses
        # dt = full_optimzed_traj["timestamp"].diff().fillna(1) / 1e9  # Avoid div by zero
        # displacements = full_optimzed_traj[["tx", "ty", "tz"]].diff().fillna(0).to_numpy()
        # speeds = np.linalg.norm(displacements, axis=1) / dt.to_numpy()
        # full_optimzed_traj["speed"] = speeds

        # # plot speeds
        # plt.figure(figsize=(10, 5))
        # plt.plot(full_optimzed_traj["timestamp"], full_optimzed_traj["speed"], label="Speed (m/s)")
        # plt.xlabel("Timestamp")
        # plt.ylabel("Speed (m/s)")
        # plt.title("Speed of Optimized Trajectory")
        # plt.legend()
        # plt.grid()
        # plt.show()

        # # plot scales
        # plt.figure(figsize=(10, 5))
        # plt.plot(range(len(scales)), scales, marker='o', label="Scale Factor")
        # plt.xlabel("Segment Index")
        # plt.ylabel("Scale Factor")
        # plt.title("Scale Factors for Each Segment")
        # plt.legend()
        # plt.grid()
        # plt.show()

        return full_optimzed_traj


    def global_optimization(self, poses_traj: pd.DataFrame, poses_anchor: pd.DataFrame) -> pd.DataFrame:
        pass
        
    def scale_trajectory_from_first_pose(self, poses_traj_segment: pd.DataFrame, scale: float) -> pd.DataFrame:
        # Extract the first pose
        first_pose = poses_traj_segment.iloc[0]
        t0 = first_pose[["tx", "ty", "tz"]].to_numpy()
        q0 = first_pose[["qx", "qy", "qz", "qw"]].to_numpy()
        R0 = Rsp.from_quat(q0).as_matrix()

        # Store transformed poses
        scaled_poses = []

        for i, row in poses_traj_segment.iterrows():
            t = row[["tx", "ty", "tz"]].to_numpy()
            q = row[["qx", "qy", "qz", "qw"]].to_numpy()

            # Transform to local frame of first pose
            t_local = R0.T @ (t - t0)

            # Apply scale in local frame
            t_scaled_local = t_local * scale

            # Transform back to world frame
            t_scaled_world = R0 @ t_scaled_local + t0

            # Keep orientation as-is (you can optionally interpolate it later if needed)
            scaled_poses.append({
                "timestamp": row["timestamp"],
                "tx": t_scaled_world[0],
                "ty": t_scaled_world[1],
                "tz": t_scaled_world[2],
                "qx": q[0],
                "qy": q[1],
                "qz": q[2],
                "qw": q[3],
            })

        return pd.DataFrame(scaled_poses)

    def compute_segment_scale(self, poses_traj_segment: pd.DataFrame, poses_anchor_segment: pd.DataFrame, moving_average: bool = True) -> float:
        # Get first and last trajectory positions
        t0 = poses_traj_segment.iloc[0][["tx", "ty", "tz"]].to_numpy()
        t1 = poses_traj_segment.iloc[-1][["tx", "ty", "tz"]].to_numpy()
        d_traj = np.linalg.norm(t1 - t0)

        # Get first and second anchor positions
        a0 = poses_anchor_segment.iloc[0][["tx", "ty", "tz"]].to_numpy()
        a1 = poses_anchor_segment.iloc[1][["tx", "ty", "tz"]].to_numpy()
        d_anchor = np.linalg.norm(a1 - a0)

        # clamp scale to avoid unrealistic speeds
        # compute average speed of the trajectory segment
        # traj_speed = d_traj / (poses_traj_segment.iloc[-1]["timestamp"] - poses_traj_segment.iloc[0]["timestamp"])
        # anchor_speed = d_anchor / (poses_anchor_segment.iloc[1]["timestamp"] - poses_anchor_segment.iloc[0]["timestamp"])


        scale = d_anchor / d_traj


        return scale

    def optimize_trajectory(self, poses_traj: pd.DataFrame, poses_anchor: pd.DataFrame) -> pd.DataFrame:
            """
            Builds a GTSAM factor graph and optimizes the trajectory.
            The scaled and reconstructed trajectory is used as the initial guess.
            The hloc poses are used as unary (prior) constraints.
            Odometry from the original trajectory is used for BetweenFactors.
            
            :return: DataFrame with the globally optimized trajectory.
            """

            graph = gtsam.NonlinearFactorGraph()
            initial_estimate = gtsam.Values()
            
            num_poses = len(poses_traj)
            
            # --- 1. Add Initial Estimate from the Trajectory ---
            for i in range(num_poses):
                pose_row = poses_traj.iloc[i]
                initial_estimate.insert(i, self._create_gtsam_pose3(pose_row))

            # --- 2. Add Odometry (Between) Factors from the ORIGINAL Trajectory ---
            print("Adding odometry constraints from original trajectory...")
            # odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.9, 0.9, 0.9, 0.01,0.01, 0.01]))
            # odometry_noise = self._robust_noise_model([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], k=1.345)
            
            vels = []
            for i in range(num_poses - 1):
                # original poses
                pose_i   = self._create_gtsam_pose3(poses_traj.iloc[i])
                pose_i1  = self._create_gtsam_pose3(poses_traj.iloc[i+1])
                # compute relative motion
                relative = pose_i.between(pose_i1)
                odom_noise = self._robust_noise_model([0.1, 0.1, 0.1, 0.01, 0.01, 0.01], k=1.345)
                graph.add(gtsam.BetweenFactorPose3(i, i + 1, relative, odom_noise))
                
            # --- 3. Add HLOC Priors as Unary Constraints ---
            anchor_pos_noise = self._robust_noise_model([1e-3,1e-3,1e-3, 1e-3,1e-3,1e-3], k=1.345)
            # hloc_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]))
            # hloc_noise = gtsam.noiseModel.Constrained.All(6)  
            traj_timestamps = poses_traj["timestamp"].to_numpy()
            for _, anchor_row in poses_anchor.iterrows():
                anchor_ts = anchor_row["timestamp"]
                
                # Find the index of the closest trajectory pose for this anchor
                idx = np.argmin(np.abs(traj_timestamps - anchor_ts))
                
                hloc_pose = self._create_gtsam_pose3(anchor_row)
                graph.add(gtsam.PriorFactorPose3(idx, hloc_pose, anchor_pos_noise))
            # Constrain only first pose fully (position + orientation)
                
            # --- 4. Run the Optimizer ---

            params = gtsam.LevenbergMarquardtParams()
            params.setVerbosity("TERMINATION")
            optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
            result = optimizer.optimize()

            
            # --- 5. Convert Result back to DataFrame ---
            optimized_traj = []
            for i in range(num_poses):
                optimized_pose = result.atPose3(i)
                row_data = [poses_traj.iloc[i]["timestamp"]]
                row_data.extend(list(optimized_pose.translation()))
                rot_mat = optimized_pose.rotation().matrix()
                rot_quat = Rsp.from_matrix(rot_mat).as_quat()
                row_data.extend(list(rot_quat))
                optimized_traj.append(row_data)
                
            columns = ["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
            optimized_df = pd.DataFrame(optimized_traj, columns=columns)
            
            self.poses_optimized = optimized_df
            return optimized_df
    
    def get_segments_between_hloc_poses(self) -> List[List[float]]:
        """
        Return segments between HLOC poses. Each segment contains all the trajectory poses
        between two consecutive anchor poses.
        Return list of list containng timestamps
        """

        segments = []
        for i in range(len(self.poses_anchor) - 1):
            start_time = self.poses_anchor.iloc[i]["timestamp"]
            end_time = self.poses_anchor.iloc[i + 1]["timestamp"]

            segment = self.poses_traj[(self.poses_traj["timestamp"] >= start_time) & 
                                      (self.poses_traj["timestamp"] <= end_time)]
            
            if not segment.empty:
                print(start_time-segment["timestamp"].iloc[0])

            if len(segment) == 0 or segment["timestamp"].iloc[0] != start_time:
                continue
            # get timestamps as a list
            segment = segment["timestamp"].tolist()

            segments.append(segment)

        self.segments = segments
        return segments
    
    def viz_poses_in_ply(self, poses_traj: pd.DataFrame, poses_anchor: pd.DataFrame, path_pcd: str | Path, stride: int = 10):

        pcd = o3d.io.read_point_cloud(str(path_pcd))
            
        frames_traj = []

        for i in range(len(poses_traj)):
            if i % stride != 0:
                continue
            qw = poses_traj["qw"].iloc[i]
            qx = poses_traj["qx"].iloc[i]
            qy = poses_traj["qy"].iloc[i]
            qz = poses_traj["qz"].iloc[i]
            tx = poses_traj["tx"].iloc[i]
            ty = poses_traj["ty"].iloc[i]
            tz = poses_traj["tz"].iloc[i]

            T_ad = np.eye(4)
            T_ad[:3, :3] = Rsp.from_quat([qx, qy, qz, qw]).as_matrix()
            T_ad[:3, 3] = np.array([tx, ty, tz])

            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            frame.transform(T_ad)

            frames_traj.append(copy.deepcopy(frame))

        frames_anchor = []
        for i in range(len(poses_anchor)):
            qw = poses_anchor["qw"].iloc[i]
            qx = poses_anchor["qx"].iloc[i]
            qy = poses_anchor["qy"].iloc[i]
            qz = poses_anchor["qz"].iloc[i]
            tx = poses_anchor["tx"].iloc[i]
            ty = poses_anchor["ty"].iloc[i]
            tz = poses_anchor["tz"].iloc[i]

            T_ad = np.eye(4)
            T_ad[:3, :3] = Rsp.from_quat([qx, qy, qz, qw]).as_matrix()
            T_ad[:3, 3] = np.array([tx, ty, tz])

            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            frame.transform(T_ad)

            frames_anchor.append(copy.deepcopy(frame))
        #
        # Create a visualizer instance
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='Trajectory Visualization')

        # Add all your geometries to the visualizer
        for geometry in [pcd] + frames_traj + frames_anchor:
            vis.add_geometry(geometry)

        # --- Set the initial viewpoint ---
        # Get the view control object
        view_control = vis.get_view_control()


        # The 'up' vector defines which direction is 'up' in the view.
        view_control.set_front([1, 0, 0])     # Look along the positive X-axis
        view_control.set_lookat([0, 0, 0])    # Focus on the world origin
        view_control.set_up([0, 0, 1])        # Set Z-axis as the 'up' direction (common for robotics/3D vision)
        view_control.set_zoom(0.8)            # Adjust the zoom level to see the scene

        # --- Render the scene ---
        vis.run()
        vis.destroy_window()
    
