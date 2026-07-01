from hoi.data_tools.data_indexer import RecordingIndex
import os
from pathlib import Path
from hoi.data_tools.data_loader_gripper import GripperData
# pip install cvxpy
import numpy as np
import cvxpy as cp
import pandas as pd
from hoi.data_tools.utils_gripper_model import GripperModel
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import shutil
import cv2
from typing import Optional, Union, Iterable, Tuple
import pickle



def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero-length vector where a direction was expected.")
    return v / n

def skew(p: np.ndarray) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix [p]_x such that [p]_x @ v == p × v."""
    px, py, pz = p
    return np.array([[0, -pz,  py],
                     [pz,   0, -px],
                     [-py, px,   0]], dtype=float)

def tangential(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Project v onto the plane orthogonal to n (n is unit)."""
    return v - n * float(n @ v)

def total_load_lower_bound(F_w, tau_w, p1, p2, n1) -> float:
    """
    Solver-free rigorous lower bound:
      L_min = 2*Fc + S_ext
    but this function returns only S_ext so you can add 2*Fc yourself.
    All inputs are 3-vectors in the FT frame. n1 points inward from finger-1 into the object.
    """
    F_w  = np.asarray(F_w,  dtype=float).reshape(3)
    tau_w = np.asarray(tau_w, dtype=float).reshape(3)
    p1   = np.asarray(p1,   dtype=float).reshape(3)
    p2   = np.asarray(p2,   dtype=float).reshape(3)
    n1   = unit(n1)

    dvec = p1 - p2
    d = np.linalg.norm(dvec)
    if d == 0:
        raise ValueError("p1 and p2 must be distinct (nonzero jaw gap).")
    b = dvec / d
    a = unit(np.cross(b, n1))          # couple axis (out of palm)
    pC = 0.5 * (p1 + p2)
    tauC_ext = tau_w - np.cross(pC, F_w)
    S_ext = 2.0 * abs(float(tauC_ext @ a)) / d
    return S_ext  # use L_min = 2*Fc + S_ext

def total_load_high_fidelity(
    F_w, tau_w, p1, p2, n1, Fc, mu,
    n2=None,
    beta=1e-2,
    nonneg_eps: float = 0.0,
    solver: str = "SCS",
) -> float:
    # beta is 3xupper limits for the standard deviation on torque noise from data sheet
    import numpy as np, cvxpy as cp

    # arrays & unit normals
    F_w   = np.asarray(F_w, float).reshape(3)
    tau_w = np.asarray(tau_w, float).reshape(3)
    p1    = np.asarray(p1,  float).reshape(3)
    p2    = np.asarray(p2,  float).reshape(3)
    n1    = np.asarray(n1,  float).reshape(3)
    n1   /= (np.linalg.norm(n1) + 1e-12)
    if n2 is None:
        n2 = -n1
    else:
        n2 = np.asarray(n2, float).reshape(3)
        n2 /= (np.linalg.norm(n2) + 1e-12)

    Fc = float(Fc); mu = float(mu); nonneg_eps = float(nonneg_eps)

    def skew(p):
        px, py, pz = p
        return np.array([[0, -pz,  py],
                         [pz,   0, -px],
                         [-py,  px,  0]], float)

    # decision vars
    u1 = cp.Variable(3)
    u2 = cp.Variable(3)
    # small moment to model noisy torque data
    # r_tau = cp.Variable(3)

    # fingertip forces
    f1 = Fc * n1 + u1
    f2 = Fc * n2 + u2

    # components for cones
    n1u1 = n1 @ u1
    n2u2 = n2 @ u2
    t1 = u1 - n1 * n1u1
    t2 = u2 - n2 * n2u2

    pC = 0.5 * (p1 + p2)
    tau_c = tau_w - np.cross(pC, F_w)

    constraints = []
    # force and torque balance 
    constraints += [u1 + u2 == -F_w]
    # constraints += [skew(p1) @ u1 + skew(p2) @ u2 + r_tau == -tau_w]
    # constraints += [skew(p1 - pC) @ u1 + skew(p2 - pC) @ u2 + r_tau == -tau_c]
    # small moment to model noisy torque data
    # constraints += [cp.norm(r_tau, 2) <= beta]

    # Coulomb cones & compressive normals
    constraints += [
        cp.norm(t1, 2) <= mu * (Fc + n1u1),
        cp.norm(t2, 2) <= mu * (Fc + n2u2),
        Fc + n1u1 >= -nonneg_eps,
        Fc + n2u2 >= -nonneg_eps,
    ]

    # objective: sum of fingertip magnitudes
    lam = 0.5  # regularization on r_tau
    prob = cp.Problem(cp.Minimize(cp.norm(f1, 2) + cp.norm(f2, 2)), constraints)
    prob.solve(solver=solver, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"SOCP did not solve to optimality: status={prob.status}")

    return float(prob.value)



def generate_contact_force_estimation_pseudo_ground_truth_from_ft_and_motor_states(
    gripper_data: GripperData,
    output_path: str,
    rec_location: str,
    interaction_indices: str
):
    """
    Generates pseudo-groundtruth contact forces from wrist FT sensor data
    and robot finger motor states using the methods defined above.
    Pseudo GT because its fused from multiple noisy sources and uses a model of the grasp,
    since we don't have real contact force sensors on the fingers.

    We load motor states and FT data from a recording, then compute the pseudo GT contact forces.
    We only use where the gripper is at least partially closed (so we have a grasp).

    We use the high-fidelity method total_load_high_fidelity() to compute L*, the total load supported by both fingertips.

    """

    THRESH_PARTIALLY_CLOSED = 65  # in degrees, gripper is partially closed above this, value is hardware dependent
    TAU = 64 # in degrees, offset from dynamixel to kinematic frame, value is hardware dependent
    L1 = 0.038 # m, gripper link length, value is hardware dependent
    L2 = 0.0444  # m, gripper link length, value is hardware dependent
    eps = 1e-9
    k1_const = 1.769  # motor current to torque constant
    k2_const = -0.2214    # motor current to torque constant
    # helper system rotation matrix from system aligned
    # with rails to FT frame
    # helper to sensor frame
    n1_left = np.array([0, 0, 1])  # inward normal at finger-1 (toward finger-2)
    n2_right = np.array([0, 0, 1]) # inward normal at finger-2 (toward finger-1)
    R_S_helper = np.array([[-0.7071, -0.7071, 0],
                            [0.7071, -0.7071, 0],
                            [0, 0, 1]])
    
    # left finger frame to helper system
    R_helper_left = np.array([[0, 0, 1],
                              [1, 0, 0],
                              [0, -1, 0]])
    
    # right finger frame to helper system
    R_helper_right = np.array([[0, 0, -1],
                               [-1, 0, 0],
                               [0, -1, 0]])

    images_left = gripper_data.get_digit_images(side="left")
    df_images_left = images_to_df(images_left)
    no_force_reference_img_left  = images_left[0]
    images_right = gripper_data.get_digit_images(side="right")
    df_images_right = images_to_df(images_right)
    no_force_reference_img_right = images_right[0]

    df_ft = gripper_data.get_force_torque_measurements() # ca 100 Hz
    df_ft = df_ft[["timestamp", "wrench_ext.force.x_filt",
                        "wrench_ext.force.y_filt",
                        "wrench_ext.force.z_filt",
                        "wrench_ext.torque.x_filt",
                        "wrench_ext.torque.y_filt",
                        "wrench_ext.torque.z_filt"]]  

    df_ms = gripper_data.get_motor_states() # ca 60 Hz
    df_ms = df_ms[["timestamp", "position.0", "effort.0", "velocity.0"]] 
    gripper_model = GripperModel()

    # get relevant timestamps where gripper is at least partially closed and near stationary
    # thrsh ttransform to ros motor states which are 180 deg offset and given in rad
    thresh = (THRESH_PARTIALLY_CLOSED + TAU - 180) * np.pi / 180
    df_ms = df_ms[df_ms["position.0"] > thresh]
    df_ms = df_ms[df_ms["velocity.0"] >= 0.0]
    df_ms = df_ms[np.abs(df_ms["effort.0"]) > 50.0]  # avoid low-effort region with poor SNR 

    # add other columns 
    df_ms["alpha.rad"] = df_ms["position.0"] + np.deg2rad(180 - TAU)  # in degrees, convert from rad and offset
    df_ms["x.single"] = gripper_model.x_of_alpha(df_ms["alpha.rad"].to_numpy(), L1, L2)   # x per side
    df_ms["gap"] = 2.0 * df_ms["x.single"]  
    #jacobian to map torques to forces
    df_ms["dg_dalpha"] = gripper_model.dg_dalpha(df_ms["alpha.rad"].to_numpy(), L1, L2)  # m/rad
    df_ms["torque.0"] = gripper_model.current_to_torque(df_ms["effort.0"]/1000.0, k1_const=k1_const, k2_const=k2_const)  # Nm, motor torque estimate

    # get binned eta values to calibrate motor current to clamp force
    
    eta_values = gripper_model.eta_of_current(df_ms["effort.0"].to_numpy())
    df_ms["Fc.per_finger"] = np.abs(eta_values * df_ms["torque.0"]) / np.maximum(np.abs(df_ms["dg_dalpha"]), eps)

    # merge the two dataframes with asof merge, nearest neighbor within tolerance
    # to get FT data at motor state timestamps
    tol_ns = int(10e6) # 10 ms tolerance for asof merge
    df = pd.merge_asof(
        df_ms.sort_values("timestamp"),
        df_ft.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=tol_ns)

    # drop rows with no FT data within tolerance
    ft_cols = ["wrench_ext.force.x_filt","wrench_ext.force.y_filt","wrench_ext.force.z_filt",
           "wrench_ext.torque.x_filt","wrench_ext.torque.y_filt","wrench_ext.torque.z_filt"]    
    df = df.dropna(subset=ft_cols)
    

    # get forces and torques in sensor frame
    Fw_S = df[["wrench_ext.force.x_filt","wrench_ext.force.y_filt","wrench_ext.force.z_filt"]].to_numpy()

    # compute total load L* 
    Ls = []
    for i in range(len(df)):
        Fw_H = R_S_helper.T @ Fw_S[i]
        # tangetial forces in digit frame
        sum_F_y = np.abs(Fw_H[2])  # helper frame z is colinear with digit y
        sum_F_x = np.abs(Fw_H[1])  # helper frame y is colinear with digit x

        # normal forces in digit frame (clamp and normal part of external force)
        sum_F_z = 2 * np.abs(df.iloc[i]["Fc.per_finger"]) + np.abs(Fw_H[0]) # helper frame x is colinear with digit z

        # also get clipped data fot fairer comparison with predictions
        # clipping values chosen based on max values in training data
        if sum_F_x > 8.0:
            sum_F_x_clipped = 8.0
        else:
            sum_F_x_clipped = sum_F_x

        if sum_F_y > 8.0:
            sum_F_y_clipped = 8.0
        else:
            sum_F_y_clipped = sum_F_y

        if sum_F_z > 10.0:
            sum_F_z_clipped = 10.0
        else:
            sum_F_z_clipped = sum_F_z

        L_star = sum_F_x + sum_F_y + sum_F_z
        L_star_clipped = sum_F_x_clipped + sum_F_y_clipped + sum_F_z_clipped
        L_star_L2 = np.sqrt(sum_F_x**2 + sum_F_y**2 + sum_F_z**2)
        L_star_L2_clipped = np.sqrt(sum_F_x_clipped**2 + sum_F_y_clipped**2 + sum_F_z_clipped**2)
        L_star_tangential_L2 = np.sqrt(sum_F_x**2 + sum_F_y**2)
        L_star_tangential_L2_clipped = np.sqrt(sum_F_x_clipped**2 + sum_F_y_clipped**2)
        Ls.append([L_star, sum_F_x, sum_F_y, sum_F_z, L_star_L2, L_star_tangential_L2,
                   L_star_clipped, sum_F_x_clipped, sum_F_y_clipped, sum_F_z_clipped,
                   L_star_L2_clipped, L_star_tangential_L2_clipped])

    Ls = np.array(Ls)
    # add to dataframe
    df["L_star"] = Ls[:,0]
    df["L_star_x"] = Ls[:,1]
    df["L_star_y"] = Ls[:,2]
    df["L_star_z"] = Ls[:,3]
    df["L_star_L2"] = Ls[:,4]
    df["L_star_tangential_L2"] = Ls[:,5]
    df["L_star_normalized"] = df["L_star"] / df["L_star"].max()
    df["L_star_clipped"] = Ls[:,6]
    df["L_star_x_clipped"] = Ls[:,7]
    df["L_star_y_clipped"] = Ls[:,8]
    df["L_star_z_clipped"] = Ls[:,9]
    df["L_star_L2_clipped"] = Ls[:,10]
    df["L_star_tangential_L2_clipped"] = Ls[:,11]
    df["L_star_normalized_clipped"] = df["L_star_clipped"] / df["L_star_clipped"].max()
    # attach nearest digit images
    df_matched = attach_nearest_force_to_digit_images(df, df_images_right, tol_ms=10)    
    # write to csv
    output_path = f"{output_path}"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_matched.to_csv(output_path, index=False)
    print(f"[✓] Wrote contact force pseudo-groundtruth to {output_path}")
    
    dump_to_sparsh_format(df_matched,
        output_root=output_path.parent / f"{rec_location}_{interaction_indices}_sparsh_format",
        gap_ns=int(2000e6),  # 200 ms
        ts_col="timestamp_img",
        right_col="image_path",
    )



    # split_and_copy_sequences(
    #     df_matched,
    #     output_root=output_path.parent,
    #     no_force_reference_img_left=no_force_reference_img_left,
    #     no_force_reference_img_right=no_force_reference_img_right,
    #     gap_ns=int(200e6),  # 100 ms
    #     side_col_right="image_path",
    #     ts_col="timestamp_img",
    # )


def dump_to_sparsh_format(
    df: pd.DataFrame,
    output_root: Path,
    gap_ns: int = int(1000e6),
    ts_col: str = "timestamp_img",
    right_col: str = "image_path",
    write_dummy_pickles: bool = True,
    force_scale_xyz = (4.0, 4.0, 5.0),
    poses_dim: int = 7,
    min_frames: int = 11,  # strictly more than 10 frames required
) -> None:
    import cv2, numpy as np, pickle, shutil, json, csv
    from pathlib import Path

    def _png_bytes(img_path: Path) -> bytes:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            raise RuntimeError(f"Failed to encode PNG for {img_path}")
        return buf.tobytes()

    # sort + split sequences
    df = df.sort_values(ts_col).reset_index(drop=True)
    gaps = df[ts_col].diff().fillna(0).to_numpy()
    seq_ids = (np.abs(gaps) > gap_ns).cumsum()

    root = Path(output_root)
    left_dir  = root / "digit_left"
    right_dir = root / "digit_right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    # aggregates for dataset_slip_forces.pkl
    left_trajectories, right_trajectories = {}, {}
    left_in_contact_global: list[float] = []
    right_in_contact_global: list[float] = []
    left_global_idx = right_global_idx = 0
    left_traj_id = right_traj_id = 0

    # per-side CSV rows
    left_index_rows, right_index_rows = [], []

    for seq_idx, seq_df in df.groupby(seq_ids):
        if seq_df.empty:
            continue

        seq_base = f"dataset_digit_{seq_idx:03d}"   # no extension
        seq_pkl  = f"{seq_base}.pkl"

        # gather paired paths
        right_paths, left_paths = [], []
        for p_right_str in seq_df[right_col].to_numpy():
            p_right = Path(str(p_right_str))
            p_left  = Path(str(p_right).replace("/right/", "/left/").replace("digit_right", "digit_left"))
            if not p_right.exists() or not p_left.exists():
                print(f"[warn] missing pair -> R:{p_right}  L:{p_left}")
                continue
            right_paths.append(p_right)
            left_paths.append(p_left)
        if not right_paths or not left_paths:
            continue

        # keep ordering consistent
        right_paths.sort(key=lambda p: p.stem)
        left_paths.sort(key=lambda p: p.stem)

        # enforce length: use the paired count
        T_pair = min(len(right_paths), len(left_paths))
        if T_pair < min_frames:
            print(f"[skip] {seq_base}: only {T_pair} frames (< {min_frames})")
            continue

        # ---------------- RIGHT sequence ----------------
        frames_r_bytes = [_png_bytes(p) for p in right_paths[:T_pair]]  # no neutral prepend
        with open(right_dir / seq_pkl, "wb") as f:
            pickle.dump(frames_r_bytes, f, protocol=5)

        right_frames_dir = right_dir / f"{seq_base}_frames"
        right_frames_dir.mkdir(exist_ok=True)
        for i, b in enumerate(frames_r_bytes):
            img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            cv2.imwrite(str(right_frames_dir / f"{i:05d}.png"), img)

        right_names = [p.stem for p in right_paths[:T_pair]]
        with open(right_dir / f"{seq_base}_index.json", "w") as jf:
            json.dump({"seq": seq_base, "names": right_names}, jf, indent=2)

        # labels & dummy arrays (RIGHT)
        T_r = len(frames_r_bytes)
        in_contact_r  = np.ones((T_r,), dtype=np.float64)
        in_contact_r[0] = 0.0  # first real frame is non-contact
        forces_r       = np.zeros((T_r, 3), dtype=np.float64)
        forces_slip_r  = np.zeros((T_r, 3), dtype=np.float64)
        delta_forces_r = np.zeros((T_r, 3), dtype=np.float64)
        delta_mag_s_r  = np.zeros((T_r,),   dtype=np.float64)
        delta_mag_n_r  = np.zeros((T_r,),   dtype=np.float64)
        slip_label_r   = np.zeros((T_r,),   dtype=np.float64)
        poses_r        = np.zeros((T_r, poses_dim), dtype=np.float64)
        coef_r         = 0.383

        idxs_r = list(range(right_global_idx, right_global_idx + T_r))
        right_global_idx += T_r
        right_in_contact_global.extend(in_contact_r.tolist())
        right_trajectories[right_traj_id] = {
            "indexes": idxs_r,
            "forces": forces_r,
            "forces_slip": forces_slip_r,
            "poses": poses_r,
            "delta_forces": delta_forces_r,
            "delta_mag_shear": delta_mag_s_r,
            "delta_mag_normal": delta_mag_n_r,
            "slip_label": slip_label_r,
            "coef_friction": float(coef_r),
        }
        for local_i, (gidx, nm, ic) in enumerate(zip(idxs_r, right_names, in_contact_r.tolist())):
            right_index_rows.append((seq_base, local_i, gidx, nm, int(ic)))
        right_traj_id += 1

        # ---------------- LEFT sequence -----------------
        frames_l_bytes = [_png_bytes(p) for p in left_paths[:T_pair]]  # no neutral prepend
        with open(left_dir / seq_pkl, "wb") as f:
            pickle.dump(frames_l_bytes, f, protocol=5)

        left_frames_dir = left_dir / f"{seq_base}_frames"
        left_frames_dir.mkdir(exist_ok=True)
        for i, b in enumerate(frames_l_bytes):
            img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            cv2.imwrite(str(left_frames_dir / f"{i:05d}.png"), img)

        left_names = [p.stem for p in left_paths[:T_pair]]
        with open(left_dir / f"{seq_base}_index.json", "w") as jf:
            json.dump({"seq": seq_base, "names": left_names}, jf, indent=2)

        # labels & dummy arrays (LEFT)
        T_l = len(frames_l_bytes)
        in_contact_l  = np.ones((T_l,), dtype=np.float64)
        in_contact_l[0] = 0.0
        forces_l       = np.zeros((T_l, 3), dtype=np.float64)
        forces_slip_l  = np.zeros((T_l, 3), dtype=np.float64)
        delta_forces_l = np.zeros((T_l, 3), dtype=np.float64)
        delta_mag_s_l  = np.zeros((T_l,),   dtype=np.float64)
        delta_mag_n_l  = np.zeros((T_l,),   dtype=np.float64)
        slip_label_l   = np.zeros((T_l,),   dtype=np.float64)
        poses_l        = np.zeros((T_l, poses_dim), dtype=np.float64)
        coef_l         = 0.383

        idxs_l = list(range(left_global_idx, left_global_idx + T_l))
        left_global_idx += T_l
        left_in_contact_global.extend(in_contact_l.tolist())
        left_trajectories[left_traj_id] = {
            "indexes": idxs_l,
            "forces": forces_l,
            "forces_slip": forces_slip_l,
            "poses": poses_l,
            "delta_forces": delta_forces_l,
            "delta_mag_shear": delta_mag_s_l,
            "delta_mag_normal": delta_mag_n_l,
            "slip_label": slip_label_l,
            "coef_friction": float(coef_l),
        }
        for local_i, (gidx, nm, ic) in enumerate(zip(idxs_l, left_names, in_contact_l.tolist())):
            left_index_rows.append((seq_base, local_i, gidx, nm, int(ic)))
        left_traj_id += 1

    if write_dummy_pickles:
        left_bundle = {
            "in_contact": np.asarray(left_in_contact_global, dtype=np.float64),
            "trajectories": left_trajectories,
        }
        right_bundle = {
            "in_contact": np.asarray(right_in_contact_global, dtype=np.float64),
            "trajectories": right_trajectories,
        }
        with open(left_dir / "dataset_slip_forces.pkl", "wb") as f:
            pickle.dump(left_bundle, f, protocol=5)
        with open(right_dir / "dataset_slip_forces.pkl", "wb") as f:
            pickle.dump(right_bundle, f, protocol=5)

        for side_dir, rows in ((left_dir, left_index_rows), (right_dir, right_index_rows)):
            if rows:
                with open(side_dir / "index_map.csv", "w", newline="") as cf:
                    w = csv.writer(cf)
                    w.writerow(["seq", "local_idx", "global_idx", "name", "in_contact"])
                    w.writerows(rows)

        print(f"[✓] wrote bundles + index maps in {left_dir} and {right_dir}")

    print(f"[✓] wrote per-sequence lists in {left_dir} and {right_dir}")


def split_and_copy_sequences(
    df_matched: pd.DataFrame,
    output_root: Path,
    no_force_reference_img_left: str,
    no_force_reference_img_right: str,
    gap_ns: int = int(40e6),      # 40 ms if timestamps are in ns
    side_col_right: str = "image_path",  # column with right image paths
    ts_col: str = "timestamp_img",
):
    # 0) sort chronologically
    df = df_matched.sort_values(ts_col).reset_index(drop=True)

    # 1) compute gaps between consecutive frames
    # gap[i] = ts[i] - ts[i-1]; first gap = 0
    gaps = df[ts_col].diff().fillna(0).to_numpy()

    # 2) start a new group when gap > threshold
    new_seq = (np.abs(gaps) > gap_ns).astype(int)
    seq_ids = new_seq.cumsum()  # 0,0,0,1,1,2,...

    # get reference images
    ref_left  = Path(no_force_reference_img_left)
    ref_right = Path(no_force_reference_img_right)

    (output_root / "digit_left").mkdir(parents=True, exist_ok=True)
    (output_root / "digit_right").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ref_left,  output_root / "digit_left"  / "no_force_reference.png")
    shutil.copy2(ref_right, output_root / "digit_right" / "no_force_reference.png")

    # 3) iterate sequences
    for seq_idx, seq_df in df.groupby(seq_ids):
        if len(seq_df) < 2:
            continue  # skip single-frame sequences

        # build output dirs (no stray spaces)
        out_right = output_root / f"digit_right/seq_{seq_idx:03d}/frames"
        out_left  = output_root / f"digit_left/seq_{seq_idx:03d}/frames"
        out_right.mkdir(parents=True, exist_ok=True)
        out_left.mkdir(parents=True, exist_ok=True)

        # 4) copy files
        for p_right in seq_df[side_col_right].to_numpy():
            p_right = Path(p_right)
            # Derive left path robustly: if you have it in the DF, use that column instead.
            # If not, and the layout is mirrored on disk:
            p_left = Path(str(p_right).replace("/right/", "/left/").replace("digit_right", "digit_left"))

            # Optionally check existence before copying
            if not p_right.exists():
                print(f"[warn] missing right image: {p_right}")
                continue
            if not p_left.exists():
                print(f"[warn] missing left image:  {p_left}")
                continue

            shutil.copy2(p_right, out_right / p_right.name)
            shutil.copy2(p_left,  out_left  / p_left.name)

def images_to_df(image_paths):
    rows = []
    for p in image_paths:
        ts_str = Path(p).stem  # "1234567890123456789"
        try:
            ts = int(ts_str)   # nanoseconds assumed
        except ValueError:
            continue
        rows.append({"timestamp": ts, "image_path": str(p)})
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def attach_nearest_force_to_digit_images(df_L, df_right, tol_ms=10):

    """ Attach nearest force value to digit images within tolerance."""
    tol_ns = int(tol_ms * 1e6)


    L = df_L.sort_values("timestamp").astype({"timestamp":"int64"})
    imgs = (df_right.sort_values("timestamp")
                  .astype({"timestamp":"int64"})
                  .rename(columns={"timestamp":"timestamp_img"}))

    matched = pd.merge_asof(
        imgs, L.rename(columns={"timestamp":"timestamp_L"}),
        left_on="timestamp_img", right_on="timestamp_L",
        direction="nearest", tolerance=tol_ns
    ).dropna(subset=["timestamp_L","L_star"])

    matched["delta_t_s"] = (matched["timestamp_img"] - matched["timestamp_L"]) * 1e-9
    return matched



if __name__ == "__main__":
    ################################################3
    # recording location
    ################################################3
    rec_location = "bedroom_4"
    base_path = Path(f"/data/ikea_recordings")
    data_indexer = RecordingIndex(
        os.path.join(str(base_path), "extracted") 
    )
    path_docker_root_odometry = Path("/exchange/hoi-dataset-tools/data_processing/docker/odometry")
    # interaction_index = "1-6"
    color = "blue"
    visualize = False

    rec_type = "gripper"
    rec_module = "gripper"
    interaction_indices = "12-14"

    gripper_data = GripperData(base_path, 
                               rec_loc=rec_location, 
                               rec_type=rec_type, 
                               rec_module=rec_module, 
                               interaction_indices=interaction_indices,
                               data_indexer=data_indexer,
                               color=color,)

    generate_contact_force_estimation_pseudo_ground_truth_from_ft_and_motor_states(
        gripper_data, 
        output_path=f"/data/evaluations/contact_force_estimation/outputs_sparsh/groundtruth/{rec_location}/{interaction_indices}_gt.csv",
        rec_location=rec_location,
        interaction_indices=interaction_indices
    )   


    a = 2