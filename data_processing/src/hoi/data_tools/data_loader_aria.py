from pathlib import Path
from typing import Optional, Tuple, List, Dict
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import RecordableTypeId, StreamId
import cv2
from tqdm import tqdm
import pandas as pd
import numpy as np
import gzip
import open3d as o3d
import os
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import json
from .utils import ensure_dir, save_image, clean_label, estimate_fps, load_sorted_images, is_valid_image
from .mps_request import MPSClient
from .data_indexer import RecordingIndex
import torch
from transformers import pipeline
from accelerate.test_utils.testing import get_backend
from PIL import Image
import matplotlib
matplotlib.use("TkAgg") 
import matplotlib.pyplot as plt

from PIL import Image
import math
from torchvision import transforms, models
import cv2
import shutil

from .utils_keyframing import KeyframeExtractor
from .utils_anonymization import EgoBlurFaceAnonymizer

class AriaData:

    ARIA_USERNAME = os.environ.get("ARIA_USERNAME")  # set via env; no credentials in source
    ARIA_PASSWORD = os.environ.get("ARIA_PASSWORD")  # MPSClient falls back to an interactive prompt if unset

    # monodepth model setup
    DEVICE, _, _ = get_backend()
    MONO_DEPTH_CHECKPOINT = "depth-anything/Depth-Anything-V2-base-hf"
    PIPE_MONO_DEPTH = pipeline("depth-estimation", model=MONO_DEPTH_CHECKPOINT, device=DEVICE)


    def __init__(self, base_path: Path, 
                 rec_loc: str, 
                 rec_type: str, 
                 rec_module: str, 
                 interaction_indices: str,
                 data_indexer: Optional[RecordingIndex] = None,
                 voxel: float = 0.02*6):

        self.voxel = voxel
        self.rec_loc = rec_loc
        self.base_path = base_path
        self.rec_module = rec_module
        self.rec_type = rec_type
        self.interaction_indices = interaction_indices

        self.extraction_path_base = self.base_path / "extracted" / self.rec_loc / self.rec_type
        self.extraction_path = self.base_path / "extracted" / self.rec_loc / self.rec_type / self.rec_module / f"{self.rec_loc}_{self.interaction_indices}_{self.rec_type}_vrs"

        self.mps_path_raw = self.base_path / "raw" / self.rec_loc / self.rec_type / self.rec_module / f"mps_{self.rec_loc}_{self.interaction_indices}_{self.rec_type}_vrs"
        self.vrs_file_raw = self.base_path / "raw" / self.rec_loc / self.rec_type / self.rec_module / f"{self.rec_loc}_{self.interaction_indices}_{self.rec_type}.vrs"
        self.mps_path_raw_all_devices = self.base_path / "raw" / self.rec_loc / "mps_all_devices"

        self.label_rgb = f"/camera_rgb"
        self.label_rgb_raw = f"/camera_rgb_raw"
        self.label_slam = f"/slam"
        self.label_hand_tracking = f"/hand_tracking"
        self.label_eye_gaze = f"/eye_gaze"
        self.label_keyframes = f"visual_registration/keyframes/rgb"
        self.label_keyframes_raw = f"/keyframes_raw/rgb"
        self.label_depth = f"/camera_depth"

        self.label_clt = f"{self.label_slam}/closed_loop_trajectory"
        self.label_sdp = f"{self.label_slam}/semidense_points"
        self.label_sdpd = f"{self.label_slam}/semidense_points_downsampled"

        self.label_clt_aligned = f"{self.label_slam}/closed_loop_trajectory_aligned"
        self.label_sdp_aligned = f"{self.label_slam}/semidense_points_aligned"
        self.label_sdpd_aligned = f"{self.label_slam}/semidense_points_downsampled_aligned"
        self.label_hand_tracking_aligned = f"{self.label_hand_tracking}/hand_tracking_aligned"
        self.label_palm_and_wrist_tracking_aligned = f"{self.label_hand_tracking}/palm_and_wrist_tracking_aligned"
        self.label_eye_gaze_aligned = f"{self.label_eye_gaze}/general_eye_gaze_aligned"

        self.semidense_points_ply_path = self.extraction_path / self.label_sdp.strip("/") / "data.ply"
        self.semidense_points_downsampled_ply_path = self.extraction_path / self.label_sdpd.strip("/") / "data.ply"
        self.visual_registration_output_path = self.extraction_path / "visual_registration"

        self.device_calib = None
        self.provider = None
        self.load_provider()

        self.calibration = self.get_calibration()

        self.extracted_vrs = Path(self.extraction_path / self.label_rgb.strip("/")).exists()
        self.extracted_vrs_raw = Path(self.extraction_path / self.label_rgb_raw.strip("/")).exists()
        self.extracted_mps = Path(self.extraction_path / self.label_slam.strip("/")).exists()
        self.time_aligned = False        

        self.data_indexer = data_indexer

        self.logging_tag = f"{self.rec_loc}_{self.rec_type}_{self.rec_module}".upper()

        self.rgb_extension = ".jpg"  

        self.statistics = {}
        self.anonym_info = {}


    def _extracted(self, label: str) -> bool:
        """
        Check if the data for the given label has been extracted.
        """
        label_path = self.extraction_path / label.strip("/")
        return label_path.exists() and any(label_path.iterdir())

    def load_provider(self):
        if not self.vrs_file_raw.exists():
            print(f"WARNING: VRS file not found: {self.vrs_file_raw}, data \
                          raw data extraction functions will not work.")
            return
        
        self.provider = data_provider.create_vrs_data_provider(str(self.vrs_file_raw))
        if not self.provider:
            raise RuntimeError(f"Failed to create data provider for {self.vrs_file_raw}")

        self.device_calib = self.provider.get_device_calibration()


    def get_calibration(self) -> dict:
        """
        Returns the intrinsic calibration dictionary for the RGB camera.
        If a calibration config file exists, it is loaded and returned.
        Otherwise, it computes the calibration, saves it, and returns it.
        """
        calib_dir = self.extraction_path / "calib"
        calib_path = calib_dir / "calib.json"
        calib_dir.mkdir(parents=True, exist_ok=True)

        # # --- Load cached calibration if present ---
        if calib_path.exists():
            with open(calib_path, "r") as f:
                calib = json.load(f)
            return _load_to_numpy(calib)

        # --- Otherwise compute calibration from device ---
        if not self.device_calib:
            raise RuntimeError("Device calibration not loaded")

        calib = {}

        # PINHOLE (linearized + rotated)
        dc_rgb = self.device_calib.get_camera_calib("camera-rgb")
        f = dc_rgb.get_focal_lengths()[0]
        h0 = dc_rgb.get_image_size()[0]
        w0 = dc_rgb.get_image_size()[1]

        pinhole = calibration.get_linear_camera_calibration(int(w0), int(h0), float(f))
        pinhole_rot = calibration.rotate_camera_calib_cw90deg(pinhole)

        f_x, f_y, c_x, c_y = pinhole_rot.get_projection_params()[:4]
        K = np.array([[f_x, 0,   c_x],
                    [0,   f_y, c_y],
                    [0,   0,   1]], dtype=np.float32)

        clb = {
            "K": K,
            "h": int(h0),
            "w": int(w0),
            "model": "PINHOLE",
            "distortion": np.zeros(5, dtype=np.float32),
            "focal_length": np.array([f_x, f_y], dtype=np.float32),
            "principal_point": np.array([c_x, c_y], dtype=np.float32),
            "pinhole_T_device_camera": pinhole_rot.get_transform_device_camera().to_matrix(),
            "T_device_camera": dc_rgb.get_transform_device_camera().to_matrix(),
            "colmap_camera_cfg": {
                "model": "PINHOLE",
                "width": int(w0),     # ensure plain int
                "height": int(h0),    # ensure plain int
                "params": [float(f_x), float(f_y), float(c_x), float(c_y)],
            },
        }
        calib["PINHOLE"] = clb

        # NON_PINHOLE (native fisheye)
        calib_rgb = dc_rgb
        h1, w1 = calib_rgb.get_image_size()
        f_x2, f_y2 = calib_rgb.get_focal_lengths()
        c_x2, c_y2 = calib_rgb.get_principal_point()

        K2 = np.array([[f_x2, 0,    c_x2],
                    [0,    f_y2,  c_y2],
                    [0,    0,     1]], dtype=np.float32)

        clb2 = {
            "K": K2,
            "h": int(h1),
            "w": int(w1),
            "model": "FISHEYE_624",
            "distortion": np.array(calib_rgb.get_projection_params()[3:7], dtype=np.float32),
            "focal_length": np.array([f_x2, f_y2], dtype=np.float32),
            "principal_point": np.array([c_x2, c_y2], dtype=np.float32),
            "T_device_camera": calib_rgb.get_transform_device_camera().to_matrix(),
            "colmap_camera_cfg": {},  # as you had it
        }
        calib["NON_PINHOLE"] = clb2

        # --- Save (after sanitizing numpy types) ---
        with open(calib_path, "w") as f:
            json.dump(_to_jsonable(calib), f, indent=2)

        return calib
        
    def request_mps(self, force: bool = False) -> None:

        if self.mps_path_raw.exists() or self._extracted(self.label_slam):
            print(f"[{self.logging_tag}] MPS data already exists at {self.mps_path_raw}")
            return
        
        if not self.vrs_file_raw.exists():
            raise FileNotFoundError(f"VRS file not found: {self.vrs_file_raw}")
        
        print(f"[{self.logging_tag}] Requesting MPS data from {self.vrs_file_raw}")

        mps_client = MPSClient()
        try: # will prompt for credentials
            mps_client.request_single(
                input_path=str(self.vrs_file_raw),
                username=self.ARIA_USERNAME,
                password=self.ARIA_PASSWORD,
                features=["SLAM", "HAND_TRACKING", "EYE_GAZE"],
                force=force,
                no_ui=True
            )
            print(f"[{self.logging_tag}] MPS data requested successfully")
        except Exception as e:
            print(f"[{self.logging_tag}] Failed to request MPS data: {e}")

    def request_mps_all_devices(self, force: bool = False) -> None:

        mps_client = MPSClient()

        all_vrs_files = self.data_indexer.vrs_files(
            location=self.rec_loc
        )

        if self.mps_path_raw_all_devices.exists() and len(list(self.mps_path_raw_all_devices.iterdir())) >= len(all_vrs_files) and not force:
            print(f"[{self.logging_tag}] MPS data for all devices already exists at {self.mps_path_raw_all_devices}")
            return
        
        if not self.mps_path_raw_all_devices.exists():
            ensure_dir(self.mps_path_raw_all_devices)
        
        # try:
        mps_client.request_multi(
            input_paths=all_vrs_files,
            output_dir=self.mps_path_raw_all_devices,
            force=force,
            no_ui=True
        )
        print(f"[{self.logging_tag}] MPS data requested successfully")
        # except Exception as e:
        #     print(f"[{self.logging_tag}] Failed to request MPS data: {e}")


    def extract_vrs(self, undistort: bool = True):

        if undistort and self.extracted_vrs:
            print(f"[{self.logging_tag}] VRS data already extracted to {self.extraction_path}")
            return
        if not undistort and self.extracted_vrs_raw:
            print(f"[{self.logging_tag}] VRS data already extracted to {self.extraction_path}")
            return

        if not self.vrs_file_raw:
            raise FileNotFoundError(f"No vrs file found for {self.vrs_file_raw}")

        if not undistort:
            out_dir = self.extraction_path / self.label_rgb_raw.strip("/")
            self.extracted_vrs_raw = True
        else:
            out_dir = self.extraction_path / self.label_rgb.strip("/")
            self.extracted_vrs = True

        ensure_dir(out_dir)

        calib = self.device_calib.get_camera_calib("camera-rgb")
        # pinhole = calibration.get_linear_camera_calibration(512, 512, 150)

        f = self.device_calib.get_camera_calib("camera-rgb").get_focal_lengths()[0]
        h = self.device_calib.get_camera_calib("camera-rgb").get_image_size()[0]
        w = self.device_calib.get_camera_calib("camera-rgb").get_image_size()[1]

        pinhole = calibration.get_linear_camera_calibration(w, h, f)
        # pinhole_rot = calibration.rotate_camera_calib_cw90deg(pinhole)

        print(f"[{self.logging_tag}] Data provider created successfully")
        stream_id = self.provider.get_stream_id_from_label("camera-rgb")

        for i in tqdm(range(0, self.provider.get_num_data(stream_id)), total=self.provider.get_num_data(stream_id)):
            image_data =  self.provider.get_image_data_by_index(stream_id, i)
            sensor_data = self.provider.get_sensor_data_by_index(stream_id, i)
            ts = sensor_data.get_time_ns(TimeDomain.DEVICE_TIME)
            image_array = image_data[0].to_numpy_array()
            if undistort:
                image_array = calibration.distort_by_calibration(image_array, pinhole, calib)
                image_array = np.rot90(image_array, k=3)
            out_file = out_dir / f"{ts}.{self.rgb_extension.lstrip('.')}"
            image_array= cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)            
            cv2.imwrite(str(out_file), image_array)

        print(f"[{self.logging_tag}] Saved RGB images to {out_dir}")

    def extract_mps(self, mps_path: Optional[str | Path | os.PathLike] = None) -> None:

        if self._extracted(self.label_slam):
            print(f"[{self.logging_tag}] MPS data already extracted to {self.extraction_path / self.label_slam.strip('/')}")
            return

        # Path to closed loop SLAM trajectory
        closed_loop_trajectory_file = self.mps_path_raw / "slam" / "closed_loop_trajectory.csv"  # fixed typo in filename
        semidense_points_file = self.mps_path_raw / "slam" / "semidense_points.csv.gz"

        try:
            df = pd.read_csv(closed_loop_trajectory_file)
        except Exception as e:
            print(f"[{self.logging_tag}] Failed to read CSV {closed_loop_trajectory_file}: {e}")
        
        try:
            df_pts = pd.read_csv(semidense_points_file, compression='gzip')
        except Exception as e:
            print(f"[{self.logging_tag}] Failed to read CSV {semidense_points_file}: {e}")

        # Normalize to 'timestamp' naming
        if "tracking_timestamp_us" in df.columns:
            df["tracking_timestamp_us"] = (df["tracking_timestamp_us"].astype(np.int64) * 1_000)
            df.rename(columns={"tracking_timestamp_us": "timestamp"}, inplace=True)

        # Save to extracted location
        label_clt = f"{self.label_slam}/closed_loop_trajectory"
        csv_dir = self.extraction_path / label_clt.strip("/")

        if not csv_dir.exists():
            ensure_dir(csv_dir)
            df.to_csv(csv_dir / "data.csv", index=False)
            print(f"[{self.logging_tag}] Saved closed loop trajectory CSV: {csv_dir}/data.csv")
        else:
            print(f"[{self.logging_tag}] Closed loop trajectory CSV already exists: {csv_dir}/data.csv")


        label_sdp = f"{self.label_slam}/semidense_points"
        csv_dir = self.extraction_path / label_sdp.strip("/")

        if not csv_dir.exists():
            ensure_dir(csv_dir)
            df_pts.to_csv(csv_dir / "data.csv", index=False)
            print(f"[{self.logging_tag}] Saved semidense points CSV: {csv_dir}/data.csv")
        else:
            print(f"[{self.logging_tag}] Semidense points CSV already exists: {csv_dir}/data.csv")

        # hand and palm tracking
        # palm_tracking_file = self.mps_path_raw / "hand_tracking" / "wrist_and_palm_poses.csv"
        hand_tracking_file = self.mps_path_raw / "hand_tracking" / "hand_tracking_results.csv"

        # try:
        #     df_palm = pd.read_csv(palm_tracking_file)
        # except Exception as e:  
        #     print(f"[{self.logging_tag}] Failed to read CSV {palm_tracking_file}: {e}")

        try:
            df_hand = pd.read_csv(hand_tracking_file)
        except Exception as e:
            print(f"[{self.logging_tag}] Failed to read CSV {hand_tracking_file}: {e}")

        # # Normalize to 'timestamp' naming
        # if "tracking_timestamp_us" in df_palm.columns:
        #     df_palm["tracking_timestamp_us"] = (df_palm["tracking_timestamp_us"].astype(np.int64) * 1_000)
        #     df_palm.rename(columns={"tracking_timestamp_us": "timestamp"}, inplace=True)

        if "tracking_timestamp_us" in df_hand.columns:
            df_hand["tracking_timestamp_us"] = (df_hand["tracking_timestamp_us"].astype(np.int64) * 1_000)
            df_hand.rename(columns={"tracking_timestamp_us": "timestamp"}, inplace=True)

        # # Save palm tracking data
        # label_palm = f"{self.label_hand_tracking}/palm_and_wrist_tracking"
        # csv_dir = self.extraction_path / label_palm.strip("/")
        # if not csv_dir.exists():
        #     ensure_dir(csv_dir)
        #     df_palm.to_csv(csv_dir / "data.csv", index=False)
        #     print(f"[{self.logging_tag}] Saved palm tracking CSV: {csv_dir}/data.csv")
        # else:
        #     print(f"[{self.logging_tag}] Palm tracking CSV already exists: {csv_dir}/data.csv")

        # Save hand tracking data
        label_hand = f"{self.label_hand_tracking}/hand_tracking"
        csv_dir = self.extraction_path / label_hand.strip("/")
        if not csv_dir.exists():
            ensure_dir(csv_dir)
            df_hand.to_csv(csv_dir / "data.csv", index=False)
            print(f"[{self.logging_tag}] Saved hand tracking CSV: {csv_dir}/data.csv")
        else:
            print(f"[{self.logging_tag}] Hand tracking CSV already exists: {csv_dir}/data.csv")

        # eye gaze tracking
        eye_gaze_file = self.mps_path_raw / "eye_gaze" / "general_eye_gaze.csv"   

        try:
            df_eye_gaze = pd.read_csv(eye_gaze_file)
        except Exception as e:
            print(f"[{self.logging_tag}] Failed to read CSV {eye_gaze_file}: {e}")

        # Normalize to 'timestamp' naming
        if "tracking_timestamp_us" in df_eye_gaze.columns: 
            df_eye_gaze["tracking_timestamp_us"] = (df_eye_gaze["tracking_timestamp_us"].astype(np.int64) * 1_000)
            df_eye_gaze.rename(columns={"tracking_timestamp_us": "timestamp"}, inplace=True)

        # Save eye gaze tracking data
        label_eye_gaze = f"{self.label_eye_gaze}/general_eye_gaze"
        csv_dir = self.extraction_path / label_eye_gaze.strip("/")
        if not csv_dir.exists():
            ensure_dir(csv_dir)
            df_eye_gaze.to_csv(csv_dir / "data.csv", index=False)
            print(f"[{self.logging_tag}] Saved eye gaze tracking CSV: {csv_dir}/data.csv")
        else:
            print(f"[{self.logging_tag}] Eye gaze tracking CSV already exists: {csv_dir}/data.csv")


        # Update extracted flag
        self.extracted_mps = True
        print(f"[{self.logging_tag}] Extracted MPS data to {csv_dir}")

    def extract_mps_multi(self, force: bool = False) -> None:
        """
        Extracts multi MPS data for this device.
        """

        all_vrs_files = self.data_indexer.vrs_files(
            location=self.rec_loc,
            interaction_index= self.interaction_indices
            
        )

        if not self.mps_path_raw_all_devices.exists() or len(list(self.mps_path_raw_all_devices.iterdir())) < len(all_vrs_files) :
            raise FileNotFoundError(f"MPS data for all devices not found at {self.mps_path_raw_all_devices}, please run request_mps_all_devices() first.")

        multi_slam_config_file = self.mps_path_raw_all_devices / "vrs_to_multi_slam.json"

        if not multi_slam_config_file.exists():
            raise FileNotFoundError(f"Multi SLAM config file not found: {multi_slam_config_file}. Please run request_mps_all_devices() first.")
        
        with open(multi_slam_config_file, "r") as f:
            multi_slam_config = json.load(f)

        multi_slam_index = multi_slam_config.get(str(self.vrs_file_raw))

        # Path to closed loop SLAM trajectory
        closed_loop_trajectory_file = self.mps_path_raw_all_devices / multi_slam_index / "slam" / "closed_loop_trajectory.csv"  # fixed typo in filename
        semidense_points_file = self.mps_path_raw_all_devices / multi_slam_index / "slam" / "semidense_points.csv.gz"

        try:
            df = pd.read_csv(closed_loop_trajectory_file)
        except Exception as e:
            print(f"[{self.logging_tag}] Failed to read CSV {closed_loop_trajectory_file}: {e}")
            return
        
        try:
            df_pts = pd.read_csv(semidense_points_file, compression='gzip')
        except Exception as e:
            print(f"[{self.logging_tag}] Failed to read CSV {semidense_points_file}: {e}")
            return

        # Normalize to 'timestamp' naming
        if "tracking_timestamp_us" in df.columns:
            df["tracking_timestamp_us"] = (df["tracking_timestamp_us"].astype(np.int64) * 1_000)
            df.rename(columns={"tracking_timestamp_us": "timestamp"}, inplace=True)

        # Save to extracted location
        label_clt = f"multi_slam/closed_loop_trajectory"
        csv_dir = self.extraction_path / label_clt.strip("/")

        if not csv_dir.exists():
            ensure_dir(csv_dir)
            df.to_csv(csv_dir / "data.csv", index=False)
            print(f"[{self.logging_tag}] Saved closed loop trajectory CSV: {csv_dir}/data.csv")
        else:
            print(f"[{self.logging_tag}] Closed loop trajectory CSV already exists: {csv_dir}/data.csv")

        label_sdp = f"multi_slam/semidense_points"
        csv_dir = self.extraction_path / label_sdp.strip("/")

        if not csv_dir.exists():
            ensure_dir(csv_dir)
            df_pts.to_csv(csv_dir / "data.csv", index=False)
            print(f"[{self.logging_tag}] Saved semidense points CSV: {csv_dir}/data.csv")
        else:
            print(f"[{self.logging_tag}] Semidense points CSV already exists: {csv_dir}/data.csv")
            return

        # TODO - add more mps data extraction as needed
        # TODO - UTC timestamp is NOT changed at the moment, only device timestamp!!!!
        
        print(f"[{self.logging_tag}] Extracted multi MPS data to {csv_dir}")


        a = 2

    def extract_video(self, out_dir: Optional[str | Path] = None, undistort: bool = True) -> None:
        """
        Extracts the video from the RGB images in the specified directory.
        """

        if out_dir is None and not undistort:
            out_dir = self.extraction_path / self.label_rgb_raw.strip("/")
        elif out_dir is None and undistort:
            out_dir = self.extraction_path / self.label_rgb.strip("/")

        video_name = out_dir / 'data.mp4'

        # Read and sort images
        images = sorted(
            [img for img in os.listdir(out_dir) if img.lower().endswith((".png", ".jpeg", ".jpg"))],
            key=lambda x: int(os.path.splitext(x)[0]))

        # Estimate average fps
        timestamps = [int(os.path.splitext(img)[0]) for img in images]
        time_diffs = np.diff(timestamps)  # nanoseconds
        avg_dt = np.mean(time_diffs)  # average nanosecond difference
        fps = 1e9 / avg_dt  # frames per second

        print(f"[{self.logging_tag}] Estimated fps: {fps:.2f}")

        # Initialize video writer
        frame = cv2.imread(os.path.join(out_dir, images[0]))
        height, width, layers = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

        # Write frames
        for image in tqdm(images, desc="Creating video", total=len(images)):
            img = cv2.imread(os.path.join(out_dir, image))
            video.write(img)

        video.release()

        print(f"[{self.logging_tag}] Saved video to {out_dir}")

    def extract_mono_depth(
        self,
        downsampling_factor: int = 2,
        batch_size: int = 8,
        force: bool = False,
    ) -> None:
        """Batch the images in pure Python, call PIPE once per batch, save each depth map."""
        if not self.extracted_vrs:
            raise FileNotFoundError(f"[{self.logging_tag}] …")

        out_dir = self.extraction_path / self.label_depth.strip("/")
        if out_dir.exists() and any(out_dir.glob("*.npy")) and not force:
            print(f"[{self.logging_tag}] already extracted → {out_dir}")
            return
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) Gather & sort all the image paths
        img_paths = sorted(
            self.extraction_path.glob(f"{self.label_rgb.strip('/')}/**/*.{self.rgb_extension.lstrip('.')}"),
        )
        total = len(img_paths)
        if total == 0:
            print(f"[{self.logging_tag}] no images found")
            return

        # 2) Process in Python chunks of batch_size
        n_batches = math.ceil(total / batch_size)
        pbar = tqdm(total=total, desc=f"[{self.logging_tag}] Processing monodepth batches", unit="batch")
        for i in range(n_batches):
            start = i * batch_size
            end   = min(start + batch_size, total)
            batch_paths = img_paths[start:end]

            # load + preprocess into PIL list
            batch_pils = []
            stems      = []
            for p in batch_paths:
                img = cv2.imread(str(p))
                if img is None:
                    print(f"[{self.logging_tag}] skipping {p}")
                    continue
                h, w = img.shape[:2]
                img = cv2.resize(img, (w // downsampling_factor, h // downsampling_factor))
                batch_pils.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                stems.append(p.stem)

            # 3) Single batched call
            with torch.no_grad():
                preds = self.PIPE_MONO_DEPTH(batch_pils)  # returns list of dicts

            # 4) Save out each result
            for pred, stem in zip(preds, stems):
                depth_map = pred["predicted_depth"].squeeze().cpu().numpy()
                np.save(out_dir / f"{stem}.npy", depth_map)

            pbar.update(len(batch_pils))

        print(f"[{self.logging_tag}] done! depth maps in {out_dir}")

    def extract_keyframes(self, stride: int = 2, n_keyframes: int = 20, force: bool = False) -> None:

        if self._extracted(self.label_keyframes) and not force:
            print(f"[{self.logging_tag}] Keyframes already extracted to {self.visual_registration_output_path / self.label_keyframes.strip('/')}")
            return
        
        if not (
            self._extracted(self.label_rgb)
            or self._extracted(self.label_depth)
            or self._extracted(self.label_slam)
        ):
            raise FileNotFoundError(
            f"[{self.logging_tag}] No RGB, depth, or SLAM data found in "
            f"{self.extraction_path / self.label_rgb.strip('/')} or "
            f"{self.extraction_path / self.label_depth.strip('/')} or "
            f"{self.extraction_path / self.label_slam.strip('/')}"
            )
        
        out_dir = self.extraction_path / self.label_keyframes.strip("/")
        ensure_dir(out_dir)

        rgb_files = sorted(
            self.extraction_path.glob(f"{self.label_rgb.strip('/')}/**/*.{self.rgb_extension.lstrip('.')}"),
            key=lambda x: int(x.stem)
        )

        depth_files = sorted(
            self.extraction_path.glob(f"{self.label_depth.strip('/')}/**/*.npy"),
            key=lambda x: int(x.stem)
        )
    
        # setup the keyframe extractor
        keyframe_extractor = KeyframeExtractor(
            rgb_files=rgb_files,
            depth_files=depth_files,
            n_keyframes=n_keyframes,
            stride=stride)

        if not any(self.statistics.values()):
            print(f"[{self.logging_tag}] No statistics found, computing...")
            self.get_statistics(stride=stride)
        
        keyframe_extractor.statistics = self.statistics
        self.selected_keyframes = keyframe_extractor.extract_keyframes()
        out_dir.mkdir(parents=True, exist_ok=True)

        # write the list
        with open(out_dir / "keyframes.txt", "w") as f:
            for p in self.selected_keyframes:
                f.write(p.name + "\n")

        # copy the actual image files
        for p in self.selected_keyframes:
            dst = out_dir / p.name
            shutil.copy2(p, dst)  # preserves metadata

        print(f"[{self.logging_tag}] Wrote {len(self.selected_keyframes)} keyframes to {out_dir}")

    def anonymize_extracted_frames(
        self,
        model_dir: Optional[Path] = None,
        tmp_dir: Optional[Path] = None,
        max_face_image_size: int = 640,
        force: bool = False,
    ) -> Dict[str, int]:
        """
        Run EgoBlur face anonymization *in place* on all extracted RGB frames.

        - Uses EgoBlurFaceAnonymizer.
        - Overwrites original JPGs only if faces are detected.
        - Writes/reads anonymization.json to avoid re-running if already done.
        """

        # --- sanity check: do we have RGB frames? ---
        if not self._extracted(self.label_rgb):
            raise FileNotFoundError(
                f"[{self.logging_tag}] No RGB data found in "
                f"{self.extraction_path / self.label_rgb.strip('/')}"
            )

        rgb_files = self.get_extracted_frames()
        if len(rgb_files) == 0:
            print(f"[{self.logging_tag}] No extracted RGB frames to anonymize.")
            return {"faces": 0}

        # --- load existing anonymization info ---
        anonym_info_path = self.extraction_path / "anonymization.json"
        self.load_anonym_info(anonym_info_path if anonym_info_path.exists() else None)

        stream_key = self.label_rgb  # e.g. "/camera_rgb"

        # if already anonymized and not forced, skip
        if not force and stream_key in self.anonym_info:
            entry = self.anonym_info[stream_key]
            if entry.get("faces_anonymized", False):
                print(
                    f"[{self.logging_tag}] Stream {stream_key} already anonymized "
                    f"(faces_anonymized=True). Skipping."
                )
                return {"faces": entry.get("num_faces", 0)}

        # --- defaults for model & tmp dirs ---
        if model_dir is None:
            model_dir = self.base_path / "ego_blur_weights"

        if tmp_dir is None:
            tmp_dir = self.extraction_path / "anonymization_cache" / "egoblur_faces"

        print(
            f"[{self.logging_tag}] Running EgoBlur face anonymization in-place on "
            f"{len(rgb_files)} frames.\n"
            f"  model_dir={model_dir}\n"
            f"  tmp_dir={tmp_dir}"
        )

        anonymizer = EgoBlurFaceAnonymizer(anonymization_dir=model_dir)

        counts = anonymizer.run_anonymization(
            image_paths=rgb_files,
            tmp_dir=tmp_dir,
            outdir=None,                 # ignored when inplace=True
            inplace=True,                # in-place overwrite
            overwrite=True,              # irrelevant for inplace but explicit
            max_face_image_size=max_face_image_size,
        )

        # --- update anonymization info & save ---
        self.anonym_info[stream_key] = {
            "faces_anonymized": True,
            "num_faces": counts.get("faces", 0),
            "num_images": len(rgb_files),
            "method": "EgoBlurFaceAnonymizer",
            "inplace": True,
        }
        self.save_anonym_info(anonym_info_path)

        print(
            f"[{self.logging_tag}] EgoBlur finished: "
            f"{counts['faces']} faces blurred across {len(rgb_files)} frames (in-place)."
        )
        return counts
    
    def get_closed_loop_trajectory(self) -> pd.DataFrame:
        """
        Returns the closed loop trajectory as a pandas DataFrame.
        """

        csv_dir = self.extraction_path / self.label_clt.strip("/") / "data.csv"

        if not Path(csv_dir).exists():
            raise FileNotFoundError(f"Closed loop trajectory CSV not found: {csv_dir}")
        
        df = pd.read_csv(csv_dir)
        return df
    
    def get_hand_tracking(self) -> pd.DataFrame:
        """
        Returns the palm tracking data as a pandas DataFrame.
        """

        csv_dir = self.extraction_path / self.label_hand_tracking.strip("/") / "hand_tracking" / "data.csv"
    
        if not csv_dir.exists():
            raise FileNotFoundError(f"Hand tracking CSV not found: {csv_dir}")
        
        df = pd.read_csv(csv_dir)
        return df
    
    def get_palm_and_wrist_tracking(self) -> pd.DataFrame:
        """
        Returns the palm tracking data as a pandas DataFrame.
        """
        
        csv_dir = self.extraction_path / self.label_hand_tracking.strip("/") / "palm_and_wrist_tracking" / "data.csv"
        if not csv_dir.exists():
            raise FileNotFoundError(f"Palm tracking CSV not found: {csv_dir}")
        
        df = pd.read_csv(csv_dir)
        return df
    

    def get_closed_loop_trajectory_aligned(self) -> pd.DataFrame:
        """ Returns the closed loop trajectory aligned as a pandas DataFrame.
        """

        csv_dir = self.extraction_path / self.label_clt_aligned.strip("/") / "data.csv"
        if not csv_dir.exists():
            raise FileNotFoundError(f"Closed loop trajectory aligned CSV not found: {csv_dir}")
        
        df = pd.read_csv(csv_dir)
        return df

    def get_seidense_points_aligned_df(self) -> pd.DataFrame:
        pass

    def get_hand_tracking_aligned_df(self) -> pd.DataFrame:
        """ Returns the hand tracking aligned data as a pandas DataFrame.
        """
        
        csv_dir = self.extraction_path / self.label_hand_tracking_aligned.strip("/") / "data.csv"
        if not csv_dir.exists():
            raise FileNotFoundError(f"Hand tracking aligned CSV not found: {csv_dir}")
        
        df = pd.read_csv(csv_dir)
        return df

    def get_palm_and_wrist_tracking_aligned(self) -> pd.DataFrame:
        """ Returns the palm tracking aligned data as a pandas DataFrame.
        """
        
        csv_dir = self.extraction_path / self.label_palm_and_wrist_tracking_aligned.strip("/") / "data.csv"
        if not csv_dir.exists():
            raise FileNotFoundError(f"Palm tracking aligned CSV not found: {csv_dir}")
        
        df = pd.read_csv(csv_dir)
        return df
    
    def get_semidense_points_df(self) -> Optional[pd.DataFrame]:
        """
        Returns the semidense points as a pandas DataFrame.
        """

        csv_dir = self.extraction_path / self.label_sdp.strip("/") / "data.csv"
        if not csv_dir.exists():
            raise FileNotFoundError(f"Semidense points CSV not found: {csv_dir}")
        
        df = pd.read_csv(csv_dir)
        return df
    
    def get_semidense_points_pcd(self, force: bool = False) -> o3d.geometry.PointCloud:
        """Return the full-resolution cloud, converting from E57 if needed."""

        if not self.semidense_points_ply_path.exists() or force:
            self._points_raw_to_ply()
        return o3d.io.read_point_cloud(str(self.semidense_points_ply_path))
    
    def get_downsampled(self, force: bool = False) -> Tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
        """
        Returns downsampled_cloud.  Caches both to disk.
        """

        if force or not (self.semidense_points_downsampled_ply_path.exists()):
            self._make_downsampled()

        down = o3d.io.read_point_cloud(str(self.semidense_points_downsampled_ply_path))

        return down
    
    def get_mps_pose_at_timestamp(self, timestamp: int, aligned: int = False) -> Optional[np.ndarray]:
        
        if not aligned:
            trajectory_df = self.get_closed_loop_trajectory()
        else:
            trajectory_df = self.get_closed_loop_trajectory_aligned()

        if trajectory_df is None:
            print(f"[!] No closed loop trajectory data found. re-extract MPS data.")
            return None
        
        # Find the closest timestamp
        # closest_index = (np.abs(trajectory_df["timestamp"] - timestamp)).idxmin()
        
        # find next larger and smaller timestamp
        closest_index_later = trajectory_df[trajectory_df["timestamp"] >= timestamp].index.min()
        closest_index_prev = trajectory_df[trajectory_df["timestamp"] < timestamp].index.max()
    
        if timestamp < trajectory_df["timestamp"].iloc[0]:
            return None
            raise ValueError(f"Timestamp {timestamp} is before the first timestamp in the trajectory.")
        
        if timestamp > trajectory_df["timestamp"].iloc[-1]:
            return None
            raise ValueError(f"Timestamp {timestamp} is after the last timestamp in the trajectory.")

        if closest_index_prev is np.nan:
            # TODO assign it timestamp it currently has, no interpoaltion
            closest_index = (np.abs(trajectory_df["timestamp"] - timestamp)).idxmin()
            closest_row = trajectory_df.iloc[closest_index]
            t_world_device = closest_row[["tx_world_device", "ty_world_device", "tz_world_device"]].to_numpy()
            q_world_device = closest_row[["qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device"]].to_numpy()

            # Convert quaternion to rotation matrix
            r = R.from_quat(q_world_device)
            R_world_device = r.as_matrix()
            T_world_device = np.eye(4)
            T_world_device[:3, :3] = R_world_device
            T_world_device[:3, 3] = t_world_device
            return T_world_device
        
        # interpolate poses
        timestamp_prev = trajectory_df["timestamp"].iloc[closest_index_prev]
        timestamp_later = trajectory_df["timestamp"].iloc[closest_index_later]

        row_prev = trajectory_df.iloc[closest_index_prev]
        row_later = trajectory_df.iloc[closest_index_later]

        t_world_device_prev = row_prev[["tx_world_device", "ty_world_device", "tz_world_device"]].to_numpy()
        t_world_device_later = row_later[["tx_world_device", "ty_world_device", "tz_world_device"]].to_numpy()
        q_world_device_prev = row_prev[["qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device"]].to_numpy()
        q_world_device_later = row_later[["qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device"]].to_numpy()

        r_prev = R.from_quat(q_world_device_prev)
        r_later = R.from_quat(q_world_device_later)

        alpha = (timestamp - timestamp_prev) / (timestamp_later - timestamp_prev) 

        t_world_device = (1 - alpha) * t_world_device_prev + alpha * t_world_device_later

        slerp = Slerp([timestamp_prev, timestamp_later], R.concatenate([r_prev, r_later]))
        rot = slerp(timestamp)

        R_world_device = rot.as_matrix()

        T_world_device = np.eye(4)
        T_world_device[:3, :3] = R_world_device
        T_world_device[:3, 3] = t_world_device

        return T_world_device
        # TODO - add more mps data extraction as needed
        # TODO interpolate pose between timestamps if needed

    def get_transform_world_query(self) -> np.ndarray:
        """
        Returns the transform from world to device coordinates.
        """

        if not Path(self.visual_registration_output_path / "T_wq.json").exists():
            raise FileNotFoundError(f"Transform file not found: {self.visual_registration_output_path / 'T_wq.json'}. Please run the visual and pointcloud registration first.")
    
        with open(self.visual_registration_output_path / "T_wq.json", "r") as f:
            transform = json.load(f)

        self.T_wq = transform["T_wq"]

        return self.T_wq

    def _points_raw_to_ply(self, voxel: float | None = None) -> None:
        """
        Converts the semidense points DataFrame to a PLY file.
        """

        # Convert DataFrame to PLY format
        df = self.get_semidense_points_df()
        xyz = df[["px_world", "py_world", "pz_world", "inv_dist_std", "dist_std"]].to_numpy(dtype=np.float32)

        xyz = xyz[~np.isnan(xyz).any(axis=1)]

        mask = (xyz[:, 3] <= 0.005) & (xyz[:, 4] <= 0.01)
        xyz = xyz[mask]
        xyz = xyz[:, :3]
        
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        if voxel:
            pcd = pcd.voxel_down_sample(voxel)
        
        with tqdm(total=1, desc="Saving PLY", unit="file") as pbar:
            o3d.io.write_point_cloud(str(self.semidense_points_ply_path), pcd, write_ascii=False)
            pbar.update(1)        
        
        print(f"[{self.logging_tag}] Saved full-resolution PLY → {self.semidense_points_ply_path}")

    def _make_downsampled(self) -> None:
        """
        Down-sample the semidense points.
        """

        ensure_dir(self.semidense_points_downsampled_ply_path.parent)

        print(f"[{self.logging_tag}] Loading semidense points...")
        full = self.get_semidense_points_pcd()  # ensures .ply exists

        with tqdm(total=4, desc="[{self.logging_tag}] Downsample", unit="step") as pbar:
            print(f"[{self.logging_tag}] Down-sampling at voxel={self.voxel:.3f}")
            down = full.voxel_down_sample(voxel_size=self.voxel)
            pbar.update(1)

            print("[{self.logging_tag}] Estimating normals...")
            down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.voxel * 2.0, max_nn=30
                )
            )
            pbar.update(1)

            print("[{self.logging_tag}] Saving downsampled cloud..") # stored in PLY comment
            o3d.io.write_point_cloud(str(self.semidense_points_downsampled_ply_path), down, write_ascii=False)
            pbar.update(1)

        print(
            f"[{self.logging_tag}] Cached ↓ cloud → {self.semidense_points_downsampled_ply_path.name}"
        )

    def get_extracted_frames(self):

        """
        Returns the list of extracted RGB frames.
        """

        if not self._extracted(self.label_rgb):
            raise FileNotFoundError(f"[{self.logging_tag}] No RGB data found in {self.extraction_path / self.label_rgb.strip('/')}")

        rgb_files = sorted(
            self.extraction_path.glob(f"{self.label_rgb.strip('/')}/**/*.{self.rgb_extension.lstrip('.')}"),
            key=lambda x: int(x.stem)
        )

        return rgb_files
    
    def get_extracted_depth_frames(self):

        """
        Returns the list of extracted depth frames.
        """

        if not self._extracted(self.label_depth):
            raise FileNotFoundError(f"[{self.logging_tag}] No depth data found in {self.extraction_path / self.label_depth.strip('/')}")

        depth_files = sorted(
            self.extraction_path.glob(f"{self.label_depth.strip('/')}/**/*.npy"),
            key=lambda x: int(x.stem)
        )

        return depth_files

    def get_statistics(self, stride: int = 1, visualize: bool = False, force: bool = False) -> None:
        """
        Computes and visualizes statistics from the RGB and depth data.
        Strides the RGB and depth files by the given stride.
        """

        if not self._extracted(self.label_depth) or not self._extracted(self.label_rgb):
            raise FileNotFoundError(f"[{self.logging_tag}] RGB or depth data not extracted to {self.extraction_path}")

        statistics_file = self.extraction_path / "statistics.json"
        if statistics_file.exists() and not force:
            print(f"[{self.logging_tag}] Statistics already computed and saved to {statistics_file}")
            self.load_statistics()
            return

        # get RGB files sorted by timestamp
        rgb_files = sorted(
            self.extraction_path.glob(f"{self.label_rgb.strip('/')}/**/*{self.rgb_extension}"),
            key=lambda x: int(x.stem)
        )

        # get depth files sorted by timestamp
        depth_files = sorted(
            self.extraction_path.glob(f"{self.label_depth.strip('/')}/**/*.npy"),
            key=lambda x: int(x.stem)
        )

        keyframe_extractor = KeyframeExtractor(
            rgb_files=rgb_files,
            depth_files=depth_files,
            n_keyframes=20,
            stride=stride
        )

        self.statistics = keyframe_extractor.get_statistics(
            force=force
        )
        
        # Save the statistics to a JSON file
        self.save_statistics(statistics_file)


    def save_statistics(self, out_path: str | Path | None = None) -> None:
        """
        Saves the computed statistics to a JSON file.
        """
        if not self.statistics:
            raise ValueError(f"[{self.logging_tag}] No statistics computed yet. Call get_statistics() first.")

        if out_path is None:
            out_path = self.extraction_path / "statistics.json"

        with open(out_path, 'w') as f:
            json.dump(self.statistics, f, indent=4)
        print(f"[{self.logging_tag}] Statistics saved to {out_path}")

    def load_statistics(self, in_path: str | Path | None = None) -> None:
        """
        Loads the statistics from a JSON file.
        """
        if in_path is None:
            in_path = self.extraction_path / "statistics.json"

        if not Path(in_path).exists():
            raise FileNotFoundError(f"[{self.logging_tag}] Statistics file not found: {in_path}")

        with open(in_path, 'r') as f:
            self.statistics = json.load(f)
        print(f"[{self.logging_tag}] Statistics loaded from {in_path}")


    def save_anonym_info(self, out_path: str | Path | None = None) -> None:
        """
        Saves anonymization metadata (what streams were anonymized) to a JSON file.
        """
        if out_path is None:
            out_path = self.extraction_path / "anonymization.json"

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w") as f:
            json.dump(self.anonym_info, f, indent=4)
        print(f"[{self.logging_tag}] Anonymization info saved to {out_path}")

    def load_anonym_info(self, in_path: str | Path | None = None) -> None:
        """
        Loads anonymization metadata from JSON into self.anonym_info.
        """
        if in_path is None:
            in_path = self.extraction_path / "anonymization.json"

        in_path = Path(in_path)
        if not in_path.exists():
            self.anonym_info = {}
            print(f"[{self.logging_tag}] No anonymization info found at {in_path}, starting fresh.")
            return

        with open(in_path, "r") as f:
            self.anonym_info = json.load(f)
        print(f"[{self.logging_tag}] Anonymization info loaded from {in_path}")

    def visualize_camera_transforms(
        self,
        axis_size: float = 0.08,
        show_pointcloud: bool = False,
        pointcloud_downsampled: bool = True,
    ) -> None:
        """
        Visualize device, raw RGB camera, and rectified pinhole camera frames.

        Convention:
            p_device = T_device_cam @ p_cam

        Therefore:
            - device frame is at identity
            - a camera frame can be placed in device coordinates by applying
            T_device_cam to its local frame mesh

        This is useful to sanity-check that:
            - T_device_camera is the raw camera extrinsic
            - pinhole_T_device_camera is the rectified camera extrinsic
            - their relative offset/orientation makes sense
        """
        calib = self.calibration["PINHOLE"]

        T_device_camRaw = np.asarray(calib["T_device_camera"], dtype=np.float64)
        T_device_camRect = np.asarray(calib["pinhole_T_device_camera"], dtype=np.float64)

        # Relative transform: raw <- rect
        T_camRaw_camRect = np.linalg.inv(T_device_camRaw) @ T_device_camRect

        print("\n=== Camera transform sanity check ===")
        print("T_device_camRaw:\n", T_device_camRaw)
        print("\nT_device_camRect:\n", T_device_camRect)
        print("\nT_camRaw_camRect = inv(T_device_camRaw) @ T_device_camRect:\n", T_camRaw_camRect)

        def make_frame(T: np.ndarray, size: float):
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
            frame.transform(T)
            return frame

        # Device frame = visualization/world origin
        device_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size * 1.5)

        # Raw and rectified camera frames placed in device coordinates
        raw_frame = make_frame(T_device_camRaw, axis_size)
        rect_frame = make_frame(T_device_camRect, axis_size * 0.8)

        # Small spheres at origins so you can see if frames nearly overlap
        def make_origin_sphere(T: np.ndarray, radius: float):
            s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            s.compute_vertex_normals()
            s.translate(T[:3, 3])
            return s

        raw_origin = make_origin_sphere(T_device_camRaw, axis_size * 0.08)
        rect_origin = make_origin_sphere(T_device_camRect, axis_size * 0.06)

        geoms = [device_frame, raw_frame, rect_frame, raw_origin, rect_origin]

        if show_pointcloud:
            try:
                if pointcloud_downsampled:
                    pcd = self.get_downsampled()
                else:
                    pcd = self.get_semidense_points_pcd()
                geoms.append(pcd)
            except Exception as e:
                print(f"[{self.logging_tag}] Could not load point cloud for visualization: {e}")

        o3d.visualization.draw_geometries(
            geoms,
            window_name="Aria camera transforms",
            width=1400,
            height=1000,
        )
    
def _to_jsonable(x):
    """Recursively convert numpy types to plain Python types for JSON."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x

def _load_to_numpy(calib_dict):
    """Optionally convert certain fields back to numpy arrays."""
    array_fields = {
        "K": np.float32,
        "distortion": np.float32,
        "focal_length": np.float32,
        "principal_point": np.float32,
        "pinhole_T_device_camera": np.float32,
        "T_device_camera": np.float32,
    }
    for _, sub in calib_dict.items():
        for key, dtype in array_fields.items():
            if key in sub:
                sub[key] = np.array(sub[key], dtype=dtype)
    return calib_dict



if __name__ == "__main__":

    test = True
    location = False
    if location:
        # extract data from a specific location
        rec_location = "bedroom_1"
        base_path = Path(f"/data/ikea_recordings")
        # rec_type_aria = "gripper"
        # rec_module = "aria_gripper"
        # interaction_indices = "1-8"
        
        data_indexer = RecordingIndex(
            os.path.join(str(base_path), "raw") 
        )

        aria_queries_at_loc = data_indexer.query(
            location=rec_location, 
            interaction=None, 
            recorder="aria*"
        )

        # Uncomment the following lines to process each aria query
        for loc, inter, rec, ii, path in aria_queries_at_loc:
            print(f"Found recorder: {rec} at {path}")

            rec_type = inter
            rec_module = rec
            interaction_indices = ii

            aria_data = AriaData(base_path, rec_location, rec_type, rec_module, interaction_indices, data_indexer)
            aria_data.request_mps(force=False)
            aria_data.request_mps_all_devices(force=False)
            
            aria_data.extract_vrs(undistort=True)
            aria_data.extract_mps()
            aria_data.extract_mps_multi(force=False)

            # can be done after time synchronization to save time
            aria_data.extract_mono_depth(force=False)

    if test:
        
        rec_location = "bedroom_6"
        base_path = Path(f"/data/ikea_recordings")
        data_indexer = RecordingIndex(
            os.path.join(str(base_path), "raw") 
        )

        aria_queries_at_loc = data_indexer.query(
            location=rec_location, 
            interaction="gripper", 
            recorder="aria_gripper"
        )

        for loc, inter, rec, ii, path in aria_queries_at_loc:
            print(f"Found recorder: {rec} at {path}")

            rec_type = inter
            rec_module = rec
            interaction_indices = ii

            aria_data = AriaData(base_path, rec_location, rec_type, rec_module, interaction_indices, data_indexer)
            aria_data.visualize_camera_transforms()
            a =2 
 
            # aria_data.extract_keyframes()

        a = 2