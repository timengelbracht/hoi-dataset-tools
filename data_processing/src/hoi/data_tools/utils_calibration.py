import cv2
from pathlib import Path
import telemetry_parser
from hoi.data_tools.utils_mp4 import get_frames_from_mp4, get_imu_from_mp4
from hoi.data_tools.utils_vrs import VRSUtils
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore
from tqdm import tqdm
import numpy as np
from hoi.data_tools.utils_bag import get_topics_from_bag
from hoi.data_tools.qrcode_detector_decoder import QRCodeDetectorDecoder
from hoi.data_tools.time_aligner import TimeAligner
from typing import List, Tuple
import cv2
from rosbags.rosbag1 import Reader, Writer
from rosbags.typesys import Stores, get_typestore
from tqdm import tqdm
from typing import Union, List, Tuple, Optional, Dict, Any
import sys
from rosbags.typesys.base import TypesysError
import pandas as pd
from hoi.data_tools.utils_yaml import load_imucam, load_camchain
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from hoi.data_tools.utils_gripper_model import GripperModel
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.optimize import least_squares
import json
import hashlib

def load_hand_eye_calibration_asl(calib_path: Path | str) -> Tuple[List[float], List[float]]:
    """
    Load hand-eye calibration from ASL format from json file.
    Returns:
        T_body_cam: 4x4 numpy array, transformation from camera to body frame

    """
    if isinstance(calib_path, str):
        calib_path = Path(calib_path)

    with open(calib_path, 'r') as f:
        data = json.load(f)

    t = data['translation']
    t = [t['x'], t['y'], t['z']]
    q = data['rotation']
    quat = [q['i'], q['j'], q['k'], q['w']]

    return t, quat


def mp4_to_rosbag(mp4_path: Path | str, 
                bag_output_path: Path | str,
                cam_topic: str = "/cam0/image_raw",
                imu_topic: str = "/imu0",
                cam_frame_id: str = "cam0",
                imu_frame_id: str = "imu0",) -> None:
    """
    Convert MP4 files to ROS bag format (for Kalibr)
    """

    imu_df = get_imu_from_mp4(mp4_path)
    frames_av, timestamps_ns = get_frames_from_mp4(mp4_path)
    
    typestore = get_typestore(Stores.ROS1_NOETIC)
    Header    = typestore.types['std_msgs/msg/Header']
    ImageMsg  = typestore.types['sensor_msgs/msg/Image']
    ImuMsg    = typestore.types['sensor_msgs/msg/Imu']
    Quaternion = typestore.types['geometry_msgs/msg/Quaternion']
    Time = typestore.types['builtin_interfaces/msg/Time']
    Vector3 = typestore.types['geometry_msgs/msg/Vector3']

    bag_path = Path(bag_output_path)
    with Writer(bag_path) as writer:

        # Register both connections once
        con_cam = writer.add_connection(
            cam_topic, ImageMsg.__msgtype__, typestore=typestore)
        con_imu = writer.add_connection(
            imu_topic, ImuMsg.__msgtype__, typestore=typestore)

        events = []

        # ---------- IMU ----------------------------------------------------
        default_covariance = np.zeros(9, dtype=np.float64)                         # list, not np.array
        default_orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        for seq, row in tqdm(enumerate(imu_df.itertuples(index=False)),
                            total=len(imu_df), desc='IMU', unit='sample'):
            t_ns = int(row.timestamp)
            stamp = Time(sec=t_ns // 1_000_000_000,
                        nanosec=t_ns % 1_000_000_000)
            header = Header(seq=seq, stamp=stamp, frame_id=imu_frame_id)
            imu_msg = ImuMsg(
                header=header,
                orientation=default_orientation,
                orientation_covariance=default_covariance,
                angular_velocity=Vector3(x=row.angular_vel_x, y=row.angular_vel_y, z=row.angular_vel_z),                
                angular_velocity_covariance=default_covariance,
                linear_acceleration=Vector3(x=row.linear_accel_x, y=row.linear_accel_y, z=row.linear_accel_z),
                linear_acceleration_covariance=default_covariance,
            )

            raw = typestore.serialize_ros1(imu_msg, ImuMsg.__msgtype__)
            events.append((t_ns, con_imu, raw))

        # ---------- Images -------------------------------------------------
        for seq, (frame, t_ns) in enumerate(
                tqdm(zip(frames_av, timestamps_ns),
                    total=len(timestamps_ns), desc='Images', unit='frame')):

            bgr = frame.to_ndarray(format='bgr24')
            h, w = bgr.shape[:2]
            stamp = Time(sec=t_ns // 1_000_000_000,
                        nanosec=t_ns % 1_000_000_000)
            header = Header(seq=seq, stamp=stamp, frame_id=cam_frame_id)
            img_msg = ImageMsg(
                header=header, height=h, width=w,
                encoding='bgr8', is_bigendian=0, step=3 * w,
                data=bgr.reshape(-1),            # ndarray view, not bytes
            )

            raw = typestore.serialize_ros1(img_msg, ImageMsg.__msgtype__)
            events.append((t_ns, con_cam, raw))

        events.sort(key=lambda e: e[0])   # sort by timestamp

        for t_ns, con, raw in tqdm(events, desc="Writing events", unit="event"):
            writer.write(con, t_ns, raw)

    print(f"[rosbags] wrote {len(timestamps_ns)} images and {len(imu_df)} IMU "
          f"samples → {bag_path}")
    
def merge_calibration_vrs_and_calibration_bag(vrs_path: Path | str, 
                                            rosbag_path: Path | str,
                                            temp_path: Path | str) -> None:
    """
    Merges the calibration data from VRS and the ROS bag.
    Background: Aria vrs and gripper back are jointly recorded, with the aria mounted on the gripper.
    First tiemstamps for the aria need to be adjusted to match the gripper timestamps by detecting
    the timestamped qr code. then aria is added to rosbag for later Kalibr calibration.
    1. Extract frames abnd timestamps from vrs into dummy directory
    2. Extract frames and timestamps from rosbag
    3. Detect the timestamped qr code in the aria frames and gripper frames, compoute offset
    4. Adjust the aria timestamps by the offset
    5. Write the adjusted aria frames and timestamps to the rosbag
    """
    
    if isinstance(vrs_path, str):
        vrs_path = Path(vrs_path)
    if isinstance(rosbag_path, str):
        rosbag_path = Path(rosbag_path)
    if isinstance(temp_path, str):
        temp_path = Path(temp_path)

    if not vrs_path.exists():
        raise FileNotFoundError(f"VRS file not found: {vrs_path}")
    
    if not rosbag_path.exists():
        raise FileNotFoundError(f"ROS bag file not found: {rosbag_path}")
    
    if not temp_path.exists():
        temp_path.mkdir(parents=True, exist_ok=True)

    temp_path_rosbag = temp_path / "rosbag"
    temp_path_vrs = temp_path / "vrs"
    temp_path_rosbag.mkdir(parents=True, exist_ok=True)
    temp_path_vrs.mkdir(parents=True, exist_ok=True)

    new_cam_topic = "/aria/camera_rgb/image_raw"
    new_cam_frame_id = "aria_camera_rgb"
    
    # Extract frames and timestamps from VRS into a temporary directory
    if not any(temp_path_vrs.glob("*")):
        vrs_utils = VRSUtils(vrs_path, undistort=False)
        _, _ = vrs_utils.get_frames_from_vrs(out_dir=temp_path_vrs)

    # Extract frames and timestamps from ROS bag
    if not any(temp_path_rosbag.glob("*")):
        get_topics_from_bag(
            image_topics=["/zedm/zed_node/left_raw/image_raw_color"],
            non_image_topics={},
            bag_path=rosbag_path,
            out_dir=temp_path_rosbag
        )

    # Detect the timestamped QR code in the VRS frames
    qr = QRCodeDetectorDecoder(frame_dir=temp_path_vrs, ext=".png")
    time_pair_aria = qr.find_first_valid_qr()

    # Detect the timestamped QR code in the ROS bag frames
    frame_dir = temp_path_rosbag / "zedm/zed_node/left_raw/image_raw_color"
    qr = QRCodeDetectorDecoder(frame_dir=frame_dir, ext=".png")
    time_pair_gripper = qr.find_first_valid_qr()

    # get the offset between the two timestamps
    if time_pair_aria is None or time_pair_gripper is None:
        raise ValueError("Could not find valid QR codes in either VRS or ROS bag frames.")
    
    # flip time pairs, so we get aria delta to gripper 
    # (unlike in data extraction, where we had gripper delta to aria)
    timealigner = TimeAligner(
        aria_pair=time_pair_gripper,
        sensor_pair=time_pair_aria,
    )
    delta = timealigner.get_delta()

    # Adjust the timestamps of the VRS frames by the delta
    for frame in temp_path_vrs.glob("*.png"):
        ts = int(frame.stem)
        adjusted_ts = ts + delta
        new_frame_name = temp_path_vrs / f"{adjusted_ts}.png"
        frame.rename(new_frame_name)

    # Write the adjusted VRS frames and timestamps to the ROS bag
    # First read the old rosbag into all_events, adding the adjusted VRS frames and sort
    output_bag_path = temp_path / "merged_calibration.bag"

    typestore = get_typestore(Stores.ROS1_NOETIC)
    Header = typestore.types['std_msgs/msg/Header']
    ImageMsg = typestore.types['sensor_msgs/msg/Image']
    Time = typestore.types['builtin_interfaces/msg/Time']


    all_events: List[Tuple[int, object, bytes]] = []

    # --- 1. Read existing messages from the original bag ---
    print(f"Reading existing messages from {rosbag_path}...")
    with Reader(rosbag_path) as reader:
        for connection, timestamp_ns, rawdata in tqdm(reader.messages(), desc="Reading existing bag"):
            all_events.append((timestamp_ns, connection, rawdata))
    
    # --- 2. Add new images from the VRS frames ---
    # todo frames used to be png, now they are jpg
    vrs_files = sorted(list(temp_path_vrs.glob("*.png")))
    for seq, img_file in enumerate(tqdm(vrs_files, desc="Processing new images")):
        try:
            # Extract timestamp from filename (assuming integer timestamp)
            t_ns = int(img_file.stem) # .stem gets the filename without suffix

            # Load image using OpenCV
            bgr_image = cv2.imread(str(img_file))
            if bgr_image is None:
                print(f"Warning: Could not read image {img_file}. Skipping.", file=sys.stderr)
                continue

            h, w = bgr_image.shape[:2]
            
            # Create ROS 1 Time object
            stamp = Time(sec=t_ns // 1_000_000_000, nanosec=t_ns % 1_000_000_000)
            
            # Create ROS 1 Header object
            header = Header(seq=seq, stamp=stamp, frame_id=new_cam_frame_id)
            
            # Create ROS 1 Image message
            img_msg = ImageMsg(
                header=header,
                height=h,
                width=w,
                encoding='bgr8', # OpenCV reads as BGR, so 'bgr8' is appropriate
                is_bigendian=0,
                step=3 * w, # 3 bytes per pixel (BGR) * width
                data=bgr_image.reshape(-1) # Flatten the numpy array to bytes
            )

            # Serialize message to raw bytes
            raw = typestore.serialize_ros1(img_msg, ImageMsg.__msgtype__)
            all_events.append((t_ns, new_cam_topic, raw)) # Store topic name for new connection

        except ValueError:
            print(f"Warning: Skipping {img_file} as its name is not a valid integer timestamp.", file=sys.stderr)
        except Exception as e:
            print(f"Error processing {img_file}: {e}. Skipping.", file=sys.stderr)

    # --- 3. Sort all events by timstamp ---
    all_events.sort(key=lambda e: e[0])

    # --- 4. Write all events to the new bag file ---
    print(f"Writing all messages to new bag file: {output_bag_path}...")
    with Writer(output_bag_path) as writer:
        # Keep track of connections for existing topics
        existing_connections = {}
        new_cam_connection = None

        for t_ns, con_or_topic, raw in tqdm(all_events, desc="Writing events to new bag"):
            if isinstance(con_or_topic, str): # This is a new camera topic
                if new_cam_connection is None:
                    new_cam_connection = writer.add_connection(
                        new_cam_topic, ImageMsg.__msgtype__, typestore=typestore
                    )
                writer.write(new_cam_connection, t_ns, raw)
            else: # This is an existing connection from the original bag
                # Add connection if not already added (e.g., first message of a topic)
                if con_or_topic.topic not in existing_connections:
                    try:
                        existing_connections[con_or_topic.topic] = writer.add_connection(
                            con_or_topic.topic, con_or_topic.msgtype, typestore=typestore
                        )
                    except TypesysError as e:
                        print(f"Warning: Skipping topic '{con_or_topic.topic}' due to unknown type '{con_or_topic.msgtype}': {e}", file=sys.stderr)
                        # Mark this connection as None to indicate it was skipped
                        existing_connections[con_or_topic.topic] = None
                        continue # Skip writing this specific message

                # Only write if the connection was successfully added (not None)
                if existing_connections[con_or_topic.topic] is not None:
                    writer.write(existing_connections[con_or_topic.topic], t_ns, raw)


def _estimate_robust_stream_offset_ns_from_time_pairs(
    time_pairs: List[Tuple[int, int]],
    logging_tag: str,
    mad_multiplier: float = 3.5,
) -> int:
    """
    Estimate one robust QR-to-device offset from many
    ``(device_timestamp_ns, qr_timestamp_ns)`` correspondences.
    """
    if not time_pairs:
        raise ValueError("Need at least one time pair to estimate a stream offset.")

    offsets = np.array(
        [int(qr_ts) - int(device_ts) for device_ts, qr_ts in time_pairs],
        dtype=np.int64,
    )
    median_offset = int(np.median(offsets))

    if len(offsets) < 3:
        print(
            f"[{logging_tag}] Using median offset from {len(offsets)} QR correspondences: "
            f"{median_offset} ns"
        )
        return median_offset

    absolute_deviation = np.abs(offsets - median_offset)
    mad = int(np.median(absolute_deviation))

    if mad == 0:
        print(
            f"[{logging_tag}] QR offsets are perfectly consistent across "
            f"{len(offsets)} correspondences."
        )
        return median_offset

    inlier_mask = absolute_deviation <= mad_multiplier * mad
    inlier_offsets = offsets[inlier_mask]
    if inlier_offsets.size == 0:
        inlier_offsets = offsets

    robust_offset = int(np.median(inlier_offsets))
    print(
        f"[{logging_tag}] Robust QR offset estimate: {robust_offset} ns "
        f"from {int(inlier_offsets.size)}/{len(offsets)} inlier correspondences "
        f"(MAD={mad} ns)"
    )
    return robust_offset


def _resolve_existing_image_extension(
    frame_dir: Path,
    preferred_ext: str,
) -> str:
    """
    Resolve the actual on-disk image extension for a frame directory.
    This is used only by the new high-precision path to stay robust to
    differences between extractors.
    """
    if not frame_dir.exists():
        raise FileNotFoundError(f"Frame directory not found: {frame_dir}")

    available_suffixes = {
        path.suffix.lower()
        for path in frame_dir.iterdir()
        if path.is_file() and path.suffix
    }
    if not available_suffixes:
        raise FileNotFoundError(f"No image files found in frame directory: {frame_dir}")

    preferred_ext = preferred_ext.lower()
    if preferred_ext in available_suffixes:
        return preferred_ext

    for candidate in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        if candidate in available_suffixes:
            print(
                f"[high_precision_alignment] Using detected extension {candidate} in "
                f"{frame_dir} instead of preferred {preferred_ext}."
            )
            return candidate

    raise FileNotFoundError(
        f"Could not find a supported image extension in {frame_dir}. "
        f"Available suffixes: {sorted(available_suffixes)}"
    )


def _load_cached_high_precision_time_pairs(
    cache_file: Path,
    *,
    frame_dir: Path,
    ext: str,
    stride: int,
    deduplicate_by_qr_timestamp: bool,
    max_unique_qr_detections: Optional[int],
) -> Optional[List[Tuple[int, int]]]:
    """
    Load cached high-precision QR time pairs when the cache metadata matches
    the current scan configuration.
    """
    if not cache_file.exists():
        return None

    try:
        with cache_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        print(f"[high_precision_alignment] Failed to read cache {cache_file}: {exc}")
        return None

    metadata = payload.get("metadata", {})
    cache_matches = (
        metadata.get("frame_dir") == str(frame_dir)
        and metadata.get("ext") == ext
        and int(metadata.get("stride", -1)) == int(stride)
        and bool(metadata.get("deduplicate_by_qr_timestamp")) == bool(deduplicate_by_qr_timestamp)
        and metadata.get("max_unique_qr_detections") == max_unique_qr_detections
    )
    if not cache_matches:
        print(
            f"[high_precision_alignment] Ignoring incompatible cache at {cache_file}."
        )
        return None

    pairs = payload.get("time_pairs", [])
    loaded_pairs = [
        (int(item["device_timestamp_ns"]), int(item["qr_timestamp_ns"]))
        for item in pairs
    ]
    print(
        f"[high_precision_alignment] Loaded {len(loaded_pairs)} cached QR time pairs "
        f"from {cache_file}."
    )
    return loaded_pairs


def _save_cached_high_precision_time_pairs(
    cache_file: Path,
    *,
    frame_dir: Path,
    ext: str,
    stride: int,
    deduplicate_by_qr_timestamp: bool,
    max_unique_qr_detections: Optional[int],
    time_pairs: List[Tuple[int, int]],
) -> None:
    """
    Save high-precision QR time pairs for reuse on later runs of the new
    dynamic calibration path.
    """
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "frame_dir": str(frame_dir),
            "ext": ext,
            "stride": int(stride),
            "deduplicate_by_qr_timestamp": bool(deduplicate_by_qr_timestamp),
            "max_unique_qr_detections": max_unique_qr_detections,
            "num_time_pairs": len(time_pairs),
        },
        "time_pairs": [
            {
                "device_timestamp_ns": int(device_timestamp_ns),
                "qr_timestamp_ns": int(qr_timestamp_ns),
            }
            for device_timestamp_ns, qr_timestamp_ns in time_pairs
        ],
    }
    with cache_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
    print(
        f"[high_precision_alignment] Saved {len(time_pairs)} QR time pairs to cache "
        f"{cache_file}."
    )


def _get_or_compute_high_precision_time_pairs(
    *,
    frame_dir: Path,
    ext: str,
    cache_file: Path,
    stride: int,
    deduplicate_by_qr_timestamp: bool,
    max_unique_qr_detections: Optional[int],
) -> List[Tuple[int, int]]:
    """
    Resolve high-precision QR time pairs from cache when available, otherwise
    run the detector and store the result in a script-local cache file.
    """
    cached_pairs = _load_cached_high_precision_time_pairs(
        cache_file,
        frame_dir=frame_dir,
        ext=ext,
        stride=stride,
        deduplicate_by_qr_timestamp=deduplicate_by_qr_timestamp,
        max_unique_qr_detections=max_unique_qr_detections,
    )
    if cached_pairs is not None:
        return cached_pairs

    detector = QRCodeDetectorDecoder(frame_dir=frame_dir, ext=ext)
    time_pairs = detector.find_all_valid_time_qrs(
        stride=stride,
        deduplicate_by_qr_timestamp=deduplicate_by_qr_timestamp,
        max_unique_qr_detections=max_unique_qr_detections,
    )
    _save_cached_high_precision_time_pairs(
        cache_file,
        frame_dir=frame_dir,
        ext=ext,
        stride=stride,
        deduplicate_by_qr_timestamp=deduplicate_by_qr_timestamp,
        max_unique_qr_detections=max_unique_qr_detections,
        time_pairs=time_pairs,
    )
    return time_pairs


def _compute_high_precision_time_delta_between_streams(
    reference_frame_dir: Path,
    sensor_frame_dir: Path,
    *,
    reference_ext: str = ".png",
    sensor_ext: str = ".png",
    reference_cache_file: Optional[Path] = None,
    sensor_cache_file: Optional[Path] = None,
    stride: int = 1,
    min_qr_pairs: int = 2,
    deduplicate_by_qr_timestamp: bool = True,
    max_unique_qr_detections: Optional[int] = None,
    mad_multiplier: float = 3.5,
) -> int:
    """
    Compute a robust additive delta that maps ``sensor`` timestamps into the
    ``reference`` time frame using many QR detections across both streams.

    Returns
    -------
    int
        ``reference_ts = sensor_ts + delta``
    """
    reference_detector = QRCodeDetectorDecoder(frame_dir=reference_frame_dir, ext=reference_ext)
    sensor_detector = QRCodeDetectorDecoder(frame_dir=sensor_frame_dir, ext=sensor_ext)

    if reference_cache_file is not None:
        reference_pairs = _get_or_compute_high_precision_time_pairs(
            frame_dir=reference_frame_dir,
            ext=reference_ext,
            cache_file=reference_cache_file,
            stride=stride,
            deduplicate_by_qr_timestamp=deduplicate_by_qr_timestamp,
            max_unique_qr_detections=max_unique_qr_detections,
        )
    else:
        reference_pairs = reference_detector.find_all_valid_time_qrs(
            stride=stride,
            deduplicate_by_qr_timestamp=deduplicate_by_qr_timestamp,
            max_unique_qr_detections=max_unique_qr_detections,
        )

    if sensor_cache_file is not None:
        sensor_pairs = _get_or_compute_high_precision_time_pairs(
            frame_dir=sensor_frame_dir,
            ext=sensor_ext,
            cache_file=sensor_cache_file,
            stride=stride,
            deduplicate_by_qr_timestamp=deduplicate_by_qr_timestamp,
            max_unique_qr_detections=max_unique_qr_detections,
        )
    else:
        sensor_pairs = sensor_detector.find_all_valid_time_qrs(
            stride=stride,
            deduplicate_by_qr_timestamp=deduplicate_by_qr_timestamp,
            max_unique_qr_detections=max_unique_qr_detections,
        )

    def _resolve_stream_offset(
        detector: QRCodeDetectorDecoder,
        pairs: List[Tuple[int, int]],
        stream_name: str,
    ) -> int:
        if len(pairs) >= min_qr_pairs:
            return _estimate_robust_stream_offset_ns_from_time_pairs(
                pairs,
                logging_tag=stream_name,
                mad_multiplier=mad_multiplier,
            )

        if len(pairs) == 1:
            device_ts, qr_ts = pairs[0]
            offset = int(qr_ts) - int(device_ts)
            print(
                f"[{stream_name}] Only one valid time QR found during full-stream scan, "
                f"using that single correspondence: {offset} ns"
            )
            return offset

        time_pair = detector.find_first_valid_qr(stride=stride)
        if time_pair[0] is None or time_pair[1] is None:
            raise ValueError(
                f"Could not find any valid QR codes for high-precision alignment in {stream_name}."
            )

        offset = int(time_pair[1]) - int(time_pair[0])
        print(
            f"[{stream_name}] No multi-QR detections found, falling back to the first "
            f"valid QR pair: {offset} ns"
        )
        return offset

    reference_offset = _resolve_stream_offset(
        reference_detector,
        reference_pairs,
        stream_name=f"{reference_frame_dir.name}-reference",
    )
    sensor_offset = _resolve_stream_offset(
        sensor_detector,
        sensor_pairs,
        stream_name=f"{sensor_frame_dir.name}-sensor",
    )

    return sensor_offset - reference_offset


def _resolve_savgol_params(
    num_samples: int,
    sg_window_length: int,
    sg_polyorder: int,
) -> Tuple[int, int]:
    """
    Resolve a valid Savitzky-Golay configuration for the available sample count.
    """
    if num_samples < 3:
        raise ValueError(
            "Need at least 3 Aria pose samples to smooth and differentiate signals."
        )

    window_length = int(sg_window_length)
    polyorder = int(sg_polyorder)

    if window_length < 3:
        window_length = 3
    if window_length % 2 == 0:
        window_length += 1

    max_window_length = num_samples if num_samples % 2 == 1 else num_samples - 1
    window_length = min(window_length, max_window_length)
    if window_length < 3:
        raise ValueError(
            "Could not resolve a valid Savitzky-Golay window length from the pose series."
        )

    polyorder = max(1, min(polyorder, window_length - 1))
    return window_length, polyorder


def _resolve_requested_dynamic_smoothing_config(
    *,
    sg_window_length: int = 21,
    sg_polyorder: int = 3,
    sg_window_length_linear: Optional[int] = None,
    sg_polyorder_linear: Optional[int] = None,
    sg_window_length_angular: Optional[int] = None,
    sg_polyorder_angular: Optional[int] = None,
) -> Dict[str, int]:
    """
    Resolve the requested smoothing configuration while preserving backward
    compatibility with the older shared Savitzky-Golay parameters.
    """
    return {
        "sg_window_length_linear": int(
            sg_window_length if sg_window_length_linear is None else sg_window_length_linear
        ),
        "sg_polyorder_linear": int(
            sg_polyorder if sg_polyorder_linear is None else sg_polyorder_linear
        ),
        "sg_window_length_angular": int(
            sg_window_length if sg_window_length_angular is None else sg_window_length_angular
        ),
        "sg_polyorder_angular": int(
            sg_polyorder if sg_polyorder_angular is None else sg_polyorder_angular
        ),
    }


def _normalize_candidate_smoothing_config(
    candidate_config: Optional[Dict[str, int]],
    base_config: Dict[str, int],
) -> Dict[str, Any]:
    """
    Normalize one candidate smoothing configuration on top of a base config.
    """
    normalized = dict(base_config)
    if candidate_config is None:
        return normalized

    if "name" in candidate_config and candidate_config["name"] is not None:
        normalized["name"] = str(candidate_config["name"])

    if "sg_window_length" in candidate_config and candidate_config["sg_window_length"] is not None:
        normalized["sg_window_length_linear"] = int(candidate_config["sg_window_length"])
        normalized["sg_window_length_angular"] = int(candidate_config["sg_window_length"])
    if "sg_polyorder" in candidate_config and candidate_config["sg_polyorder"] is not None:
        normalized["sg_polyorder_linear"] = int(candidate_config["sg_polyorder"])
        normalized["sg_polyorder_angular"] = int(candidate_config["sg_polyorder"])

    for key in [
        "sg_window_length_linear",
        "sg_polyorder_linear",
        "sg_window_length_angular",
        "sg_polyorder_angular",
    ]:
        if key in candidate_config and candidate_config[key] is not None:
            normalized[key] = int(candidate_config[key])
    return normalized


def _extract_smoothing_kwargs(
    smoothing_config: Dict[str, Any],
) -> Dict[str, int]:
    return {
        key: int(smoothing_config[key])
        for key in [
            "sg_window_length_linear",
            "sg_polyorder_linear",
            "sg_window_length_angular",
            "sg_polyorder_angular",
        ]
        if key in smoothing_config and smoothing_config[key] is not None
    }


def _dynamic_smoothing_config_key(
    smoothing_config: Dict[str, Any],
) -> Tuple[int, int, int, int]:
    smoothing_kwargs = _extract_smoothing_kwargs(smoothing_config)
    return tuple(
        int(smoothing_kwargs[key])
        for key in [
            "sg_window_length_linear",
            "sg_polyorder_linear",
            "sg_window_length_angular",
            "sg_polyorder_angular",
        ]
    )


def _default_candidate_smoothing_configs(
    base_config: Dict[str, int],
) -> List[Dict[str, Any]]:
    default_candidates = [
        {
            "name": "L21_P3__A21_P3",
            "sg_window_length_linear": 21,
            "sg_polyorder_linear": 3,
            "sg_window_length_angular": 21,
            "sg_polyorder_angular": 3,
        },
        {
            "name": "L31_P3__A31_P3",
            "sg_window_length_linear": 31,
            "sg_polyorder_linear": 3,
            "sg_window_length_angular": 31,
            "sg_polyorder_angular": 3,
        },
        {
            "name": "L41_P3__A31_P3",
            "sg_window_length_linear": 41,
            "sg_polyorder_linear": 3,
            "sg_window_length_angular": 31,
            "sg_polyorder_angular": 3,
        },
        {
            "name": "L41_P2__A41_P2",
            "sg_window_length_linear": 41,
            "sg_polyorder_linear": 2,
            "sg_window_length_angular": 41,
            "sg_polyorder_angular": 2,
        },
    ]

    candidates = [dict(base_config, name=_format_dynamic_smoothing_config_name(base_config))]
    seen = {_dynamic_smoothing_config_key(base_config)}
    for candidate in default_candidates:
        normalized = _normalize_candidate_smoothing_config(candidate, base_config)
        key = _dynamic_smoothing_config_key(normalized)
        if key not in seen:
            seen.add(key)
            candidates.append(normalized)
    return candidates


def _format_dynamic_smoothing_config_label(smoothing_config: Dict[str, int]) -> str:
    return (
        f"L{int(smoothing_config['sg_window_length_linear'])}/P{int(smoothing_config['sg_polyorder_linear'])} "
        f"| A{int(smoothing_config['sg_window_length_angular'])}/P{int(smoothing_config['sg_polyorder_angular'])}"
    )


def _format_dynamic_smoothing_config_name(smoothing_config: Dict[str, int]) -> str:
    return (
        f"L{int(smoothing_config['sg_window_length_linear'])}_P{int(smoothing_config['sg_polyorder_linear'])}"
        f"__A{int(smoothing_config['sg_window_length_angular'])}_P{int(smoothing_config['sg_polyorder_angular'])}"
    )


def _interp_vector_series(
    source_timestamps_ns: np.ndarray,
    values: np.ndarray,
    target_timestamps_ns: np.ndarray,
) -> np.ndarray:
    """
    Component-wise linear interpolation with endpoint clamping, matching the
    timestamp handling used by ``_slerp_pose_series_to_targets``.
    """
    source_timestamps_ns = np.asarray(source_timestamps_ns, dtype=np.int64)
    target_timestamps_ns = np.asarray(target_timestamps_ns, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)

    if source_timestamps_ns.ndim != 1:
        raise ValueError("source_timestamps_ns must be a 1D array.")
    if target_timestamps_ns.ndim != 1:
        raise ValueError("target_timestamps_ns must be a 1D array.")
    if values.ndim != 2:
        raise ValueError("values must be a 2D array of shape (N, C).")
    if values.shape[0] != source_timestamps_ns.shape[0]:
        raise ValueError(
            "values and source_timestamps_ns must have the same leading dimension."
        )
    if source_timestamps_ns.size == 0:
        raise ValueError("Cannot interpolate an empty source series.")
    if np.any(np.diff(source_timestamps_ns) <= 0):
        raise ValueError("source_timestamps_ns must be strictly increasing.")

    source_timestamps_interp = source_timestamps_ns.astype(np.float64)
    target_timestamps_interp = np.clip(
        target_timestamps_ns.astype(np.float64),
        source_timestamps_interp[0],
        source_timestamps_interp[-1],
    )

    interpolated = np.empty((target_timestamps_ns.size, values.shape[1]), dtype=np.float64)
    for channel_idx in range(values.shape[1]):
        interpolated[:, channel_idx] = np.interp(
            target_timestamps_interp,
            source_timestamps_interp,
            values[:, channel_idx],
        )
    return interpolated


def _finalize_debug_figure(
    fig: plt.Figure,
    *,
    visualize: bool,
    visualize_out_dir: Optional[Path],
    filename: str,
) -> None:
    fig.tight_layout()
    if visualize_out_dir is not None:
        visualize_out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(visualize_out_dir / filename, dpi=200, bbox_inches="tight")
    if visualize:
        plt.show()
    plt.close(fig)


def _visualize_dynamic_signals_from_aria(
    dynamic_signals: Dict[str, np.ndarray],
    *,
    visualize: bool,
    visualize_out_dir: Optional[Path],
) -> None:
    """
    Debug plots for the Aria-derived dynamic signals. Plots are shown when
    ``visualize`` is true and saved when ``visualize_out_dir`` is provided.
    """
    if not visualize and visualize_out_dir is None:
        return

    pose_time_s = dynamic_signals["pose_time_s"]
    wrench_time_s = dynamic_signals["wrench_time_s"]

    linear_speed_device = np.linalg.norm(dynamic_signals["v_D"], axis=1)
    linear_speed_world_raw = np.linalg.norm(dynamic_signals["v_W_raw"], axis=1)
    linear_speed_world_smooth = np.linalg.norm(dynamic_signals["v_W_smooth"], axis=1)
    linear_acc_world = np.linalg.norm(dynamic_signals["a_W_pose"], axis=1)
    linear_acc_device = np.linalg.norm(dynamic_signals["a_D_pose"], axis=1)
    angular_speed_device_raw = np.linalg.norm(dynamic_signals["omega_D_raw"], axis=1)
    angular_speed_device_smooth = np.linalg.norm(dynamic_signals["omega_D_pose"], axis=1)
    angular_acc_device = np.linalg.norm(dynamic_signals["alpha_D_pose"], axis=1)

    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)

    axes[0].plot(pose_time_s, linear_speed_device, label="|v_D| raw", color="tab:blue", lw=1.5)
    axes[0].plot(
        pose_time_s,
        linear_speed_world_raw,
        label="|v_W| raw",
        color="tab:green",
        lw=1.2,
        alpha=0.8,
    )
    axes[0].plot(
        pose_time_s,
        linear_speed_world_smooth,
        label="|v_W| smooth",
        color="tab:red",
        lw=1.8,
    )
    axes[0].set_ylabel("Speed [m/s]")
    axes[0].set_title("Aria linear velocity magnitude")
    axes[0].legend(loc="upper right")
    axes[0].grid(True)

    axes[1].plot(
        pose_time_s,
        linear_acc_world,
        label="|a_W|",
        color="tab:purple",
        lw=1.5,
    )
    axes[1].plot(
        pose_time_s,
        linear_acc_device,
        label="|a_D|",
        color="tab:brown",
        lw=1.5,
    )
    axes[1].set_ylabel("Accel. [m/s$^2$]")
    axes[1].set_title("Aria linear acceleration magnitude")
    axes[1].legend(loc="upper right")
    axes[1].grid(True)

    axes[2].plot(
        pose_time_s,
        angular_speed_device_raw,
        label="|omega_D| raw",
        color="tab:orange",
        lw=1.2,
    )
    axes[2].plot(
        pose_time_s,
        angular_speed_device_smooth,
        label="|omega_D| smooth",
        color="tab:red",
        lw=1.8,
    )
    axes[2].set_ylabel("Angular vel. [rad/s]")
    axes[2].set_title("Aria angular velocity magnitude")
    axes[2].legend(loc="upper right")
    axes[2].grid(True)

    axes[3].plot(
        pose_time_s,
        angular_acc_device,
        label="|alpha_D|",
        color="tab:cyan",
        lw=1.5,
    )
    axes[3].set_xlabel("Time [s]")
    axes[3].set_ylabel("Angular accel. [rad/s$^2$]")
    axes[3].set_title("Aria angular acceleration magnitude")
    axes[3].legend(loc="upper right")
    axes[3].grid(True)

    _finalize_debug_figure(
        fig,
        visualize=visualize,
        visualize_out_dir=visualize_out_dir,
        filename="aria_dynamic_signal_magnitudes.png",
    )

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    world_labels = ["x", "y", "z"]
    world_colors = ["tab:red", "tab:green", "tab:blue"]
    for axis_idx, (label, color) in enumerate(zip(world_labels, world_colors)):
        axes[axis_idx].plot(
            pose_time_s,
            dynamic_signals["v_W_raw"][:, axis_idx],
            label=f"v_W raw {label}",
            color=color,
            lw=1.0,
            alpha=0.6,
        )
        axes[axis_idx].plot(
            pose_time_s,
            dynamic_signals["v_W_smooth"][:, axis_idx],
            label=f"v_W smooth {label}",
            color=color,
            lw=1.8,
        )
        axes[axis_idx].set_ylabel(f"{label} [m/s]")
        axes[axis_idx].legend(loc="upper right")
        axes[axis_idx].grid(True)
    axes[0].set_title("World-frame linear velocity: raw vs Savitzky-Golay smoothed")
    axes[-1].set_xlabel("Time [s]")

    _finalize_debug_figure(
        fig,
        visualize=visualize,
        visualize_out_dir=visualize_out_dir,
        filename="aria_world_velocity_raw_vs_smoothed.png",
    )

    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    component_series = [
        ("v_D [m/s]", dynamic_signals["v_D"], None),
        ("a_D [m/s$^2$]", dynamic_signals["a_D_pose"], None),
        ("omega_D [rad/s]", dynamic_signals["omega_D_raw"], dynamic_signals["omega_D_pose"]),
        ("alpha_D [rad/s$^2$]", dynamic_signals["alpha_D_pose"], None),
    ]
    for axis_idx, (ylabel, series_raw, series_smooth) in enumerate(component_series):
        for comp_idx, color in enumerate(world_colors):
            label = ["x", "y", "z"][comp_idx]
            axes[axis_idx].plot(
                pose_time_s,
                series_raw[:, comp_idx],
                color=color,
                lw=1.0,
                alpha=0.65,
                label=f"{label} raw" if series_smooth is not None else label,
            )
            if series_smooth is not None:
                axes[axis_idx].plot(
                    pose_time_s,
                    series_smooth[:, comp_idx],
                    color=color,
                    lw=1.8,
                    label=f"{label} smooth",
                )
        axes[axis_idx].set_ylabel(ylabel)
        axes[axis_idx].grid(True)
        axes[axis_idx].legend(loc="upper right", ncol=3)
    axes[0].set_title("Device-frame dynamic signal components")
    axes[-1].set_xlabel("Time [s]")

    _finalize_debug_figure(
        fig,
        visualize=visualize,
        visualize_out_dir=visualize_out_dir,
        filename="aria_dynamic_signal_components_device_frame.png",
    )

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(
        wrench_time_s,
        np.linalg.norm(dynamic_signals["a_S"], axis=1),
        color="tab:blue",
        lw=1.5,
        label="|a_S|",
    )
    axes[0].set_ylabel("Accel. [m/s$^2$]")
    axes[0].set_title("Sensor-frame signals at wrench timestamps")
    axes[0].grid(True)
    axes[0].legend(loc="upper right")

    axes[1].plot(
        wrench_time_s,
        np.linalg.norm(dynamic_signals["omega_S"], axis=1),
        color="tab:orange",
        lw=1.5,
        label="|omega_S|",
    )
    axes[1].set_ylabel("Angular vel. [rad/s]")
    axes[1].grid(True)
    axes[1].legend(loc="upper right")

    axes[2].plot(
        wrench_time_s,
        np.linalg.norm(dynamic_signals["alpha_S"], axis=1),
        color="tab:green",
        lw=1.5,
        label="|alpha_S|",
    )
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Angular accel. [rad/s$^2$]")
    axes[2].grid(True)
    axes[2].legend(loc="upper right")

    _finalize_debug_figure(
        fig,
        visualize=visualize,
        visualize_out_dir=visualize_out_dir,
        filename="aria_dynamic_signal_sensor_frame_magnitudes.png",
    )


def compute_dynamic_signals_from_aria(
    poses_aria: pd.DataFrame,
    wrench_timestamps_ns: np.ndarray,
    T_ariadevice_ft: np.ndarray,
    *,
    sg_window_length: int = 21,
    sg_polyorder: int = 3,
    sg_window_length_linear: Optional[int] = None,
    sg_polyorder_linear: Optional[int] = None,
    sg_window_length_angular: Optional[int] = None,
    sg_polyorder_angular: Optional[int] = None,
    visualize: bool = True,
    visualize_out_dir: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """
    Build Aria-derived dynamic signals for the force/torque calibration pipeline.

    Frame assumptions
    -----------------
    The current pipeline convention treats ``T_ariadevice_ft`` as the rigid
    transform from the FT sensor frame into the Aria device frame. Under that
    convention:

    - ``T_ariadevice_ft[:3, :3]`` is ``R_D_S``, rotating sensor-frame vectors
      into the device frame.
    - ``T_ariadevice_ft[:3, 3]`` is the vector from the device origin to the
      sensor origin expressed in the device frame, i.e. ``r_D_to_S``.

    Linear velocity is differentiated in the world frame because differentiating
    body-frame linear velocity directly would mix translational acceleration with
    rotational transport terms. Angular velocity is differentiated in the device
    frame because the downstream rigid-body transport equation is expressed in
    the body/device frame.
    """
    if visualize_out_dir is not None and not isinstance(visualize_out_dir, Path):
        visualize_out_dir = Path(visualize_out_dir)

    requested_smoothing_config = _resolve_requested_dynamic_smoothing_config(
        sg_window_length=sg_window_length,
        sg_polyorder=sg_polyorder,
        sg_window_length_linear=sg_window_length_linear,
        sg_polyorder_linear=sg_polyorder_linear,
        sg_window_length_angular=sg_window_length_angular,
        sg_polyorder_angular=sg_polyorder_angular,
    )

    required_columns = [
        "timestamp",
        "qx_world_device",
        "qy_world_device",
        "qz_world_device",
        "qw_world_device",
        "device_linear_velocity_x_device",
        "device_linear_velocity_y_device",
        "device_linear_velocity_z_device",
        "angular_velocity_x_device",
        "angular_velocity_y_device",
        "angular_velocity_z_device",
    ]
    missing_columns = [column for column in required_columns if column not in poses_aria.columns]
    if missing_columns:
        raise KeyError(
            f"poses_aria is missing required columns for dynamic signal creation: {missing_columns}"
        )

    pose_columns = required_columns.copy()
    for optional_translation_column in [
        "tx_world_device",
        "ty_world_device",
        "tz_world_device",
    ]:
        if optional_translation_column in poses_aria.columns:
            pose_columns.append(optional_translation_column)

    pose_df = poses_aria[pose_columns].copy()
    pose_df = pose_df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    pose_timestamps_ns = pose_df["timestamp"].to_numpy(dtype=np.int64)
    wrench_timestamps_ns = np.asarray(wrench_timestamps_ns, dtype=np.int64)
    T_ariadevice_ft = np.asarray(T_ariadevice_ft, dtype=np.float64)

    if pose_timestamps_ns.ndim != 1 or pose_timestamps_ns.size < 3:
        raise ValueError("Need at least 3 unique pose timestamps to compute dynamic signals.")
    if wrench_timestamps_ns.ndim != 1 or wrench_timestamps_ns.size == 0:
        raise ValueError("wrench_timestamps_ns must be a non-empty 1D array.")
    if np.any(np.diff(pose_timestamps_ns) <= 0):
        raise ValueError("Pose timestamps must be strictly increasing.")
    if np.any(np.diff(wrench_timestamps_ns) < 0):
        raise ValueError("wrench_timestamps_ns must be sorted in non-decreasing order.")
    if T_ariadevice_ft.shape != (4, 4):
        raise ValueError("T_ariadevice_ft must be a 4x4 homogeneous transform.")

    R_D_S = T_ariadevice_ft[:3, :3]
    r_D_to_S = T_ariadevice_ft[:3, 3].astype(np.float64)
    if not np.allclose(R_D_S.T @ R_D_S, np.eye(3), atol=1e-6):
        raise ValueError("T_ariadevice_ft[:3, :3] is not a valid rotation matrix.")
    if not np.isclose(np.linalg.det(R_D_S), 1.0, atol=1e-6):
        raise ValueError("T_ariadevice_ft[:3, :3] must have determinant +1.")
    R_S_D = R_D_S.T

    quat_world_device = pose_df[
        [
            "qx_world_device",
            "qy_world_device",
            "qz_world_device",
            "qw_world_device",
        ]
    ].to_numpy(dtype=np.float64)
    quat_norm = np.linalg.norm(quat_world_device, axis=1, keepdims=True)
    if np.any(quat_norm <= 0.0):
        raise ValueError("Encountered a zero-norm quaternion in poses_aria.")
    quat_world_device = quat_world_device / quat_norm

    linear_velocity_device = pose_df[
        [
            "device_linear_velocity_x_device",
            "device_linear_velocity_y_device",
            "device_linear_velocity_z_device",
        ]
    ].to_numpy(dtype=np.float64)
    angular_velocity_device = pose_df[
        [
            "angular_velocity_x_device",
            "angular_velocity_y_device",
            "angular_velocity_z_device",
        ]
    ].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(linear_velocity_device)):
        raise ValueError("poses_aria contains non-finite linear velocity values.")
    if not np.all(np.isfinite(angular_velocity_device)):
        raise ValueError("poses_aria contains non-finite angular velocity values.")

    R_W_D_pose = R.from_quat(quat_world_device).as_matrix()
    if R_W_D_pose.shape != (pose_timestamps_ns.size, 3, 3):
        raise AssertionError("Unexpected rotation shape while building R_W_D_pose.")

    v_W_raw = np.einsum("nij,nj->ni", R_W_D_pose, linear_velocity_device)
    if not np.allclose(
        np.linalg.norm(linear_velocity_device, axis=1),
        np.linalg.norm(v_W_raw, axis=1),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise AssertionError("Velocity magnitude should be preserved by the device-to-world rotation.")

    pose_time_s = (pose_timestamps_ns - pose_timestamps_ns[0]).astype(np.float64) * 1e-9
    dt_pose_s = np.diff(pose_time_s)
    if np.any(dt_pose_s <= 0.0):
        raise ValueError("Pose timestamps must map to strictly increasing seconds.")

    linear_window_length, linear_polyorder = _resolve_savgol_params(
        num_samples=pose_timestamps_ns.size,
        sg_window_length=requested_smoothing_config["sg_window_length_linear"],
        sg_polyorder=requested_smoothing_config["sg_polyorder_linear"],
    )
    angular_window_length, angular_polyorder = _resolve_savgol_params(
        num_samples=pose_timestamps_ns.size,
        sg_window_length=requested_smoothing_config["sg_window_length_angular"],
        sg_polyorder=requested_smoothing_config["sg_polyorder_angular"],
    )

    # Savitzky-Golay assumes approximately uniform sample spacing for the smoothing
    # stage. We use it only for denoising, then differentiate with the true
    # timestamps to stay robust to small timestamp jitter.
    v_W_smooth = savgol_filter(
        v_W_raw,
        window_length=linear_window_length,
        polyorder=linear_polyorder,
        axis=0,
        mode="interp",
    )
    a_W_pose = np.gradient(
        v_W_smooth,
        pose_time_s,
        axis=0,
        edge_order=2 if pose_timestamps_ns.size >= 3 else 1,
    )

    R_D_W_pose = np.transpose(R_W_D_pose, axes=(0, 2, 1))
    a_D_pose = np.einsum("nij,nj->ni", R_D_W_pose, a_W_pose)

    omega_D_pose = savgol_filter(
        angular_velocity_device,
        window_length=angular_window_length,
        polyorder=angular_polyorder,
        axis=0,
        mode="interp",
    )
    alpha_D_pose = np.gradient(
        omega_D_pose,
        pose_time_s,
        axis=0,
        edge_order=2 if pose_timestamps_ns.size >= 3 else 1,
    )

    R_W_D, _ = _slerp_pose_series_to_targets(pose_df, wrench_timestamps_ns)
    R_W_D = np.asarray(R_W_D, dtype=np.float64)

    a_D = _interp_vector_series(pose_timestamps_ns, a_D_pose, wrench_timestamps_ns)
    omega_D = _interp_vector_series(pose_timestamps_ns, omega_D_pose, wrench_timestamps_ns)
    alpha_D = _interp_vector_series(pose_timestamps_ns, alpha_D_pose, wrench_timestamps_ns)

    a_sensor_origin_D = (
        a_D
        + np.cross(alpha_D, r_D_to_S[None, :])
        + np.cross(omega_D, np.cross(omega_D, r_D_to_S[None, :]))
    )

    a_S = np.einsum("ij,nj->ni", R_S_D, a_sensor_origin_D)
    omega_S = np.einsum("ij,nj->ni", R_S_D, omega_D)
    alpha_S = np.einsum("ij,nj->ni", R_S_D, alpha_D)

    R_W_S = np.einsum("nij,jk->nik", R_W_D, R_D_S)
    R_S_W = np.transpose(R_W_S, axes=(0, 2, 1))

    dynamic_signals = {
        "timestamps_ns_pose": pose_timestamps_ns,
        "timestamps_ns_wrench": wrench_timestamps_ns,
        "pose_time_s": pose_time_s,
        "wrench_time_s": (wrench_timestamps_ns - wrench_timestamps_ns[0]).astype(np.float64) * 1e-9,
        "smoothing_config_requested": requested_smoothing_config,
        "smoothing_config_resolved": {
            "sg_window_length_linear": int(linear_window_length),
            "sg_polyorder_linear": int(linear_polyorder),
            "sg_window_length_angular": int(angular_window_length),
            "sg_polyorder_angular": int(angular_polyorder),
        },
        "v_D": linear_velocity_device,
        "v_W_raw": v_W_raw,
        "v_W_smooth": v_W_smooth,
        "a_W_pose": a_W_pose,
        "a_D_pose": a_D_pose,
        "omega_D_raw": angular_velocity_device,
        "omega_D_pose": omega_D_pose,
        "alpha_D_pose": alpha_D_pose,
        "r_D_to_S": r_D_to_S,
        "R_D_S": R_D_S,
        "R_S_D": R_S_D,
        "a_D": a_D,
        "omega_D": omega_D,
        "alpha_D": alpha_D,
        "a_S": a_S,
        "omega_S": omega_S,
        "alpha_S": alpha_S,
        "R_W_D": R_W_D,
        "R_S_W": R_S_W,
    }

    for key in ["a_D", "omega_D", "alpha_D", "a_S", "omega_S", "alpha_S"]:
        if dynamic_signals[key].shape != (wrench_timestamps_ns.size, 3):
            raise AssertionError(f"{key} has an unexpected shape: {dynamic_signals[key].shape}")
        if not np.all(np.isfinite(dynamic_signals[key])):
            raise ValueError(f"{key} contains non-finite values.")

    if dynamic_signals["R_W_D"].shape != (wrench_timestamps_ns.size, 3, 3):
        raise AssertionError("R_W_D has an unexpected shape.")
    if dynamic_signals["R_S_W"].shape != (wrench_timestamps_ns.size, 3, 3):
        raise AssertionError("R_S_W has an unexpected shape.")

    _visualize_dynamic_signals_from_aria(
        dynamic_signals,
        visualize=visualize,
        visualize_out_dir=visualize_out_dir,
    )
    return dynamic_signals


def _build_dynamic_recording_cache_dir(
    *,
    temp_path: Path,
    vrs_path: Path,
    mps_file: Path,
    rosbag_path: Path,
    camchain_imucam: Path,
    stride: int,
    min_qr_pairs: int,
    deduplicate_by_qr_timestamp: bool,
    max_unique_qr_detections: Optional[int],
    mad_multiplier: float,
) -> Path:
    cache_key_payload = {
        "vrs_path": str(vrs_path.resolve()),
        "mps_file": str(mps_file.resolve()),
        "rosbag_path": str(rosbag_path.resolve()),
        "camchain_imucam": str(camchain_imucam.resolve()),
        "stride": int(stride),
        "min_qr_pairs": int(min_qr_pairs),
        "deduplicate_by_qr_timestamp": bool(deduplicate_by_qr_timestamp),
        "max_unique_qr_detections": (
            None if max_unique_qr_detections is None else int(max_unique_qr_detections)
        ),
        "mad_multiplier": float(mad_multiplier),
    }
    cache_key = hashlib.sha1(
        json.dumps(cache_key_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    recording_cache_dir = temp_path / "_dynamic_recording_cache" / cache_key
    recording_cache_dir.mkdir(parents=True, exist_ok=True)

    cache_info_file = recording_cache_dir / "cache_info.json"
    with cache_info_file.open("w", encoding="utf-8") as f:
        json.dump(cache_key_payload, f, indent=4)

    return recording_cache_dir


def _load_dynamic_recording_data(
    vrs_path: Path | str,
    mps_file: Path | str,
    rosbag_path: Path | str,
    temp_path: Path | str,
    camchain_imucam: Path | str,
    *,
    stride: int = 1,
    min_qr_pairs: int = 2,
    deduplicate_by_qr_timestamp: bool = True,
    max_unique_qr_detections: Optional[int] = 50,
    mad_multiplier: float = 3.5,
    visualize_aria_velocities: bool = False,
    sg_window_length: int = 21,
    sg_polyorder: int = 3,
    sg_window_length_linear: Optional[int] = None,
    sg_polyorder_linear: Optional[int] = None,
    sg_window_length_angular: Optional[int] = None,
    sg_polyorder_angular: Optional[int] = None,
    dynamic_signal_time_offset_ns: int = 0,
    dynamic_signal_visualize: bool = False,
    dynamic_signal_visualize_out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    if isinstance(vrs_path, str):
        vrs_path = Path(vrs_path)
    if isinstance(rosbag_path, str):
        rosbag_path = Path(rosbag_path)
    if isinstance(temp_path, str):
        temp_path = Path(temp_path)
    if isinstance(mps_file, str):
        mps_file = Path(mps_file)
    if isinstance(camchain_imucam, str):
        camchain_imucam = Path(camchain_imucam)
    if dynamic_signal_visualize_out_dir is not None and not isinstance(dynamic_signal_visualize_out_dir, Path):
        dynamic_signal_visualize_out_dir = Path(dynamic_signal_visualize_out_dir)

    if not vrs_path.exists():
        raise FileNotFoundError(f"VRS file not found: {vrs_path}")
    if not rosbag_path.exists():
        raise FileNotFoundError(f"ROS bag file not found: {rosbag_path}")
    if not temp_path.exists():
        temp_path.mkdir(parents=True, exist_ok=True)
    if not mps_file.exists():
        raise FileNotFoundError(f"MPS file not found: {mps_file}")
    if not camchain_imucam.exists():
        raise FileNotFoundError(f"Camera-IMU calibration file not found: {camchain_imucam}")

    cam_calib = load_camchain(camchain_imucam, cam_name="cam2")
    imu_calib = load_imucam(camchain_imucam, imu_name="cam2")

    recording_cache_dir = _build_dynamic_recording_cache_dir(
        temp_path=temp_path,
        vrs_path=vrs_path,
        mps_file=mps_file,
        rosbag_path=rosbag_path,
        camchain_imucam=camchain_imucam,
        stride=stride,
        min_qr_pairs=min_qr_pairs,
        deduplicate_by_qr_timestamp=deduplicate_by_qr_timestamp,
        max_unique_qr_detections=max_unique_qr_detections,
        mad_multiplier=mad_multiplier,
    )

    temp_path_rosbag = recording_cache_dir / "rosbag"
    temp_path_vrs = recording_cache_dir / "vrs_high_precision"
    temp_path_rosbag.mkdir(parents=True, exist_ok=True)
    temp_path_vrs.mkdir(parents=True, exist_ok=True)

    mps_data = pd.read_csv(mps_file)

    vrs_utils = VRSUtils(vrs_path, undistort=False)
    if not any(temp_path_vrs.glob("*")):
        _, _ = vrs_utils.get_frames_from_vrs(out_dir=temp_path_vrs)

    force_torque_topic = "/force_torque/ft_sensor0/ft_sensor_readings/wrench"
    temperature_topic = "/force_torque/ft_sensor0/ft_sensor_readings/temperature"
    if not any(temp_path_rosbag.glob("*")):
        get_topics_from_bag(
            image_topics=["/zedm/zed_node/left_raw/image_raw_color"],
            non_image_topics={
                force_torque_topic: "geometry_msgs/WrenchStamped",
                temperature_topic: "sensor_msgs/Temperature",
            },
            bag_path=rosbag_path,
            out_dir=temp_path_rosbag,
        )

    adjusted_mps_file = recording_cache_dir / "adjusted_mps_high_precision.csv"
    alignment_info_file = recording_cache_dir / "adjusted_mps_high_precision_alignment.json"
    gripper_qr_pairs_cache_file = recording_cache_dir / "gripper_high_precision_time_qr_pairs.json"
    vrs_qr_pairs_cache_file = recording_cache_dir / "vrs_high_precision_time_qr_pairs.json"
    delta = None

    if not adjusted_mps_file.exists():
        frame_dir_gripper = temp_path_rosbag / "zedm/zed_node/left_raw/image_raw_color"
        reference_ext = _resolve_existing_image_extension(
            frame_dir_gripper,
            preferred_ext=".jpg",
        )
        sensor_ext = _resolve_existing_image_extension(
            temp_path_vrs,
            preferred_ext=".png",
        )
        delta = _compute_high_precision_time_delta_between_streams(
            reference_frame_dir=frame_dir_gripper,
            sensor_frame_dir=temp_path_vrs,
            reference_ext=reference_ext,
            sensor_ext=sensor_ext,
            reference_cache_file=gripper_qr_pairs_cache_file,
            sensor_cache_file=vrs_qr_pairs_cache_file,
            stride=stride,
            min_qr_pairs=min_qr_pairs,
            deduplicate_by_qr_timestamp=deduplicate_by_qr_timestamp,
            max_unique_qr_detections=max_unique_qr_detections,
            mad_multiplier=mad_multiplier,
        )
        print(f"High-precision time delta between VRS and ROS bag: {delta} ns")

        for frame in temp_path_vrs.glob(f"*{sensor_ext}"):
            ts = int(frame.stem)
            adjusted_ts = ts + delta
            new_frame_name = temp_path_vrs / f"{adjusted_ts}{sensor_ext}"
            frame.rename(new_frame_name)

        adjusted_mps_data = mps_data.copy()
        adjusted_mps_data["tracking_timestamp_us"] = (
            adjusted_mps_data["tracking_timestamp_us"] * 1_000
        ) + delta
        adjusted_mps_data.rename(columns={"tracking_timestamp_us": "timestamp"}, inplace=True)
        adjusted_mps_data.to_csv(adjusted_mps_file, index=False)

        with alignment_info_file.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "delta_ns": int(delta),
                    "stride": stride,
                    "min_qr_pairs": min_qr_pairs,
                    "deduplicate_by_qr_timestamp": deduplicate_by_qr_timestamp,
                    "max_unique_qr_detections": max_unique_qr_detections,
                    "mad_multiplier": mad_multiplier,
                },
                f,
                indent=4,
            )
    else:
        adjusted_mps_data = pd.read_csv(adjusted_mps_file)
        if alignment_info_file.exists():
            with alignment_info_file.open("r", encoding="utf-8") as f:
                delta = json.load(f).get("delta_ns")

    force_torque_df = pd.read_csv(temp_path_rosbag / force_torque_topic.strip("/") / "data.csv")
    temperature_df = pd.read_csv(temp_path_rosbag / temperature_topic.strip("/") / "data.csv")

    T_ariacam_imuft = imu_calib.T_cam_imu
    T_imuft_ft = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0.0257],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    ariacam_calibration = vrs_utils.device_calib.get_camera_calib("camera-rgb")
    T_ariadevice_ariacam = ariacam_calibration.get_transform_device_camera().to_matrix()
    T_ariadevice_ft = T_ariadevice_ariacam @ T_ariacam_imuft @ T_imuft_ft

    wrench_ft = force_torque_df[
        [
            "timestamp",
            "wrench.force.x",
            "wrench.force.y",
            "wrench.force.z",
            "wrench.torque.x",
            "wrench.torque.y",
            "wrench.torque.z",
        ]
    ]
    wrench_timestamps_ns = wrench_ft["timestamp"].to_numpy(dtype=np.int64)
    dynamic_signal_sample_timestamps_ns = wrench_timestamps_ns + int(dynamic_signal_time_offset_ns)
    temperature_ft = temperature_df[["timestamp", "temperature"]]

    poses_aria = adjusted_mps_data[
        [
            "timestamp",
            "tx_world_device",
            "ty_world_device",
            "tz_world_device",
            "qx_world_device",
            "qy_world_device",
            "qz_world_device",
            "qw_world_device",
            "device_linear_velocity_x_device",
            "device_linear_velocity_y_device",
            "device_linear_velocity_z_device",
            "angular_velocity_x_device",
            "angular_velocity_y_device",
            "angular_velocity_z_device",
        ]
    ]

    if visualize_aria_velocities:
        visualize_aria_velocity_magnitudes(
            poses_aria,
            title_prefix="Aria adjusted MPS",
        )

    dynamic_signals = compute_dynamic_signals_from_aria(
        poses_aria=poses_aria,
        wrench_timestamps_ns=dynamic_signal_sample_timestamps_ns,
        T_ariadevice_ft=T_ariadevice_ft,
        sg_window_length=sg_window_length,
        sg_polyorder=sg_polyorder,
        sg_window_length_linear=sg_window_length_linear,
        sg_polyorder_linear=sg_polyorder_linear,
        sg_window_length_angular=sg_window_length_angular,
        sg_polyorder_angular=sg_polyorder_angular,
        visualize=dynamic_signal_visualize,
        visualize_out_dir=dynamic_signal_visualize_out_dir,
    )

    f_meas_S = wrench_ft[
        ["wrench.force.x", "wrench.force.y", "wrench.force.z"]
    ].to_numpy(dtype=np.float64)
    tau_meas_S = wrench_ft[
        ["wrench.torque.x", "wrench.torque.y", "wrench.torque.z"]
    ].to_numpy(dtype=np.float64)

    return {
        "cam_calib": cam_calib,
        "imu_calib": imu_calib,
        "vrs_utils": vrs_utils,
        "temp_path": temp_path,
        "recording_cache_dir": recording_cache_dir,
        "temp_path_rosbag": temp_path_rosbag,
        "temp_path_vrs": temp_path_vrs,
        "force_torque_topic": force_torque_topic,
        "temperature_topic": temperature_topic,
        "adjusted_mps_file": adjusted_mps_file,
        "adjusted_mps_data": adjusted_mps_data,
        "poses_aria": poses_aria,
        "force_torque_df": force_torque_df,
        "temperature_df": temperature_df,
        "wrench_ft": wrench_ft,
        "temperature_ft": temperature_ft,
        "wrench_timestamps_ns": wrench_timestamps_ns,
        "dynamic_signal_sample_timestamps_ns": dynamic_signal_sample_timestamps_ns,
        "T_ariadevice_ft": T_ariadevice_ft,
        "dynamic_signals": dynamic_signals,
        "f_meas_S": f_meas_S,
        "tau_meas_S": tau_meas_S,
        "delta_ns": delta,
        "gripper_qr_pairs_cache_file": gripper_qr_pairs_cache_file,
        "vrs_qr_pairs_cache_file": vrs_qr_pairs_cache_file,
    }


def _compute_dynamic_quasistatic_bias_state(
    *,
    adjusted_mps_data: pd.DataFrame,
    poses_aria: pd.DataFrame,
    force_torque_df: pd.DataFrame,
    temperature_df: pd.DataFrame,
    temperature_ft: pd.DataFrame,
    f_meas_S: np.ndarray,
    tau_meas_S: np.ndarray,
    dynamic_signals: Dict[str, np.ndarray],
    dynamic_signal_sample_timestamps_ns: np.ndarray,
    m_known: float,
    c_S_known: np.ndarray,
    dynamic_fit_cog: bool = False,
    epsilon_quasistatic_linear: float = 0.1,
    epsilon_quasistatic_angular: float = 0.2,
    epsilon_quasistatic_linear_acc: float = 0.5,
    epsilon_quasistatic_angular_acc: float = 1.0,
    g: float = 9.81,
) -> Dict[str, Any]:
    adjusted_mps_data_quasistatic = adjusted_mps_data.copy()
    poses_aria_quasistatic = poses_aria.copy()
    force_torque_df_quasistatic = force_torque_df.copy()
    temperature_df_quasistatic = temperature_df.copy()

    pose_timestamps_ns = poses_aria["timestamp"].to_numpy(dtype=np.int64)
    linear_velocity_device = poses_aria[
        [
            "device_linear_velocity_x_device",
            "device_linear_velocity_y_device",
            "device_linear_velocity_z_device",
        ]
    ].to_numpy(dtype=np.float64)
    angular_velocity_device = poses_aria[
        [
            "angular_velocity_x_device",
            "angular_velocity_y_device",
            "angular_velocity_z_device",
        ]
    ].to_numpy(dtype=np.float64)

    linear_speed_mps = np.linalg.norm(linear_velocity_device, axis=1)
    angular_speed_radps = np.linalg.norm(angular_velocity_device, axis=1)

    pose_quasistatic_mask = (
        (linear_speed_mps <= float(epsilon_quasistatic_linear))
        & (angular_speed_radps <= float(epsilon_quasistatic_angular))
    )
    adjusted_mps_data_quasistatic = adjusted_mps_data_quasistatic.loc[pose_quasistatic_mask].copy()
    poses_aria_quasistatic = poses_aria_quasistatic.loc[pose_quasistatic_mask].copy()

    if pose_timestamps_ns.size == 0:
        raise ValueError("No Aria poses available for quasi-static filtering.")

    pose_timestamps_interp = pose_timestamps_ns.astype(np.float64)
    wrench_linear_speed_mps = np.interp(
        np.asarray(dynamic_signal_sample_timestamps_ns, dtype=np.int64).astype(np.float64),
        pose_timestamps_interp,
        linear_speed_mps,
    )
    wrench_angular_speed_radps = np.interp(
        np.asarray(dynamic_signal_sample_timestamps_ns, dtype=np.int64).astype(np.float64),
        pose_timestamps_interp,
        angular_speed_radps,
    )
    quasistatic_wrench_mask = (
        (wrench_linear_speed_mps <= float(epsilon_quasistatic_linear))
        & (wrench_angular_speed_radps <= float(epsilon_quasistatic_angular))
    )

    temperature_timestamps_ns = temperature_ft["timestamp"].to_numpy(dtype=np.int64)
    temperature_linear_speed_mps = np.interp(
        temperature_timestamps_ns.astype(np.float64),
        pose_timestamps_interp,
        linear_speed_mps,
    )
    temperature_angular_speed_radps = np.interp(
        temperature_timestamps_ns.astype(np.float64),
        pose_timestamps_interp,
        angular_speed_radps,
    )
    quasistatic_temperature_mask = (
        (temperature_linear_speed_mps <= float(epsilon_quasistatic_linear))
        & (temperature_angular_speed_radps <= float(epsilon_quasistatic_angular))
    )

    force_torque_df_quasistatic = force_torque_df_quasistatic.loc[quasistatic_wrench_mask].copy()
    temperature_df_quasistatic = temperature_df_quasistatic.loc[quasistatic_temperature_mask].copy()

    R_S_W = np.asarray(dynamic_signals["R_S_W"], dtype=np.float64)
    a_D = np.asarray(dynamic_signals["a_D"], dtype=np.float64)
    alpha_D = np.asarray(dynamic_signals["alpha_D"], dtype=np.float64)

    R_ft_ariaworld_quasistatic = R_S_W[quasistatic_wrench_mask]
    f_meas_S_quasistatic = f_meas_S[quasistatic_wrench_mask]
    tau_meas_S_quasistatic = tau_meas_S[quasistatic_wrench_mask]
    quasistatic_sample_count = int(quasistatic_wrench_mask.sum())
    if quasistatic_sample_count == 0:
        raise ValueError(
            "Quasi-static filtering removed all wrench samples. "
            "Consider increasing epsilon_quasistatic_linear and/or "
            "epsilon_quasistatic_angular."
        )
    quasistatic_params_velocity_only = _estimate_tool_params_ls(
        F_meas_S=f_meas_S_quasistatic,
        tau_meas_S=tau_meas_S_quasistatic,
        R_S_W_list=R_ft_ariaworld_quasistatic,
        m_known=m_known,
        c_S_known=None if dynamic_fit_cog else c_S_known,
        g=g,
    )

    print(
        "Quasi-static filtering kept "
        f"{quasistatic_sample_count}/{len(quasistatic_wrench_mask)} wrench samples, "
        f"{int(quasistatic_temperature_mask.sum())}/{len(quasistatic_temperature_mask)} "
        "temperature samples, and "
        f"{int(pose_quasistatic_mask.sum())}/{len(pose_quasistatic_mask)} Aria poses "
        f"(eps_linear={epsilon_quasistatic_linear}, "
        f"eps_angular={epsilon_quasistatic_angular})."
    )

    wrench_linear_acc_mps2 = np.linalg.norm(a_D, axis=1)
    wrench_angular_acc_radps2 = np.linalg.norm(alpha_D, axis=1)
    quasistatic_wrench_mask_dyn = (
        quasistatic_wrench_mask
        & (wrench_linear_acc_mps2 <= float(epsilon_quasistatic_linear_acc))
        & (wrench_angular_acc_radps2 <= float(epsilon_quasistatic_angular_acc))
    )
    quasistatic_wrench_mask_dyn_count = int(quasistatic_wrench_mask_dyn.sum())

    if quasistatic_wrench_mask_dyn_count == 0:
        print(
            "Dynamic quasi-static refinement removed all wrench samples; "
            "falling back to the velocity-only quasi-static initialization."
        )
        quasistatic_wrench_mask_used = quasistatic_wrench_mask
        force_torque_df_quasistatic_dyn = force_torque_df_quasistatic.copy()
        R_S_W_quasistatic_dyn = R_ft_ariaworld_quasistatic
        f_meas_S_quasistatic_dyn = f_meas_S_quasistatic
        tau_meas_S_quasistatic_dyn = tau_meas_S_quasistatic
        params = quasistatic_params_velocity_only
    else:
        quasistatic_wrench_mask_used = quasistatic_wrench_mask_dyn
        force_torque_df_quasistatic_dyn = force_torque_df.loc[quasistatic_wrench_mask_dyn].copy()
        R_S_W_quasistatic_dyn = R_S_W[quasistatic_wrench_mask_dyn]
        f_meas_S_quasistatic_dyn = f_meas_S[quasistatic_wrench_mask_dyn]
        tau_meas_S_quasistatic_dyn = tau_meas_S[quasistatic_wrench_mask_dyn]
        params = _estimate_tool_params_ls(
            F_meas_S=f_meas_S_quasistatic_dyn,
            tau_meas_S=tau_meas_S_quasistatic_dyn,
            R_S_W_list=R_S_W_quasistatic_dyn,
            m_known=m_known,
            c_S_known=None if dynamic_fit_cog else c_S_known,
            g=g,
        )

    print(
        "Dynamic quasi-static refinement kept "
        f"{quasistatic_wrench_mask_dyn_count}/{len(quasistatic_wrench_mask_dyn)} wrench samples "
        f"(eps_linear_acc={epsilon_quasistatic_linear_acc}, "
        f"eps_angular_acc={epsilon_quasistatic_angular_acc})."
    )

    pose_timestamp_range_ns = [
        int(np.min(pose_timestamps_ns)),
        int(np.max(pose_timestamps_ns)),
    ]
    dynamic_signal_timestamp_range_ns = [
        int(np.min(np.asarray(dynamic_signal_sample_timestamps_ns, dtype=np.int64))),
        int(np.max(np.asarray(dynamic_signal_sample_timestamps_ns, dtype=np.int64))),
    ]
    valid_dynamic_mask = (
        (np.asarray(dynamic_signal_sample_timestamps_ns, dtype=np.int64) >= pose_timestamp_range_ns[0])
        & (np.asarray(dynamic_signal_sample_timestamps_ns, dtype=np.int64) <= pose_timestamp_range_ns[1])
    )
    valid_dynamic_sample_count = int(valid_dynamic_mask.sum())
    invalid_dynamic_sample_count = int((~valid_dynamic_mask).sum())
    print(
        "Dynamic valid-overlap mask: "
        f"pose_ns=[{pose_timestamp_range_ns[0]}, {pose_timestamp_range_ns[1]}], "
        f"dynamic_signal_ns=[{dynamic_signal_timestamp_range_ns[0]}, {dynamic_signal_timestamp_range_ns[1]}], "
        f"valid={valid_dynamic_sample_count}, invalid_excluded={invalid_dynamic_sample_count}."
    )
    if valid_dynamic_sample_count == 0:
        raise ValueError(
            "No valid pose/dynamic-signal overlap found for the dynamic fit. "
            "Cannot run dynamic least-squares without overlapping timestamps."
        )

    c_S_dynamic_init = np.asarray(
        params["c_S"] if dynamic_fit_cog else c_S_known,
        dtype=np.float64,
    ).reshape(3,)

    return {
        "adjusted_mps_data_quasistatic": adjusted_mps_data_quasistatic,
        "poses_aria_quasistatic": poses_aria_quasistatic,
        "force_torque_df_quasistatic": force_torque_df_quasistatic,
        "force_torque_df_quasistatic_dyn": force_torque_df_quasistatic_dyn,
        "temperature_df_quasistatic": temperature_df_quasistatic,
        "quasistatic_params": params,
        "quasistatic_params_velocity_only": quasistatic_params_velocity_only,
        "quasistatic_wrench_mask": quasistatic_wrench_mask,
        "quasistatic_wrench_mask_dyn": quasistatic_wrench_mask_dyn,
        "quasistatic_wrench_mask_used": quasistatic_wrench_mask_used,
        "quasistatic_temperature_mask": quasistatic_temperature_mask,
        "quasistatic_pose_mask": pose_quasistatic_mask,
        "quasistatic_wrench_sample_count_velocity_only": quasistatic_sample_count,
        "quasistatic_wrench_sample_count_dyn": quasistatic_wrench_mask_dyn_count,
        "epsilon_quasistatic_linear": float(epsilon_quasistatic_linear),
        "epsilon_quasistatic_angular": float(epsilon_quasistatic_angular),
        "epsilon_quasistatic_linear_acc": float(epsilon_quasistatic_linear_acc),
        "epsilon_quasistatic_angular_acc": float(epsilon_quasistatic_angular_acc),
        "pose_timestamp_range_ns": pose_timestamp_range_ns,
        "dynamic_signal_timestamp_range_ns": dynamic_signal_timestamp_range_ns,
        "valid_dynamic_mask": valid_dynamic_mask,
        "valid_dynamic_sample_count": valid_dynamic_sample_count,
        "invalid_dynamic_sample_count": invalid_dynamic_sample_count,
        "c_S_dynamic_init": c_S_dynamic_init,
    }


def estimate_tool_params_dynamic(
    vrs_path: Path | str,
    mps_file: Path | str,
    rosbag_path: Path | str,
    temp_path: Path | str,
    camchain_imucam: Path | str,
    *,
    stride: int = 1,
    min_qr_pairs: int = 2,
    deduplicate_by_qr_timestamp: bool = True,
    max_unique_qr_detections: Optional[int] = 50,
    mad_multiplier: float = 3.5,
    visualize_aria_velocities: bool = False,
    epsilon_quasistatic_linear: float = 0.1,
    epsilon_quasistatic_angular: float = 0.2,
    epsilon_quasistatic_linear_acc: float = 0.5,
    epsilon_quasistatic_angular_acc: float = 1.0,
    m_known: float = 0.490,
    c_S_known: Optional[np.ndarray] = None,
    dynamic_fit_cog: bool = False,
    dynamic_torque_weight: float = 1.0,
    dynamic_use_gyro_term: bool = True,
    compare_dynamic_gyro: bool = False,
    gyro_selection_negligible_rmsT_tolerance: float = 1e-4,
    compare_dynamic_time_offset: bool = False,
    candidate_time_offsets_ms: Optional[List[float]] = None,
    time_offset_selection_negligible_rmsT_tolerance: float = 1e-4,
    sg_window_length: int = 21,
    sg_polyorder: int = 3,
    sg_window_length_linear: Optional[int] = None,
    sg_polyorder_linear: Optional[int] = None,
    sg_window_length_angular: Optional[int] = None,
    sg_polyorder_angular: Optional[int] = None,
    evaluate_candidate_smoothing: bool = False,
    candidate_smoothing_configs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Prepare the same calibration inputs as ``estimate_tool_params`` up to the
    first force/torque CSV load, but use the new high-precision multi-QR
    alignment instead of the legacy first-hit alignment.

    This function is intentionally additive and does not modify the existing
    static estimation path.
    """
    temp_path_for_debug = Path(temp_path) if isinstance(temp_path, str) else temp_path
    if c_S_known is None:
        c_S_known = np.array([0.021, 0.020, 0.059], dtype=np.float64)
    else:
        c_S_known = np.asarray(c_S_known, dtype=np.float64).reshape(3,)

    dynamic_signal_debug_dir = temp_path_for_debug / "dynamic_signal_debug"
    run_smoothing_sweep = bool(evaluate_candidate_smoothing or candidate_smoothing_configs is not None)
    primary_smoothing_config = _resolve_requested_dynamic_smoothing_config(
        sg_window_length=sg_window_length,
        sg_polyorder=sg_polyorder,
        sg_window_length_linear=sg_window_length_linear,
        sg_polyorder_linear=sg_polyorder_linear,
        sg_window_length_angular=sg_window_length_angular,
        sg_polyorder_angular=sg_polyorder_angular,
    )
    dynamic_signal_debug_plot_paths = [
        dynamic_signal_debug_dir / "aria_dynamic_signal_magnitudes.png",
        dynamic_signal_debug_dir / "aria_world_velocity_raw_vs_smoothed.png",
        dynamic_signal_debug_dir / "aria_dynamic_signal_components_device_frame.png",
        dynamic_signal_debug_dir / "aria_dynamic_signal_sensor_frame_magnitudes.png",
    ]
    prep_state = _load_dynamic_recording_data(
        vrs_path=vrs_path,
        mps_file=mps_file,
        rosbag_path=rosbag_path,
        temp_path=temp_path,
        camchain_imucam=camchain_imucam,
        stride=stride,
        min_qr_pairs=min_qr_pairs,
        deduplicate_by_qr_timestamp=deduplicate_by_qr_timestamp,
        max_unique_qr_detections=max_unique_qr_detections,
        mad_multiplier=mad_multiplier,
        visualize_aria_velocities=visualize_aria_velocities,
        sg_window_length=sg_window_length,
        sg_polyorder=sg_polyorder,
        sg_window_length_linear=sg_window_length_linear,
        sg_polyorder_linear=sg_polyorder_linear,
        sg_window_length_angular=sg_window_length_angular,
        sg_polyorder_angular=sg_polyorder_angular,
        dynamic_signal_time_offset_ns=0,
        dynamic_signal_visualize=bool(visualize_aria_velocities) and not run_smoothing_sweep,
        dynamic_signal_visualize_out_dir=dynamic_signal_debug_dir if not run_smoothing_sweep else None,
    )

    cam_calib = prep_state["cam_calib"]
    imu_calib = prep_state["imu_calib"]
    vrs_utils = prep_state["vrs_utils"]
    temp_path = prep_state["temp_path"]
    temp_path_rosbag = prep_state["temp_path_rosbag"]
    temp_path_vrs = prep_state["temp_path_vrs"]
    force_torque_topic = prep_state["force_torque_topic"]
    temperature_topic = prep_state["temperature_topic"]
    adjusted_mps_file = prep_state["adjusted_mps_file"]
    adjusted_mps_data = prep_state["adjusted_mps_data"]
    poses_aria = prep_state["poses_aria"]
    force_torque_df = prep_state["force_torque_df"]
    temperature_df = prep_state["temperature_df"]
    temperature_ft = prep_state["temperature_ft"]
    wrench_timestamps_ns = prep_state["wrench_timestamps_ns"]
    dynamic_signals = prep_state["dynamic_signals"]
    T_ariadevice_ft = prep_state["T_ariadevice_ft"]
    f_meas_S = prep_state["f_meas_S"]
    tau_meas_S = prep_state["tau_meas_S"]
    delta = prep_state["delta_ns"]
    gripper_qr_pairs_cache_file = prep_state["gripper_qr_pairs_cache_file"]
    vrs_qr_pairs_cache_file = prep_state["vrs_qr_pairs_cache_file"]

    bias_state = _compute_dynamic_quasistatic_bias_state(
        adjusted_mps_data=adjusted_mps_data,
        poses_aria=poses_aria,
        force_torque_df=force_torque_df,
        temperature_df=temperature_df,
        temperature_ft=temperature_ft,
        f_meas_S=f_meas_S,
        tau_meas_S=tau_meas_S,
        dynamic_signals=dynamic_signals,
        dynamic_signal_sample_timestamps_ns=prep_state["dynamic_signal_sample_timestamps_ns"],
        m_known=m_known,
        c_S_known=c_S_known,
        dynamic_fit_cog=dynamic_fit_cog,
        epsilon_quasistatic_linear=epsilon_quasistatic_linear,
        epsilon_quasistatic_angular=epsilon_quasistatic_angular,
        epsilon_quasistatic_linear_acc=epsilon_quasistatic_linear_acc,
        epsilon_quasistatic_angular_acc=epsilon_quasistatic_angular_acc,
        g=9.81,
    )

    adjusted_mps_data_quasistatic = bias_state["adjusted_mps_data_quasistatic"]
    poses_aria_quasistatic = bias_state["poses_aria_quasistatic"]
    force_torque_df_quasistatic = bias_state["force_torque_df_quasistatic"]
    force_torque_df_quasistatic_dyn = bias_state["force_torque_df_quasistatic_dyn"]
    temperature_df_quasistatic = bias_state["temperature_df_quasistatic"]
    params = bias_state["quasistatic_params"]
    quasistatic_params_velocity_only = bias_state["quasistatic_params_velocity_only"]
    quasistatic_wrench_mask = bias_state["quasistatic_wrench_mask"]
    quasistatic_wrench_mask_dyn = bias_state["quasistatic_wrench_mask_dyn"]
    quasistatic_wrench_mask_used = bias_state["quasistatic_wrench_mask_used"]
    quasistatic_temperature_mask = bias_state["quasistatic_temperature_mask"]
    pose_quasistatic_mask = bias_state["quasistatic_pose_mask"]
    quasistatic_sample_count = bias_state["quasistatic_wrench_sample_count_velocity_only"]
    quasistatic_wrench_mask_dyn_count = bias_state["quasistatic_wrench_sample_count_dyn"]
    pose_timestamp_range_ns = bias_state["pose_timestamp_range_ns"]
    wrench_timestamp_range_ns = [
        int(np.min(wrench_timestamps_ns)),
        int(np.max(wrench_timestamps_ns)),
    ]
    valid_dynamic_mask = bias_state["valid_dynamic_mask"]
    valid_dynamic_sample_count = bias_state["valid_dynamic_sample_count"]
    invalid_dynamic_sample_count = bias_state["invalid_dynamic_sample_count"]
    wrench_timestamps_ns_dynamic_valid = wrench_timestamps_ns[valid_dynamic_mask]
    b_f_init = np.asarray(params["f0"], dtype=np.float64)
    b_tau_init = np.asarray(params["tau0"], dtype=np.float64)
    c_S_dynamic_init = bias_state["c_S_dynamic_init"]

    a_D = dynamic_signals["a_D"]
    omega_D = dynamic_signals["omega_D"]
    alpha_D = dynamic_signals["alpha_D"]
    a_S = dynamic_signals["a_S"]
    omega_S = dynamic_signals["omega_S"]
    alpha_S = dynamic_signals["alpha_S"]
    R_S_W = dynamic_signals["R_S_W"]
    dynamic_smoothing_debug_dir = temp_path / "dynamic_smoothing_debug"
    dynamic_smoothing_debug_plot_paths: List[Path] = []
    dynamic_smoothing_candidate_results: List[Dict[str, Any]] = []
    dynamic_smoothing_best_index: Optional[int] = None
    dynamic_smoothing_best_config: Optional[Dict[str, int]] = None

    if run_smoothing_sweep:
        if candidate_smoothing_configs is None:
            candidate_smoothing_configs_normalized = _default_candidate_smoothing_configs(
                primary_smoothing_config
            )
        else:
            candidate_smoothing_configs_normalized = [
                _normalize_candidate_smoothing_config(candidate_config, primary_smoothing_config)
                for candidate_config in candidate_smoothing_configs
            ]
            candidate_smoothing_configs_normalized = _default_candidate_smoothing_configs(
                primary_smoothing_config
            ) if len(candidate_smoothing_configs_normalized) == 0 else candidate_smoothing_configs_normalized

        baseline_config_key = _dynamic_smoothing_config_key(primary_smoothing_config)
        candidate_configs_unique: List[Dict[str, Any]] = []
        seen_candidate_keys = set()
        for candidate_config in [primary_smoothing_config, *candidate_smoothing_configs_normalized]:
            candidate_key = _dynamic_smoothing_config_key(candidate_config)
            if candidate_key not in seen_candidate_keys:
                seen_candidate_keys.add(candidate_key)
                candidate_configs_unique.append(candidate_config)

        for candidate_index, candidate_config in enumerate(candidate_configs_unique):
            candidate_key = _dynamic_smoothing_config_key(candidate_config)
            candidate_name = str(
                candidate_config.get(
                    "name",
                    _format_dynamic_smoothing_config_name(_extract_smoothing_kwargs(candidate_config)),
                )
            )
            if candidate_key == baseline_config_key:
                candidate_dynamic_signals = dynamic_signals
            else:
                candidate_dynamic_signals = compute_dynamic_signals_from_aria(
                    poses_aria=poses_aria,
                    wrench_timestamps_ns=wrench_timestamps_ns,
                    T_ariadevice_ft=T_ariadevice_ft,
                    visualize=False,
                    visualize_out_dir=None,
                    **_extract_smoothing_kwargs(candidate_config),
                )

            candidate_signal_stats = _summarize_dynamic_signal_stats(
                candidate_dynamic_signals,
                valid_dynamic_mask,
            )
            candidate_fit_bundle = _run_dynamic_fit_for_dynamic_signals(
                dynamic_signals=candidate_dynamic_signals,
                valid_dynamic_mask=valid_dynamic_mask,
                F_meas_S=f_meas_S,
                tau_meas_S=tau_meas_S,
                m_known=m_known,
                c_S_known=c_S_known,
                dynamic_fit_cog=dynamic_fit_cog,
                c_S_init=c_S_dynamic_init,
                b_f_init=b_f_init,
                b_tau_init=b_tau_init,
                dynamic_torque_weight=dynamic_torque_weight,
                dynamic_use_gyro_term=dynamic_use_gyro_term,
                dynamic_fit_debug_dir=None,
                visualize=False,
            )

            dynamic_smoothing_candidate_results.append(
                {
                    "candidate_index": int(candidate_index),
                    "name": candidate_name,
                    "requested_smoothing_config": candidate_config,
                    "resolved_smoothing_config": candidate_dynamic_signals["smoothing_config_resolved"],
                    "signal_stats": candidate_signal_stats,
                    "fit_metrics": candidate_fit_bundle["dynamic_fit_residual_stats"],
                    "dynamic_fit": candidate_fit_bundle["dynamic_fit"],
                    "dynamic_fit_success": bool(candidate_fit_bundle["dynamic_fit"]["success"]),
                    "dynamic_fit_nfev": int(candidate_fit_bundle["dynamic_fit"]["nfev"]),
                    "dynamic_fit_cost": float(candidate_fit_bundle["dynamic_fit"]["cost"]),
                    "inertia_eigenvalues": np.asarray(
                        candidate_fit_bundle["dynamic_fit"]["inertia_eigenvalues"],
                        dtype=np.float64,
                    ).tolist(),
                    "inertia_det": float(candidate_fit_bundle["dynamic_fit"]["inertia_det"]),
                    "dynamic_fit_bundle": candidate_fit_bundle,
                    "dynamic_signals": candidate_dynamic_signals,
                }
            )
            inertia_eigenvalues_str_candidate = ", ".join(
                f"{float(value):.3e}"
                for value in np.asarray(
                    candidate_fit_bundle["dynamic_fit"]["inertia_eigenvalues"],
                    dtype=np.float64,
                )
            )
            print(
                f"[dynamic_smoothing] {candidate_name}: "
                f"rmsF {candidate_fit_bundle['dynamic_fit_residual_stats']['rmsF_before']:.4f}"
                f" -> {candidate_fit_bundle['dynamic_fit_residual_stats']['rmsF_after']:.4f}, "
                f"rmsT {candidate_fit_bundle['dynamic_fit_residual_stats']['rmsT_before']:.4f}"
                f" -> {candidate_fit_bundle['dynamic_fit_residual_stats']['rmsT_after']:.4f}, "
                f"mean||a_D||={candidate_signal_stats['a_D_norm_mean']:.4f}, "
                f"max||a_D||={candidate_signal_stats['a_D_norm_max']:.4f}, "
                f"eig=[{inertia_eigenvalues_str_candidate}]"
            )

        dynamic_smoothing_best_index = _select_best_dynamic_smoothing_candidate(
            dynamic_smoothing_candidate_results
        )
        best_smoothing_candidate = dynamic_smoothing_candidate_results[dynamic_smoothing_best_index]
        dynamic_smoothing_best_config = best_smoothing_candidate["resolved_smoothing_config"]
        print(
            "[dynamic_smoothing] selected best candidate: "
            f"{best_smoothing_candidate['name']} "
            f"({_format_dynamic_smoothing_config_label(dynamic_smoothing_best_config)}) "
            "because it achieved the lowest stable torque RMS after fit."
        )
        dynamic_signals = best_smoothing_candidate["dynamic_signals"]
        dynamic_signal_stats = best_smoothing_candidate["signal_stats"]
        dynamic_fit_bundle = best_smoothing_candidate["dynamic_fit_bundle"]
        dynamic_smoothing_debug_plot_paths = _visualize_dynamic_smoothing_candidates(
            dynamic_smoothing_candidate_results,
            dynamic_smoothing_best_index,
            visualize=bool(visualize_aria_velocities),
            visualize_out_dir=dynamic_smoothing_debug_dir,
        )
        _visualize_dynamic_signals_from_aria(
            dynamic_signals,
            visualize=bool(visualize_aria_velocities),
            visualize_out_dir=dynamic_signal_debug_dir,
        )
    else:
        dynamic_signal_stats = _summarize_dynamic_signal_stats(
            dynamic_signals,
            valid_dynamic_mask,
        )
        dynamic_fit_bundle = _run_dynamic_fit_for_dynamic_signals(
            dynamic_signals=dynamic_signals,
            valid_dynamic_mask=valid_dynamic_mask,
            F_meas_S=f_meas_S,
            tau_meas_S=tau_meas_S,
            m_known=m_known,
            c_S_known=c_S_known,
            dynamic_fit_cog=dynamic_fit_cog,
            c_S_init=c_S_dynamic_init,
            b_f_init=b_f_init,
            b_tau_init=b_tau_init,
            dynamic_torque_weight=dynamic_torque_weight,
            dynamic_use_gyro_term=dynamic_use_gyro_term,
            dynamic_fit_debug_dir=temp_path / "dynamic_fit_debug",
            visualize=bool(visualize_aria_velocities),
        )
        dynamic_smoothing_best_config = dynamic_signals["smoothing_config_resolved"]

    dynamic_time_offset_debug_dir = temp_path / "dynamic_time_offset_debug"
    dynamic_time_offset_debug_plot_paths: List[Path] = []
    time_offset_candidate_results: List[Dict[str, Any]] = []
    time_offset_selected_index: Optional[int] = None
    time_offset_selected_reason: Optional[str] = None
    selected_time_offset_ms: float = 0.0
    selected_time_offset_ns: int = 0
    selected_dynamic_signal_timestamps_ns = np.asarray(
        dynamic_signals["timestamps_ns_wrench"],
        dtype=np.int64,
    )
    selected_dynamic_use_gyro_term_for_time_offset = bool(dynamic_use_gyro_term)
    selected_dynamic_smoothing_kwargs = _extract_smoothing_kwargs(
        dynamic_signals["smoothing_config_resolved"]
    )
    if not compare_dynamic_time_offset:
        candidate_time_offsets_ms_values = [0.0]
    elif candidate_time_offsets_ms is None:
        candidate_time_offsets_ms_values = [0.0, 5.0, 10.0, 15.0, 20, 25, 30, 35]
    else:
        candidate_time_offsets_ms_values = [float(value) for value in candidate_time_offsets_ms]
        if len(candidate_time_offsets_ms_values) == 0:
            candidate_time_offsets_ms_values = [0.0]
    candidate_time_offsets_ms_unique: List[float] = []
    seen_time_offsets_ns = set()
    for candidate_offset_ms in candidate_time_offsets_ms_values:
        candidate_offset_ns = int(round(candidate_offset_ms * 1e6))
        if candidate_offset_ns not in seen_time_offsets_ns:
            seen_time_offsets_ns.add(candidate_offset_ns)
            candidate_time_offsets_ms_unique.append(candidate_offset_ms)

    # Time-offset convention: a positive offset means we sample the Aria/MPS
    # dynamic signals at ``wrench_timestamps_ns + offset_ns``. In other words,
    # the same FT sample is compared against a later dynamic state.
    for time_offset_candidate_index, candidate_offset_ms in enumerate(candidate_time_offsets_ms_unique):
        candidate_offset_ns = int(round(candidate_offset_ms * 1e6))
        candidate_dynamic_sampling_timestamps_ns = wrench_timestamps_ns + candidate_offset_ns
        candidate_valid_dynamic_mask = (
            (candidate_dynamic_sampling_timestamps_ns >= pose_timestamp_range_ns[0])
            & (candidate_dynamic_sampling_timestamps_ns <= pose_timestamp_range_ns[1])
        )
        candidate_valid_sample_count = int(candidate_valid_dynamic_mask.sum())

        if candidate_offset_ns == 0 and not bool(dynamic_fit_bundle["dynamic_fit"]["use_gyro_term"]):
            candidate_dynamic_signals = dynamic_signals
            candidate_signal_stats = _summarize_dynamic_signal_stats(
                candidate_dynamic_signals,
                candidate_valid_dynamic_mask,
            )
            candidate_fit_bundle = dynamic_fit_bundle
        else:
            candidate_dynamic_signals = compute_dynamic_signals_from_aria(
                poses_aria=poses_aria,
                wrench_timestamps_ns=candidate_dynamic_sampling_timestamps_ns,
                T_ariadevice_ft=T_ariadevice_ft,
                visualize=False,
                visualize_out_dir=None,
                **selected_dynamic_smoothing_kwargs,
            )
            candidate_signal_stats = _summarize_dynamic_signal_stats(
                candidate_dynamic_signals,
                candidate_valid_dynamic_mask,
            )
            candidate_fit_bundle = _run_dynamic_fit_for_dynamic_signals(
                dynamic_signals=candidate_dynamic_signals,
                valid_dynamic_mask=candidate_valid_dynamic_mask,
                F_meas_S=f_meas_S,
                tau_meas_S=tau_meas_S,
                m_known=m_known,
                c_S_known=c_S_known,
                dynamic_fit_cog=dynamic_fit_cog,
                c_S_init=c_S_dynamic_init,
                b_f_init=b_f_init,
                b_tau_init=b_tau_init,
                dynamic_torque_weight=dynamic_torque_weight,
                dynamic_use_gyro_term=selected_dynamic_use_gyro_term_for_time_offset,
                dynamic_fit_debug_dir=None,
                visualize=False,
            )

        candidate_dynamic_fit = candidate_fit_bundle["dynamic_fit"]
        candidate_fit_metrics = candidate_fit_bundle["dynamic_fit_residual_stats"]
        candidate_time_offset_result = {
            "candidate_index": int(time_offset_candidate_index),
            "offset_ms": float(candidate_offset_ms),
            "offset_ns": int(candidate_offset_ns),
            "valid_sample_count": int(candidate_valid_sample_count),
            "invalid_sample_count": int((~candidate_valid_dynamic_mask).sum()),
            "fit_metrics": candidate_fit_metrics,
            "dynamic_fit": candidate_dynamic_fit,
            "dynamic_fit_bundle": candidate_fit_bundle,
            "dynamic_signals": candidate_dynamic_signals,
            "signal_stats": candidate_signal_stats,
            "valid_dynamic_mask": candidate_valid_dynamic_mask,
            "sample_timestamps_ns": candidate_dynamic_sampling_timestamps_ns,
            "inertia_eigenvalues": np.asarray(
                candidate_dynamic_fit["inertia_eigenvalues"],
                dtype=np.float64,
            ).tolist(),
            "inertia_det": float(candidate_dynamic_fit["inertia_det"]),
        }
        time_offset_candidate_results.append(candidate_time_offset_result)

        inertia_eigenvalues_str_time_offset = ", ".join(
            f"{float(value):.3e}"
            for value in np.asarray(candidate_dynamic_fit["inertia_eigenvalues"], dtype=np.float64)
        )
        print(
            f"[dynamic_time_offset] dt={candidate_offset_ms:.0f} ms: "
            f"rmsF {candidate_fit_metrics['rmsF_before']:.4f} -> {candidate_fit_metrics['rmsF_after']:.4f}, "
            f"rmsT {candidate_fit_metrics['rmsT_before']:.4f} -> {candidate_fit_metrics['rmsT_after']:.4f}, "
            f"valid={candidate_valid_sample_count}, eig=[{inertia_eigenvalues_str_time_offset}]"
        )

    if compare_dynamic_time_offset:
        time_offset_selected_index, time_offset_selected_reason = _select_best_dynamic_time_offset_candidate(
            time_offset_candidate_results,
            negligible_rmsT_tolerance=time_offset_selection_negligible_rmsT_tolerance,
        )
    else:
        time_offset_selected_index = next(
            (
                result["candidate_index"]
                for result in time_offset_candidate_results
                if abs(float(result["offset_ms"])) < 1e-12
            ),
            0,
        )
        time_offset_selected_reason = "Time-offset comparison disabled; keeping the zero-shift model."

    selected_time_offset_result = time_offset_candidate_results[time_offset_selected_index]
    selected_time_offset_ms = float(selected_time_offset_result["offset_ms"])
    selected_time_offset_ns = int(selected_time_offset_result["offset_ns"])
    dynamic_signals = selected_time_offset_result["dynamic_signals"]
    dynamic_signal_stats = selected_time_offset_result["signal_stats"]
    dynamic_fit_bundle = selected_time_offset_result["dynamic_fit_bundle"]
    valid_dynamic_mask = np.asarray(selected_time_offset_result["valid_dynamic_mask"], dtype=bool)
    valid_dynamic_sample_count = int(selected_time_offset_result["valid_sample_count"])
    invalid_dynamic_sample_count = int(selected_time_offset_result["invalid_sample_count"])
    selected_dynamic_signal_timestamps_ns = np.asarray(
        selected_time_offset_result["sample_timestamps_ns"],
        dtype=np.int64,
    )
    wrench_timestamps_ns_dynamic_valid = wrench_timestamps_ns[valid_dynamic_mask]
    dynamic_time_offset_debug_plot_paths = _visualize_dynamic_time_offset_candidates(
        time_offset_candidate_results,
        time_offset_selected_index,
        visualize=bool(visualize_aria_velocities),
        visualize_out_dir=dynamic_time_offset_debug_dir,
    )
    print(
        f"[dynamic_time_offset] selected dt={selected_time_offset_ms:.0f} ms: "
        f"{time_offset_selected_reason}"
    )

    a_D = dynamic_signals["a_D"]
    omega_D = dynamic_signals["omega_D"]
    alpha_D = dynamic_signals["alpha_D"]
    a_S = dynamic_signals["a_S"]
    omega_S = dynamic_signals["omega_S"]
    alpha_S = dynamic_signals["alpha_S"]
    R_S_W = dynamic_signals["R_S_W"]

    dynamic_gyro_debug_dir = temp_path / "dynamic_gyro_debug"
    dynamic_gyro_debug_plot_paths: List[Path] = []
    gyro_comparison_results: List[Dict[str, Any]] = []
    gyro_selected_index: Optional[int] = None
    gyro_selected_reason: Optional[str] = None

    gyro_flags_to_evaluate = [False, True] if compare_dynamic_gyro else [bool(dynamic_use_gyro_term)]
    baseline_dynamic_use_gyro_term = bool(dynamic_fit_bundle["dynamic_fit"]["use_gyro_term"])
    for gyro_candidate_index, gyro_flag in enumerate(gyro_flags_to_evaluate):
        if bool(gyro_flag) == baseline_dynamic_use_gyro_term:
            gyro_candidate_fit_bundle = dynamic_fit_bundle
        else:
            gyro_candidate_fit_bundle = _run_dynamic_fit_for_dynamic_signals(
                dynamic_signals=dynamic_signals,
                valid_dynamic_mask=valid_dynamic_mask,
                F_meas_S=f_meas_S,
                tau_meas_S=tau_meas_S,
                m_known=m_known,
                c_S_known=c_S_known,
                dynamic_fit_cog=dynamic_fit_cog,
                c_S_init=c_S_dynamic_init,
                b_f_init=b_f_init,
                b_tau_init=b_tau_init,
                dynamic_torque_weight=dynamic_torque_weight,
                dynamic_use_gyro_term=bool(gyro_flag),
                dynamic_fit_debug_dir=None,
                visualize=False,
            )

        gyro_dynamic_fit = gyro_candidate_fit_bundle["dynamic_fit"]
        gyro_fit_metrics = gyro_candidate_fit_bundle["dynamic_fit_residual_stats"]
        gyro_candidate_result = {
            "candidate_index": int(gyro_candidate_index),
            "use_gyro_term": bool(gyro_flag),
            "fit_metrics": gyro_fit_metrics,
            "dynamic_fit": gyro_dynamic_fit,
            "dynamic_fit_bundle": gyro_candidate_fit_bundle,
            "b_f": np.asarray(gyro_dynamic_fit["b_f"], dtype=np.float64).tolist(),
            "b_tau": np.asarray(gyro_dynamic_fit["b_tau"], dtype=np.float64).tolist(),
            "inertia_eigenvalues": np.asarray(
                gyro_dynamic_fit["inertia_eigenvalues"],
                dtype=np.float64,
            ).tolist(),
            "inertia_det": float(gyro_dynamic_fit["inertia_det"]),
        }
        gyro_comparison_results.append(gyro_candidate_result)

        inertia_eigenvalues_str_gyro = ", ".join(
            f"{float(value):.3e}"
            for value in np.asarray(gyro_dynamic_fit["inertia_eigenvalues"], dtype=np.float64)
        )
        print(
            f"[dynamic_gyro_compare] gyro={bool(gyro_flag)}: "
            f"rmsF {gyro_fit_metrics['rmsF_before']:.4f} -> {gyro_fit_metrics['rmsF_after']:.4f}, "
            f"rmsT {gyro_fit_metrics['rmsT_before']:.4f} -> {gyro_fit_metrics['rmsT_after']:.4f}, "
            f"success={gyro_dynamic_fit['success']}, status={gyro_dynamic_fit['status']}, "
            f"nfev={gyro_dynamic_fit['nfev']}, cost={gyro_dynamic_fit['cost']:.6e}, "
            f"det={gyro_dynamic_fit['inertia_det']:.6e}, eig=[{inertia_eigenvalues_str_gyro}]"
        )

    if compare_dynamic_gyro:
        gyro_selected_index, gyro_selected_reason = _select_best_dynamic_gyro_candidate(
            gyro_comparison_results,
            negligible_rmsT_tolerance=gyro_selection_negligible_rmsT_tolerance,
        )
        selected_gyro_result = gyro_comparison_results[gyro_selected_index]
        dynamic_fit_bundle = selected_gyro_result["dynamic_fit_bundle"]
        dynamic_use_gyro_term_selected = bool(selected_gyro_result["use_gyro_term"])
        dynamic_gyro_debug_plot_paths = _visualize_dynamic_gyro_comparison(
            gyro_comparison_results,
            gyro_selected_index,
            visualize=bool(visualize_aria_velocities),
            visualize_out_dir=dynamic_gyro_debug_dir,
        )
        print(
            f"[dynamic_gyro_compare] selected gyro={dynamic_use_gyro_term_selected}: "
            f"{gyro_selected_reason}"
        )
    else:
        gyro_selected_index = 0
        gyro_selected_reason = "Gyro comparison disabled; using the requested dynamic_use_gyro_term setting."
        dynamic_use_gyro_term_selected = bool(gyro_comparison_results[gyro_selected_index]["use_gyro_term"])

    dynamic_fit_debug_dir = temp_path / "dynamic_fit_debug"
    if run_smoothing_sweep or compare_dynamic_time_offset or compare_dynamic_gyro:
        _visualize_dynamic_fit_residuals(
            wrench_time_s=dynamic_fit_bundle["wrench_time_s_dynamic_valid"],
            dynamic_fit=dynamic_fit_bundle["dynamic_fit"],
            visualize=bool(visualize_aria_velocities),
            visualize_out_dir=dynamic_fit_debug_dir,
        )
        dynamic_fit_bundle["dynamic_fit_debug_plot_paths"] = [
            dynamic_fit_debug_dir / "dynamic_fit_residual_magnitudes.png",
            dynamic_fit_debug_dir / "dynamic_fit_residual_components.png",
        ]

    dynamic_fit = dynamic_fit_bundle["dynamic_fit"]
    dynamic_fit_residual_stats = dynamic_fit_bundle["dynamic_fit_residual_stats"]
    dynamic_fit_debug_plot_paths = dynamic_fit_bundle["dynamic_fit_debug_plot_paths"]
    f_model = dynamic_fit_bundle["f_model"]
    tau_model = dynamic_fit_bundle["tau_model"]
    f_ext_hat = dynamic_fit_bundle["f_ext_hat"]
    tau_ext_hat = dynamic_fit_bundle["tau_ext_hat"]
    f_ext_hat_init = dynamic_fit_bundle["f_ext_hat_init"]
    tau_ext_hat_init = dynamic_fit_bundle["tau_ext_hat_init"]
    f_model_valid = dynamic_fit_bundle["f_model_valid"]
    tau_model_valid = dynamic_fit_bundle["tau_model_valid"]
    f_ext_hat_valid = dynamic_fit_bundle["f_ext_hat_valid"]
    tau_ext_hat_valid = dynamic_fit_bundle["tau_ext_hat_valid"]
    f_ext_hat_init_valid = dynamic_fit_bundle["f_ext_hat_init_valid"]
    tau_ext_hat_init_valid = dynamic_fit_bundle["tau_ext_hat_init_valid"]

    print(
        "Dynamic LS fit RMS residuals: "
        f"force {dynamic_fit['rmsF_init']:.4f} -> {dynamic_fit['rmsF']:.4f}, "
        f"torque {dynamic_fit['rmsT_init']:.4f} -> {dynamic_fit['rmsT']:.4f} "
        f"on {valid_dynamic_sample_count} valid overlap samples "
        f"(gyro_term={dynamic_use_gyro_term_selected}, time_offset_ms={selected_time_offset_ms:.3f}, "
        f"torque_weight={dynamic_torque_weight})."
    )

    smoothing_summary_label = (
        _format_dynamic_smoothing_config_label(dynamic_smoothing_best_config)
        if dynamic_smoothing_best_config is not None
        else "n/a"
    )
    inertia_eigenvalues_str = ", ".join(
        f"{float(value):.6e}" for value in np.asarray(dynamic_fit["inertia_eigenvalues"], dtype=np.float64)
    )
    summary_lines = [
        "=== Dynamic FT Calibration Summary ===",
        (
            "Time overlap: "
            f"pose_ns=[{pose_timestamp_range_ns[0]}, {pose_timestamp_range_ns[1]}], "
            f"wrench_ns=[{wrench_timestamp_range_ns[0]}, {wrench_timestamp_range_ns[1]}], "
            f"valid={valid_dynamic_sample_count}, invalid_excluded={invalid_dynamic_sample_count}"
        ),
        (
            "Quasi-static masks: "
            f"velocity_only={quasistatic_sample_count}, "
            f"velocity_plus_acc={quasistatic_wrench_mask_dyn_count}"
        ),
        (
            "Smoothing: "
            f"selected={smoothing_summary_label}, "
            f"candidate_count={len(dynamic_smoothing_candidate_results) if run_smoothing_sweep else 1}"
        ),
        (
            "Dynamic signal stats: "
            f"mean||a_D||={dynamic_signal_stats['a_D_norm_mean']:.4f}, "
            f"max||a_D||={dynamic_signal_stats['a_D_norm_max']:.4f}, "
            f"mean||a_S||={dynamic_signal_stats['a_S_norm_mean']:.4f}, "
            f"max||a_S||={dynamic_signal_stats['a_S_norm_max']:.4f}, "
            f"mean||alpha_D||={dynamic_signal_stats['alpha_D_norm_mean']:.4f}, "
            f"max||alpha_D||={dynamic_signal_stats['alpha_D_norm_max']:.4f}, "
            f"mean||alpha_S||={dynamic_signal_stats['alpha_S_norm_mean']:.4f}, "
            f"max||alpha_S||={dynamic_signal_stats['alpha_S_norm_max']:.4f}"
        ),
        (
            "Dynamic fit: "
            f"success={dynamic_fit['success']}, status={dynamic_fit['status']}, "
            f"nfev={dynamic_fit['nfev']}, cost={dynamic_fit['cost']:.6e}, "
            f"gyro_term={dynamic_use_gyro_term_selected}, "
            f"time_offset_ms={selected_time_offset_ms:.3f}, "
            f"torque_weight={dynamic_torque_weight}"
        ),
        (
            "Residual RMS: "
            f"force {dynamic_fit['rmsF_init']:.4f} -> {dynamic_fit['rmsF']:.4f}, "
            f"torque {dynamic_fit['rmsT_init']:.4f} -> {dynamic_fit['rmsT']:.4f}"
        ),
        (
            "Time offset: "
            f"candidate_count={len(time_offset_candidate_results)}, "
            f"selected_ms={selected_time_offset_ms:.3f}, "
            f"reason={time_offset_selected_reason}"
        ),
        (
            "Gyro compare: "
            f"candidate_count={len(gyro_comparison_results)}, "
            f"selected={dynamic_use_gyro_term_selected}, "
            f"reason={gyro_selected_reason}"
        ),
        (
            "CoG: "
            f"fit_enabled={dynamic_fit_cog}, "
            f"init={np.asarray(c_S_dynamic_init, dtype=np.float64).tolist()}, "
            f"selected={np.asarray(dynamic_fit['c_S'], dtype=np.float64).tolist()}"
        ),
        (
            "Inertia: "
            f"spd={dynamic_fit['inertia_is_spd']}, det={dynamic_fit['inertia_det']:.6e}, "
            f"eig=[{inertia_eigenvalues_str}]"
        ),
        (
            "Debug dirs: "
            f"signals={dynamic_signal_debug_dir}, "
            f"fit={dynamic_fit_debug_dir}, "
            f"time_offset={dynamic_time_offset_debug_dir if compare_dynamic_time_offset else 'not_used'}, "
            f"smoothing={dynamic_smoothing_debug_dir if run_smoothing_sweep else 'not_used'}, "
            f"gyro={dynamic_gyro_debug_dir if compare_dynamic_gyro else 'not_used'}"
        ),
    ]
    dynamic_params_file = temp_path / "tool_params_dynamic_estimate.json"
    dynamic_params_payload = {
        "frame": "sensor",
        "mass_kg": float(dynamic_fit["m_known"]),
        "center_of_gravity_m_sensor": np.asarray(dynamic_fit["c_S"], dtype=np.float64).tolist(),
        "force_bias_N_sensor": np.asarray(dynamic_fit["b_f"], dtype=np.float64).tolist(),
        "torque_bias_Nm_sensor": np.asarray(dynamic_fit["b_tau"], dtype=np.float64).tolist(),
        "inertia_tensor_kgm2_sensor": np.asarray(dynamic_fit["I_C"], dtype=np.float64).tolist(),
        "inertia_eigenvalues_kgm2": np.asarray(dynamic_fit["inertia_eigenvalues"], dtype=np.float64).tolist(),
        "inertia_principal_moments_kgm2": np.asarray(
            dynamic_fit["inertia_principal_moments"],
            dtype=np.float64,
        ).tolist(),
        "inertia_det": float(dynamic_fit["inertia_det"]),
        "inertia_is_spd": bool(dynamic_fit["inertia_is_spd"]),
        "fit_cog": bool(dynamic_fit.get("fit_c_S", dynamic_fit_cog)),
        "cog_init_m_sensor": np.asarray(dynamic_fit.get("c_S_init", c_S_dynamic_init), dtype=np.float64).tolist(),
        "selected_smoothing_config": dynamic_smoothing_best_config,
        "selected_time_offset_ms": float(selected_time_offset_ms),
        "selected_time_offset_ns": int(selected_time_offset_ns),
        "use_gyro_term": bool(dynamic_use_gyro_term_selected),
        "torque_weight": float(dynamic_torque_weight),
        "rms_force_before_N": float(dynamic_fit["rmsF_init"]),
        "rms_force_after_N": float(dynamic_fit["rmsF"]),
        "rms_torque_before_Nm": float(dynamic_fit["rmsT_init"]),
        "rms_torque_after_Nm": float(dynamic_fit["rmsT"]),
        "optimizer_success": bool(dynamic_fit["success"]),
        "optimizer_status": int(dynamic_fit["status"]),
        "optimizer_message": str(dynamic_fit["message"]),
        "optimizer_nfev": int(dynamic_fit["nfev"]),
        "optimizer_cost": float(dynamic_fit["cost"]),
    }
    with dynamic_params_file.open("w", encoding="utf-8") as f:
        json.dump(dynamic_params_payload, f, indent=4)
    summary_lines.append(f"Dynamic params saved: {dynamic_params_file}")
    print("\n".join(summary_lines))

    return {
        "cam_calib": cam_calib,
        "imu_calib": imu_calib,
        "vrs_utils": vrs_utils,
        "temp_path_rosbag": temp_path_rosbag,
        "temp_path_vrs": temp_path_vrs,
        "force_torque_topic": force_torque_topic,
        "temperature_topic": temperature_topic,
        "adjusted_mps_file": adjusted_mps_file,
        "adjusted_mps_data": adjusted_mps_data,
        "poses_aria": poses_aria,
        "force_torque_df": force_torque_df,
        "temperature_df": temperature_df,
        "wrench_timestamps_ns": wrench_timestamps_ns,
        "T_ariadevice_ft": T_ariadevice_ft,
        "dynamic_signals": dynamic_signals,
        "dynamic_signal_debug_dir": dynamic_signal_debug_dir,
        "dynamic_signal_debug_plot_paths": dynamic_signal_debug_plot_paths,
        "dynamic_signal_stats": dynamic_signal_stats,
        "primary_smoothing_config": primary_smoothing_config,
        "dynamic_smoothing_best_config": dynamic_smoothing_best_config,
        "dynamic_smoothing_best_index": dynamic_smoothing_best_index,
        "dynamic_smoothing_candidate_results": [
            {
                "candidate_index": candidate["candidate_index"],
                "name": candidate["name"],
                "requested_smoothing_config": candidate["requested_smoothing_config"],
                "resolved_smoothing_config": candidate["resolved_smoothing_config"],
                "signal_stats": candidate["signal_stats"],
                "fit_metrics": candidate["fit_metrics"],
                "dynamic_fit_success": candidate["dynamic_fit_success"],
                "dynamic_fit_nfev": candidate["dynamic_fit_nfev"],
                "dynamic_fit_cost": candidate["dynamic_fit_cost"],
                "inertia_eigenvalues": candidate["inertia_eigenvalues"],
                "inertia_det": candidate["inertia_det"],
            }
            for candidate in dynamic_smoothing_candidate_results
        ],
        "smoothing_candidates": [
            {
                "candidate_index": candidate["candidate_index"],
                "name": candidate["name"],
                "smoothing_config": candidate["resolved_smoothing_config"],
                "signal_stats": candidate["signal_stats"],
                "fit_metrics": candidate["fit_metrics"],
                "optimizer_success": candidate["dynamic_fit_success"],
                "optimizer_nfev": candidate["dynamic_fit_nfev"],
                "optimizer_cost": candidate["dynamic_fit_cost"],
                "inertia_eigenvalues": candidate["inertia_eigenvalues"],
                "inertia_det": candidate["inertia_det"],
            }
            for candidate in dynamic_smoothing_candidate_results
        ],
        "smoothing_selected": (
            {
                "candidate_index": dynamic_smoothing_candidate_results[dynamic_smoothing_best_index]["candidate_index"],
                "name": dynamic_smoothing_candidate_results[dynamic_smoothing_best_index]["name"],
                "smoothing_config": dynamic_smoothing_candidate_results[dynamic_smoothing_best_index]["resolved_smoothing_config"],
                "signal_stats": dynamic_smoothing_candidate_results[dynamic_smoothing_best_index]["signal_stats"],
                "fit_metrics": dynamic_smoothing_candidate_results[dynamic_smoothing_best_index]["fit_metrics"],
                "optimizer_success": dynamic_smoothing_candidate_results[dynamic_smoothing_best_index]["dynamic_fit_success"],
                "optimizer_nfev": dynamic_smoothing_candidate_results[dynamic_smoothing_best_index]["dynamic_fit_nfev"],
                "optimizer_cost": dynamic_smoothing_candidate_results[dynamic_smoothing_best_index]["dynamic_fit_cost"],
                "inertia_eigenvalues": dynamic_smoothing_candidate_results[dynamic_smoothing_best_index]["inertia_eigenvalues"],
                "inertia_det": dynamic_smoothing_candidate_results[dynamic_smoothing_best_index]["inertia_det"],
            }
            if dynamic_smoothing_best_index is not None
            else None
        ),
        "smoothing_selected_name": (
            dynamic_smoothing_candidate_results[dynamic_smoothing_best_index]["name"]
            if dynamic_smoothing_best_index is not None
            else None
        ),
        "smoothing_candidate_count": (
            len(dynamic_smoothing_candidate_results) if run_smoothing_sweep else 1
        ),
        "dynamic_smoothing_debug_dir": dynamic_smoothing_debug_dir,
        "dynamic_smoothing_debug_plot_paths": dynamic_smoothing_debug_plot_paths,
        "time_offset_candidates": [
            {
                "candidate_index": result["candidate_index"],
                "offset_ms": result["offset_ms"],
                "offset_ns": result["offset_ns"],
                "valid_sample_count": result["valid_sample_count"],
                "invalid_sample_count": result["invalid_sample_count"],
                "fit_metrics": result["fit_metrics"],
                "optimizer_success": bool(result["dynamic_fit"]["success"]),
                "optimizer_status": int(result["dynamic_fit"]["status"]),
                "optimizer_nfev": int(result["dynamic_fit"]["nfev"]),
                "optimizer_cost": float(result["dynamic_fit"]["cost"]),
                "inertia_eigenvalues": result["inertia_eigenvalues"],
                "inertia_det": result["inertia_det"],
                "inertia_is_spd": bool(result["dynamic_fit"]["inertia_is_spd"]),
                "valid_for_selection": bool(result.get("valid_for_selection", False)),
            }
            for result in time_offset_candidate_results
        ],
        "time_offset_selected": (
            {
                "candidate_index": time_offset_candidate_results[time_offset_selected_index]["candidate_index"],
                "offset_ms": time_offset_candidate_results[time_offset_selected_index]["offset_ms"],
                "offset_ns": time_offset_candidate_results[time_offset_selected_index]["offset_ns"],
                "valid_sample_count": time_offset_candidate_results[time_offset_selected_index]["valid_sample_count"],
                "invalid_sample_count": time_offset_candidate_results[time_offset_selected_index]["invalid_sample_count"],
                "fit_metrics": time_offset_candidate_results[time_offset_selected_index]["fit_metrics"],
                "optimizer_success": bool(time_offset_candidate_results[time_offset_selected_index]["dynamic_fit"]["success"]),
                "optimizer_status": int(time_offset_candidate_results[time_offset_selected_index]["dynamic_fit"]["status"]),
                "optimizer_nfev": int(time_offset_candidate_results[time_offset_selected_index]["dynamic_fit"]["nfev"]),
                "optimizer_cost": float(time_offset_candidate_results[time_offset_selected_index]["dynamic_fit"]["cost"]),
                "inertia_eigenvalues": time_offset_candidate_results[time_offset_selected_index]["inertia_eigenvalues"],
                "inertia_det": time_offset_candidate_results[time_offset_selected_index]["inertia_det"],
                "inertia_is_spd": bool(time_offset_candidate_results[time_offset_selected_index]["dynamic_fit"]["inertia_is_spd"]),
                "valid_for_selection": bool(time_offset_candidate_results[time_offset_selected_index].get("valid_for_selection", False)),
                "selection_reason": time_offset_selected_reason,
            }
            if time_offset_selected_index is not None
            else None
        ),
        "time_offset_selected_ms": float(selected_time_offset_ms),
        "time_offset_selected_ns": int(selected_time_offset_ns),
        "time_offset_selection_reason": time_offset_selected_reason,
        "dynamic_time_offset_debug_dir": dynamic_time_offset_debug_dir,
        "dynamic_time_offset_debug_plot_paths": dynamic_time_offset_debug_plot_paths,
        "gyro_comparison": [
            {
                "candidate_index": result["candidate_index"],
                "use_gyro_term": result["use_gyro_term"],
                "fit_metrics": result["fit_metrics"],
                "optimizer_success": bool(result["dynamic_fit"]["success"]),
                "optimizer_status": int(result["dynamic_fit"]["status"]),
                "optimizer_nfev": int(result["dynamic_fit"]["nfev"]),
                "optimizer_cost": float(result["dynamic_fit"]["cost"]),
                "b_f": result["b_f"],
                "b_tau": result["b_tau"],
                "inertia_eigenvalues": result["inertia_eigenvalues"],
                "inertia_det": result["inertia_det"],
                "inertia_is_spd": bool(result["dynamic_fit"]["inertia_is_spd"]),
                "valid_for_selection": bool(result.get("valid_for_selection", False)),
            }
            for result in gyro_comparison_results
        ],
        "gyro_selected": (
            {
                "candidate_index": gyro_comparison_results[gyro_selected_index]["candidate_index"],
                "use_gyro_term": gyro_comparison_results[gyro_selected_index]["use_gyro_term"],
                "fit_metrics": gyro_comparison_results[gyro_selected_index]["fit_metrics"],
                "optimizer_success": bool(gyro_comparison_results[gyro_selected_index]["dynamic_fit"]["success"]),
                "optimizer_status": int(gyro_comparison_results[gyro_selected_index]["dynamic_fit"]["status"]),
                "optimizer_nfev": int(gyro_comparison_results[gyro_selected_index]["dynamic_fit"]["nfev"]),
                "optimizer_cost": float(gyro_comparison_results[gyro_selected_index]["dynamic_fit"]["cost"]),
                "b_f": gyro_comparison_results[gyro_selected_index]["b_f"],
                "b_tau": gyro_comparison_results[gyro_selected_index]["b_tau"],
                "inertia_eigenvalues": gyro_comparison_results[gyro_selected_index]["inertia_eigenvalues"],
                "inertia_det": gyro_comparison_results[gyro_selected_index]["inertia_det"],
                "inertia_is_spd": bool(gyro_comparison_results[gyro_selected_index]["dynamic_fit"]["inertia_is_spd"]),
                "valid_for_selection": bool(gyro_comparison_results[gyro_selected_index].get("valid_for_selection", False)),
                "selection_reason": gyro_selected_reason,
            }
            if gyro_selected_index is not None
            else None
        ),
        "gyro_selected_flag": bool(dynamic_use_gyro_term_selected),
        "gyro_selection_reason": gyro_selected_reason,
        "dynamic_gyro_debug_dir": dynamic_gyro_debug_dir,
        "dynamic_gyro_debug_plot_paths": dynamic_gyro_debug_plot_paths,
        "dynamic_fit": dynamic_fit,
        "dynamic_params_file": dynamic_params_file,
        "dynamic_params_payload": dynamic_params_payload,
        "dynamic_fit_debug_dir": dynamic_fit_debug_dir,
        "dynamic_fit_debug_plot_paths": dynamic_fit_debug_plot_paths,
        "dynamic_fit_residual_stats": dynamic_fit_residual_stats,
        "valid_dynamic_mask": valid_dynamic_mask,
        "valid_dynamic_sample_count": valid_dynamic_sample_count,
        "invalid_dynamic_sample_count": invalid_dynamic_sample_count,
        "pose_timestamp_range_ns": pose_timestamp_range_ns,
        "wrench_timestamp_range_ns": wrench_timestamp_range_ns,
        "wrench_timestamps_ns_dynamic_valid": wrench_timestamps_ns_dynamic_valid,
        "dynamic_signal_sample_timestamps_ns": selected_dynamic_signal_timestamps_ns,
        "a_D": a_D,
        "omega_D": omega_D,
        "alpha_D": alpha_D,
        "a_S": a_S,
        "omega_S": omega_S,
        "alpha_S": alpha_S,
        "R_S_W": R_S_W,
        "f_model": f_model,
        "tau_model": tau_model,
        "f_ext_hat": f_ext_hat,
        "tau_ext_hat": tau_ext_hat,
        "f_ext_hat_init": f_ext_hat_init,
        "tau_ext_hat_init": tau_ext_hat_init,
        "f_model_valid": f_model_valid,
        "tau_model_valid": tau_model_valid,
        "f_ext_hat_valid": f_ext_hat_valid,
        "tau_ext_hat_valid": tau_ext_hat_valid,
        "f_ext_hat_init_valid": f_ext_hat_init_valid,
        "tau_ext_hat_init_valid": tau_ext_hat_init_valid,
        "adjusted_mps_data_quasistatic": adjusted_mps_data_quasistatic,
        "poses_aria_quasistatic": poses_aria_quasistatic,
        "force_torque_df_quasistatic": force_torque_df_quasistatic,
        "force_torque_df_quasistatic_dyn": force_torque_df_quasistatic_dyn,
        "temperature_df_quasistatic": temperature_df_quasistatic,
        "quasistatic_params": params,
        "quasistatic_params_velocity_only": quasistatic_params_velocity_only,
        "quasistatic_wrench_mask": quasistatic_wrench_mask,
        "quasistatic_wrench_mask_dyn": quasistatic_wrench_mask_dyn,
        "quasistatic_wrench_mask_used": quasistatic_wrench_mask_used,
        "quasistatic_temperature_mask": quasistatic_temperature_mask,
        "quasistatic_pose_mask": pose_quasistatic_mask,
        "quasistatic_wrench_sample_count_velocity_only": quasistatic_sample_count,
        "quasistatic_wrench_sample_count_dyn": quasistatic_wrench_mask_dyn_count,
        "epsilon_quasistatic_linear": float(epsilon_quasistatic_linear),
        "epsilon_quasistatic_angular": float(epsilon_quasistatic_angular),
        "epsilon_quasistatic_linear_acc": float(epsilon_quasistatic_linear_acc),
        "epsilon_quasistatic_angular_acc": float(epsilon_quasistatic_angular_acc),
        "m_known": float(m_known),
        "c_S_known": c_S_known,
        "dynamic_fit_cog": bool(dynamic_fit_cog),
        "c_S_dynamic_init": c_S_dynamic_init.copy(),
        "c_S_selected": np.asarray(dynamic_fit["c_S"], dtype=np.float64).copy(),
        "dynamic_torque_weight": float(dynamic_torque_weight),
        "dynamic_use_gyro_term": bool(dynamic_use_gyro_term),
        "delta_ns": delta,
        "gripper_qr_pairs_cache_file": gripper_qr_pairs_cache_file,
        "vrs_qr_pairs_cache_file": vrs_qr_pairs_cache_file,
    }


def _load_saved_dynamic_compensation_config(
    config_json: Path | str,
) -> Dict[str, Any]:
    config_path = Path(config_json) if isinstance(config_json, str) else config_json
    if not config_path.exists():
        raise FileNotFoundError(f"Dynamic compensation config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw_config = json.load(f)

    if raw_config.get("frame") not in (None, "sensor"):
        raise ValueError(
            "Only sensor-frame dynamic compensation configs are supported. "
            f"Found frame={raw_config.get('frame')!r} in {config_path}."
        )

    def _get_first_present(
        candidate_keys: List[str],
        *,
        required: bool = True,
        default: Any = None,
    ) -> Any:
        for key in candidate_keys:
            if key in raw_config and raw_config[key] is not None:
                return raw_config[key]
        if required:
            raise KeyError(
                f"Could not find any of the required config keys {candidate_keys} in {config_path}."
            )
        return default

    default_smoothing_config = _resolve_requested_dynamic_smoothing_config(
        sg_window_length=21,
        sg_polyorder=3,
        sg_window_length_linear=None,
        sg_polyorder_linear=None,
        sg_window_length_angular=None,
        sg_polyorder_angular=None,
    )
    smoothing_config = _normalize_candidate_smoothing_config(
        _get_first_present(
            ["selected_smoothing_config", "smoothing_config"],
            required=False,
            default=default_smoothing_config,
        ),
        default_smoothing_config,
    )

    selected_time_offset_ms = float(
        _get_first_present(
            ["selected_time_offset_ms", "time_offset_ms"],
            required=False,
            default=0.0,
        )
    )
    selected_time_offset_ns = int(
        _get_first_present(
            ["selected_time_offset_ns", "time_offset_ns"],
            required=False,
            default=int(round(selected_time_offset_ms * 1e6)),
        )
    )
    if "selected_time_offset_ms" not in raw_config and "time_offset_ms" not in raw_config:
        selected_time_offset_ms = float(selected_time_offset_ns) * 1e-6

    force_bias_saved = _get_first_present(
        ["force_bias_N_sensor", "force_bias_N"],
        required=False,
        default=None,
    )
    torque_bias_saved = _get_first_present(
        ["torque_bias_Nm_sensor", "torque_bias_Nm"],
        required=False,
        default=None,
    )
    T_ariadevice_ft_saved = _get_first_present(
        ["T_ariadevice_ft", "T_device_sensor", "T_ariadevice_sensor"],
        required=False,
        default=None,
    )

    inertia_sensor_kgm2 = np.asarray(
        _get_first_present(["inertia_tensor_kgm2_sensor", "inertia_sensor_kgm2"]),
        dtype=np.float64,
    ).reshape(3, 3)
    if not np.all(np.isfinite(inertia_sensor_kgm2)):
        raise ValueError(f"Saved inertia tensor contains non-finite entries in {config_path}.")
    if not np.allclose(inertia_sensor_kgm2, inertia_sensor_kgm2.T, atol=1e-12, rtol=1e-9):
        raise ValueError(f"Saved inertia tensor must be symmetric in {config_path}.")
    inertia_sensor_kgm2 = 0.5 * (inertia_sensor_kgm2 + inertia_sensor_kgm2.T)
    inertia_eigenvalues = np.linalg.eigvalsh(inertia_sensor_kgm2)
    if not np.all(np.isfinite(inertia_eigenvalues)):
        raise ValueError(f"Saved inertia tensor contains non-finite eigenvalues in {config_path}.")
    if not np.all(inertia_eigenvalues > 0.0):
        raise ValueError(
            f"Saved inertia tensor must be symmetric positive definite in {config_path}. "
            "Negative off-diagonal entries are allowed, but all principal moments must stay positive."
        )

    loaded_config = {
        "config_path": config_path,
        "raw_config": raw_config,
        "mass_kg": float(_get_first_present(["mass_kg"])),
        "com_sensor_m": np.asarray(
            _get_first_present(["center_of_gravity_m_sensor", "com_sensor_m"]),
            dtype=np.float64,
        ).reshape(3,),
        "inertia_sensor_kgm2": inertia_sensor_kgm2,
        "force_bias_N_sensor": (
            None
            if force_bias_saved is None
            else np.asarray(force_bias_saved, dtype=np.float64).reshape(3,)
        ),
        "torque_bias_Nm_sensor": (
            None
            if torque_bias_saved is None
            else np.asarray(torque_bias_saved, dtype=np.float64).reshape(3,)
        ),
        "selected_smoothing_config": smoothing_config,
        "selected_time_offset_ms": selected_time_offset_ms,
        "selected_time_offset_ns": selected_time_offset_ns,
        "use_gyro_term": bool(
            _get_first_present(
                ["use_gyro_term", "dynamic_use_gyro_term"],
                required=False,
                default=False,
            )
        ),
        "gravity_mps2": float(
            _get_first_present(
                ["gravity_mps2", "gravity_m_s2", "gravity"],
                required=False,
                default=9.81,
            )
        ),
        "T_ariadevice_ft_saved": (
            None
            if T_ariadevice_ft_saved is None
            else np.asarray(T_ariadevice_ft_saved, dtype=np.float64).reshape(4, 4)
        ),
        "frame": raw_config.get("frame", "sensor"),
    }
    loaded_config["inertia_principal_moments_kgm2"] = inertia_eigenvalues.copy()

    return loaded_config


def _compute_wrench_stage_metrics(
    force_values: np.ndarray,
    torque_values: np.ndarray,
) -> Dict[str, Any]:
    force_values = np.asarray(force_values, dtype=np.float64)
    torque_values = np.asarray(torque_values, dtype=np.float64)
    if force_values.ndim != 2 or force_values.shape[1] != 3:
        raise ValueError("force_values must have shape (N, 3).")
    if torque_values.ndim != 2 or torque_values.shape[1] != 3:
        raise ValueError("torque_values must have shape (N, 3).")
    if force_values.shape[0] != torque_values.shape[0]:
        raise ValueError("force_values and torque_values must have the same number of samples.")
    if force_values.shape[0] == 0:
        raise ValueError("Need at least one sample to compute evaluation metrics.")
    if not np.all(np.isfinite(force_values)):
        raise ValueError("force_values contains non-finite values.")
    if not np.all(np.isfinite(torque_values)):
        raise ValueError("torque_values contains non-finite values.")

    force_norm = np.linalg.norm(force_values, axis=1)
    torque_norm = np.linalg.norm(torque_values, axis=1)
    return {
        "sample_count": int(force_values.shape[0]),
        "force_rms_norm": float(np.sqrt(np.mean(np.sum(force_values ** 2, axis=1)))),
        "torque_rms_norm": float(np.sqrt(np.mean(np.sum(torque_values ** 2, axis=1)))),
        "force_norm_mean": float(np.mean(force_norm)),
        "torque_norm_mean": float(np.mean(torque_norm)),
        "force_norm_std": float(np.std(force_norm)),
        "torque_norm_std": float(np.std(torque_norm)),
        "force_norm_max": float(np.max(force_norm)),
        "torque_norm_max": float(np.max(torque_norm)),
        "force_axis_mean": np.mean(force_values, axis=0).tolist(),
        "force_axis_std": np.std(force_values, axis=0).tolist(),
        "torque_axis_mean": np.mean(torque_values, axis=0).tolist(),
        "torque_axis_std": np.std(torque_values, axis=0).tolist(),
    }


def _estimate_recording_specific_bias_from_low_dynamics(
    *,
    bias_state: Dict[str, Any],
    dynamic_signals: Dict[str, np.ndarray],
    f_meas_S: np.ndarray,
    tau_meas_S: np.ndarray,
    m_known: float,
    c_S_known: np.ndarray,
    g: float,
) -> Dict[str, Any]:
    valid_dynamic_mask = np.asarray(bias_state["valid_dynamic_mask"], dtype=bool)
    quasistatic_wrench_mask_used = np.asarray(bias_state["quasistatic_wrench_mask_used"], dtype=bool)
    quasistatic_wrench_mask_velocity = np.asarray(bias_state["quasistatic_wrench_mask"], dtype=bool)

    bias_mask = quasistatic_wrench_mask_used & valid_dynamic_mask
    mask_source = "velocity_plus_acc"
    if int(bias_mask.sum()) == 0:
        bias_mask = quasistatic_wrench_mask_velocity & valid_dynamic_mask
        mask_source = "velocity_only_fallback"
    if int(bias_mask.sum()) == 0:
        raise ValueError(
            "No overlap-valid low-dynamics wrench samples remain for recording-specific "
            "bias estimation."
        )

    bias_params = _estimate_tool_params_ls(
        F_meas_S=np.asarray(f_meas_S, dtype=np.float64)[bias_mask],
        tau_meas_S=np.asarray(tau_meas_S, dtype=np.float64)[bias_mask],
        R_S_W_list=np.asarray(dynamic_signals["R_S_W"], dtype=np.float64)[bias_mask],
        m_known=m_known,
        c_S_known=np.asarray(c_S_known, dtype=np.float64).reshape(3,),
        g=g,
    )
    return {
        "bias_params": bias_params,
        "bias_mask": bias_mask,
        "bias_mask_source": mask_source,
        "bias_sample_count": int(bias_mask.sum()),
    }


def _visualize_saved_dynamic_compensation_evaluation(
    *,
    wrench_time_s: np.ndarray,
    F_raw: np.ndarray,
    tau_raw: np.ndarray,
    F_qs: np.ndarray,
    tau_qs: np.ndarray,
    F_dyn: np.ndarray,
    tau_dyn: np.ndarray,
    visualize_out_dir: Path,
) -> List[Path]:
    visualize_out_dir = Path(visualize_out_dir)
    plot_paths = [
        visualize_out_dir / "force_norms_over_time.png",
        visualize_out_dir / "torque_norms_over_time.png",
        visualize_out_dir / "dynamic_force_components.png",
        visualize_out_dir / "dynamic_torque_components.png",
        visualize_out_dir / "compensation_norm_histograms.png",
    ]

    wrench_time_s = np.asarray(wrench_time_s, dtype=np.float64)
    F_raw = np.asarray(F_raw, dtype=np.float64)
    tau_raw = np.asarray(tau_raw, dtype=np.float64)
    F_qs = np.asarray(F_qs, dtype=np.float64)
    tau_qs = np.asarray(tau_qs, dtype=np.float64)
    F_dyn = np.asarray(F_dyn, dtype=np.float64)
    tau_dyn = np.asarray(tau_dyn, dtype=np.float64)

    stage_specs = [
        ("Raw", F_raw, tau_raw, "tab:gray"),
        ("Quasi-static", F_qs, tau_qs, "tab:blue"),
        ("Dynamic", F_dyn, tau_dyn, "tab:orange"),
    ]

    fig, ax = plt.subplots(figsize=(13, 5))
    for label, force_values, _, color in stage_specs:
        ax.plot(
            wrench_time_s,
            np.linalg.norm(force_values, axis=1),
            label=label,
            color=color,
            lw=1.5,
        )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Force norm [N]")
    ax.set_title("Raw vs quasi-static vs dynamic force norm")
    ax.grid(True)
    ax.legend(loc="upper right")
    _finalize_debug_figure(
        fig,
        visualize=False,
        visualize_out_dir=visualize_out_dir,
        filename=plot_paths[0].name,
    )

    fig, ax = plt.subplots(figsize=(13, 5))
    for label, _, torque_values, color in stage_specs:
        ax.plot(
            wrench_time_s,
            np.linalg.norm(torque_values, axis=1),
            label=label,
            color=color,
            lw=1.5,
        )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Torque norm [Nm]")
    ax.set_title("Raw vs quasi-static vs dynamic torque norm")
    ax.grid(True)
    ax.legend(loc="upper right")
    _finalize_debug_figure(
        fig,
        visualize=False,
        visualize_out_dir=visualize_out_dir,
        filename=plot_paths[1].name,
    )

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    component_labels = ["x", "y", "z"]
    for component_idx, axis in enumerate(axes):
        axis.plot(wrench_time_s, F_dyn[:, component_idx], color="tab:orange", lw=1.4)
        axis.set_ylabel(f"F {component_labels[component_idx]} [N]")
        axis.grid(True)
    axes[0].set_title("Dynamic compensated force components")
    axes[-1].set_xlabel("Time [s]")
    _finalize_debug_figure(
        fig,
        visualize=False,
        visualize_out_dir=visualize_out_dir,
        filename=plot_paths[2].name,
    )

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    for component_idx, axis in enumerate(axes):
        axis.plot(wrench_time_s, tau_dyn[:, component_idx], color="tab:orange", lw=1.4)
        axis.set_ylabel(f"Tau {component_labels[component_idx]} [Nm]")
        axis.grid(True)
    axes[0].set_title("Dynamic compensated torque components")
    axes[-1].set_xlabel("Time [s]")
    _finalize_debug_figure(
        fig,
        visualize=False,
        visualize_out_dir=visualize_out_dir,
        filename=plot_paths[3].name,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for label, force_values, _, color in stage_specs:
        axes[0].hist(
            np.linalg.norm(force_values, axis=1),
            bins=50,
            alpha=0.5,
            color=color,
            label=label,
        )
    axes[0].set_xlabel("Force norm [N]")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Force norm histogram")
    axes[0].grid(True)
    axes[0].legend(loc="upper right")

    for label, _, torque_values, color in stage_specs:
        axes[1].hist(
            np.linalg.norm(torque_values, axis=1),
            bins=50,
            alpha=0.5,
            color=color,
            label=label,
        )
    axes[1].set_xlabel("Torque norm [Nm]")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Torque norm histogram")
    axes[1].grid(True)
    axes[1].legend(loc="upper right")
    _finalize_debug_figure(
        fig,
        visualize=False,
        visualize_out_dir=visualize_out_dir,
        filename=plot_paths[4].name,
    )

    return plot_paths


def evaluate_saved_dynamic_compensation_on_recording(
    config_json: Path | str,
    vrs_path: Path | str,
    mps_file: Path | str,
    rosbag_path: Path | str,
    temp_path: Path | str,
    camchain_imucam: Path | str,
    *,
    recompute_bias_for_recording: bool = True,
    stride: int = 1,
    min_qr_pairs: int = 2,
    deduplicate_by_qr_timestamp: bool = True,
    max_unique_qr_detections: Optional[int] = 50,
    mad_multiplier: float = 3.5,
) -> Dict[str, Any]:
    saved_config = _load_saved_dynamic_compensation_config(config_json)
    temp_path = Path(temp_path) if isinstance(temp_path, str) else temp_path
    debug_dir = temp_path / "saved_dynamic_comp_eval"
    dynamic_signal_debug_dir = debug_dir / "dynamic_signals"

    smoothing_config = saved_config["selected_smoothing_config"]
    prep_state = _load_dynamic_recording_data(
        vrs_path=vrs_path,
        mps_file=mps_file,
        rosbag_path=rosbag_path,
        temp_path=temp_path,
        camchain_imucam=camchain_imucam,
        stride=stride,
        min_qr_pairs=min_qr_pairs,
        deduplicate_by_qr_timestamp=deduplicate_by_qr_timestamp,
        max_unique_qr_detections=max_unique_qr_detections,
        mad_multiplier=mad_multiplier,
        visualize_aria_velocities=False,
        sg_window_length=int(smoothing_config["sg_window_length_linear"]),
        sg_polyorder=int(smoothing_config["sg_polyorder_linear"]),
        sg_window_length_linear=int(smoothing_config["sg_window_length_linear"]),
        sg_polyorder_linear=int(smoothing_config["sg_polyorder_linear"]),
        sg_window_length_angular=int(smoothing_config["sg_window_length_angular"]),
        sg_polyorder_angular=int(smoothing_config["sg_polyorder_angular"]),
        dynamic_signal_time_offset_ns=int(saved_config["selected_time_offset_ns"]),
        dynamic_signal_visualize=False,
        dynamic_signal_visualize_out_dir=dynamic_signal_debug_dir,
    )

    bias_state = _compute_dynamic_quasistatic_bias_state(
        adjusted_mps_data=prep_state["adjusted_mps_data"],
        poses_aria=prep_state["poses_aria"],
        force_torque_df=prep_state["force_torque_df"],
        temperature_df=prep_state["temperature_df"],
        temperature_ft=prep_state["temperature_ft"],
        f_meas_S=prep_state["f_meas_S"],
        tau_meas_S=prep_state["tau_meas_S"],
        dynamic_signals=prep_state["dynamic_signals"],
        dynamic_signal_sample_timestamps_ns=prep_state["dynamic_signal_sample_timestamps_ns"],
        m_known=float(saved_config["mass_kg"]),
        c_S_known=np.asarray(saved_config["com_sensor_m"], dtype=np.float64),
        dynamic_fit_cog=False,
        g=float(saved_config["gravity_mps2"]),
    )

    valid_dynamic_mask = np.asarray(bias_state["valid_dynamic_mask"], dtype=bool)
    valid_dynamic_sample_count = int(bias_state["valid_dynamic_sample_count"])
    invalid_dynamic_sample_count = int(bias_state["invalid_dynamic_sample_count"])
    if valid_dynamic_sample_count == 0:
        raise ValueError(
            "No valid overlap between wrench timestamps and Aria pose timestamps on the "
            "evaluation recording."
        )

    if recompute_bias_for_recording:
        # Bias often drifts across recordings due to temperature, electronics, and
        # sensor zero-offset changes. The physical parameters can transfer, while
        # the per-recording bias is usually safer to refresh from low-dynamics
        # no-contact windows on the new recording itself.
        recording_bias_state = _estimate_recording_specific_bias_from_low_dynamics(
            bias_state=bias_state,
            dynamic_signals=prep_state["dynamic_signals"],
            f_meas_S=prep_state["f_meas_S"],
            tau_meas_S=prep_state["tau_meas_S"],
            m_known=float(saved_config["mass_kg"]),
            c_S_known=np.asarray(saved_config["com_sensor_m"], dtype=np.float64),
            g=float(saved_config["gravity_mps2"]),
        )
        bias_params = recording_bias_state["bias_params"]
        b_f = np.asarray(bias_params["f0"], dtype=np.float64)
        b_tau = np.asarray(bias_params["tau0"], dtype=np.float64)
        bias_source = "recomputed"
        bias_sample_count = int(recording_bias_state["bias_sample_count"])
        bias_mask_source = recording_bias_state["bias_mask_source"]
        bias_mask = recording_bias_state["bias_mask"]
    else:
        if saved_config["force_bias_N_sensor"] is None or saved_config["torque_bias_Nm_sensor"] is None:
            raise ValueError(
                "The saved dynamic config does not contain force/torque bias values, "
                "so recompute_bias_for_recording=False cannot be used."
            )
        b_f = np.asarray(saved_config["force_bias_N_sensor"], dtype=np.float64)
        b_tau = np.asarray(saved_config["torque_bias_Nm_sensor"], dtype=np.float64)
        bias_source = "saved"
        bias_sample_count = 0
        bias_mask_source = "saved"
        bias_mask = np.zeros_like(valid_dynamic_mask, dtype=bool)

    f_meas_S = np.asarray(prep_state["f_meas_S"], dtype=np.float64)
    tau_meas_S = np.asarray(prep_state["tau_meas_S"], dtype=np.float64)
    dynamic_signals = prep_state["dynamic_signals"]
    R_S_W = np.asarray(dynamic_signals["R_S_W"], dtype=np.float64)
    a_S = np.asarray(dynamic_signals["a_S"], dtype=np.float64)
    omega_S = np.asarray(dynamic_signals["omega_S"], dtype=np.float64)
    alpha_S = np.asarray(dynamic_signals["alpha_S"], dtype=np.float64)
    wrench_timestamps_ns = np.asarray(prep_state["wrench_timestamps_ns"], dtype=np.int64)

    F_raw = f_meas_S[valid_dynamic_mask]
    tau_raw = tau_meas_S[valid_dynamic_mask]
    R_S_W_valid = R_S_W[valid_dynamic_mask]
    a_S_valid = a_S[valid_dynamic_mask]
    omega_S_valid = omega_S[valid_dynamic_mask]
    alpha_S_valid = alpha_S[valid_dynamic_mask]

    zero_kinematics = np.zeros_like(a_S_valid)
    quasi_static_model = _predict_dynamic_internal_wrench(
        R_S_W_list=R_S_W_valid,
        a_S=zero_kinematics,
        omega_S=zero_kinematics,
        alpha_S=zero_kinematics,
        m_known=float(saved_config["mass_kg"]),
        c_S_known=np.asarray(saved_config["com_sensor_m"], dtype=np.float64),
        b_f=b_f,
        b_tau=b_tau,
        I_C=np.zeros((3, 3), dtype=np.float64),
        g=float(saved_config["gravity_mps2"]),
        use_gyro_term=False,
    )
    dynamic_model = _predict_dynamic_internal_wrench(
        R_S_W_list=R_S_W_valid,
        a_S=a_S_valid,
        omega_S=omega_S_valid,
        alpha_S=alpha_S_valid,
        m_known=float(saved_config["mass_kg"]),
        c_S_known=np.asarray(saved_config["com_sensor_m"], dtype=np.float64),
        b_f=b_f,
        b_tau=b_tau,
        I_C=np.asarray(saved_config["inertia_sensor_kgm2"], dtype=np.float64),
        g=float(saved_config["gravity_mps2"]),
        use_gyro_term=bool(saved_config["use_gyro_term"]),
    )

    F_qs = F_raw - np.asarray(quasi_static_model["F_model"], dtype=np.float64)
    tau_qs = tau_raw - np.asarray(quasi_static_model["tau_model"], dtype=np.float64)
    F_dyn = F_raw - np.asarray(dynamic_model["F_model"], dtype=np.float64)
    tau_dyn = tau_raw - np.asarray(dynamic_model["tau_model"], dtype=np.float64)

    metrics_raw = _compute_wrench_stage_metrics(F_raw, tau_raw)
    metrics_quasi_static = _compute_wrench_stage_metrics(F_qs, tau_qs)
    metrics_dynamic = _compute_wrench_stage_metrics(F_dyn, tau_dyn)

    wrench_timestamps_ns_valid = wrench_timestamps_ns[valid_dynamic_mask]
    wrench_time_s_valid = (
        (wrench_timestamps_ns_valid - wrench_timestamps_ns_valid[0]).astype(np.float64) * 1e-9
    )
    debug_plot_paths = _visualize_saved_dynamic_compensation_evaluation(
        wrench_time_s=wrench_time_s_valid,
        F_raw=F_raw,
        tau_raw=tau_raw,
        F_qs=F_qs,
        tau_qs=tau_qs,
        F_dyn=F_dyn,
        tau_dyn=tau_dyn,
        visualize_out_dir=debug_dir,
    )
    dynamic_signal_debug_plot_paths = [
        dynamic_signal_debug_dir / "aria_dynamic_signal_magnitudes.png",
        dynamic_signal_debug_dir / "aria_world_velocity_raw_vs_smoothed.png",
        dynamic_signal_debug_dir / "aria_dynamic_signal_components_device_frame.png",
        dynamic_signal_debug_dir / "aria_dynamic_signal_sensor_frame_magnitudes.png",
    ]

    print("=== Saved Dynamic Compensation Evaluation ===")
    print(f"bias source: {bias_source}")
    print(
        "raw:         "
        f"force_rms={metrics_raw['force_rms_norm']:.4f} N, "
        f"torque_rms={metrics_raw['torque_rms_norm']:.4f} Nm"
    )
    print(
        "quasi-static "
        f"force_rms={metrics_quasi_static['force_rms_norm']:.4f} N, "
        f"torque_rms={metrics_quasi_static['torque_rms_norm']:.4f} Nm"
    )
    print(
        "dynamic:     "
        f"force_rms={metrics_dynamic['force_rms_norm']:.4f} N, "
        f"torque_rms={metrics_dynamic['torque_rms_norm']:.4f} Nm"
    )
    print(
        "config:      "
        f"smoothing={_format_dynamic_smoothing_config_label(saved_config['selected_smoothing_config'])}, "
        f"time_offset_ms={saved_config['selected_time_offset_ms']:.3f}, "
        f"gyro_term={saved_config['use_gyro_term']}, "
        f"valid_samples={valid_dynamic_sample_count}, invalid_excluded={invalid_dynamic_sample_count}"
    )

    return {
        "config_path": saved_config["config_path"],
        "saved_config": {
            "mass_kg": float(saved_config["mass_kg"]),
            "com_sensor_m": np.asarray(saved_config["com_sensor_m"], dtype=np.float64).tolist(),
            "inertia_sensor_kgm2": np.asarray(saved_config["inertia_sensor_kgm2"], dtype=np.float64).tolist(),
            "selected_smoothing_config": saved_config["selected_smoothing_config"],
            "selected_time_offset_ms": float(saved_config["selected_time_offset_ms"]),
            "selected_time_offset_ns": int(saved_config["selected_time_offset_ns"]),
            "use_gyro_term": bool(saved_config["use_gyro_term"]),
            "gravity_mps2": float(saved_config["gravity_mps2"]),
            "T_ariadevice_ft_saved": (
                None
                if saved_config["T_ariadevice_ft_saved"] is None
                else np.asarray(saved_config["T_ariadevice_ft_saved"], dtype=np.float64).tolist()
            ),
        },
        "recompute_bias_for_recording": bool(recompute_bias_for_recording),
        "bias_used": {
            "force_bias_N": np.asarray(b_f, dtype=np.float64).tolist(),
            "torque_bias_Nm": np.asarray(b_tau, dtype=np.float64).tolist(),
            "source": bias_source,
            "low_dynamics_sample_count": int(bias_sample_count),
            "low_dynamics_mask_source": bias_mask_source,
        },
        "metrics_raw": metrics_raw,
        "metrics_quasi_static": metrics_quasi_static,
        "metrics_dynamic": metrics_dynamic,
        "valid_dynamic_mask": valid_dynamic_mask,
        "valid_dynamic_sample_count": valid_dynamic_sample_count,
        "invalid_dynamic_sample_count": invalid_dynamic_sample_count,
        "pose_timestamp_range_ns": bias_state["pose_timestamp_range_ns"],
        "wrench_timestamp_range_ns": [
            int(np.min(wrench_timestamps_ns)),
            int(np.max(wrench_timestamps_ns)),
        ],
        "debug_dir": debug_dir,
        "debug_plot_paths": debug_plot_paths,
        "dynamic_signal_debug_dir": dynamic_signal_debug_dir,
        "dynamic_signal_debug_plot_paths": dynamic_signal_debug_plot_paths,
        "bias_recompute_mask": bias_mask,
        "T_ariadevice_ft": np.asarray(prep_state["T_ariadevice_ft"], dtype=np.float64),
        "quasi_static_model_valid": {
            "F_model": np.asarray(quasi_static_model["F_model"], dtype=np.float64),
            "tau_model": np.asarray(quasi_static_model["tau_model"], dtype=np.float64),
        },
        "dynamic_model_valid": {
            "F_model": np.asarray(dynamic_model["F_model"], dtype=np.float64),
            "tau_model": np.asarray(dynamic_model["tau_model"], dtype=np.float64),
        },
    }


def visualize_aria_velocity_magnitudes(
    dynamic_result_or_mps: Union[Dict[str, Any], pd.DataFrame],
    *,
    title_prefix: str = "Aria",
) -> Dict[str, np.ndarray]:
    """
    Visualize the magnitudes of Aria linear and angular velocity from the
    adjusted MPS trajectory.

    Parameters
    ----------
    dynamic_result_or_mps:
        Either the dictionary returned by ``estimate_tool_params_dynamic`` or a
        DataFrame containing the adjusted MPS trajectory columns.

    Returns
    -------
    Dict[str, np.ndarray]
        Arrays for downstream inspection:
        ``time_s``, ``linear_speed_mps``, ``angular_speed_radps``.
    """
    if isinstance(dynamic_result_or_mps, dict):
        if "adjusted_mps_data" not in dynamic_result_or_mps:
            raise KeyError(
                "Expected 'adjusted_mps_data' in the dynamic result dictionary."
            )
        adjusted_mps_data = dynamic_result_or_mps["adjusted_mps_data"]
    else:
        adjusted_mps_data = dynamic_result_or_mps

    required_columns = [
        "timestamp",
        "device_linear_velocity_x_device",
        "device_linear_velocity_y_device",
        "device_linear_velocity_z_device",
        "angular_velocity_x_device",
        "angular_velocity_y_device",
        "angular_velocity_z_device",
    ]
    missing_columns = [
        column for column in required_columns if column not in adjusted_mps_data.columns
    ]
    if missing_columns:
        raise KeyError(
            f"Adjusted MPS data is missing required velocity columns: {missing_columns}"
        )

    velocity_df = adjusted_mps_data[required_columns].copy()
    velocity_df = velocity_df.sort_values("timestamp").drop_duplicates("timestamp")

    timestamps_ns = velocity_df["timestamp"].to_numpy(dtype=np.int64)
    time_s = (timestamps_ns - timestamps_ns[0]).astype(np.float64) * 1e-9

    linear_velocity_device = velocity_df[
        [
            "device_linear_velocity_x_device",
            "device_linear_velocity_y_device",
            "device_linear_velocity_z_device",
        ]
    ].to_numpy(dtype=np.float64)
    angular_velocity_device = velocity_df[
        [
            "angular_velocity_x_device",
            "angular_velocity_y_device",
            "angular_velocity_z_device",
        ]
    ].to_numpy(dtype=np.float64)

    linear_speed_mps = np.linalg.norm(linear_velocity_device, axis=1)
    angular_speed_radps = np.linalg.norm(angular_velocity_device, axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(time_s, linear_speed_mps, lw=1.8, color="tab:blue")
    axes[0].set_ylabel("Linear speed [m/s]")
    axes[0].set_title(f"{title_prefix} linear velocity magnitude")
    axes[0].grid(True)

    axes[1].plot(time_s, angular_speed_radps, lw=1.8, color="tab:orange")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Angular speed [rad/s]")
    axes[1].set_title(f"{title_prefix} angular velocity magnitude")
    axes[1].grid(True)

    fig.tight_layout()
    plt.show()

    return {
        "time_s": time_s,
        "linear_speed_mps": linear_speed_mps,
        "angular_speed_radps": angular_speed_radps,
    }

def estimate_tool_params(vrs_path: Path | str,
                        mps_file: Path | str, 
                        rosbag_path: Path | str,
                        temp_path: Path | str,
                        camchain_imucam: Path,
                        forceless_time_intervall: List[int]) -> None:
    """
    Estimate tool mass parameters to later compensate the tool from the force/torque readings.
    Background: Aria vrs and gripper back are jointly recorded, with the aria mounted on the gripper.
    First tiemstamps for the aria need to be adjusted to match the gripper timestamps by detecting
    the timestamped qr code. mps is extracted and timestamps are again adjusted by the offset.
    1. Extract frames abnd timestamps from vrs into dummy directory
    2. Extract frames and timestamps from rosbag
    3. Detect the timestamped qr code in the aria frames and gripper frames, compoute offset
    4. Adjust the aria timestamps (and mps) by the offset. This brings forcec/torque readings and 
       slam poses into the same time frame.
    5. load transform between force/torque and aria slam poses from prior calibration
    6. cut off part of the recording where the gripper touches the floor (external force, do it manually 
       by looing for the first frame when the gripper is lifted and the last frame when the gripper
       is lowered)
    6. compute the tool parameters from the force/torque readings and slam poses.
    """
    

    if isinstance(vrs_path, str):
        vrs_path = Path(vrs_path)
    if isinstance(rosbag_path, str):
        rosbag_path = Path(rosbag_path)
    if isinstance(temp_path, str):
        temp_path = Path(temp_path)
    if isinstance(mps_file, str):
        mps_file = Path(mps_file)
    if isinstance(camcahin_imucam, str):
        camchain_imucam = Path(camcahin_imucam)

    if not vrs_path.exists():
        raise FileNotFoundError(f"VRS file not found: {vrs_path}")
    
    if not rosbag_path.exists():
        raise FileNotFoundError(f"ROS bag file not found: {rosbag_path}")
    
    if not temp_path.exists():
        temp_path.mkdir(parents=True, exist_ok=True)

    if not mps_file.exists():
        raise FileNotFoundError(f"MPS file not found: {mps_file}")
    
    if not camchain_imucam.exists():
        raise FileNotFoundError(f"Camera-IMU calibration file not found: {camchain_imucam}")

    # Load camera-IMU calibration
    cam_calib = load_camchain(camchain_imucam, cam_name="cam2")
    imu_calib = load_imucam(camchain_imucam, imu_name="cam2")

    temp_path_rosbag = temp_path / "rosbag"
    temp_path_vrs = temp_path / "vrs"
    temp_path_rosbag.mkdir(parents=True, exist_ok=True)
    temp_path_vrs.mkdir(parents=True, exist_ok=True)

    # Load the MPS file
    if not mps_file.exists():
        raise FileNotFoundError(f"MPS file not found: {mps_file}")
    mps_data = pd.read_csv(mps_file)
    
    # Extract frames and timestamps from VRS into a temporary directory
    vrs_utils = VRSUtils(vrs_path, undistort=False)
    if not any(temp_path_vrs.glob("*")):
        _, _ = vrs_utils.get_frames_from_vrs(out_dir=temp_path_vrs)

    # Extract frames and timestamps from ROS bag
    force_torque_topic = "/force_torque/ft_sensor0/ft_sensor_readings/wrench"
    temperature_topic = "/force_torque/ft_sensor0/ft_sensor_readings/temperature"
    if not any(temp_path_rosbag.glob("*")):
        get_topics_from_bag(
            image_topics=["/zedm/zed_node/left_raw/image_raw_color"],
            non_image_topics={force_torque_topic: "geometry_msgs/WrenchStamped",
                              temperature_topic: "sensor_msgs/Temperature"},
            bag_path=rosbag_path,
            out_dir=temp_path_rosbag
        )

    adjusted_mps_file = temp_path / "adjusted_mps.csv"
    if not adjusted_mps_file.exists():
        # Detect the timestamped QR code in the VRS frames
        qr = QRCodeDetectorDecoder(frame_dir=temp_path_vrs, ext=".png")
        time_pair_aria = qr.find_first_valid_qr()

        # Detect the timestamped QR code in the ROS bag frames
        frame_dir = temp_path_rosbag / "zedm/zed_node/left_raw/image_raw_color"
        qr = QRCodeDetectorDecoder(frame_dir=frame_dir, ext=".png")
        time_pair_gripper = qr.find_first_valid_qr()

        # get the offset between the two timestamps
        if time_pair_aria is None or time_pair_gripper is None:
            raise ValueError("Could not find valid QR codes in either VRS or ROS bag frames.")
        
        # flip time pairs, so we get aria delta to gripper 
        # (unlike in data extraction, where we had gripper delta to aria)
        timealigner = TimeAligner(
            aria_pair=time_pair_gripper,
            sensor_pair=time_pair_aria,
        )
        delta = timealigner.get_delta()
        print(f"Time delta between VRS and ROS bag: {delta} ns")

        # Adjust the timestamps of the VRS frames by the delta
        for frame in temp_path_vrs.glob("*.png"):
            ts = int(frame.stem)
            adjusted_ts = ts + delta
            new_frame_name = temp_path_vrs / f"{adjusted_ts}.png"
            frame.rename(new_frame_name)

        # Adjust the MPS timestamps by the delta
        adjusted_mps_data = mps_data.copy()
        adjusted_mps_data["tracking_timestamp_us"] = (adjusted_mps_data["tracking_timestamp_us"] * 1_000) + delta
        #rename the timestamp column to match the adjusted format
        adjusted_mps_data.rename(columns={"tracking_timestamp_us": "timestamp"}, inplace=True)
        adjusted_mps_data.to_csv(adjusted_mps_file, index=False)
    else:
        adjusted_mps_data = pd.read_csv(adjusted_mps_file)
    
    # read force/torque readings from the rosbag    
    force_torque_df = pd.read_csv(temp_path_rosbag / force_torque_topic.strip("/") / "data.csv")
    temperature_df = pd.read_csv(temp_path_rosbag / temperature_topic.strip("/") / "data.csv")
    # cut off part of the recording where the gripper touches the floor
    t_start = forceless_time_intervall[0]
    t_end = forceless_time_intervall[1]

    force_torque_df_cut = force_torque_df[
        (force_torque_df["timestamp"] >= t_start) & 
        (force_torque_df["timestamp"] <= t_end)
    ]
    adjusted_mps_data_cut = adjusted_mps_data[
        (adjusted_mps_data["timestamp"] >= t_start) & 
        (adjusted_mps_data["timestamp"] <= t_end)
    ]

    temperature_df_cut = temperature_df[
        (temperature_df["timestamp"] >= t_start) &
        (temperature_df["timestamp"] <= t_end)
    ]

    # load transform from cam aria to force/torque sensor
    T_ariacam_imuft = imu_calib.T_cam_imu

    # trafo from imu of forcetorque to forcetorque frame (offset)
    # exampple: (0,0,0) in ft frame is (0,0,0.0257) in imu frame
    T_imuft_ft = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0.0257],
        [0, 0, 0, 1]
    ], dtype=np.float64)

    ariacam_calibration = vrs_utils.device_calib.get_camera_calib("camera-rgb")
    T_ariadevice_ariacam = ariacam_calibration.get_transform_device_camera().to_matrix()

    # wrench in ft frame (sensor)
    wrench_ft = force_torque_df_cut[["timestamp", "wrench.force.x", "wrench.force.y", "wrench.force.z",
                                      "wrench.torque.x", "wrench.torque.y", "wrench.torque.z"]]
    wrench_timestamps_ns = wrench_ft["timestamp"].to_numpy(dtype=np.int64)
    temperature_ft = temperature_df_cut[["timestamp", "temperature"]]

    # aria slam poses in aria world frame
    poses_aria = adjusted_mps_data_cut[["timestamp", "tx_world_device", "ty_world_device", "tz_world_device",
                                        "qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device"]]
    
    T_ariadevice_ft = T_ariadevice_ariacam @ T_ariacam_imuft @ T_imuft_ft

    # slerp interpolate poeses_aria to get the poses at the timestamps of the wrench_ft
    R_ariaworld_ariadevice_list, t_ariaworld_ariadevice_list = _slerp_pose_series_to_targets(poses_aria, wrench_timestamps_ns)

    # convert to numpy arrays
    R_ariaworld_ariadevice = np.array(R_ariaworld_ariadevice_list, dtype=np.float64)  # (N, 3, 3)
    R_ariadevice_ft = T_ariadevice_ft[:3, :3]  # (3, 3)

    # compute the rotation from aria world to force/torque sensor frame
    R_ariaworld_ft = R_ariaworld_ariadevice @ R_ariadevice_ft  # (N, 3, 3)
    R_ft_ariaworld = np.transpose(R_ariaworld_ft, axes=(0, 2, 1))  # (N, 3, 3)


    f_meas_S = wrench_ft[["wrench.force.x", "wrench.force.y", "wrench.force.z"]].to_numpy(dtype=np.float64)  # (N, 3)
    tau_meas_S = wrench_ft[["wrench.torque.x", "wrench.torque.y", "wrench.torque.z"]].to_numpy(dtype=np.float64)
    T_meas_S = temperature_ft["temperature"].to_numpy(dtype=np.float64)  # (N,)

    params = _estimate_tool_params_ls(
        F_meas_S=f_meas_S,
        tau_meas_S=tau_meas_S,
        R_S_W_list=R_ft_ariaworld,
        m_known=0.490,
        g=9.81
    )

    # save the parameters to a file
    params_file = temp_path / "tool_params_static_estimate.json"
    with open(params_file, 'w') as f:
        import json
        json.dump(params, f, indent=4)

    print(f"Tool parameters estimated and saved to {params_file}")
    print(f"Mass: {params['m']} kg")
    print(f"Center of mass in sensor frame: {params['c_S']}")
    print(f"Force offset: {params['f0']}")
    print(f"Torque offset: {params['tau0']}")
    print(f"RMS Force error: {params['rmsF']}")
    print(f"RMS Torque error: {params['rmsT']}")


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0, -z,  y],
                     [z,  0, -x],
                     [-y, x,  0]], dtype=float)


def _estimate_tool_params_ls(
    F_meas_S,
    tau_meas_S,
    R_S_W_list,
    g: float = 9.81,
    m_known: float | None = None,
    c_S_known: np.ndarray | None = None,
):
    """
    Linear LS on quasi-static model with optional known mass and/or CoG.

    Model:
      F   = f0 + m * g_S
      tau = tau0 - [g_S]_x * (m * c_S)      (with p := m c_S)

    Unknowns depend on what is known:
      - If m and c_S unknown:           theta = [f0(3), m(1), p(3), tau0(3)]          -> 10 params (original)
      - If m known, c_S unknown:        theta = [f0(3), c_S(3), tau0(3)]              ->  9 params
      - If m unknown, c_S known:        theta = [f0(3), m(1), tau0(3)]                ->  7 params
      - If both m and c_S known:        theta = [f0(3), tau0(3)]                      ->  6 params

    Inputs:
      F_meas_S      : (N,3) forces in sensor frame S
      tau_meas_S    : (N,3) torques in sensor frame S
      R_S_W_list    : (N,3,3) rotations world->sensor per sample
      g             : gravity magnitude
      m_known       : float or None
      c_S_known     : (3,) or None  (CoG in sensor frame S, meters)

    Returns dict with:
      m, c_S, f0, tau0, rmsF, rmsT, theta (all lists except scalars), plus bookkeeping.
    """
    N = F_meas_S.shape[0]
    assert tau_meas_S.shape[0] == N and R_S_W_list.shape[0] == N
    gW = np.array([0.0, 0.0, -g], dtype=float)

    # Normalize c_S_known shape if provided
    if c_S_known is not None:
        c_S_known = np.asarray(c_S_known, dtype=float).reshape(3,)

    # Helper to build skew-symmetric matrix
    def _skew(v: np.ndarray) -> np.ndarray:
        x, y, z = v
        return np.array([[0.0, -z,   y],
                         [z,    0.0, -x],
                         [-y,   x,   0.0]], dtype=float)

    rows = []
    ys   = []

    # We’ll assemble a design matrix with named blocks so we can parse theta cleanly.
    col_index = 0
    cols = {}  # name -> (start, length)

    def add_cols(name, ncols):
        nonlocal col_index
        cols[name] = (col_index, ncols)
        col_index += ncols

    # Decide which blocks are present based on known/unknown settings
    # f0 and tau0 are ALWAYS estimated
    add_cols("f0",   3)

    if m_known is None and c_S_known is None:
        # Original: unknown m and p = m c_S
        add_cols("m",    1)
        add_cols("p",    3)   # first moment
        add_cols("tau0", 3)
        param_count = col_index
        for i in range(N):
            g_S = R_S_W_list[i] @ gW

            AF = np.zeros((3, param_count))
            i_f0, _ = cols["f0"]
            i_m,  _ = cols["m"]
            AF[:, i_f0:i_f0+3] = np.eye(3)            # f0
            AF[:, i_m:i_m+1]   = g_S.reshape(3,1)     # m
            rows.append(AF); ys.append(F_meas_S[i])

            AT = np.zeros((3, param_count))
            i_p, _    = cols["p"]
            i_tau0, _ = cols["tau0"]
            AT[:, i_p:i_p+3]      = -_skew(g_S)       # p
            AT[:, i_tau0:i_tau0+3]= np.eye(3)         # tau0
            rows.append(AT); ys.append(tau_meas_S[i])

    elif (m_known is not None) and (c_S_known is None):
        # Known mass, unknown c_S
        add_cols("c_S",  3)
        add_cols("tau0", 3)
        param_count = col_index
        m = float(m_known)

        for i in range(N):
            g_S = R_S_W_list[i] @ gW

            # Forces: F = f0 + m * g_S  -> move known gravity to RHS
            AF = np.zeros((3, param_count))
            i_f0, _ = cols["f0"]
            AF[:, i_f0:i_f0+3] = np.eye(3)            # f0
            rows.append(AF); ys.append(F_meas_S[i] - m * g_S)

            # Torques: tau = tau0 - m [g_S]_x c_S
            AT = np.zeros((3, param_count))
            i_cS, _   = cols["c_S"]
            i_tau0, _ = cols["tau0"]
            AT[:, i_cS:i_cS+3]    = -m * _skew(g_S)   # c_S
            AT[:, i_tau0:i_tau0+3]= np.eye(3)         # tau0
            rows.append(AT); ys.append(tau_meas_S[i])

    elif (m_known is None) and (c_S_known is not None):
        # Unknown mass, known c_S
        add_cols("m",    1)
        add_cols("tau0", 3)
        param_count = col_index
        cS = c_S_known

        for i in range(N):
            g_S = R_S_W_list[i] @ gW

            # Forces: F = f0 + m * g_S
            AF = np.zeros((3, param_count))
            i_f0, _ = cols["f0"]
            i_m,  _ = cols["m"]
            AF[:, i_f0:i_f0+3] = np.eye(3)            # f0
            AF[:, i_m:i_m+1]   = g_S.reshape(3,1)     # m
            rows.append(AF); ys.append(F_meas_S[i])

            # Torques: tau = tau0 - m [g_S]_x c_S  -> torque is linear in m with vector -(g_S × c_S)
            AT = np.zeros((3, param_count))
            i_m,  _    = cols["m"]
            i_tau0, _  = cols["tau0"]
            vec = -(_skew(g_S) @ cS).reshape(3,1)     # 3x1 column multiplying m
            AT[:, i_m:i_m+1]     = vec               # m
            AT[:, i_tau0:i_tau0+3]= np.eye(3)         # tau0
            rows.append(AT); ys.append(tau_meas_S[i])

    else:
        # Both m and c_S known: estimate only f0 and tau0
        add_cols("tau0", 3)
        param_count = col_index
        m = float(m_known)
        cS = c_S_known

        for i in range(N):
            g_S = R_S_W_list[i] @ gW
            # Forces: F - m g_S = f0
            AF = np.zeros((3, param_count))
            i_f0, _ = cols["f0"]
            AF[:, i_f0:i_f0+3] = np.eye(3)
            rows.append(AF); ys.append(F_meas_S[i] - m * g_S)

            # Torques: tau - ( - [g_S]_x (m c_S) ) = tau0
            AT = np.zeros((3, param_count))
            i_tau0, _ = cols["tau0"]
            AT[:, i_tau0:i_tau0+3] = np.eye(3)
            tau_g = -_skew(g_S) @ (m * cS)
            rows.append(AT); ys.append(tau_meas_S[i] - tau_g)

    # Stack and solve
    A = np.vstack(rows)          # (6N, P)
    y = np.hstack(ys)            # (6N,)
    lam = 1e-6
    ATA = A.T @ A + lam * np.eye(A.shape[1])
    ATy = A.T @ y
    theta = np.linalg.solve(ATA, ATy)

    # Parse solution into outputs
    out = {}
    # f0
    i, L = cols["f0"];  f0 = theta[i:i+L]
    out["f0"] = f0.tolist()

    if m_known is None and c_S_known is None:
        i_m, _ = cols["m"]; m_est = float(theta[i_m])
        i_p, _ = cols["p"]; p = theta[i_p:i_p+3]
        cS_est = (p / m_est) if abs(m_est) > 1e-9 else np.zeros(3)
        i_t0,_ = cols["tau0"]; tau0 = theta[i_t0:i_t0+3]
        out.update(m=m_est, c_S=cS_est.tolist(), tau0=tau0.tolist())

    elif (m_known is not None) and (c_S_known is None):
        m_est = float(m_known)
        i_cS,_ = cols["c_S"]; cS_est = theta[i_cS:i_cS+3]
        i_t0,_ = cols["tau0"]; tau0 = theta[i_t0:i_t0+3]
        out.update(m=m_est, c_S=cS_est.tolist(), tau0=tau0.tolist())

    elif (m_known is None) and (c_S_known is not None):
        cS_est = c_S_known
        i_m,_  = cols["m"]; m_est = float(theta[i_m])
        i_t0,_ = cols["tau0"]; tau0 = theta[i_t0:i_t0+3]
        out.update(m=m_est, c_S=cS_est.tolist(), tau0=tau0.tolist())

    else:
        # both known
        m_est = float(m_known)
        cS_est = c_S_known
        i_t0,_ = cols["tau0"]; tau0 = theta[i_t0:i_t0+3]
        out.update(m=m_est, c_S=cS_est.tolist(), tau0=tau0.tolist())

    # Diagnostics: compute residuals
    # Reconstruct predictions using the final (m, c_S, f0, tau0)
    m_use  = out["m"]
    cS_use = np.asarray(out["c_S"], float)
    tau0   = np.asarray(out["tau0"], float)
    f0     = np.asarray(out["f0"], float)

    F_pred = np.empty_like(F_meas_S)
    T_pred = np.empty_like(tau_meas_S)
    for i in range(N):
        g_S = R_S_W_list[i] @ gW
        F_pred[i] = f0 + m_use * g_S
        tau_g = -_skew(g_S) @ (m_use * cS_use)
        T_pred[i] = tau0 + tau_g

    rmsF = float(np.sqrt(np.mean(np.sum((F_meas_S - F_pred)**2, axis=1))))
    rmsT = float(np.sqrt(np.mean(np.sum((tau_meas_S - T_pred)**2, axis=1))))

    out.update(rmsF=rmsF, rmsT=rmsT, theta=theta.tolist())
    return out


def _spd_inertia_matrix_from_cholesky_params(cholesky_params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cholesky_params = np.asarray(cholesky_params, dtype=np.float64).reshape(-1)
    if cholesky_params.size != 6:
        raise ValueError(
            "Expected 6 Cholesky-style inertia parameters: l11, l22, l33, l21, l31, l32."
        )
    l11, l22, l33, l21, l31, l32 = cholesky_params
    L = np.array(
        [
            [np.exp(l11), 0.0, 0.0],
            [l21, np.exp(l22), 0.0],
            [l31, l32, np.exp(l33)],
        ],
        dtype=np.float64,
    )
    return L @ L.T, L


def _spd_inertia_cholesky_params_from_matrix(
    I_C: Optional[np.ndarray],
    *,
    fallback_diagonal: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Convert an inertia initialization into the Cholesky-style SPD parameterization.

    Returns
    -------
    params : (6,) np.ndarray
        Log-diagonal/off-diagonal parameterization of the lower-triangular factor.
    I_C_spd : (3,3) np.ndarray
        SPD inertia matrix used for initialization.
    used_fallback : bool
        Whether the function had to fall back to a small diagonal SPD matrix.
    """
    if fallback_diagonal is None:
        fallback_diagonal = np.array([1e-4, 1e-4, 1e-4], dtype=np.float64)
    fallback_diagonal = np.asarray(fallback_diagonal, dtype=np.float64).reshape(3,)
    fallback_matrix = np.diag(fallback_diagonal)

    used_fallback = False
    if I_C is None:
        I_C_matrix = fallback_matrix
        used_fallback = True
    else:
        I_C = np.asarray(I_C, dtype=np.float64)
        if I_C.shape == (6,):
            Ixx, Iyy, Izz, Ixy, Ixz, Iyz = I_C
            I_C_matrix = np.array(
                [
                    [Ixx, Ixy, Ixz],
                    [Ixy, Iyy, Iyz],
                    [Ixz, Iyz, Izz],
                ],
                dtype=np.float64,
            )
        elif I_C.shape == (3, 3):
            I_C_matrix = I_C.copy()
        else:
            raise ValueError("I_init must be None, a 3x3 inertia matrix, or a 6-vector of symmetric entries.")
        I_C_matrix = 0.5 * (I_C_matrix + I_C_matrix.T)

    try:
        L = np.linalg.cholesky(I_C_matrix)
        I_C_spd = I_C_matrix
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(fallback_matrix)
        I_C_spd = fallback_matrix
        used_fallback = True

    params = np.array(
        [
            np.log(L[0, 0]),
            np.log(L[1, 1]),
            np.log(L[2, 2]),
            L[1, 0],
            L[2, 0],
            L[2, 1],
        ],
        dtype=np.float64,
    )
    return params, I_C_spd, used_fallback


def _predict_dynamic_internal_wrench(
    R_S_W_list: np.ndarray,
    a_S: np.ndarray,
    omega_S: np.ndarray,
    alpha_S: np.ndarray,
    *,
    m_known: float,
    c_S_known: np.ndarray,
    b_f: np.ndarray,
    b_tau: np.ndarray,
    I_C: np.ndarray,
    g: float = 9.81,
    use_gyro_term: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Predict the internal wrench of the no-contact calibration sequence in the
    FT sensor frame.
    """
    R_S_W_list = np.asarray(R_S_W_list, dtype=np.float64)
    a_S = np.asarray(a_S, dtype=np.float64)
    omega_S = np.asarray(omega_S, dtype=np.float64)
    alpha_S = np.asarray(alpha_S, dtype=np.float64)
    c_S_known = np.asarray(c_S_known, dtype=np.float64).reshape(3,)
    b_f = np.asarray(b_f, dtype=np.float64).reshape(3,)
    b_tau = np.asarray(b_tau, dtype=np.float64).reshape(3,)
    I_C = np.asarray(I_C, dtype=np.float64).reshape(3, 3)

    num_samples = R_S_W_list.shape[0]
    if R_S_W_list.shape != (num_samples, 3, 3):
        raise ValueError("R_S_W_list must have shape (N, 3, 3).")
    for name, array in [("a_S", a_S), ("omega_S", omega_S), ("alpha_S", alpha_S)]:
        if array.shape != (num_samples, 3):
            raise ValueError(f"{name} must have shape ({num_samples}, 3).")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values.")
    if not np.all(np.isfinite(R_S_W_list)):
        raise ValueError("R_S_W_list contains non-finite values.")
    if not np.all(np.isfinite(I_C)):
        raise ValueError("I_C contains non-finite values.")

    g_W = np.array([0.0, 0.0, -g], dtype=np.float64)
    g_S = np.einsum("nij,j->ni", R_S_W_list, g_W)

    a_C_S = (
        a_S
        + np.cross(alpha_S, c_S_known[None, :])
        + np.cross(omega_S, np.cross(omega_S, c_S_known[None, :]))
    )
    F_internal_S = float(m_known) * (g_S + a_C_S)

    I_alpha = np.einsum("ij,nj->ni", I_C, alpha_S)
    tau_internal_S = np.cross(
        np.broadcast_to(c_S_known, F_internal_S.shape),
        F_internal_S,
    ) + I_alpha
    if use_gyro_term:
        I_omega = np.einsum("ij,nj->ni", I_C, omega_S)
        tau_internal_S = tau_internal_S + np.cross(omega_S, I_omega)

    F_model = F_internal_S + b_f[None, :]
    tau_model = tau_internal_S + b_tau[None, :]
    return {
        "g_S": g_S,
        "a_C_S": a_C_S,
        "F_internal_S": F_internal_S,
        "tau_internal_S": tau_internal_S,
        "F_model": F_model,
        "tau_model": tau_model,
    }


def _estimate_tool_params_dynamic_ls(
    F_meas_S: np.ndarray,
    tau_meas_S: np.ndarray,
    R_S_W_list: np.ndarray,
    a_S: np.ndarray,
    omega_S: np.ndarray,
    alpha_S: np.ndarray,
    *,
    m_known: float,
    c_S_known: np.ndarray,
    fit_c_S: bool = False,
    c_S_init: Optional[np.ndarray] = None,
    b_f_init: np.ndarray,
    b_tau_init: np.ndarray,
    I_init: Optional[np.ndarray] = None,
    g: float = 9.81,
    torque_weight: float = 1.0,
    use_gyro_term: bool = False,
) -> Dict[str, Any]:
    """
    Robust nonlinear least-squares fit of dynamic FT sensor biases and inertia
    with fixed known mass and CoG.
    """
    F_meas_S = np.asarray(F_meas_S, dtype=np.float64)
    tau_meas_S = np.asarray(tau_meas_S, dtype=np.float64)
    R_S_W_list = np.asarray(R_S_W_list, dtype=np.float64)
    a_S = np.asarray(a_S, dtype=np.float64)
    omega_S = np.asarray(omega_S, dtype=np.float64)
    alpha_S = np.asarray(alpha_S, dtype=np.float64)
    c_S_known = np.asarray(c_S_known, dtype=np.float64).reshape(3,)
    b_f_init = np.asarray(b_f_init, dtype=np.float64).reshape(3,)
    b_tau_init = np.asarray(b_tau_init, dtype=np.float64).reshape(3,)
    if c_S_init is None:
        c_S_init = c_S_known.copy()
    else:
        c_S_init = np.asarray(c_S_init, dtype=np.float64).reshape(3,)

    num_samples = F_meas_S.shape[0]
    if F_meas_S.shape != (num_samples, 3):
        raise ValueError("F_meas_S must have shape (N, 3).")
    if tau_meas_S.shape != (num_samples, 3):
        raise ValueError("tau_meas_S must have shape (N, 3).")
    if R_S_W_list.shape != (num_samples, 3, 3):
        raise ValueError("R_S_W_list must have shape (N, 3, 3).")
    for name, array in [("a_S", a_S), ("omega_S", omega_S), ("alpha_S", alpha_S)]:
        if array.shape != (num_samples, 3):
            raise ValueError(f"{name} must have shape ({num_samples}, 3).")
    if num_samples == 0:
        raise ValueError("Need at least one sample for dynamic least-squares fitting.")
    if torque_weight <= 0.0:
        raise ValueError("torque_weight must be strictly positive.")
    for name, array in [
        ("F_meas_S", F_meas_S),
        ("tau_meas_S", tau_meas_S),
        ("R_S_W_list", R_S_W_list),
        ("a_S", a_S),
        ("omega_S", omega_S),
        ("alpha_S", alpha_S),
        ("b_f_init", b_f_init),
        ("b_tau_init", b_tau_init),
        ("c_S_init", c_S_init),
    ]:
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values.")

    if I_init is None:
        I_init_cholesky_params, I_init_matrix, inertia_init_used_fallback = _spd_inertia_cholesky_params_from_matrix(
            None
        )
    else:
        I_init_cholesky_params, I_init_matrix, inertia_init_used_fallback = _spd_inertia_cholesky_params_from_matrix(
            I_init
        )
    if fit_c_S:
        theta0 = np.hstack(
            [
                b_f_init,
                b_tau_init,
                c_S_init,
                I_init_cholesky_params,
            ]
        )
    else:
        theta0 = np.hstack(
            [
                b_f_init,
                b_tau_init,
                I_init_cholesky_params,
            ]
        )

    def residual_vector(theta: np.ndarray) -> np.ndarray:
        b_f = theta[:3]
        b_tau = theta[3:6]
        if fit_c_S:
            c_S_current = theta[6:9]
            inertia_theta = theta[9:]
        else:
            c_S_current = c_S_known
            inertia_theta = theta[6:]
        I_C, _ = _spd_inertia_matrix_from_cholesky_params(inertia_theta)
        model = _predict_dynamic_internal_wrench(
            R_S_W_list=R_S_W_list,
            a_S=a_S,
            omega_S=omega_S,
            alpha_S=alpha_S,
            m_known=m_known,
            c_S_known=c_S_current,
            b_f=b_f,
            b_tau=b_tau,
            I_C=I_C,
            g=g,
            use_gyro_term=use_gyro_term,
        )
        F_residual = F_meas_S - model["F_model"]
        tau_residual = tau_meas_S - model["tau_model"]
        return np.hstack(
            [
                F_residual.reshape(-1),
                float(torque_weight) * tau_residual.reshape(-1),
            ]
        )

    result = least_squares(
        residual_vector,
        theta0,
        loss="soft_l1",
        method="trf",
    )

    b_f_fit = result.x[:3]
    b_tau_fit = result.x[3:6]
    if fit_c_S:
        c_S_fit = result.x[6:9]
        inertia_theta_fit = result.x[9:]
    else:
        c_S_fit = c_S_known.copy()
        inertia_theta_fit = result.x[6:]
    I_C_fit, L_fit = _spd_inertia_matrix_from_cholesky_params(inertia_theta_fit)
    I_init_cholesky_matrix = np.linalg.cholesky(I_init_matrix)

    model_init = _predict_dynamic_internal_wrench(
        R_S_W_list=R_S_W_list,
        a_S=a_S,
        omega_S=omega_S,
        alpha_S=alpha_S,
        m_known=m_known,
        c_S_known=c_S_init if fit_c_S else c_S_known,
        b_f=b_f_init,
        b_tau=b_tau_init,
        I_C=I_init_matrix,
        g=g,
        use_gyro_term=use_gyro_term,
    )
    model_fit = _predict_dynamic_internal_wrench(
        R_S_W_list=R_S_W_list,
        a_S=a_S,
        omega_S=omega_S,
        alpha_S=alpha_S,
        m_known=m_known,
        c_S_known=c_S_fit,
        b_f=b_f_fit,
        b_tau=b_tau_fit,
        I_C=I_C_fit,
        g=g,
        use_gyro_term=use_gyro_term,
    )

    F_residual_init = F_meas_S - model_init["F_model"]
    tau_residual_init = tau_meas_S - model_init["tau_model"]
    F_residual = F_meas_S - model_fit["F_model"]
    tau_residual = tau_meas_S - model_fit["tau_model"]

    rmsF_init = float(np.sqrt(np.mean(np.sum(F_residual_init ** 2, axis=1))))
    rmsT_init = float(np.sqrt(np.mean(np.sum(tau_residual_init ** 2, axis=1))))
    rmsF = float(np.sqrt(np.mean(np.sum(F_residual ** 2, axis=1))))
    rmsT = float(np.sqrt(np.mean(np.sum(tau_residual ** 2, axis=1))))

    inertia_eigenvalues = np.linalg.eigvalsh(I_C_fit)
    inertia_principal_moments = np.sort(inertia_eigenvalues)
    inertia_det = float(np.linalg.det(I_C_fit))
    inertia_is_spd = bool(np.all(inertia_eigenvalues > 0.0))

    return {
        "b_f": b_f_fit,
        "b_tau": b_tau_fit,
        "c_S": c_S_fit.copy(),
        "c_S_init": c_S_init.copy(),
        "I_C": I_C_fit,
        "L": L_fit,
        "I_init": I_init_matrix,
        "L_init": I_init_cholesky_matrix,
        "inertia_init_used_fallback": bool(inertia_init_used_fallback),
        "inertia_cholesky_params": inertia_theta_fit.copy(),
        "F_model": model_fit["F_model"],
        "tau_model": model_fit["tau_model"],
        "F_residual": F_residual,
        "tau_residual": tau_residual,
        "F_model_init": model_init["F_model"],
        "tau_model_init": model_init["tau_model"],
        "F_residual_init": F_residual_init,
        "tau_residual_init": tau_residual_init,
        "rmsF_init": rmsF_init,
        "rmsT_init": rmsT_init,
        "rmsF": rmsF,
        "rmsT": rmsT,
        "status": int(result.status),
        "success": bool(result.success),
        "message": result.message,
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "theta": result.x.copy(),
        "use_gyro_term": bool(use_gyro_term),
        "torque_weight": float(torque_weight),
        "m_known": float(m_known),
        "fit_c_S": bool(fit_c_S),
        "c_S_known": c_S_known.copy(),
        "inertia_eigenvalues": inertia_eigenvalues,
        "inertia_principal_moments": inertia_principal_moments,
        "inertia_det": inertia_det,
        "inertia_is_spd": inertia_is_spd,
    }


def _visualize_dynamic_fit_residuals(
    wrench_time_s: np.ndarray,
    dynamic_fit: Dict[str, Any],
    *,
    visualize: bool,
    visualize_out_dir: Optional[Path],
) -> None:
    if not visualize and visualize_out_dir is None:
        return

    wrench_time_s = np.asarray(wrench_time_s, dtype=np.float64)
    F_residual_init = np.asarray(dynamic_fit["F_residual_init"], dtype=np.float64)
    tau_residual_init = np.asarray(dynamic_fit["tau_residual_init"], dtype=np.float64)
    F_residual = np.asarray(dynamic_fit["F_residual"], dtype=np.float64)
    tau_residual = np.asarray(dynamic_fit["tau_residual"], dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(
        wrench_time_s,
        np.linalg.norm(F_residual_init, axis=1),
        label="Before dynamic fit",
        color="tab:gray",
        lw=1.2,
    )
    axes[0].plot(
        wrench_time_s,
        np.linalg.norm(F_residual, axis=1),
        label="After dynamic fit",
        color="tab:blue",
        lw=1.6,
    )
    axes[0].set_ylabel("Force residual [N]")
    axes[0].set_title("Force residual magnitude before vs after dynamic fit")
    axes[0].grid(True)
    axes[0].legend(loc="upper right")

    axes[1].plot(
        wrench_time_s,
        np.linalg.norm(tau_residual_init, axis=1),
        label="Before dynamic fit",
        color="tab:gray",
        lw=1.2,
    )
    axes[1].plot(
        wrench_time_s,
        np.linalg.norm(tau_residual, axis=1),
        label="After dynamic fit",
        color="tab:orange",
        lw=1.6,
    )
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Torque residual [Nm]")
    axes[1].set_title("Torque residual magnitude before vs after dynamic fit")
    axes[1].grid(True)
    axes[1].legend(loc="upper right")

    _finalize_debug_figure(
        fig,
        visualize=visualize,
        visualize_out_dir=visualize_out_dir,
        filename="dynamic_fit_residual_magnitudes.png",
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    component_labels = ["x", "y", "z"]
    colors_before = "tab:gray"
    colors_after = "tab:blue"
    for component_idx, label in enumerate(component_labels):
        axes[0, component_idx].plot(
            wrench_time_s,
            F_residual_init[:, component_idx],
            color=colors_before,
            lw=1.0,
            label="Before",
        )
        axes[0, component_idx].plot(
            wrench_time_s,
            F_residual[:, component_idx],
            color=colors_after,
            lw=1.4,
            label="After",
        )
        axes[0, component_idx].set_ylabel(f"F {label} [N]")
        axes[0, component_idx].grid(True)
        axes[0, component_idx].legend(loc="upper right")

        axes[1, component_idx].plot(
            wrench_time_s,
            tau_residual_init[:, component_idx],
            color=colors_before,
            lw=1.0,
            label="Before",
        )
        axes[1, component_idx].plot(
            wrench_time_s,
            tau_residual[:, component_idx],
            color="tab:orange",
            lw=1.4,
            label="After",
        )
        axes[1, component_idx].set_ylabel(f"Tau {label} [Nm]")
        axes[1, component_idx].grid(True)
        axes[1, component_idx].legend(loc="upper right")
        axes[1, component_idx].set_xlabel("Time [s]")

    axes[0, 0].set_title("Per-axis force residuals")
    axes[1, 0].set_title("Per-axis torque residuals")

    _finalize_debug_figure(
        fig,
        visualize=visualize,
        visualize_out_dir=visualize_out_dir,
        filename="dynamic_fit_residual_components.png",
    )


def _summarize_dynamic_signal_stats(
    dynamic_signals: Dict[str, np.ndarray],
    valid_dynamic_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    if valid_dynamic_mask is None:
        valid_dynamic_mask = np.ones(dynamic_signals["a_D"].shape[0], dtype=bool)
    else:
        valid_dynamic_mask = np.asarray(valid_dynamic_mask, dtype=bool)

    stats = {}
    metric_sources = {
        "a_D": np.asarray(dynamic_signals["a_D"], dtype=np.float64),
        "a_S": np.asarray(dynamic_signals["a_S"], dtype=np.float64),
        "alpha_D": np.asarray(dynamic_signals["alpha_D"], dtype=np.float64),
        "alpha_S": np.asarray(dynamic_signals["alpha_S"], dtype=np.float64),
        "omega_D": np.asarray(dynamic_signals["omega_D"], dtype=np.float64),
        "omega_S": np.asarray(dynamic_signals["omega_S"], dtype=np.float64),
    }
    for metric_name, metric_values in metric_sources.items():
        metric_values = metric_values[valid_dynamic_mask]
        metric_norm = np.linalg.norm(metric_values, axis=1)
        stats[f"{metric_name}_norm_mean"] = float(np.mean(metric_norm))
        stats[f"{metric_name}_norm_max"] = float(np.max(metric_norm))
    return stats


def _run_dynamic_fit_for_dynamic_signals(
    dynamic_signals: Dict[str, np.ndarray],
    valid_dynamic_mask: np.ndarray,
    F_meas_S: np.ndarray,
    tau_meas_S: np.ndarray,
    *,
    m_known: float,
    c_S_known: np.ndarray,
    dynamic_fit_cog: bool,
    c_S_init: np.ndarray,
    b_f_init: np.ndarray,
    b_tau_init: np.ndarray,
    dynamic_torque_weight: float,
    dynamic_use_gyro_term: bool,
    dynamic_fit_debug_dir: Optional[Path] = None,
    visualize: bool = False,
) -> Dict[str, Any]:
    valid_dynamic_mask = np.asarray(valid_dynamic_mask, dtype=bool)
    wrench_time_s_dynamic_valid = np.asarray(dynamic_signals["wrench_time_s"], dtype=np.float64)[valid_dynamic_mask]
    F_meas_S_dynamic_valid = np.asarray(F_meas_S, dtype=np.float64)[valid_dynamic_mask]
    tau_meas_S_dynamic_valid = np.asarray(tau_meas_S, dtype=np.float64)[valid_dynamic_mask]
    R_S_W_dynamic_valid = np.asarray(dynamic_signals["R_S_W"], dtype=np.float64)[valid_dynamic_mask]
    a_S_dynamic_valid = np.asarray(dynamic_signals["a_S"], dtype=np.float64)[valid_dynamic_mask]
    omega_S_dynamic_valid = np.asarray(dynamic_signals["omega_S"], dtype=np.float64)[valid_dynamic_mask]
    alpha_S_dynamic_valid = np.asarray(dynamic_signals["alpha_S"], dtype=np.float64)[valid_dynamic_mask]

    dynamic_fit = _estimate_tool_params_dynamic_ls(
        F_meas_S=F_meas_S_dynamic_valid,
        tau_meas_S=tau_meas_S_dynamic_valid,
        R_S_W_list=R_S_W_dynamic_valid,
        a_S=a_S_dynamic_valid,
        omega_S=omega_S_dynamic_valid,
        alpha_S=alpha_S_dynamic_valid,
        m_known=m_known,
        c_S_known=c_S_known,
        fit_c_S=dynamic_fit_cog,
        c_S_init=c_S_init,
        b_f_init=b_f_init,
        b_tau_init=b_tau_init,
        g=9.81,
        torque_weight=dynamic_torque_weight,
        use_gyro_term=dynamic_use_gyro_term,
    )
    f_model_valid = np.asarray(dynamic_fit["F_model"], dtype=np.float64)
    tau_model_valid = np.asarray(dynamic_fit["tau_model"], dtype=np.float64)
    f_ext_hat_valid = np.asarray(dynamic_fit["F_residual"], dtype=np.float64)
    tau_ext_hat_valid = np.asarray(dynamic_fit["tau_residual"], dtype=np.float64)
    f_ext_hat_init_valid = np.asarray(dynamic_fit["F_residual_init"], dtype=np.float64)
    tau_ext_hat_init_valid = np.asarray(dynamic_fit["tau_residual_init"], dtype=np.float64)

    f_model = np.full_like(F_meas_S, np.nan, dtype=np.float64)
    tau_model = np.full_like(tau_meas_S, np.nan, dtype=np.float64)
    f_ext_hat = np.full_like(F_meas_S, np.nan, dtype=np.float64)
    tau_ext_hat = np.full_like(tau_meas_S, np.nan, dtype=np.float64)
    f_ext_hat_init = np.full_like(F_meas_S, np.nan, dtype=np.float64)
    tau_ext_hat_init = np.full_like(tau_meas_S, np.nan, dtype=np.float64)
    f_model[valid_dynamic_mask] = f_model_valid
    tau_model[valid_dynamic_mask] = tau_model_valid
    f_ext_hat[valid_dynamic_mask] = f_ext_hat_valid
    tau_ext_hat[valid_dynamic_mask] = tau_ext_hat_valid
    f_ext_hat_init[valid_dynamic_mask] = f_ext_hat_init_valid
    tau_ext_hat_init[valid_dynamic_mask] = tau_ext_hat_init_valid

    dynamic_fit_debug_plot_paths = []
    if dynamic_fit_debug_dir is not None:
        dynamic_fit_debug_plot_paths = [
            dynamic_fit_debug_dir / "dynamic_fit_residual_magnitudes.png",
            dynamic_fit_debug_dir / "dynamic_fit_residual_components.png",
        ]
    if dynamic_fit_debug_dir is not None or visualize:
        _visualize_dynamic_fit_residuals(
            wrench_time_s=wrench_time_s_dynamic_valid,
            dynamic_fit=dynamic_fit,
            visualize=visualize,
            visualize_out_dir=dynamic_fit_debug_dir,
        )

    dynamic_fit_residual_stats = {
        "rmsF_before": float(dynamic_fit["rmsF_init"]),
        "rmsT_before": float(dynamic_fit["rmsT_init"]),
        "rmsF_after": float(dynamic_fit["rmsF"]),
        "rmsT_after": float(dynamic_fit["rmsT"]),
        "force_residual_norm_mean_before": float(np.mean(np.linalg.norm(f_ext_hat_init_valid, axis=1))),
        "force_residual_norm_mean_after": float(np.mean(np.linalg.norm(f_ext_hat_valid, axis=1))),
        "torque_residual_norm_mean_before": float(np.mean(np.linalg.norm(tau_ext_hat_init_valid, axis=1))),
        "torque_residual_norm_mean_after": float(np.mean(np.linalg.norm(tau_ext_hat_valid, axis=1))),
    }

    return {
        "dynamic_fit": dynamic_fit,
        "dynamic_fit_residual_stats": dynamic_fit_residual_stats,
        "dynamic_fit_debug_plot_paths": dynamic_fit_debug_plot_paths,
        "f_model": f_model,
        "tau_model": tau_model,
        "f_ext_hat": f_ext_hat,
        "tau_ext_hat": tau_ext_hat,
        "f_ext_hat_init": f_ext_hat_init,
        "tau_ext_hat_init": tau_ext_hat_init,
        "f_model_valid": f_model_valid,
        "tau_model_valid": tau_model_valid,
        "f_ext_hat_valid": f_ext_hat_valid,
        "tau_ext_hat_valid": tau_ext_hat_valid,
        "f_ext_hat_init_valid": f_ext_hat_init_valid,
        "tau_ext_hat_init_valid": tau_ext_hat_init_valid,
        "wrench_time_s_dynamic_valid": wrench_time_s_dynamic_valid,
    }


def _select_best_dynamic_smoothing_candidate(
    candidate_results: List[Dict[str, Any]],
) -> int:
    """
    Pick the best smoothing candidate.

    Selection rule:
    1. Prefer successful optimizer runs.
    2. Reject obviously unstable candidates with extreme acceleration peaks
       relative to the candidate set median.
    3. Among the remaining candidates, prioritize lower ``rmsT_after``.
    4. If several candidates are within 5% of the best torque RMS, prefer the
       more conservative smoothing (larger total window length, then lower total
       polynomial order).
    """
    if not candidate_results:
        raise ValueError("Need at least one smoothing candidate to select a best result.")

    successful_candidates = [
        candidate for candidate in candidate_results if candidate["dynamic_fit"]["success"]
    ]
    candidate_pool = successful_candidates or candidate_results

    a_S_max_values = np.array(
        [candidate["signal_stats"]["a_S_norm_max"] for candidate in candidate_pool],
        dtype=np.float64,
    )
    alpha_S_max_values = np.array(
        [candidate["signal_stats"]["alpha_S_norm_max"] for candidate in candidate_pool],
        dtype=np.float64,
    )
    accel_limit = 3.0 * float(np.median(a_S_max_values))
    angular_accel_limit = 3.0 * float(np.median(alpha_S_max_values))
    stable_candidates = [
        candidate
        for candidate in candidate_pool
        if candidate["signal_stats"]["a_S_norm_max"] <= accel_limit
        and candidate["signal_stats"]["alpha_S_norm_max"] <= angular_accel_limit
    ]
    candidate_pool = stable_candidates or candidate_pool

    best_rmsT_after = min(
        candidate["fit_metrics"]["rmsT_after"] for candidate in candidate_pool
    )
    close_candidates = [
        candidate
        for candidate in candidate_pool
        if candidate["fit_metrics"]["rmsT_after"] <= 1.05 * best_rmsT_after
    ]

    def selection_key(candidate: Dict[str, Any]) -> Tuple[float, float, int, int]:
        smoothing_config = candidate["resolved_smoothing_config"]
        total_window_length = int(smoothing_config["sg_window_length_linear"]) + int(
            smoothing_config["sg_window_length_angular"]
        )
        total_polyorder = int(smoothing_config["sg_polyorder_linear"]) + int(
            smoothing_config["sg_polyorder_angular"]
        )
        return (
            float(candidate["fit_metrics"]["rmsT_after"]),
            float(candidate["fit_metrics"]["rmsF_after"]),
            -total_window_length,
            total_polyorder,
        )

    best_candidate = min(close_candidates, key=selection_key)
    return int(best_candidate["candidate_index"])


def _visualize_dynamic_smoothing_candidates(
    candidate_results: List[Dict[str, Any]],
    best_candidate_index: int,
    *,
    visualize: bool,
    visualize_out_dir: Optional[Path],
) -> List[Path]:
    if not visualize and visualize_out_dir is None:
        return []

    labels = [
        _format_dynamic_smoothing_config_label(candidate["resolved_smoothing_config"])
        for candidate in candidate_results
    ]
    rmsF_after = [candidate["fit_metrics"]["rmsF_after"] for candidate in candidate_results]
    rmsT_after = [candidate["fit_metrics"]["rmsT_after"] for candidate in candidate_results]
    a_S_mean = [candidate["signal_stats"]["a_S_norm_mean"] for candidate in candidate_results]
    a_S_max = [candidate["signal_stats"]["a_S_norm_max"] for candidate in candidate_results]
    alpha_S_mean = [candidate["signal_stats"]["alpha_S_norm_mean"] for candidate in candidate_results]
    alpha_S_max = [candidate["signal_stats"]["alpha_S_norm_max"] for candidate in candidate_results]
    candidate_indices = np.arange(len(candidate_results))
    bar_colors = [
        "tab:orange" if index == best_candidate_index else "tab:blue"
        for index in range(len(candidate_results))
    ]

    output_paths: List[Path] = []
    if visualize_out_dir is not None:
        output_paths = [
            visualize_out_dir / "dynamic_smoothing_candidate_residuals.png",
            visualize_out_dir / "dynamic_smoothing_candidate_signal_stats.png",
        ]

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    axes[0].bar(candidate_indices, rmsT_after, color=bar_colors, alpha=0.85)
    axes[0].set_ylabel("Torque RMS after [Nm]")
    axes[0].set_title("Dynamic smoothing candidates: torque residual comparison")
    axes[0].grid(True, axis="y")

    axes[1].bar(candidate_indices, rmsF_after, color=bar_colors, alpha=0.85)
    axes[1].set_ylabel("Force RMS after [N]")
    axes[1].set_title("Dynamic smoothing candidates: force residual comparison")
    axes[1].grid(True, axis="y")
    axes[1].set_xticks(candidate_indices)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_xlabel("Smoothing configuration")

    _finalize_debug_figure(
        fig,
        visualize=visualize,
        visualize_out_dir=visualize_out_dir,
        filename="dynamic_smoothing_candidate_residuals.png",
    )

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(candidate_indices, a_S_mean, marker="o", label="mean ||a_S||", color="tab:green")
    axes[0].plot(candidate_indices, a_S_max, marker="o", label="max ||a_S||", color="tab:red")
    axes[0].set_ylabel("Linear accel. [m/s$^2$]")
    axes[0].set_title("Dynamic smoothing candidates: linear acceleration statistics")
    axes[0].grid(True)
    axes[0].legend(loc="upper right")

    axes[1].plot(candidate_indices, alpha_S_mean, marker="o", label="mean ||alpha_S||", color="tab:purple")
    axes[1].plot(candidate_indices, alpha_S_max, marker="o", label="max ||alpha_S||", color="tab:orange")
    axes[1].set_ylabel("Angular accel. [rad/s$^2$]")
    axes[1].set_title("Dynamic smoothing candidates: angular acceleration statistics")
    axes[1].grid(True)
    axes[1].legend(loc="upper right")
    axes[1].set_xticks(candidate_indices)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_xlabel("Smoothing configuration")

    _finalize_debug_figure(
        fig,
        visualize=visualize,
        visualize_out_dir=visualize_out_dir,
        filename="dynamic_smoothing_candidate_signal_stats.png",
    )
    return output_paths


def _select_best_dynamic_gyro_candidate(
    gyro_results: List[Dict[str, Any]],
    *,
    negligible_rmsT_tolerance: float = 1e-4,
) -> Tuple[int, str]:
    """
    Pick between gyro-disabled and gyro-enabled fits.

    Selection rule:
    1. A candidate must have a successful optimizer result, a finite SPD inertia
       tensor, and finite residual statistics to be considered valid.
    2. Prefer the valid candidate with lower ``rmsT_after``.
    3. If the torque-RMS improvement of ``gyro=True`` over ``gyro=False`` is no
       larger than ``negligible_rmsT_tolerance`` [Nm], keep ``gyro=False`` for
       the simpler model.
    """
    if len(gyro_results) == 0:
        raise ValueError("Need at least one gyro comparison result to select a model.")

    gyro_results_by_flag = {bool(result["use_gyro_term"]): result for result in gyro_results}
    if False not in gyro_results_by_flag or True not in gyro_results_by_flag:
        raise ValueError("Expected both gyro=False and gyro=True candidates.")

    for result in gyro_results:
        fit_metrics = result["fit_metrics"]
        dynamic_fit = result["dynamic_fit"]
        inertia_eigenvalues = np.asarray(dynamic_fit["inertia_eigenvalues"], dtype=np.float64)
        is_valid = (
            bool(dynamic_fit["success"])
            and bool(dynamic_fit["inertia_is_spd"])
            and np.isfinite(float(dynamic_fit["inertia_det"]))
            and np.all(np.isfinite(inertia_eigenvalues))
            and np.all(np.isfinite(np.asarray(dynamic_fit["b_f"], dtype=np.float64)))
            and np.all(np.isfinite(np.asarray(dynamic_fit["b_tau"], dtype=np.float64)))
            and np.isfinite(float(fit_metrics["rmsF_after"]))
            and np.isfinite(float(fit_metrics["rmsT_after"]))
        )
        result["valid_for_selection"] = bool(is_valid)

    gyro_false_result = gyro_results_by_flag[False]
    gyro_true_result = gyro_results_by_flag[True]

    if gyro_false_result["valid_for_selection"] and not gyro_true_result["valid_for_selection"]:
        return int(gyro_false_result["candidate_index"]), "gyro=True fit was invalid; keeping gyro=False."
    if gyro_true_result["valid_for_selection"] and not gyro_false_result["valid_for_selection"]:
        return int(gyro_true_result["candidate_index"]), "gyro=False fit was invalid; selecting gyro=True."
    if not gyro_false_result["valid_for_selection"] and not gyro_true_result["valid_for_selection"]:
        fallback_candidate = min(
            gyro_results,
            key=lambda result: (
                float(result["fit_metrics"]["rmsT_after"]),
                float(result["fit_metrics"]["rmsF_after"]),
            ),
        )
        return (
            int(fallback_candidate["candidate_index"]),
            "Neither gyro candidate passed validity checks; fell back to the lower residual result.",
        )

    rmsT_false = float(gyro_false_result["fit_metrics"]["rmsT_after"])
    rmsT_true = float(gyro_true_result["fit_metrics"]["rmsT_after"])
    torque_improvement_with_gyro = rmsT_false - rmsT_true

    if torque_improvement_with_gyro > float(negligible_rmsT_tolerance):
        return (
            int(gyro_true_result["candidate_index"]),
            f"gyro=True reduced torque RMS by {torque_improvement_with_gyro:.6f} Nm, which exceeds the negligible threshold.",
        )
    if torque_improvement_with_gyro >= 0.0:
        return (
            int(gyro_false_result["candidate_index"]),
            f"gyro=True improvement {torque_improvement_with_gyro:.6f} Nm did not exceed the negligible threshold; keeping gyro=False.",
        )
    return (
        int(gyro_false_result["candidate_index"]),
        f"gyro=False already achieved lower torque RMS by {-torque_improvement_with_gyro:.6f} Nm.",
    )


def _visualize_dynamic_gyro_comparison(
    gyro_results: List[Dict[str, Any]],
    best_candidate_index: int,
    *,
    visualize: bool,
    visualize_out_dir: Optional[Path],
) -> List[Path]:
    if not visualize and visualize_out_dir is None:
        return []

    labels = [
        "gyro=True" if bool(result["use_gyro_term"]) else "gyro=False"
        for result in gyro_results
    ]
    rmsF_after = [float(result["fit_metrics"]["rmsF_after"]) for result in gyro_results]
    rmsT_after = [float(result["fit_metrics"]["rmsT_after"]) for result in gyro_results]
    candidate_indices = np.arange(len(gyro_results))
    bar_colors = [
        "tab:orange" if int(result["candidate_index"]) == int(best_candidate_index) else "tab:blue"
        for result in gyro_results
    ]

    output_paths: List[Path] = []
    if visualize_out_dir is not None:
        visualize_out_dir.mkdir(parents=True, exist_ok=True)
        output_paths = [
            visualize_out_dir / "dynamic_gyro_residual_comparison.png",
            visualize_out_dir / "dynamic_gyro_comparison.txt",
        ]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].bar(candidate_indices, rmsT_after, color=bar_colors, alpha=0.85)
    axes[0].set_ylabel("Torque RMS after [Nm]")
    axes[0].set_title("Dynamic gyro comparison: torque residual")
    axes[0].grid(True, axis="y")

    axes[1].bar(candidate_indices, rmsF_after, color=bar_colors, alpha=0.85)
    axes[1].set_ylabel("Force RMS after [N]")
    axes[1].set_title("Dynamic gyro comparison: force residual")
    axes[1].grid(True, axis="y")
    axes[1].set_xticks(candidate_indices)
    axes[1].set_xticklabels(labels)
    axes[1].set_xlabel("Gyro term setting")

    _finalize_debug_figure(
        fig,
        visualize=visualize,
        visualize_out_dir=visualize_out_dir,
        filename="dynamic_gyro_residual_comparison.png",
    )

    if visualize_out_dir is not None:
        summary_lines = []
        for result in gyro_results:
            dynamic_fit = result["dynamic_fit"]
            fit_metrics = result["fit_metrics"]
            inertia_eigenvalues_str = ", ".join(
                f"{float(value):.6e}"
                for value in np.asarray(dynamic_fit["inertia_eigenvalues"], dtype=np.float64)
            )
            summary_lines.append(
                f"{'gyro=True' if result['use_gyro_term'] else 'gyro=False'} | "
                f"success={dynamic_fit['success']} | status={dynamic_fit['status']} | "
                f"nfev={dynamic_fit['nfev']} | cost={dynamic_fit['cost']:.6e} | "
                f"rmsF={fit_metrics['rmsF_before']:.6f}->{fit_metrics['rmsF_after']:.6f} | "
                f"rmsT={fit_metrics['rmsT_before']:.6f}->{fit_metrics['rmsT_after']:.6f} | "
                f"det={dynamic_fit['inertia_det']:.6e} | eig=[{inertia_eigenvalues_str}]"
            )
        output_paths[1].write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return output_paths


def _select_best_dynamic_time_offset_candidate(
    time_offset_results: List[Dict[str, Any]],
    *,
    negligible_rmsT_tolerance: float = 1e-4,
) -> Tuple[int, str]:
    """
    Pick the best constant time offset candidate.

    Selection rule:
    1. A candidate must have a successful optimizer result, a finite SPD inertia
       tensor, and at least one valid overlap sample to be considered valid.
    2. Prefer the valid candidate with lower ``rmsT_after``.
    3. If a candidate is within ``negligible_rmsT_tolerance`` [Nm] of the best
       torque RMS, prefer the smaller absolute offset, and in particular keep
       ``0 ms`` when it is effectively tied for best.
    """
    if len(time_offset_results) == 0:
        raise ValueError("Need at least one time-offset candidate to select a best result.")

    for result in time_offset_results:
        dynamic_fit = result["dynamic_fit"]
        fit_metrics = result["fit_metrics"]
        inertia_eigenvalues = np.asarray(dynamic_fit["inertia_eigenvalues"], dtype=np.float64)
        is_valid = (
            int(result["valid_sample_count"]) > 0
            and bool(dynamic_fit["success"])
            and bool(dynamic_fit["inertia_is_spd"])
            and np.isfinite(float(dynamic_fit["inertia_det"]))
            and np.all(np.isfinite(inertia_eigenvalues))
            and np.all(np.isfinite(np.asarray(dynamic_fit["b_f"], dtype=np.float64)))
            and np.all(np.isfinite(np.asarray(dynamic_fit["b_tau"], dtype=np.float64)))
            and np.isfinite(float(fit_metrics["rmsF_after"]))
            and np.isfinite(float(fit_metrics["rmsT_after"]))
        )
        result["valid_for_selection"] = bool(is_valid)

    valid_candidates = [result for result in time_offset_results if result["valid_for_selection"]]
    candidate_pool = valid_candidates or time_offset_results
    best_rmsT_after = min(float(result["fit_metrics"]["rmsT_after"]) for result in candidate_pool)
    close_candidates = [
        result
        for result in candidate_pool
        if float(result["fit_metrics"]["rmsT_after"]) <= best_rmsT_after + float(negligible_rmsT_tolerance)
    ]

    zero_offset_candidate = next(
        (result for result in close_candidates if abs(float(result["offset_ms"])) < 1e-12),
        None,
    )
    if zero_offset_candidate is not None:
        return (
            int(zero_offset_candidate["candidate_index"]),
            "0 ms stayed within the negligible torque-RMS tolerance of the best candidate, so the unshifted model was kept.",
        )

    def selection_key(result: Dict[str, Any]) -> Tuple[float, float, float, int]:
        return (
            float(result["fit_metrics"]["rmsT_after"]),
            abs(float(result["offset_ms"])),
            float(result["fit_metrics"]["rmsF_after"]),
            -int(result["valid_sample_count"]),
        )

    best_candidate = min(close_candidates, key=selection_key)
    return (
        int(best_candidate["candidate_index"]),
        f"Selected the lowest stable torque RMS candidate at {float(best_candidate['offset_ms']):.3f} ms.",
    )


def _visualize_dynamic_time_offset_candidates(
    time_offset_results: List[Dict[str, Any]],
    best_candidate_index: int,
    *,
    visualize: bool,
    visualize_out_dir: Optional[Path],
) -> List[Path]:
    if not visualize and visualize_out_dir is None:
        return []

    offsets_ms = np.array([float(result["offset_ms"]) for result in time_offset_results], dtype=np.float64)
    rmsT_after = np.array(
        [float(result["fit_metrics"]["rmsT_after"]) for result in time_offset_results],
        dtype=np.float64,
    )
    rmsF_after = np.array(
        [float(result["fit_metrics"]["rmsF_after"]) for result in time_offset_results],
        dtype=np.float64,
    )
    point_colors = [
        "tab:orange" if int(result["candidate_index"]) == int(best_candidate_index) else "tab:blue"
        for result in time_offset_results
    ]

    output_paths: List[Path] = []
    if visualize_out_dir is not None:
        visualize_out_dir.mkdir(parents=True, exist_ok=True)
        output_paths = [visualize_out_dir / "dynamic_time_offset_residuals.png"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(offsets_ms, rmsT_after, color="tab:orange", lw=1.5)
    axes[0].scatter(offsets_ms, rmsT_after, c=point_colors, s=55, zorder=3)
    axes[0].axvline(0.0, color="tab:gray", ls="--", lw=1.0)
    axes[0].set_ylabel("Torque RMS after [Nm]")
    axes[0].set_title("Dynamic time-offset sweep: torque residual vs offset")
    axes[0].grid(True)

    axes[1].plot(offsets_ms, rmsF_after, color="tab:blue", lw=1.5)
    axes[1].scatter(offsets_ms, rmsF_after, c=point_colors, s=55, zorder=3)
    axes[1].axvline(0.0, color="tab:gray", ls="--", lw=1.0)
    axes[1].set_ylabel("Force RMS after [N]")
    axes[1].set_xlabel("Constant time offset [ms]")
    axes[1].set_title("Dynamic time-offset sweep: force residual vs offset")
    axes[1].grid(True)

    _finalize_debug_figure(
        fig,
        visualize=visualize,
        visualize_out_dir=visualize_out_dir,
        filename="dynamic_time_offset_residuals.png",
    )
    return output_paths


def _slerp_pose_series_to_targets(poses_df: pd.DataFrame, target_ts_ns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    poses_df columns:
      required: ['timestamp','qx_world_device','qy_world_device','qz_world_device','qw_world_device']
      optional: ['tx_world_device','ty_world_device','tz_world_device']
    Returns arrays aligned to target_ts_ns:
      R_W_D_list (N,3,3), t_W_D_list (N,3)

    If the translation columns are absent, the returned translations are zeros.
    This keeps the helper reusable in places where only orientation interpolation
    is needed.
    """
    if len(poses_df) < 2:
        raise ValueError("Not enough poses for interpolation")

    # sort and unique
    poses_df = poses_df.sort_values('timestamp').drop_duplicates('timestamp')

    t_ref = poses_df['timestamp'].to_numpy(dtype=np.int64)
    quat  = poses_df[['qx_world_device','qy_world_device','qz_world_device','qw_world_device']].to_numpy(dtype=float)

    if {'tx_world_device', 'ty_world_device', 'tz_world_device'}.issubset(poses_df.columns):
        pos = poses_df[['tx_world_device', 'ty_world_device', 'tz_world_device']].to_numpy(dtype=float)
    else:
        pos = np.zeros((len(poses_df), 3), dtype=float)

    # Normalize quaternions (format [x,y,z,w])
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)

    # Clamp targets to [t0, tf]
    t0 = t_ref[0]
    tf = t_ref[-1]
    if tf == t0:
        raise ValueError("Pose timestamps are constant")
    ts = np.clip(target_ts_ns.astype(np.int64), t0, tf)

    # For each target, find surrounding indices
    idx_right = np.searchsorted(t_ref, ts, side='left')
    idx_left  = np.clip(idx_right - 1, 0, len(t_ref)-1)
    idx_right = np.clip(idx_right,     0, len(t_ref)-1)

    # avoid identical indices (degenerate): push apart
    same = (idx_left == idx_right)
    idx_right[same] = np.clip(idx_right[same] + 1, 0, len(t_ref)-1)
    idx_left[same]  = np.clip(idx_left[same] - 1, 0, len(t_ref)-1)

    tL = t_ref[idx_left].astype(np.float64)
    tR = t_ref[idx_right].astype(np.float64)
    w  = np.zeros_like(ts, dtype=np.float64)
    mask = (tR != tL)
    w[mask] = (ts[mask] - tL[mask]) / (tR[mask] - tL[mask])
    w = np.clip(w, 0.0, 1.0)

    # ----- Vectorized SLERP between quat[idx_left] and quat[idx_right] with weight w -----
    qL = quat[idx_left]    # (N,4)
    qR = quat[idx_right]   # (N,4)

    # Ensure shortest path
    dot = np.sum(qL * qR, axis=1)
    sign = np.where(dot < 0.0, -1.0, 1.0)
    qR = qR * sign[:, None]
    dot = np.abs(dot)

    # Avoid numerical issues
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    # Where angle is tiny, use LERP; else SLERP
    eps = 1e-8
    use_lerp = sin_theta < eps

    s0 = np.empty_like(w, dtype=float)
    s1 = np.empty_like(w, dtype=float)

    # SLERP weights
    idx = ~use_lerp
    s0[idx] = np.sin((1.0 - w[idx]) * theta[idx]) / sin_theta[idx]
    s1[idx] = np.sin(w[idx] * theta[idx]) / sin_theta[idx]

    # LERP fallback
    s0[use_lerp] = 1.0 - w[use_lerp]
    s1[use_lerp] = w[use_lerp]

    q_interp = (s0[:, None] * qL) + (s1[:, None] * qR)
    q_interp = q_interp / np.linalg.norm(q_interp, axis=1, keepdims=True)

    # Convert to rotation matrices
    R_W_D_list = R.from_quat(q_interp).as_matrix()

    # LERP translations
    pL = pos[idx_left]
    pR = pos[idx_right]
    t_W_D_list = (1.0 - w)[:, None] * pL + w[:, None] * pR

    return R_W_D_list, t_W_D_list


def compensate_wrench_batch(F_meas_S, tau_meas_S, R_S_W, params, g=9.81):
    # F_meas_S: (N,3), tau_meas_S: (N,3), R_S_W: (N,3,3)
    gW = np.array([0,0,-g], float)
    m    = params["m"]
    c_S  = params["c_S"]
    f0   = params["f0"]
    tau0 = params["tau0"]

    g_S = R_S_W @ gW                   # (N,3)
    Fg  = m * g_S                      # (N,3)
    taug = np.cross(np.broadcast_to(c_S, Fg.shape), Fg)

    F_contact   = F_meas_S  - f0   - Fg
    tau_contact = tau_meas_S - tau0 - taug
    return F_contact, tau_contact


def find_contact_free_segments(
    timestamps_ns: np.ndarray,   # (N,)
    F_meas_S: np.ndarray,        # (N,3)
    tau_meas_S: np.ndarray,      # (N,3)
    R_S_W: np.ndarray,           # (N,3,3)  world->sensor
    *,
    m: float,                    # known (or from prior offline est.)
    c_S: np.ndarray,             # (3,) known CoG in sensor frame
    g: float = 9.81,
    use_torque: bool = True,     # True: include torque residuals in the score
    smooth_len: int = 15,        # odd; moving median length (samples)
    k_thresh: float = 1.0,       # robust threshold multiplier (MAD)
    min_free_sec: float = 10.0,  # minimum window duration to keep (before erosion)
    erode_sec: float = 0.0,      # erode each free window by this much on both sides
) -> Tuple[List[Tuple[int,int]], np.ndarray, Dict[str,Any]]:
    """
    Contact-free detector with optional erosion:
      - Enforces min_free_sec first
      - Then erodes each window by erode_sec on both sides
    """
    N = len(timestamps_ns)
    assert F_meas_S.shape == (N,3) and tau_meas_S.shape == (N,3) and R_S_W.shape == (N,3,3)
    t = timestamps_ns.astype(np.int64)

    # 1) Gravity terms in sensor frame
    gW = np.array([0,0,-g], float)
    g_S = (R_S_W @ gW).reshape(N,3)           # (N,3)
    Fg  = m * g_S
    taug= np.cross(np.broadcast_to(c_S, Fg.shape), Fg)  # (N,3)

    # 2) Global bias
    f0   = np.median(F_meas_S - Fg, axis=0)
    tau0 = np.median(tau_meas_S - taug, axis=0)

    # 3) Residuals
    F_res   = F_meas_S  - (f0 + Fg)
    tau_res = tau_meas_S - (tau0 + taug)

    # scalar score per-sample
    if use_torque:
        L = 0.1
        r = np.sqrt(np.sum(F_res**2, axis=1) + np.sum((tau_res / L)**2, axis=1))
    else:
        r = np.linalg.norm(F_res, axis=1)

    # 4) Smooth
    if smooth_len > 1:
        k = max(1, int(smooth_len) | 1)
        pad = k // 2
        r_pad = np.pad(r, (pad, pad), mode="edge")
        r_s  = np.empty_like(r)
        for i in range(N):
            r_s[i] = np.median(r_pad[i:i+k])
    else:
        r_s = r

    # 5) Threshold
    med = float(np.median(r_s))
    mad = float(np.median(np.abs(r_s - med)) or 1e-9)
    thr = med + k_thresh * 1.4826 * mad
    free_mask = r_s <= thr

    # 6) Merge to windows
    Ts = float(np.median(np.diff(t))) / 1e9 if N > 1 else 0.01
    windows: List[Tuple[int,int]] = []
    i = 0
    while i < N:
        if free_mask[i]:
            j = i + 1
            while j < N and free_mask[j]:
                j += 1
            t0 = int(t[i])
            t1 = int(t[j-1] + max(1, int(round(Ts*1e9))))

            # enforce min duration BEFORE erosion
            if (t1 - t0)/1e9 >= min_free_sec:
                if erode_sec > 0:
                    margin = int(round(erode_sec / Ts))
                    if (j - i) > 2*margin:
                        i_e = i + margin
                        j_e = j - margin
                        t0 = int(t[i_e])
                        t1 = int(t[j_e-1] + max(1, int(round(Ts*1e9))))
                    else:
                        # window too short after erosion → drop
                        i = j
                        continue
                windows.append((t0, t1))
            i = j
        else:
            i += 1

    debug = dict(
        f0=f0, tau0=tau0, thr=thr, med=med, mad=mad,
        r=r, r_s=r_s, Ts_est_s=Ts,
    )
    return windows, free_mask, debug

def calibrate_gripper_motor_and_kinematics(
        rosbag_path: Path | str,
        temp_path: Path | str,
        single_eta: bool = False,
):


    if isinstance(rosbag_path, str):
        rosbag_path = Path(rosbag_path)
    if isinstance(temp_path, str):
        temp_path = Path(temp_path)
  
    if not rosbag_path.exists():
        raise FileNotFoundError(f"ROS bag file not found: {rosbag_path}")
    
    if not temp_path.exists():
        temp_path.mkdir(parents=True, exist_ok=True)


    temp_path_rosbag = temp_path / "rosbag"
    temp_path_rosbag.mkdir(parents=True, exist_ok=True)

    # Extract frames and timestamps from ROS bag
    force_torque_topic = "/force_torque/ft_sensor0/ft_sensor_readings/wrench"
    motor_state_topic = "/dynamixel_workbench/joint_states"

    if not any(temp_path_rosbag.glob("*")):
        get_topics_from_bag(
            image_topics=["/digit/left/image_raw",
                          "/digit/right/image_raw"],
            non_image_topics={force_torque_topic: "geometry_msgs/WrenchStamped",
                              motor_state_topic: "sensor_msgs/JointState"},
            bag_path=rosbag_path,
            out_dir=temp_path_rosbag
        )

    force_torque_df = pd.read_csv(temp_path_rosbag / force_torque_topic.strip("/") / "data.csv")
    motor_state_df = pd.read_csv(temp_path_rosbag / motor_state_topic.strip("/") / "data.csv")

    # Initialize gripper model for kinematics and motor conversions
    gripper_model = GripperModel()

    df_ft = force_torque_df # ca 100 Hz
    df_ft = df_ft[["timestamp", "wrench.force.x",
                    "wrench.force.y", "wrench.force.z",
                    "wrench.torque.x", "wrench.torque.y", "wrench.torque.z"]]

    df_ms = motor_state_df # ca 60 Hz
    df_ms = df_ms[["timestamp", "position.0", "effort.0", "velocity.0"]] 


    # add other columns 
    df_ms["alpha.rad"] = df_ms["position.0"] + np.deg2rad(180 - gripper_model.TAU)  # in degrees, convert from rad and offset
    df_ms["x.single"] = gripper_model.x_of_alpha(df_ms["alpha.rad"].to_numpy())   # x per side
    df_ms["gap"] = 2.0 * df_ms["x.single"]  
    #jacobian to map torques to forces
    df_ms["dg_dalpha"] = gripper_model.dg_dalpha(df_ms["alpha.rad"].to_numpy())  # m/rad
    df_ms["torque.0"] = gripper_model.current_to_torque(df_ms["effort.0"])  # Nm, motor torque estimate
    df_ms["Fc.per_finger"] = np.abs(df_ms["torque.0"]) / np.maximum(np.abs(df_ms["dg_dalpha"]), gripper_model.eps)

    # merge the two dataframes with asof merge, nearest neighbor within tolerance
    # to get FT data at motor state timestamps
    tol_ns = int(5e6) # 5 ms tolerance for asof merge
    df = pd.merge_asof(
        df_ms.sort_values("timestamp"),
        df_ft.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=tol_ns)

    # drop rows with no FT data within tolerance
    ft_cols = ["wrench.force.x", "wrench.force.y", "wrench.force.z",
               "wrench.torque.x", "wrench.torque.y", "wrench.torque.z"]  
    df = df.dropna(subset=ft_cols)

    mask_nc = (np.abs(df["velocity.0"]) < 0.1) & (np.abs(df["effort.0"]) < 30)
    ft_cols_F = ["wrench.force.x", "wrench.force.y", "wrench.force.z"]
    ft_cols_T = ["wrench.torque.x","wrench.torque.y","wrench.torque.z"]

    # use median for robustness
    bias_F = df.loc[mask_nc, ft_cols_F].median()
    bias_T = df.loc[mask_nc, ft_cols_T].median()

    # subtract bias from the entire run
    df.loc[:, ft_cols_F] = df.loc[:, ft_cols_F].subtract(bias_F, axis=1)
    df.loc[:, ft_cols_T] = df.loc[:, ft_cols_T].subtract(bias_T, axis=1)

    # filter out data where gripper is not sufficiently closed or moving too fast
    thresh = (gripper_model.THRESH_PARTIALLY_CLOSED + gripper_model.TAU - 180) * np.pi / 180
    # df = df[df["position.0"] > thresh]
    # df = df[np.abs(df["velocity.0"]) < 0.3] 
    # df = df[np.abs(df["effort.0"]) > 10]  # avoid low-effort region with poor SNR


    # measured force magnitude (N), half for single gripper finger
    Fz = df["wrench.force.z"].to_numpy()
    Fc_meas = 0.5 * np.abs(Fz)       # N

    # kinematics Jacobian
    J = df["dg_dalpha"].to_numpy()

    # regressor: motor torque (already in N·m) -> rocker torque
    tau = df["torque.0"].to_numpy()

    # target: J * Fc_meas (N·m)
    y = np.abs(J) * Fc_meas
    X = tau 

    # Use magnitudes for the fit
    X_fit = np.abs(X)
    y_fit = np.abs(y)
    den = float(np.dot(X_fit, X_fit))
    if den == 0:
        raise RuntimeError("Zero denominator in η fit.")
    eta = float(np.dot(X_fit, y_fit) / den)
    yhat = eta * X_fit  # for plotting the torque-domain magnitude
    rmse = float(np.sqrt(np.mean((y_fit - yhat)**2)))

    # ----- binned η(I) -----
    I = df["effort.0"].to_numpy().astype(float)  # mA
    I_min, I_max, nbins = -200.0, 1200.0, 20
    edges   = np.linspace(I_min, I_max, nbins+1)
    centers = 0.5*(edges[:-1] + edges[1:])
    eta_bins = np.empty(nbins, float)

    # recommend magnitudes for plate test (avoids sign issues)
    X_abs = np.abs(X)
    y_abs = np.abs(y)

    for k in range(nbins):
        mask = (I >= edges[k]) & (I < edges[k+1] if k < nbins-1 else I <= edges[k+1])
        if mask.sum() >= 5 and float(np.dot(X_abs[mask], X_abs[mask])) > 0:
            eta_bins[k] = float(np.dot(X_abs[mask], y_abs[mask]) /
                                np.dot(X_abs[mask], X_abs[mask]))
        else:
            eta_bins[k] = np.nan
    # fill any empty bins by interpolation; clip to plausible range
    valid = np.isfinite(eta_bins)
    eta_bins[~valid] = np.interp(centers[~valid], centers[valid], eta_bins[valid])
    eta_bins = np.clip(eta_bins, 0.0, 1.5)

    def eta_of_current(mA):
        return np.interp(mA, centers, eta_bins, left=eta_bins[0], right=eta_bins[-1])

    # --- use binned η(I) to predict ---
    eta_binned = eta_of_current(I)                  # per-sample η from effort [mA]
    yhat_binned = eta_binned * np.abs(X)            # torque domain prediction (N·m)

    # per-finger clamp prediction with binned η(I)
    Fc_pred_binned = (eta_binned * np.abs(tau)) / np.maximum(np.abs(J), 1e-9)

    # quick metrics
    rmse_binned = float(np.sqrt(np.mean((np.abs(y) - yhat_binned)**2)))
    print(f"single-η RMSE={rmse:.3f} N·m   binned-η RMSE={rmse_binned:.3f} N·m")

    t = (df["timestamp"].to_numpy() - df["timestamp"].iloc[0]) * 1e-9  # if in ns


    print("median |tau| [N·m]:", np.median(X_abs))
    print("median |J| [m/rad]:", np.median(np.abs(J)))
    print("median Fc_meas [N]:", np.median(Fc_meas))
    print("rough η ≈ med(|J|*Fc)/med(|tau|):", np.median(y_abs) / np.median(X_abs))

    import json
    eta_model = {
        "edges_mA": edges.tolist(),        # length = nbins+1
        "eta_bins": eta_bins.tolist(),     # length = nbins
        "version": 1,
    }
    (out_dir := temp_path / "calibration").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "eta_binned.json", "w") as f:
        json.dump(eta_model, f, indent=2)
    print(f"[calibration] saved binned η to {out_dir/'eta_binned.json'}")


    t = (df_ms["timestamp"].to_numpy() - df_ms["timestamp"].iloc[0]) * 1e-9  # ns → s

    plt.figure(figsize=(12,6))

    # --- subplot 1: motor current/effort ---
    plt.subplot(2,1,1)
    plt.plot(t, df_ms["effort.0"], label="effort raw", alpha=0.5)
    if "effort.0_filt" in df_ms:
        plt.plot(t, df_ms["effort.0_filt"], label="effort filt", lw=2)
    plt.ylabel("effort [mA]")
    plt.legend()
    plt.grid(True)

    # --- subplot 2: motor position ---
    plt.subplot(2,1,2)
    plt.plot(t, df_ms["position.0"], label="position raw", alpha=0.5)
    if "position.0_filt" in df_ms:
        plt.plot(t, df_ms["position.0_filt"], label="position filt", lw=2)
    plt.xlabel("time [s]")
    plt.ylabel("position [rad]")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


    t = (df["timestamp"].to_numpy() - df["timestamp"].iloc[0]) * 1e-9  # ns → s

    plt.figure(figsize=(12, 8))

    # plot measured vs predicted torque demand
    plt.subplot(3, 1, 1)
    plt.plot(t, y, label="J * Fc_meas (target)", lw=2)
    plt.plot(t, yhat, label=f"η * τ (fit, η={eta:.2f})", lw=2)
    plt.ylabel("Torque [N·m]")
    plt.legend()
    plt.grid(True)

    # plot raw clamp forces
    plt.subplot(3, 1, 2)
    plt.plot(t, Fc_meas, label="Fc_meas (per finger)", lw=2)
    plt.plot(t, df["Fc.per_finger"], label="Fc_pred (model)", lw=2)
    plt.ylabel("Clamp [N]")
    plt.legend()
    plt.grid(True)

    # plot motor torque
    plt.subplot(3, 1, 3)
    plt.plot(t, tau, label="τ motor", lw=2)
    plt.xlabel("time [s]")
    plt.ylabel("Torque [N·m]")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    t = (df["timestamp"].to_numpy() - df["timestamp"].iloc[0]) * 1e-9

    plt.figure(figsize=(12, 10))

    # 1) torque balance
    plt.subplot(4,1,1)
    plt.plot(t, y, label="J*Fc_meas", lw=2)
    plt.plot(t, yhat, label=f"η*τ (single η={eta:.2f})", lw=2, alpha=0.8)
    plt.plot(t, yhat_binned, "--", label="η(I)*τ (binned)", lw=2)
    plt.ylabel("Torque [N·m]"); plt.legend(); plt.grid(True)

    # 2) clamp force
    plt.subplot(4,1,2)
    plt.plot(t, Fc_meas, label="Fc_meas (per finger)", lw=2)
    plt.plot(t, df["Fc.per_finger"], label="Fc_pred (single η)", lw=2, alpha=0.8)
    plt.plot(t, Fc_pred_binned, "--", label="Fc_pred (binned η(I))", lw=2)
    plt.ylabel("Clamp [N]"); plt.legend(); plt.grid(True)

    # 3) effort trace with bin lines
    plt.subplot(4,1,3)
    plt.plot(t, I, lw=1.5, label="effort [mA]")
    for e in edges: plt.axhline(e, color="k", alpha=0.12, lw=0.8)
    plt.ylabel("effort [mA]"); plt.legend(); plt.grid(True)

    # 4) η(I): bin medians + interpolant
    plt.subplot(4,1,4)
    I_grid = np.linspace(I_min, I_max, 400)
    plt.plot(centers, eta_bins, "o", label="η bin medians")
    plt.plot(I_grid, np.interp(I_grid, centers, eta_bins), "-", label="η(I) interp")
    plt.xlabel("effort [mA]"); plt.ylabel("η"); plt.grid(True); plt.legend()

    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    # Example usage
    # mp4_path = Path("/exchange/calib/calib_yellow.MP4")
    # bag_output_path = Path("/exchange/calib/calib_yellow.bag")

    # mp4_to_rosbag(mp4_path, bag_output_path)
    # print(f"Converted {mp4_path} to ROS bag at {bag_output_path}")

    # Example usage for merging VRS and ROS bag calibration
    # vrs_path = Path("/exchange/calib/calib_gripper_blue/calib_250827_2.vrs")
    # rosbag_path = Path("/exchange/calib/calib_gripper_blue/calib_2025-08-27_13-08-22.bag")
    # temp_path = Path("/exchange/calib/calib_gripper_blue/temp")
    # merge_calibration_vrs_and_calibration_bag(vrs_path, rosbag_path, temp_path)

    # mps_file = Path("/exchange/calib/gripper_yellow_gravity_compensation/mps_calib_2025-07-10_2_vrs/slam/closed_loop_trajectory.csv")
    # vrs_path = Path("/exchange/calib/gripper_yellow_gravity_compensation/calib_2025-07-10_2.vrs")
    # rosbag_path = Path("/exchange/calib/gripper_yellow_gravity_compensation/calib_2025-07-10_14-54-50.bag")
    # temp_path = Path("/exchange/calib/gripper_yellow_gravity_compensation/temp")
    # camcahin_imucam = Path("/exchange/calib/gripper_yellow_gravity_compensation/merged_calibration-camchain-imucam.yaml")
    # forceless_time_intervall = [1752159291893020650, 1752159352194227650]
    # estimate_tool_params(vrs_path, mps_file, rosbag_path, temp_path, camcahin_imucam, forceless_time_intervall)
    # print(f"Estimated tool parameters from {vrs_path}, {mps_file}, and {rosbag_path}")
    
    # rosbag_path = Path("/exchange/calib/motor_calib/motor_calib_2025-09-18_08-09-30.bag")
    # temp_path = Path("/exchange/calib/motor_calib/temp")
    # calibrate_gripper_motor_and_kinematics(rosbag_path, temp_path)


    res = estimate_tool_params_dynamic(
        vrs_path=Path("/data/ikea_recordings/raw/dynamic_frontend_id_2/dynamic_frontend_id_2.vrs"),
        mps_file=Path("/data/ikea_recordings/raw/dynamic_frontend_id_2/mps_dynamic_frontend_id_2_vrs/slam/closed_loop_trajectory.csv"),
        rosbag_path=Path("/data/ikea_recordings/raw/dynamic_frontend_id_2/dynamic_frontend_id_2026-04-20_11-31-57.bag"),
        temp_path=Path("/data/ikea_recordings/raw/dynamic_frontend_id_2/temp"),
        camchain_imucam=Path("/data/ikea_recordings/raw/calib/gripper_blue/merged_calibration-camchain-imucam.yaml"),
        visualize_aria_velocities=False,
        epsilon_quasistatic_linear=0.1,
        epsilon_quasistatic_angular=0.2,
        evaluate_candidate_smoothing=True,
    )
    
    # res = evaluate_saved_dynamic_compensation_on_recording(
    #     config_json="/data/ikea_recordings/raw/dynamic_frontend_id_2/temp/tool_params_dynamic_estimate.json",
    #     vrs_path="/data/ikea_recordings/raw/dynamic_frontend_id_1/dynamic_frontend_id.vrs",
    #     mps_file="/data/ikea_recordings/raw/dynamic_frontend_id_1/mps_dynamic_frontend_id_vrs/slam/closed_loop_trajectory.csv",
    #     rosbag_path="/data/ikea_recordings/raw/dynamic_frontend_id_1/dynamic_frontend_id_2026-04-20_08-32-19.bag",
    #     temp_path="/data/ikea_recordings/raw/dynamic_frontend_id_1/temp",
    #     camchain_imucam="/data/ikea_recordings/raw/calib/gripper_blue/merged_calibration-camchain-imucam.yaml",
    #     recompute_bias_for_recording=True,
    # )
    
    a =2 
