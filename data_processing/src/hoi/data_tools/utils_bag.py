from typing import Dict, List, Any
from pathlib import Path
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg" if you have PyQt5 installed
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List, Any
from rosbags.highlevel import AnyReader
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore
import cv2
from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage
from tqdm import tqdm
import pandas as pd
import numpy as np
import open3d as o3d
from .utils_parsing import flatten_dict, ros_to_dict, ROS_MESSAGE_PARSING_CONFIG, ros_message_to_dict_recursive
from .data_indexer import RecordingIndex
from typing import Tuple, Iterator
import sys
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from rosbags.typesys import Stores, get_typestore
from rosbags.rosbag1 import Writer
from rosbags.serde import serialize_ros1


ts = get_typestore(Stores.ROS1_NOETIC)
Header     = ts.types['std_msgs/msg/Header']
Time       = ts.types['builtin_interfaces/msg/Time']
ImageMsg   = ts.types['sensor_msgs/msg/Image']
ImuMsg     = ts.types['sensor_msgs/msg/Imu']
Quaternion = ts.types['geometry_msgs/msg/Quaternion']
Vector3    = ts.types['geometry_msgs/msg/Vector3']


def get_topics_from_bag(image_topics: List[str],
                     non_image_topics: Dict[str, str],
                     bag_path: Path | str,
                     out_dir: Path | str, 
                     image_extension: str = ".jpg") -> None:
    
    if not isinstance(bag_path, Path):
        bag_path = Path(bag_path)

    if not bag_path or not bag_path.is_file():
        raise FileNotFoundError(f"Bag file not found: {bag_path}")

    print(f"[ROSBAGS] Reading from: {bag_path}")

    # This will now store lists of *nested dictionaries* for each topic
    parsed_non_image_data: Dict[str, List[Dict[str, Any]]] = \
        {topic: [] for topic in non_image_topics.keys()}

    with AnyReader([bag_path]) as reader:
        for conn, bag_time, raw in tqdm(reader.messages(),
                                        total=getattr(reader, "message_count", None)):
            topic = conn.topic
            
            # Filter for topics we explicitly listed
            if topic not in image_topics and topic not in non_image_topics.keys():
                continue

            msg = reader.deserialize(raw, conn.msgtype)

            # --- Universal Timestamp Extraction ---
            ts: int
            if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
                # ROS 2 message timestamps usually directly accessible
                if hasattr(msg.header.stamp, 'sec') and hasattr(msg.header.stamp, 'nanosec'):
                    ts = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                else: # Fallback for ROS 1 or other stamp types with to_nsec()
                        try:
                            ts = msg.header.stamp.to_nsec()
                        except AttributeError:
                            print(f"Warning: Could not get nsec from {topic} header.stamp. Falling back to bag_time.")
                            ts = bag_time.to_nsec() if hasattr(bag_time, "to_nsec") else int(bag_time)
            else: # No header, use bag message time
                ts = bag_time.to_nsec() if hasattr(bag_time, "to_nsec") else int(bag_time)
            
            ts_str = str(ts)

            # --- IMAGE TOPICS Handling ---
            if topic in image_topics:
                try:
                    img = message_to_cvimage(msg)
                    img_dir = out_dir / topic.strip("/")
                    img_dir.mkdir(parents=True, exist_ok=True)

                    if hasattr(msg, 'encoding') and msg.encoding == "32FC1": # Depth images
                        np.save(img_dir / f"{ts_str}.npy", img)

                        # Visualization for depth
                        vis_dir = out_dir / f"{topic.strip('/')}_visualization"
                        vis_dir.mkdir(parents=True, exist_ok=True)
                        
                        min_d, max_d = 0.0, 5.0
                        norm = np.clip(np.nan_to_num(img), min_d, max_d)
                        norm = ((norm - min_d) / (max_d - min_d) * 255).astype(np.uint8)
                        heat = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)
                        cv2.imwrite(vis_dir / f"{ts_str}{image_extension}", heat)
                    elif hasattr(msg, 'encoding') and msg.encoding == "bgra8": # BGRA images
                        cv2.imwrite(img_dir / f"{ts_str}{image_extension}", cv2.cvtColor(img, cv2.COLOR_BGRA2BGR))
                    else: # Default to BGR8 (common for color images)
                        cv2.imwrite(img_dir / f"{ts_str}{image_extension}", img)
                except Exception as e:
                    print(f"[!] Failed to decode image @ {ts} on {topic}: {e}")
                continue # Image topics are handled, move to the next message

            # --- NON-IMAGE TOPICS - Direct Parsing and Storage ---
            try:
                # Convert the ROS message object directly to a nested Python dictionary
                parsed_msg_dict = ros_message_to_dict_recursive(msg)
                parsed_msg_dict['timestamp'] = ts

                parsed_non_image_data[topic].append(parsed_msg_dict)
            except Exception as e:
                print(f"[!] Failed to parse and extract data from {topic}: {e}")

    # --- Dump Parsed Data to Structured Files ---
    # Instead of CSV, we'll save to JSON Lines or Parquet.
    for topic_name, messages_list in parsed_non_image_data.items():
        if not messages_list:
            print(f"[INFO] No data extracted for topic: {topic_name}. Skipping file dump.")
            continue

        output_dir = out_dir / topic_name.strip("/")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Flatten each dict in the list
            flat_messages = [flatten_dict(msg_dict) for msg_dict in messages_list]
            df = pd.DataFrame(flat_messages)
            csv_path = output_dir / "data.csv"
            df.to_csv(csv_path, index=False)
            print(f"[✓] Saved CSV: {csv_path}")
        except Exception as e:
            print(f"[!] Failed to save CSV for {topic_name}: {e}")



def gopro_to_bag(
    out_bag: Path | str,
    rgb_dir : Path | str | None = None,
    imu_csv : Path | str | None = None,
    img_stride: int   = 1,
    img_scale : float | None = None,
) -> None:
    """Convert GoPro JPG directory + IMU CSV → ROS1 bag.
       Either stream may be None (images-only or imu-only)."""

    out_bag = Path(out_bag)

    cam_topic = '/cam0/image_raw'; cam_frame = 'cam0'
    imu_topic = '/imu0';           imu_frame = 'imu0'

    # ───── Build iterators (images, imu) ──────────────────────────────
    if rgb_dir is not None:
        rgb_dir  = Path(rgb_dir)
        jpgs     = sorted(rgb_dir.glob('*.jpg'), key=lambda p: int(p.stem))[::img_stride]
        img_iter = enumerate(jpgs)
        img_seq, next_img = next(img_iter, (None, None))
    else:
        jpgs = []
        img_iter = iter(())
        img_seq  = next_img = None

    if imu_csv is not None:
        imu_df   = pd.read_csv(imu_csv)
        imu_iter = enumerate(imu_df.itertuples(index=False))
        imu_seq, next_imu = next(imu_iter, (None, None))
    else:
        imu_df   = pd.DataFrame()
        imu_iter = iter(())
        imu_seq  = next_imu = None

    if not jpgs and imu_df.empty:
        raise ValueError("Nothing to write: both rgb_dir and imu_csv are None/empty.")

    # ───── Write bag ──────────────────────────────────────────────────
    with Writer(out_bag) as bag:

        if jpgs:
            con_cam = bag.add_connection(cam_topic, ImageMsg.__msgtype__, typestore=ts)
        if not imu_df.empty:
            con_imu = bag.add_connection(imu_topic, ImuMsg.__msgtype__, typestore=ts)

        total_msgs = len(jpgs) + len(imu_df)
        for _ in tqdm(range(total_msgs), unit='msg', desc='Writing bag'):

            # ??? which message comes next chronologically?
            choose_img = (
                next_img is not None and
                (next_imu is None or int(next_img.stem) <= int(next_imu.timestamp))
            )

            if choose_img:
                t_ns = int(next_img.stem)
                bgr  = cv2.imread(str(next_img), cv2.IMREAD_COLOR)
                if bgr is not None:
                    if img_scale and img_scale != 1.0:
                        bgr = cv2.resize(bgr, None, fx=img_scale, fy=img_scale,
                                         interpolation=cv2.INTER_AREA)
                    msg = make_img_msg(bgr, t_ns, img_seq, frame_id=cam_frame)
                    bag.write(con_cam, t_ns,
                              serialize_ros1(msg, ImageMsg.__msgtype__, ts))
                img_seq, next_img = next(img_iter, (None, None))

            else:  # write IMU
                t_ns = int(next_imu.timestamp)
                msg  = make_imu_msg(next_imu, t_ns, imu_seq, frame_id=imu_frame)
                bag.write(con_imu, t_ns,
                          serialize_ros1(msg, ImuMsg.__msgtype__, ts))
                imu_seq, next_imu = next(imu_iter, (None, None))

    print(f"[✓] bag written: {out_bag} "
          f"({len(jpgs)} images | {len(imu_df)} imu msgs | stride={img_stride} | scale={img_scale})")


    
def make_img_msg(bgr, t_ns: int, seq: int, *, frame_id: str) -> ImageMsg:
    h, w  = bgr.shape[:2]
    stamp = Time(sec=t_ns // 1_000_000_000,
                 nanosec=t_ns % 1_000_000_000)
    header = Header(seq=seq, stamp=stamp, frame_id=frame_id)
    return ImageMsg(header=header, height=h, width=w,
                    encoding='bgr8', is_bigendian=0, step=3*w,
                    data=bgr.reshape(-1))

def make_imu_msg(row, t_ns: int, seq: int, *, frame_id: str) -> ImuMsg:
    stamp  = Time(sec=t_ns // 1_000_000_000,
                  nanosec=t_ns % 1_000_000_000)
    header = Header(seq=seq, stamp=stamp, frame_id=frame_id)
    zero9  = np.zeros(9, dtype=np.float64)
    return ImuMsg(
        header=header,
        orientation=Quaternion(x=0., y=0., z=0., w=1.),
        orientation_covariance=zero9,
        angular_velocity=Vector3(x=row.angular_vel_x,
                                 y=row.angular_vel_y,
                                 z=row.angular_vel_z),
        angular_velocity_covariance=zero9,
        linear_acceleration=Vector3(x=row.linear_accel_x,
                                    y=row.linear_accel_y,
                                    z=row.linear_accel_z),
        linear_acceleration_covariance=zero9,
    )




if __name__ == "__main__":
    # Example usage
    imu_file_path = Path("/exchange/calib/calib_yellow_pinhole-equi/imu/data.csv")
    rgb_dir = Path("/exchange/calib/calib_yellow_pinhole-equi/rgb")
    out_bag_path = Path("/exchange/calib/calib_yellow_pinhole-equi/data.bag")

    gopro_to_bag(rgb_dir=rgb_dir, imu_csv=imu_file_path, out_bag=out_bag_path, img_stride=2, img_scale=1.0)