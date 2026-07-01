import os
import rosbag
import cv2
import pandas as pd
from tqdm import tqdm
from cv_bridge import CvBridge

# --- CONFIG ---
bag_path = "/bags/dlab_testing_2_2025-04-08-15-06-43.bag"
output_dir = "/exchange/extracted_dataset/episode_001"
os.makedirs(output_dir, exist_ok=True)
for sub in ["digit_left", "digit_right", "zed_left", "zed_right"]:
    os.makedirs(f"{output_dir}/{sub}", exist_ok=True)

bridge = CvBridge()

# --- MESSAGE BUFFERS ---
digit_left_msgs, digit_right_msgs = [], []
zed_left_msgs, zed_right_msgs = [], []
force_msgs, joint_msgs, imu_msgs = [], [], []

# --- STEP 1: Load messages ---
print("🔍 Reading messages from bag...")
with rosbag.Bag(bag_path, 'r') as bag:
    for topic, msg, t in tqdm(bag.read_messages(), total=68376):
        ts = msg.header.stamp.to_sec() if hasattr(msg, 'header') else t.to_sec()
        if topic == "/digit/left/image_raw":
            digit_left_msgs.append((ts, msg))
        elif topic == "/digit/right/image_raw":
            digit_right_msgs.append((ts, msg))
        elif topic == "/zed/left/image_raw":
            zed_left_msgs.append((ts, msg))
        elif topic == "/zed/right/image_raw":
            zed_right_msgs.append((ts, msg))
        elif topic == "/gripper_force_trigger":
            force_msgs.append((ts, msg.data))
        elif topic == "/joint_states":
            pos = msg.position[0] if msg.position else 0.0
            effort = msg.effort[0] if msg.effort else 0.0
            joint_msgs.append((ts, pos, effort))
        elif topic == "/zed/imu/data_raw":
            imu_msgs.append((ts,
                msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
                msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z))

# --- Helper: Closest timestamp match ---
def find_closest(ts, messages, max_diff=0.2):
    if not messages:
        return None
    closest = min(messages, key=lambda x: abs(x[0] - ts))
    return closest if abs(closest[0] - ts) <= max_diff else None

# --- STEP 2: Synchronize + Save ---
print("📦 Synchronizing and extracting...")
synced_records = []
skipped_count = 0

for ts, left_msg in tqdm(digit_left_msgs, desc="Processing left_digit"):
    right = find_closest(ts, digit_right_msgs)
    zed_l = find_closest(ts, zed_left_msgs)
    zed_r = find_closest(ts, zed_right_msgs)
    force = find_closest(ts, force_msgs)
    joint = find_closest(ts, joint_msgs)

    if not all([right, zed_l, zed_r, force, joint]):
        print(f"[!] Skipping frame @ {ts:.3f}s: "
              f"right={bool(right)}, zed_left={bool(zed_l)}, zed_right={bool(zed_r)}, "
              f"force={bool(force)}, joint={bool(joint)}")
        skipped_count += 1
        continue

    base_name = f"{ts:.6f}.png"

    cv2.imwrite(f"{output_dir}/digit_left/{base_name}", bridge.imgmsg_to_cv2(left_msg, "bgr8"))
    cv2.imwrite(f"{output_dir}/digit_right/{base_name}", bridge.imgmsg_to_cv2(right[1], "bgr8"))
    cv2.imwrite(f"{output_dir}/zed_left/{base_name}", bridge.imgmsg_to_cv2(zed_l[1], "bgr8"))
    cv2.imwrite(f"{output_dir}/zed_right/{base_name}", bridge.imgmsg_to_cv2(zed_r[1], "bgr8"))

    synced_records.append({
        "timestamp": ts,
        "digit_left_img": base_name,
        "digit_right_img": base_name,
        "zed_left_img": base_name,
        "zed_right_img": base_name,
        "force_trigger": force[1],
        "joint_position_rad": joint[1],
        "joint_effort_A": joint[2]
    })

# Save metadata
meta_df = pd.DataFrame(synced_records)
meta_df.to_csv(f"{output_dir}/metadata.csv", index=False)

# Save IMU separately
imu_df = pd.DataFrame(imu_msgs, columns=[
    "timestamp", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"
])
imu_df.to_csv(f"{output_dir}/imu_data.csv", index=False)

# --- Done ---
print(f"\n✅ DONE: Saved {len(synced_records)} synchronized samples")
print(f"⚠️ Skipped {skipped_count} due to missing data")
print(f"📁 Dataset saved in: {output_dir}")
