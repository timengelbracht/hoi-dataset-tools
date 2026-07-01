#!/usr/bin/env python3
import os
import cv2
import rosbag
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from tqdm import tqdm

# --- CONFIG ---
bag_path = "/bags/dlab_testing_2_2025-04-08-15-06-43.bag"  # ← CHANGE THIS
topic = "/zed/left/image_raw"  # ← CHANGE THIS
output_video = "/bags/zed_left.mp4"
fps = 60  # ← Adjust to your topic's rate

# --- Setup ---
bridge = CvBridge()
images = []

print(f"📦 Reading images from: {topic}")
with rosbag.Bag(bag_path, 'r') as bag:
    for topic_name, msg, t in tqdm(bag.read_messages(topics=[topic])):
        try:
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            images.append(cv_img)
        except Exception as e:
            print(f"[!] Skipping a frame: {e}")

# --- Write video ---
if not images:
    print("❌ No images found in topic.")
    exit()

height, width, _ = images[0].shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

print(f"🎥 Writing video to: {output_video}")
for img in images:
    video.write(img)

video.release()
print("✅ Done.")
