#!/usr/bin/env bash
#
# Usage: ./record_zed_spot_rosbag.sh <experiment_name> [--svo]
# Output: /ssd/data/<experiment_name>_<YYYY-MM-DD_HH-MM-SS>.bag  [+ .svo if --svo]

set -euo pipefail

#############################
# 1 – paths / config
#############################
TARGET_DIR="/ssd/data"
mkdir -p "$TARGET_DIR"
export ZED_SDK_SVO_VERSION=1    # keep SVO v1 like your first script

#############################
# 2 – argument parsing
#############################
if [[ $# -lt 1 ]]; then
    echo "[ERROR] Please provide a base name for the recording."
    echo "Usage: $0 <experiment_name> [--svo]"
    exit 1
fi

NAME="$1"
ENABLE_SVO=false
if [[ "${2:-}" == "--svo" ]]; then
    ENABLE_SVO=true
fi

TIMESTAMP=$(date +%F_%H-%M-%S)
FULL_NAME="${TARGET_DIR}/${NAME}_${TIMESTAMP}"

#############################
# 3 – graceful shutdown
#############################
cleanup() {
    echo -e "\n[INFO] Caught exit signal. Cleaning up…"

    if [[ -n "${ROSBAG_PID:-}" ]]; then
        echo "[INFO] Stopping rosbag (PID $ROSBAG_PID)…"
        kill "$ROSBAG_PID" 2>/dev/null || true
        wait "$ROSBAG_PID" 2>/dev/null || true
    fi

    if $ENABLE_SVO; then
        echo "[INFO] Stopping SVO recording…"
        rosservice call /zedm/zed_node/stop_svo_recording \
            || echo "[WARN] Failed to stop SVO recording."
        rosservice call /zed2i/zed_node/stop_svo_recording \
            || echo "[WARN] Failed to stop SVO recording (zed2i)."
    fi

    echo "[INFO] Cleanup complete."
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

#############################
# 4 – topic sets
#############################
# Common helper: list of ZED* base topics (left/right/rgb already include color+gray/raw in wrapper defaults)
ZED_TOPICS=(
  # -- IMU, pose, odom
  "/%s/zed_node/imu/data_raw"
  "/%s/zed_node/imu/data"
  "/%s/zed_node/odom"
  "/%s/zed_node/pose"
  "/%s/zed_node/pose_with_covariance"
  # -- images / depth
  "/%s/zed_node/left_raw/image_raw_color"
  "/%s/zed_node/right_raw/image_raw_color"
  "/%s/zed_node/depth/depth_registered"
  "/%s/zed_node/confidence/confidence_map"
  # -- camera infos
  "/%s/zed_node/left_raw/camera_info"
  "/%s/zed_node/right_raw/camera_info"
  "/%s/zed_node/depth/camera_info"
)

declare -a ZEDM_TOPICS
declare -a ZED2I_TOPICS
for t in "${ZED_TOPICS[@]}"; do
    ZEDM_TOPICS+=("$(printf "$t" zedm)")
    ZED2I_TOPICS+=("$(printf "$t" zed2i)")
done

# Spot topics (add/remove as needed)
SPOT_TOPICS=(
  /spot/odometry
  /spot/odometry/twist
  /spot/odometry_corrected
  /spot/lidar/points
  /spot/status/battery_states
  /spot/status/system_faults
  /spot/status/behavior_faults
  /spot/status/estop
  /spot/status/power_state
  /spot/camera/back/image
  /spot/camera/frontleft/image
  /spot/camera/frontright/image
  /spot/camera/left/image
  /spot/camera/right/image
  /spot/camera/hand_color/image
  /spot/camera/hand_mono/image
  /spot/camera/back/camera_info
  /spot/camera/frontleft/camera_info
  /spot/camera/frontright/camera_info
  /spot/camera/left/camera_info
  /spot/camera/right/camera_info
  /spot/camera/hand_color/camera_info
  /spot/camera/hand_mono/camera_info
)

#############################
# 5 – (optional) start SVO recordings
#############################
if $ENABLE_SVO; then
    echo "[INFO] Waiting for ZED SVO services…"
    until rosservice list | grep -q /zedm/zed_node/start_svo_recording; do sleep 0.5; done
    until rosservice list | grep -q /zed2i/zed_node/start_svo_recording; do sleep 0.5; done

    echo "[INFO] Starting SVO on zedm and zed2i…"
    rosservice call /zedm/zed_node/start_svo_recording "{svo_filename: \"${FULL_NAME}_zedm.svo\"}" \
      && echo "[✓] zedm SVO started."
    rosservice call /zed2i/zed_node/start_svo_recording "{svo_filename: \"${FULL_NAME}_zed2i.svo\"}" \
      && echo "[✓] zed2i SVO started."
fi

#############################
# 6 – start rosbag
#############################
echo "[INFO] Recording to ${FULL_NAME}.bag"
rosbag record -O "${FULL_NAME}.bag" \
    --chunksize=8192 \
    --buffsize=104857600 \
    /tf /tf_static \
    "${ZEDM_TOPICS[@]}" \
    "${ZED2I_TOPICS[@]}" \
    "${SPOT_TOPICS[@]}" &

ROSBAG_PID=$!
wait "$ROSBAG_PID"
