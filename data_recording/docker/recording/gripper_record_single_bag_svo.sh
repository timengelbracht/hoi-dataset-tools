#!/usr/bin/env bash
#
# Usage: ./record_svo_rosbag.sh <experiment_name> [--svo]
# Output: /ssd/data/<experiment_name>_<YYYY-MM-DD_HH-MM-SS>.bag  [+ .svo if --svo]

set -euo pipefail

#############################
# 1 – configuration
#############################
TARGET_DIR="/ssd/data"
mkdir -p "$TARGET_DIR"
export ZED_SDK_SVO_VERSION=1

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
    fi

    echo "[INFO] Cleanup complete."
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

#############################
# 4 – start recordings
#############################
echo "[INFO] Saving to: $FULL_NAME.{bag,$([[ $ENABLE_SVO == true ]] && echo svo)}"

if $ENABLE_SVO; then
    echo "[INFO] Waiting for /zedm/zed_node/start_svo_recording service…"
    until rosservice list | grep -q /zedm/zed_node/start_svo_recording; do
        sleep 0.5
    done

    echo "[INFO] Starting SVO recording…"
    rosservice call /zedm/zed_node/start_svo_recording \
    "{svo_filename: \"${FULL_NAME}.svo\"}" \
      && echo "[✓] SVO recording started." \
      || { echo "[✗] Failed to start SVO recording."; exit 1; }
fi

echo "[INFO] Starting rosbag recording…"
rosbag record -O "${FULL_NAME}.bag" \
    --chunksize=8192 \
    --buffsize=1024 \
    /digit/left/image_raw \
    /digit/right/image_raw \
    /gripper_force_trigger \
    /zedm/zed_node/imu/data \
    /zedm/zed_node/depth/depth_registered \
    /zedm/zed_node/left_raw/image_raw_color \
    /zedm/zed_node/right_raw/image_raw_color \
    /tf_static \
    /dynamixel_workbench/dynamixel_state \
    /dynamixel_workbench/joint_states \
    /force_torque/ft_sensor0/ft_sensor_readings/temperature \
    /force_torque/ft_sensor0/ft_sensor_readings/wrench \
    /force_torque/ft_sensor0/ft_sensor_readings/imu &
ROSBAG_PID=$!
wait "$ROSBAG_PID"
#/zedm/zed_node/imu/data_raw \
    #/zedm/zed_node/odom \
    #/zedm/zed_node/pose \
    #/zedm/zed_node/pose_with_covariance \
    #/tf \
#/diagnostics 
