#!/usr/bin/env bash
SESSION="gripper_recording"
CONTAINER="gripper_recording_nano"

# Recording / environment name (e.g. kitchen_1). Pass as the first argument.
RECORDING_NAME="${1:-recording}"

# Load per-rig hardware identifiers (DIGIT ids, USB serials, tick limits, F/T bus).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$SCRIPT_DIR/hardware.env" ]]; then
    echo "[ERROR] $SCRIPT_DIR/hardware.env not found — edit it for your rig first." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$SCRIPT_DIR/hardware.env"

# roslaunch invocation with per-rig hardware overrides (from hardware.env)
ROSLAUNCH_CMD="roslaunch gripper_force_controller gripper_launch_single_force.launch \
digit_left_id:=${DIGIT_LEFT_ID} digit_right_id:=${DIGIT_RIGHT_ID} \
dxl_device:=${DXL_PORT} usb_port:=${DXL_PORT} serial_port:=${LOADCELL_PORT} \
min_ticks:=${MIN_TICKS} max_ticks:=${MAX_TICKS} ethercat_bus:=${ETHERCAT_BUS}"

# 1) Bring up the container
docker compose up -d recording_gripper_nano

# 2) Start a new tmux session (detached), initial host shell in pane 0
tmux new-session -d -s "$SESSION"

# 3) Split to get a 2×2 grid of host panes:
#    - split horizontally to make pane 1
tmux split-window -h -t "$SESSION":0.0
#    - split pane 0 vertically to make pane 2
tmux split-window -v -t "$SESSION":0.0
#    - split pane 1 vertically to make pane 3
tmux split-window -v -t "$SESSION":0.1

# (Now you have panes 0,1 in top row and 2,3 in bottom row—all host shells.)

# 4) Pane 0 (top-left) → container + clear + type roslaunch
tmux send-keys -t "$SESSION":0.0 "docker exec -it $CONTAINER bash" C-m
sleep 0.1
tmux send-keys -t "$SESSION":0.0 "clear" C-m
sleep 0.05
tmux send-keys -t "$SESSION":0.0 "$ROSLAUNCH_CMD" C-m

# 5) Pane 1 (top-right) → container + clear + type recording
tmux send-keys -t "$SESSION":0.1 "docker exec -it $CONTAINER bash" C-m
sleep 0.1
tmux send-keys -t "$SESSION":0.1 "clear" C-m
sleep 0.05
tmux send-keys -t "$SESSION":0.1 -l "./start_recording.sh $RECORDING_NAME"

# 6) Pane 2 (bottom-left) → container + clear + type rostopic echo
tmux send-keys -t "$SESSION":0.2 "docker exec -it $CONTAINER bash" C-m
sleep 0.1
tmux send-keys -t "$SESSION":0.2 "clear" C-m
sleep 0.05
tmux send-keys -t "$SESSION":0.2 "rostopic echo /zedm/zed_node/left_raw/image_raw_color" C-m

# 7) Pane 3 (bottom-right) → host + clear + type jtop
tmux send-keys -t "$SESSION":0.3 "clear" C-m
sleep 0.05
tmux send-keys -t "$SESSION":0.3 "jtop" C-m

# 8) (Re-tile just in case)
tmux select-layout -t "$SESSION":0 tiled

# 9) Attach to the session
tmux attach -t "$SESSION":0
