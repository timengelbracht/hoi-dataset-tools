#!/usr/bin/env bash
SESSION="gripper_recording"
CONTAINER="gripper_recording_nano"
RECORDING_NAME="<ENV NAME (e.g. kitchen_1)>"

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
tmux send-keys -t "$SESSION":0.0 "roslaunch gripper_force_controller gripper_launch_single_force.launch" C-m

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
