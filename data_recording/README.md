# data_recording

The **hardware capture side** of Hoi!: recording data with the Hoi! gripper rig
(ZED Mini, force/torque, tactile, motor) on a Jetson. Output is raw
rosbags/SVOs that `../data_processing` then ingests.

> **Supported target: `recording_gripper_nano`** (in `docker/recording/`, built
> from `Dockerfile-nano`, ROS Noetic on a Jetson). The other services in
> `docker/recording/docker-compose.yml` (`testing`, `jetson`, `rpi/light`,
> `spot_agx`, `aria`) are variants/experiments — use the nano service unless you
> know you need another.

## Sensors on the rig
- **ZED Mini** — stereo RGB + depth + IMU (auto-detected, no serial to set)
- **2× GelSight DIGIT** — tactile (per-sensor ids)
- **Dynamixel** — gripper actuation (FTDI USB-serial)
- **Bota / Rokubimini** — 6-axis force/torque (EtherCAT)
- **Load cell + Arduino** — grasp-trigger force (rosserial)

## Layout
```
docker/recording/
  Dockerfile-nano                     # the supported Jetson image
  docker-compose.yml                  # services (use recording_gripper_nano)
  hardware.env                        # ← per-rig identifiers you EDIT
  start_recording_interface_gripper.sh# one-command tmux recording UI
  gripper_launch_single_force.launch  # ROS launch (generic; overridden by hardware.env)
  gripper_record_single_bag_svo.sh    # the rosbag/SVO recorder (start_recording.sh)
  zedm.yaml zed2i.yaml zed_common.yaml basic.yaml
arduino/                              # load-cell firmware + calibration (see its README)
ros/                                  # lenai_description.tar.xz (gripper URDF)
```

## Setup (once per Jetson)

**1. Assemble the gripper** — 3D-printed body, Dynamixel motor, load cell, Bota
F/T sensor, ZED Mini, 2× DIGIT. (CAD / bill-of-materials: TODO — provided
separately.)

**2. Load cell** — flash the firmware and calibrate the scale factor following
[`arduino/README.md`](arduino/README.md).

**3. Clone this repo on the Jetson:**
```bash
git clone <this-repo> hoi-dataset-tools
cd hoi-dataset-tools/data_recording/docker/recording
```

**4. Set your hardware identifiers** — edit `hardware.env`. Every value differs
per rig; the file documents how to find each:
```bash
DIGIT_LEFT_ID / DIGIT_RIGHT_ID   # printed on each DIGIT
DXL_PORT / LOADCELL_PORT          # ls /dev/serial/by-id/
MIN_TICKS / MAX_TICKS             # from the gripper motor calibration
ETHERCAT_BUS                      # ip link  (NIC wired to the F/T sensor)
```
Also review the host paths in `docker-compose.yml` (e.g. the `/ssd` recordings
volume) for your machine.

**5. Build the image:**
```bash
docker compose build recording_gripper_nano
```
This clones + builds the ROS workspace (zed-ros-wrapper, `gelsight_digit_ros`,
`gripper_force_controller`, dynamixel-workbench, Bota `bota_driver`). Note it
pulls the `timengelbracht/*` forks — they must be reachable.

## Recording
From `docker/recording/`:
```bash
./start_recording_interface_gripper.sh <env_name>      # e.g. kitchen_1
```
This brings up the container and opens a tmux 2×2:
- **pane 0** — `roslaunch …` starts all sensors (with your `hardware.env` values)
- **pane 1** — pre-types `./start_recording.sh <env_name>`; **press Enter** to start recording once the sensors are up
- **pane 2** — `rostopic echo` of the ZED left image (liveness check)
- **pane 3** — `jtop` (Jetson resource monitor)

Stop recording with `Ctrl-C` in pane 1 (it shuts down cleanly). Bags land in
`/ssd/data/<env_name>_<timestamp>.bag` (add `--svo` in `start_recording.sh` for
a ZED SVO alongside).

### Recorded topics
DIGIT L/R images, `/gripper_force_trigger`, ZED left/right raw + depth + IMU,
Dynamixel state/joint_states, Bota F/T wrench/temperature/imu, `/tf_static`.

## Notes
- Bind-mount paths in `docker-compose.yml` (`/ssd`, data volumes) are
  machine-specific — adjust for your host.
- Camera/IMU calibration is handled in `../calibration`.
