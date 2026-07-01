# data_recording

The **hardware capture side** of Hoi!: everything used to record data with the
Hoi! gripper rig — ZED camera(s), the force/torque load-cell trigger, Spot
integration, and the Aria. Output is raw rosbags/SVOs that `../data_processing`
then ingests.

## Layout
```
docker/recording/        # recording containers + ROS launch/record scripts
arduino/                 # load-cell force-trigger firmware + wiring/calibration (see its README)
ros/                     # lenai_description.tar.xz  (gripper URDF/description)
```

## docker/recording
Per-device Dockerfiles (the rig runs on Jetson-class hardware):
- `Dockerfile-jetson`, `Dockerfile-nano`, `Dockerfile-agx-orin-spot`, `Dockerfile-light`, `Dockerfile-aria`

ROS launch + record scripts:
- `gripper_launch_single_force.launch`, `gripper_record_single_bag_svo.sh`,
  `start_recording_interface_gripper.sh` — gripper + force recording
- `spot_launch_full.launch`, `spot_launch_teleop.launch`, `spot_record_full.sh`,
  `setup_spot_nat_from_agx.sh` — Spot integration
- ZED configs: `zedm.yaml`, `zed2i.yaml`, `zed_common.yaml`

Typical flow (see the top-level `README.md` for the full command sequence):
```bash
cd docker/recording
docker compose build
docker compose up -d
docker exec -it spot_aria_gripper_recorder /bin/bash
# ... roslaunch / rosbag record inside the container
```

## arduino/
Load-cell "trigger" firmware (HX711 → Arduino → `/gripper_force_trigger`).
Mechanical install, calibration, and flashing steps are in
[`arduino/README.md`](arduino/README.md).

## ros/
`lenai_description.tar.xz` — the gripper URDF/description used by the recording
and visualization stack. Extract in place when needed.

## Notes
- The bind-mount paths in `docker/recording/docker-compose.yml` (data drives,
  devices) are machine-specific — adjust for your host.
- Calibration of the recorded cameras/IMU is handled in `../calibration`.
