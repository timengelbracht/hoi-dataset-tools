# Hoi! Dataset Tools

Tooling to **record** and **process** the Hoi! dataset of human–object interactions,
captured with multiple synchronized sensors (Project Aria glasses, a handheld
UMI GoPro gripper, a force/torque gripper, iPhone RGB-D, and Leica laser scans)
across real environments, all registered into a shared world frame.

This repo has two independent parts (plus shared calibration):

| part | what it is | start here |
|---|---|---|
| **[`data_processing/`](data_processing/README.md)** | Python package + Docker to turn raw recordings into the processed/release dataset (extract → time-align → spatially register → split interactions → package). | to **use the data** or **reproduce processing** |
| **[`data_recording/`](data_recording/README.md)** | The gripper capture rig — record your own data on a Jetson (ZED, force/torque, tactile, motor). | to **record data** |
| **[`calibration/`](calibration/README.md)** | Stock open-source camera/IMU calibration (Kalibr + allan_variance_ros) that produces the calib the pipeline consumes. | for **camera/IMU calibration** |

## Quickstart

**Process a recording location.** Bring up the `data_processing` **VS Code dev
container** (`docker/aria/.devcontainer`), **editing the mounts for your dataset
and GPU first** — see
[what the dev container mounts](data_processing/README.md#what-the-dev-container-mounts-edit-sources-for-your-machine).
Then, inside the container:
```bash
python -m hoi.data_tools.extraction_pipeline \
    --config data_processing/configs/extraction_example.yaml
```
The config picks the location, interaction indices, gripper color, and which
stages to run. Package a processed location for release with:
```bash
python -m hoi.data_tools.package_dataset_release <extracted_loc> <release_loc>
```

**Record your own data** (on a Jetson — see [the recording README](data_recording/README.md)):
```bash
cd data_recording/docker/recording
# edit hardware.env for your rig (DIGIT ids, USB serials, tick limits, F/T bus)
docker compose build recording_gripper_nano
./start_recording_interface_gripper.sh <env_name>
```

## Processing pipeline at a glance
1. **Extract** raw streams (Aria VRS+MPS, UMI GoPro, gripper ZED/F-T bag, iPhone RGB-D, Leica) — `data_loader_*`, `data_indexer`.
2. **Time-align** the streams within a recording — `time_align_extracted_single_recording` (`Datasyncer`).
3. **Spatially register** every stream into the shared Leica world frame — `spatial_registrator` (hloc/InLoc anchors; GTSAM pose-graph over ORB-SLAM3 for UMI).
4. **Split interactions** into per-interaction windows (+ some manual annotation).
5. **Package** for release — `package_dataset_release`.

See [`data_processing/README.md`](data_processing/README.md) for the module map,
expected raw layout, Aria MPS credentials, and the odometry container.

## Requirements
Everything runs in Docker (per-part Dockerfiles); no host installs beyond Docker
+ the NVIDIA container runtime. The `data_processing` package pins its
dependencies in `data_processing/docker/aria/Dockerfile`.

## Status
Research codebase, released so the community can see and use the pipeline. It
works end-to-end, but some cleanup is still in progress — see [`TODO.md`](TODO.md)
for known open items (e.g. the evaluation-tools cleanup and some machine-specific
paths in Docker mounts). Issues and PRs welcome.

## License & citation
TODO: add a license and citation before wide distribution.
