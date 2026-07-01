# calibration

Camera / IMU calibration for the Hoi! recording rig. This is the bridge between
the two other parts: `../data_recording` produces the calibration bags, and the
calibration outputs (the `*-camchain.yaml` files) are consumed by
`../data_processing`.

The container here builds [`allan_variance_ros`](https://github.com/ori-drs/allan_variance_ros)
for IMU-noise characterization, which feeds Kalibr for camera/IMU extrinsics.

## Contents
```
Dockerfile           # builds the allan_variance_ros calibration container
docker-compose.yml   # service `allan_variance_ros` (mounts /bags and /calib)
```

## Workflow
1. **Record a long static IMU bag** (≥ ~3 h) with the recording rig.
2. **Build + enter the container:**
   ```bash
   cd calibration
   docker compose build
   docker compose up -d
   docker exec -it allan_variance_ros /bin/bash
   ```
3. **Estimate IMU noise** (Allan variance) → produces the IMU noise YAML used by
   Kalibr:
   ```bash
   rosrun allan_variance_ros allan_variance /bags/cooked_rosbag.bag \
       /hoi-dataset-tools/config/imu_noise/witmotion_imu.yaml
   ```
4. **Run Kalibr** for camera + camera/IMU calibration to produce the
   `*-camchain.yaml` / `*-camchain-imucam.yaml` files.

See the top-level `README.md` for the detailed IMU-noise and motor/trigger
recipes.

## Outputs
The resulting camchain YAMLs are what `data_processing` loads as its camera
model (fisheye/equidistant intrinsics + `T_cam_imu`). Note the bind-mount paths
in `docker-compose.yml` (`/bags`, `/calib`) are machine-specific.
