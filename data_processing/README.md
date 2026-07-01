# data_processing

Software for turning **raw Hoi! recordings into the processed / release dataset**:
frame + telemetry extraction, monocular ORB-SLAM odometry, spatial registration
to the Leica map, anonymization, packaging, and the evaluation tools.

This is the part you need if you only want to *use* the released data or reproduce
the processing — it does not touch the recording hardware (`../data_recording`).

## Layout
```
src/hoi/                 # the installable `hoi` Python package
  data_tools/            # loaders + the extraction pipeline (Aria / UMI / gripper / iPhone / Leica)
  evaluation_tools/      # benchmark / evaluation scripts
  annotation_tools/      # interactive annotation GUIs (Tkinter) + SAM service
docker/aria/             # dev container the pipeline runs in (.devcontainer + docker-compose)
docker/odometry/         # ORB-SLAM3 / VINS / OpenVINS containers (invoked by the UMI pipeline)
third_party/open_vins/   # OpenVINS source (odometry dependency)
configs/                 # example run configs
```

## Environment
The package is developed and run inside the **aria dev container**
(`docker/aria/`). Open `docker/aria/.devcontainer` in VS Code ("Reopen in
Container"), or build via `docker/aria/docker-compose.yml`. The container's
`postCreateCommand` runs `pip install -e /exchange/hoi-dataset-tools`, so the
`hoi` package is importable on start. All Python dependencies are pinned in
`docker/aria/Dockerfile`.

## Running the extraction pipeline
Config-driven (one YAML per run/location — see `configs/extraction_example.yaml`):
```bash
python -m hoi.data_tools.extraction_pipeline --config configs/extraction_example.yaml
```
The config selects the location, interaction indices, gripper color, and which
`stages` to run (`mps`, `leica`, `hand`, `gripper`, `wrist`, `umi`).

The `umi` stage shells into `docker/odometry/` to run monocular ORB-SLAM3; that
container is torn down and recreated per run. See `docker/odometry/` for the
standalone odometry/VINS/OpenVINS setups.

## Packaging a release
`package_dataset_release.py` is a standalone CLI that copies one extracted
location into a release-ready tree:
```bash
python -m hoi.data_tools.package_dataset_release \
    /data/ikea_recordings/extracted/office_1 \
    /data/ikea_recordings/release/office_1 \
    [--dry-run] [--overwrite] [--double-zip]
```

## Calibration inputs
The pipeline consumes camera/IMU calibration produced by `../calibration`
(the `*-camchain.yaml` files under the dataset's `raw/.../calib`).
