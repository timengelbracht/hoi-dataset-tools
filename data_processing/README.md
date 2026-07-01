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
docker/odometry/         # ORB-SLAM3 container (used by the UMI pipeline). See note below.
third_party/open_vins/   # OpenVINS source (legacy, see note below)
configs/                 # example run configs
```

## Pipeline overview
Processing one recording location runs, in order, through a small set of modules
that each own one alignment step:

1. **Extract raw data** — the per-sensor loaders
   (`data_loader_aria.py`, `data_loader_umi.py`, `data_loader_gripper.py`,
   `data_loader_iphone.py`, `data_loader_leica.py`) decode each stream (Aria VRS
   + MPS, UMI GoPro MP4, gripper ZED/force-torque bag, iPhone RGB-D, Leica scans)
   into a common on-disk layout. `data_indexer.py` discovers what exists for a
   location/interaction.
2. **Time-align the streams** — `time_align_extracted_single_recording.py`
   (`Datasyncer`) computes per-stream time offsets and crops all streams of a
   recording to a common window.
3. **Spatially align the streams** — `spatial_registrator.py` registers each
   trajectory into the shared Leica world frame (hloc/InLoc anchors; for UMI a
   GTSAM pose-graph over the ORB-SLAM odometry).
4. **Split interactions** — the aligned recording is cut into individual
   interaction windows (followed by some **manual annotation**, see
   `annotation_tools/`).
5. **Bundle for release** — `package_dataset_release.py` copies/anonymizes an
   extracted location into the release tree.

The `run_pipeline_*` functions in `extraction_pipeline.py` orchestrate steps 1–4
per recorder; step 5 is a separate CLI.

## Odometry (ORB-SLAM3 only)
The `umi` stage shells into `docker/odometry/` and runs **monocular ORB-SLAM3**
(`mono_euroc` in the `orbslam3-melodic` container); the container is torn down and
recreated per run. **Note:** the VINS-Fusion / OpenVINS Dockerfiles in
`docker/odometry/` and `third_party/open_vins/` are **legacy experiments and are
not used by the current pipeline** — only ORB-SLAM3 is invoked. They can be
removed if we don't intend to revisit inertial odometry.

## Environment
The package runs inside the **aria dev container** (`docker/aria/`). Two ways to
launch it:

- **VS Code dev container** (recommended): open `docker/aria/.devcontainer` and
  "Reopen in Container". Its `postCreateCommand` runs `pip install -e <repo>`, so
  `hoi` is importable on start.
- **docker compose**:
  ```bash
  cd data_processing/docker/aria
  docker compose up -d aria_dev
  docker exec -it aria_dev bash
  pip install -e /path/to/hoi-dataset-tools   # once, if not already
  ```

All Python dependencies are pinned in `docker/aria/Dockerfile`.

### What to mount (edit for your machine)
The mount sources in `.devcontainer/devcontainer.json` and `docker-compose.yml`
point at the original authors' paths — **change them to yours**. What the
container actually needs:

| mount | why |
|---|---|
| your **dataset root** → `/data` | the pipeline reads/writes here; the run config's `base_path` points at it (e.g. `base_path: /data/ikea_recordings`) |
| this **repo** → the workspace | so the `hoi` package is importable (`pip install -e`); the dev container mounts it as `workspaceFolder` |
| **GPU** — `--gpus all` / nvidia runtime | torch + CUDA Open3D |
| **`/var/run/docker.sock`** | the `umi` stage launches the ORB-SLAM container *from inside* this container, so it needs the host Docker socket |
| **X11** — `/tmp/.X11-unix` + `DISPLAY` | optional; only for the Open3D visualization windows |

For headless processing you only need the **dataset mount**, the **GPU**, and —
for the UMI stage — the **docker socket**. The `/dev`, udev and Vulkan mounts in
the shipped configs are for live-device / GUI use and aren't required to process
existing recordings.

## Credentials (Aria MPS)
The `mps` stage of the extraction pipeline requests Aria Machine Perception
Services and needs **Project Aria account credentials**. No credentials are
stored in the repo; they are resolved in this order:

1. **YAML file** (handy for debugging) — copy `aria_credentials.example.yaml`
   to `aria_credentials.yaml` (git-ignored) at the repo root and fill in
   `username` / `password`. Or point `ARIA_CREDENTIALS_FILE` at a file elsewhere.
2. **Environment variables** — `ARIA_USERNAME` / `ARIA_PASSWORD`.
3. **Interactive prompt** — if neither of the above is set.

Never commit a filled-in `aria_credentials.yaml`. Stages that don't touch Aria
MPS don't need credentials.

## Expected raw data layout
The pipeline parses a fixed directory structure under `<base_path>/raw/`. Each
recording file/folder is named `<location>_<interaction_range>_<recorder>`
(e.g. `shoerack_1_1-7_umi`); `data_indexer.py` takes the interaction range from
the token after the location.

```
raw/
├── <location>/                          # e.g. shoerack_1
│   ├── gripper/                          # interaction type
│   │   ├── gripper/                      #   ZED + force/torque rosbag(s)
│   │   │   └── <loc>_<range>_<date>.bag
│   │   ├── aria_gripper/                 #   Aria mounted on the gripper
│   │   │   ├── <loc>_<range>_gripper.vrs
│   │   │   └── mps_<loc>_<range>_gripper_vrs/   # Aria MPS output
│   │   ├── aria_human/                   #   Aria worn by the operator
│   │   └── "iphone_1 (darkblue)"/  "iphone_2 (green)"/
│   │       └── <loc>_<range>_gripper/    #   iPhone RGB-D recording
│   ├── umi/
│   │   ├── umi_gripper/                  #   handheld UMI GoPro
│   │   │   └── <loc>_<range>_umi.MP4
│   │   ├── aria_human/
│   │   │   ├── <loc>_<range>_umi.vrs
│   │   │   └── mps_<loc>_<range>_umi_vrs/
│   │   └── iphone_1/  iphone_2/
│   ├── hand/
│   │   ├── aria_human/
│   │   └── "iphone_1 (darkblue)"/  "iphone_2 (green)"/
│   ├── wrist/
│   │   ├── aria_wrist/
│   │   └── aria_human/
│   └── leica/                            # Leica scans: "Job NNN- Setup NNN.{e57,png,txt,...}"
├── calib/                               # gripper_blue/ gripper_yellow/ blue/ yellow/ hand_eye/
└── umi_meta/                            # calib/{blue,yellow}/  slam_config/  umi_mask.png
```
Not every location has every recorder/interaction; the indexer only processes
what is present.

## Running the extraction pipeline
Config-driven (one YAML per run/location — see `configs/extraction_example.yaml`):
```bash
python -m hoi.data_tools.extraction_pipeline --config configs/extraction_example.yaml
```
The config selects the location, interaction indices, gripper color, and which
`stages` to run (`mps`, `leica`, `hand`, `gripper`, `wrist`, `umi`).

`index_from` controls which subtree under `base_path` is scanned for recordings:
`raw` (default) for discovering recordings to extract, or `extracted` when the
raw data is no longer present and you only want to (re)run downstream steps on
already-extracted data. Both trees share the same
`<location>/<interaction>/<recorder>/` layout, so discovery works from either.

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
