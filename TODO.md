# Release TODO

## Evaluation code cleanup  ← look into this
`data_processing/src/hoi/evaluation_tools/` needs a cleanup pass before release:
- Hardcoded absolute paths (`/data/...`, `/exchange/...`) and `sys.path.insert(...)`
  hacks pointing at the vendored sparsh under `evaluations/`.
- Data-dir references literally named `hej` in eval output paths.
- Decide which eval scripts ship publicly vs. stay internal.

## Raw-free capability (run downstream steps on extracted-only data)
- [x] `index_from: extracted` config option for discovery.
- [x] Guard `UmiData.extract_umi_meta_data()` to skip the raw/umi_meta copy when
      the extracted calib+mask already exist.
- [ ] **Aria: verify no `self.provider`-when-`None` deref.** `load_provider()`
      degrades gracefully (warns, leaves `self.provider = None`) and
      `get_calibration()` returns the cached `extracted/calib/calib.json`, so
      downstream registration should work raw-free. Confirm no downstream path
      dereferences `self.provider` when the raw VRS is absent, and soften the
      warning to read as "expected in extracted-only mode".
- [ ] Gripper: give a clear error when both raw and extracted calib are missing
      (currently `shutil.copytree` crashes on the missing raw source).
- Note: the `extract_*` stages inherently need raw; raw-free applies only to
  re-running time-align / spatial-register / package on already-extracted data.

## Recording setup (data_recording)
- [x] Per-rig hardware identifiers centralized in `hardware.env` + interface script.
- [x] `data_recording/README.md` end-to-end setup guide.
- [ ] Add CAD / 3D-print files + bill-of-materials for the gripper assembly (not in repo).
- [ ] Verify the `timengelbracht/*` forks are public and reachable
      (`gelsight_digit_ros`, `gripper_force_controller`) — required for the nano build.
- [ ] Decide fate of the non-nano compose services (`testing`, `jetson`, `rpi/light`,
      `spot_agx`, `aria`): keep as documented variants or trim.

## Other open items (from the release cleanup pass)
- [ ] Delete dead dependency files: `data_processing/docker/aria/.devcontainer/Dockerfile`
      (unused duplicate nothing builds) and top-level `requirements.txt`
      (stale: pins projectaria 1.5.7, actual is 1.5.6).
- [ ] Keep/delete decision: `data_processing/src/hoi/data_tools/extract_raw_single_location.py`
      (standalone, possibly stale) and `data_processing/data_loader.py` (dead-code candidate).
- [ ] Pin gtsam (currently builds master; the `4.3.0` tag never existed — use `4.2.2` /
      `4.3a1` / the working `/gtsam` commit).
- [ ] docker-compose bind-mounts are machine-specific (`/media/...`) — move to env / `.env`.
- [ ] Restructure the top-level `README.md` into a project intro + links to the part READMEs.
- [ ] `evaluations/` (210 MB, currently gitignored) — decide what ships.
- [ ] Full secret sweep before the first push (device serials in `*.vrs.json`, emails,
      `~/.projectaria` tokens) — the hardcoded Aria credentials are already removed + scrubbed
      from history.
- [ ] Remove the legacy VINS-Fusion / OpenVINS Dockerfiles + `third_party/open_vins` if we
      don't intend to revisit inertial odometry (pipeline uses ORB-SLAM3 only).
