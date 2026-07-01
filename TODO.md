# Release TODO

## Evaluation code cleanup  ← look into this
`data_processing/src/hoi/evaluation_tools/` needs a cleanup pass before release:
- Hardcoded absolute paths (`/data/...`, `/exchange/...`) and `sys.path.insert(...)`
  hacks pointing at the vendored sparsh under `evaluations/`.
- Data-dir references literally named `hej` in eval output paths.
- Decide which eval scripts ship publicly vs. stay internal.

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
