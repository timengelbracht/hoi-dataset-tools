import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import pandas as pd
import telemetry_parser
import av
import subprocess
import tempfile
import shutil
import re
from typing import Optional, Iterable, Sequence
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import rcParams
import matplotlib
matplotlib.use("Agg")


def get_frames_from_mp4(
    mp4_path: str | Path,
    outdir: str | Path | None = None, ext: str = "png") -> Tuple[List[av.video.frame.VideoFrame], List[int]]:
    """
    Decode every video frame in an MP4 and return
        • a list of PyAV VideoFrame objects  (empty if `outdir` is set)
        • a parallel list of timestamps in **nanoseconds**
    
    When `outdir` is provided, frames are **written straight to disk**
    as JPEGs named   <timestamp_ns>.jpg   and not kept in RAM.
    This keeps memory usage low for long clips.
    """
    mp4_path = Path(mp4_path)
    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

    # check if the file exists
    if not mp4_path.is_file():
        raise FileNotFoundError(f"MP4 file not found: {mp4_path}")

    container = av.open(mp4_path)
    vstream   = container.streams.video[0]

    ns_per_tick = float(vstream.time_base) * 1e9     # scalar once
    total       = vstream.frames or 0                # may be 0 if unknown

    frame_ts: list[int] = []
    frame_objs: list[av.video.frame.VideoFrame] = []

    for frame in tqdm(container.decode(video=0), total=total,
                      unit="frame", desc=f"Extracting {mp4_path.name}"):
        if frame.pts is None:      # should not happen, but be safe
            continue

        ts_ns = int(frame.pts * ns_per_tick)
        frame_ts.append(ts_ns)

        if outdir is None:
            frame_objs.append(frame)                 # keep in RAM
        else:
            # write JPEG without extra conversions; PIL handles RGB24
            if ext.lower() == "jpg":
                frame.to_image().save(outdir / f"{ts_ns}.jpg", format="JPEG")
            elif ext.lower() == "png":
                # PNG is lossless, but larger than JPEG
                frame.to_image().save(outdir / f"{ts_ns}.png", format="PNG")

    return frame_objs, frame_ts


def get_imu_from_mp4(mp4_file: str | Path,
                    outdir: str| Path| None = None) -> pd.DataFrame:
    """ Extract IMU data from a GoPro MP4 file and return it as a DataFrame (timestamps in nanosecs).
    Args:
        mp4_file (str | Path): Path to the MP4 file.
    Returns:
        pd.DataFrame: DataFrame containing the IMU data with columns for timestamps, gyro, and accelerometer.
    """

    if isinstance(mp4_file, Path):
        mp4_file = str(mp4_file)

    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

    # Extract telemetry data
    tp = telemetry_parser.Parser(mp4_file)

    telemetry = tp.telemetry()
    imu = tp.normalized_imu()

    # parse to DataFrame
    timestamps_ms = np.fromiter((p["timestamp_ms"] for p in imu), dtype=np.float64)
    timestamps_ns = timestamps_ms * 1e6 

    # gyro and accel come as tuples
    gyro_arr = np.stack([p['gyro'] for p in imu])     
    acc_arr  = np.stack([p['accl'] for p in imu])      

    # ── 2. build the DataFrame column‑wise (no Python loop) ───────────────
    deg2rad = np.pi / 180.0
    df = pd.DataFrame({
        'timestamp'  : timestamps_ns.astype(np.int64),
        'angular_vel_x'  : gyro_arr[:,0] * deg2rad,
        'angular_vel_y'  : gyro_arr[:,1] * deg2rad,
        'angular_vel_z'  : gyro_arr[:,2] * deg2rad,
        'linear_accel_x' : acc_arr[:,0],
        'linear_accel_y' : acc_arr[:,1],
        'linear_accel_z' : acc_arr[:,2],
    })

    if outdir is not None:
        # save to CSV in the output directory
        csv_path = outdir / f"data.csv"
        df.to_csv(csv_path, index=False)
        print(f"IMU data saved to {csv_path}")

    return df


def make_synced_video_from_timestamped_frames(
    frames_dir: Path | str,
    output: Path | str,
    ext: Optional[str] = None,
    crf: int = 18,
    preset: str = "slow",
    pix_fmt: str = "yuv420p",
    extra_ffmpeg_args: Optional[Iterable[str]] = None,
    # NEW:
    skip_bad: bool = True,                  # skip unreadable/missing images
    validate: str = "exists",               # "exists" or "open" (uses Pillow if available)
    show_progress: bool = True,
) -> Path:
    """
    Create a VFR MP4 from frames named by nanosecond timestamps, preserving real timing.
    Skips bad frames (missing/corrupted) if skip_bad=True and keeps timing by spanning
    the gap to the next good frame.

    Raises:
        FileNotFoundError: If no *valid* frames found or ffmpeg missing.
        RuntimeError: If ffmpeg fails.
        ValueError: If filenames lack numeric timestamps.
    """
    try:
        from PIL import Image  # optional, only needed if validate="open"
        pil_available = True
    except Exception:
        pil_available = False

    frames_dir = Path(frames_dir)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg not found on PATH.")

    # Collect frames
    exts = [ext] if ext else ["png", "jpg", "jpeg"]
    frames: List[Path] = []
    for e in exts:
        frames.extend(sorted(frames_dir.glob(f"*.{e}")))
    if not frames:
        raise FileNotFoundError(f"No frames found in {frames_dir} with ext(s) {exts}.")

    # Parse timestamps from filename stems
    def parse_ns(p: Path) -> int:
        m = re.search(r"(\d+)", p.stem)
        if not m:
            raise ValueError(f"Filename lacks numeric timestamp: {p.name}")
        return int(m.group(1))

    frames.sort(key=parse_ns)

    # Validate frames and build good lists
    good_frames: List[Path] = []
    good_ts: List[int] = []

    def is_good(p: Path) -> bool:
        if validate == "exists":
            return p.is_file() and p.stat().st_size > 0
        elif validate == "open":
            if not p.is_file() or p.stat().st_size == 0:
                return False
            if not pil_available:
                # Fall back to existence check if Pillow missing
                return True
            try:
                with Image.open(p) as im:
                    im.verify()  # quick header check
                return True
            except Exception:
                return False
        else:
            return p.is_file()

    bad_count = 0
    for p in frames:
        if is_good(p):
            good_frames.append(p)
            good_ts.append(parse_ns(p))
        else:
            bad_count += 1

    if not good_frames:
        raise FileNotFoundError("No readable frames after validation; nothing to encode.")

    if skip_bad and bad_count > 0 and show_progress:
        print(f"[make_synced] Skipped {bad_count} bad frame(s). Proceeding with {len(good_frames)} valid frame(s).")

    # Compute per-frame durations from deltas of the *good* timestamps
    durs: List[float] = []
    for i in range(len(good_ts) - 1):
        dt_ns = good_ts[i + 1] - good_ts[i]
        if dt_ns <= 0:
            dt_ns = 1  # keep positive
        durs.append(dt_ns / 1e9)

    # Edge cases
    if len(good_frames) == 1:
        # give a minimal duration so it displays
        durs = [1 / 30.0]  # ~33ms
        # DO NOT duplicate in ffconcat; we will handle last-frame logic below

    # Write ffconcat (exact real-time: last frame once, NO duration)
    with tempfile.TemporaryDirectory() as tmpdir:
        concat_path = Path(tmpdir) / (output.stem + ".ffconcat")
        with concat_path.open("w", encoding="utf-8") as f:
            f.write("ffconcat version 1.0\n")
            if len(good_frames) == 1:
                # single frame case: assign a tiny duration then write the file once more
                f.write(f"file '{good_frames[0].as_posix()}'\n")
                f.write(f"duration {durs[0]:.9f}\n")
                f.write(f"file '{good_frames[0].as_posix()}'\n")
            else:
                for i in range(len(good_frames) - 1):
                    f.write(f"file '{good_frames[i].as_posix()}'\n")
                    f.write(f"duration {durs[i]:.9f}\n")
                # last frame ONCE, no duration
                f.write(f"file '{good_frames[-1].as_posix()}'\n")

        # Build ffmpeg command
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_path),
            "-vsync", "vfr",
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            "-pix_fmt", pix_fmt,
            "-movflags", "+faststart",
            str(output),
        ]
        if extra_ffmpeg_args:
            cmd = cmd[:-1] + list(extra_ffmpeg_args) + [str(output)]

        if show_progress:
            proc = subprocess.run(cmd)
        else:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed with code {proc.returncode}")

    return output


def timeseries_csv_to_synced_video(
    csv_path: Path | str,
    output: Path | str,
    time_col: str = "timestamp",            # nanoseconds
    value_cols: Optional[Sequence[str]] = None,  # None -> plot all non-time numeric cols
    labels: Optional[Sequence[str]] = None,      # Legend labels (optional)
    # Visuals
    figsize: Tuple[float, float] = (12.8, 7.2),  # in inches; 12.8x7.2 @ 100 dpi ~ 1280x720
    dpi: int = 100,
    linewidth: float = 1.5,
    ylim: Optional[Tuple[float, float]] = None,  # fix y-limits if you want stable scale
    bg_color: str = "white",
    cursor_color: str = "black",
    cursor_width: float = 1.5,
    font_size: int = 12,
    x_margin_frac: float = 0.02,                 # add a small x margin on both sides
    # Performance / size
    max_fps: Optional[float] = None,             # e.g., 30. Cap snapshot rate to reduce frames
    # Encoding
    codec: str = "libx264",                      # "libx264", "libx265", "libaom-av1"
    crf: Optional[int] = None,                   # None -> sensible default per codec
    preset: str = "slow",
    pix_fmt: str = "yuv420p",
    scale: Optional[str] = None,                 # e.g., "-2:720" or "1280:-2"
    extra_ffmpeg_args: Optional[Iterable[str]] = None,
    show_progress: bool = True,
) -> Path:
    """
    Render a CSV time series to a synced VFR MP4 using nanosecond timestamps.
    The plot shows the entire series with a moving vertical time cursor; each
    exported frame uses the sample's timestamp for exact inter-frame timing.

    csv_path: CSV with at least a 'timestamp' (ns) column.
    value_cols: Which columns to plot. Default: all numeric columns except time_col.
    max_fps: If set, decimates snapshots so consecutive frames are >= 1/max_fps seconds apart.
    """
    csv_path = Path(csv_path)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg not found on PATH.")

    # -------- Load & prepare data --------
    df = pd.read_csv(csv_path)
    if time_col not in df.columns:
        raise ValueError(f"'{time_col}' not found in CSV columns: {list(df.columns)}")

    # Keep only rows with finite timestamps
    ts_ns = pd.to_numeric(df[time_col], errors="coerce").astype("Int64").dropna().astype(np.int64).values
    order = np.argsort(ts_ns)
    ts_ns = ts_ns[order]

    # Choose value columns
    if value_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        value_cols = [c for c in numeric_cols if c != time_col]
        if not value_cols:
            raise ValueError("No numeric columns to plot besides the timestamp.")
    values = df.loc[order, value_cols].astype(float).values

    # Drop exact-duplicate timestamps (keep first)
    unique_mask = np.ones(len(ts_ns), dtype=bool)
    unique_mask[1:] = ts_ns[1:] != ts_ns[:-1]
    ts_ns = ts_ns[unique_mask]
    values = values[unique_mask]

    if len(ts_ns) == 0:
        raise ValueError("No valid timestamps found.")
    if len(ts_ns) == 1:
        # Duplicate the single sample so ffmpeg has a duration to show
        ts_ns = np.array([ts_ns[0], ts_ns[0] + 33_000_000], dtype=np.int64)  # +33ms
        values = np.vstack([values[0], values[0]])

    # Optional decimation to cap frame rate
    if max_fps and max_fps > 0:
        min_dt_ns = int(1e9 / max_fps)
        keep_idx = [0]
        last_ts = ts_ns[0]
        for i in range(1, len(ts_ns)):
            if ts_ns[i] - last_ts >= min_dt_ns:
                keep_idx.append(i)
                last_ts = ts_ns[i]
        ts_ns = ts_ns[keep_idx]
        values = values[keep_idx]

    # Relative time in seconds (for x-axis)
    t0 = float(ts_ns[0]) / 1e9
    t_rel = (ts_ns / 1e9) - t0
    total_time = t_rel[-1] - t_rel[0]

    # -------- Precompute durations from deltas (for VFR) --------
    durs = np.diff(ts_ns).astype(np.int64)
    durs[durs <= 0] = 1  # avoid zero/negative
    durs_s = durs / 1e9
    # Edge: ensure a last duration
    if len(durs_s) == 0:
        durs_s = np.array([1/30], dtype=float)

    # -------- Draw background plot (whole series) --------
    plt.rcParams.update({"font.size": font_size})
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    for j in range(values.shape[1]):
        ax.plot(t_rel, values[:, j], linewidth=linewidth, label=(labels[j] if labels and j < len(labels) else value_cols[j]))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")
    if ylim is not None:
        ax.set_ylim(*ylim)
    # Stable x-limits so the view doesn't pan
    x_min, x_max = t_rel[0], t_rel[-1]
    x_pad = (x_max - x_min) * x_margin_frac
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.legend(loc="best")
    fig.tight_layout()

    # -------- Generate frames with moving cursor --------
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        frame_files: List[Path] = []
        # We'll write one image per sample, named by its original ns timestamp
        # to preserve timing in the ffconcat step.
        if show_progress:
            print(f"[timeseries] Rendering {len(ts_ns)} frames (~{total_time:.2f}s)…")

        for i, t in enumerate(t_rel):
            # Cursor & annotation
            cursor = ax.axvline(t, color=cursor_color, linewidth=cursor_width, alpha=0.9)
            # Current values for label readout
            cur_vals = values[i, :]
            txt_lines = [f"t = {t:.3f} s"]
            for j in range(values.shape[1]):
                name = labels[j] if labels and j < len(labels) else value_cols[j]
                txt_lines.append(f"{name}: {cur_vals[j]:.4g}")
            text = ax.text(0.99, 0.02, "\n".join(txt_lines), transform=ax.transAxes,
                           ha="right", va="bottom", fontsize=font_size, bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.3"))

            # Save frame
            frame_path = tmpdir / f"{int(ts_ns[i])}.png"
            fig.savefig(frame_path, facecolor=fig.get_facecolor(), dpi=dpi)
            frame_files.append(frame_path)

            # Clean up dynamic artists
            text.remove()
            cursor.remove()

            if show_progress and (i % max(1, len(ts_ns)//50) == 0):
                # ~2% steps
                print(f"  [{i+1}/{len(ts_ns)}] t={t:.3f}s", end="\r", flush=True)

        if show_progress:
            print("\n[timeseries] Frames done. Encoding…")

        # -------- Build ffconcat --------
        concat_path = tmpdir / (output.stem + ".ffconcat")
        with concat_path.open("w", encoding="utf-8") as f:
            f.write("ffconcat version 1.0\n")
            for i in range(len(frame_files) - 1):
                f.write(f"file '{frame_files[i].as_posix()}'\n")
                f.write(f"duration {durs_s[i]:.9f}\n")
            # Repeat last frame so last duration applies
            last_dur = float(durs_s[-1]) if len(durs_s) > 0 else (1/30.0)
            f.write(f"file '{frame_files[-1].as_posix()}'\n")
            f.write(f"duration {last_dur:.9f}\n")
            f.write(f"file '{frame_files[-1].as_posix()}'\n")

        # -------- Sensible CRF defaults per codec --------
        if crf is None:
            if codec == "libx264":
                crf = 24
            elif codec == "libx265":
                crf = 30
            elif codec == "libaom-av1":
                crf = 34
            else:
                crf = 24

        # Optional scaling
        vf = []
        if scale:
            vf.append(f"scale={scale}")
        vf_arg = ["-vf", ",".join(vf)] if vf else []

        # Codec-specific extras
        codec_args: List[str] = []
        if codec == "libaom-av1":
            codec_args += ["-cpu-used", "5"]  # speed/quality trade-off

        # -------- Encode with ffmpeg (VFR) --------
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_path),
            "-vsync", "vfr",
            "-c:v", "libx264",      # <--- encoder first
            "-preset", preset,
            "-crf", str(crf),
            "-pix_fmt", pix_fmt,    # <--- pixel format here, not as codec
            "-movflags", "+faststart",
            str(output),
        ]
        if extra_ffmpeg_args:
            cmd = cmd[:-1] + list(extra_ffmpeg_args) + [str(output)]

        if show_progress:
            # Show ffmpeg console output (progress)
            proc = subprocess.run(cmd)
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg failed with code {proc.returncode}")
        else:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    "ffmpeg failed:\n"
                    f"CMD: {' '.join(cmd)}\n"
                    f"STDOUT:\n{proc.stdout}\n"
                    f"STDERR:\n{proc.stderr}"
                )

    plt.close(fig)


    return output


if __name__ == "__main__":
    pass
    # Example usage
    # mp4_path = Path("/exchange/calib/calib_yellow_pinhole-equi/data.MP4")
    # imu_outpath = Path("/exchange/calib/calib_yellow_pinhole-equi/imu")
    # rgb_dir = Path("/exchange/calib/calib_yellow_pinhole-equi/rgb")
    # imu_data = get_imu_from_mp4(mp4_path, outdir=imu_outpath)
    # frames, timestamps = get_frames_from_mp4(mp4_path, outdir=rgb_dir, ext="jpg")

    #make_synced_video_from_timestamped_frames(
    #    frames_dir="/data/ikea_recordings/extracted/bedroom_1/gripper/gripper/bedroom_1_1-8_gripper_bag/zedm/zed_node/right_raw/image_raw_color",
    #    output="/data/ikea_recordings/extracted/bedroom_1/gripper/gripper/bedroom_1_1-8_gripper_bag/video/zed_right.mp4",
    #    ext=None,          # auto-detect png/jpg/jpeg
    #    crf=18,
    #    preset="slow",
    #)

    timeseries_csv_to_synced_video(
    csv_path="/data/ikea_recordings/extracted/bedroom_1/gripper/gripper/bedroom_1_1-8_gripper_bag/force_torque/ft_sensor0/ft_sensor_readings/wrench/data.csv",
    output="/data/ikea_recordings/extracted/bedroom_1/gripper/gripper/bedroom_1_1-8_gripper_bag/video/force.mp4",
    time_col="timestamp",           # ns
    value_cols=["wrench_ext.force.x", "wrench_ext.force.y", "wrench_ext.force.z"],  # or None to plot all numeric cols (except timestamp)
    labels=["fx", "fy", "fz"],      # optional legend labels
    max_fps=30,                     # cap snapshot rate (keeps sync, shrinks size)
    codec="yuv420p",                # great size/quality
    crf=30,
    preset="slow",
    ylim=(-10, 15),                # fix Y if you want stable scale
    figsize=(12.8, 7.2), dpi=100,   # ~1280x720
    )

