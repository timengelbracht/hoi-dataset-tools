from hoi.data_tools.data_indexer import RecordingIndex
from pathlib import Path
import os
import fnmatch
from typing import Dict, Tuple, Any, List
import re
import csv
import math
import json

def _classify_viewpoint_by_parts(stream: Path) -> Tuple[str, str]:
    """
    Inspects individual path segments (case-insensitive) to classify viewpoint.
    Returns (viewpoint, matched_module_hint).
    Priority order is important if multiple cues exist.
    """
    parts = [p.lower() for p in stream.parts]
    path_str = "/".join(parts)  # only for hints

    # 1) Ego: any 'aria_human' segment anywhere
    if "aria_human" in parts:
        return "ego", "aria_human"

    # 2) Wrist: any 'aria_wrist' segment anywhere
    if "aria_wrist" in parts:
        return "wrist", "aria_wrist"

    # 3) Gripper: explicit gripper sources
    if "aria_gripper" in parts:
        return "gripper", "aria_gripper"
    if "umi_gripper" in parts:
        return "gripper", "umi_gripper"

    # 3b) Special: 'gripper/gripper' consecutive segments
    gripper_idxs = [i for i, s in enumerate(parts) if s == "gripper"]
    for i in range(len(gripper_idxs) - 1):
        if gripper_idxs[i+1] == gripper_idxs[i] + 1:
            return "gripper", "gripper/gripper"

    # 4) Third person: any segment that starts with 'iphone'
    if any(seg.startswith("iphone") for seg in parts):
        return "third_person", "iphone"

    # (Optional) ZED/other camera heuristics — if path already indicates a gripper rig
    if "zedm" in parts or "zed_node" in parts:
        # If this lives under a 'gripper' subtree, you probably want gripper
        if "gripper" in parts:
            return "gripper", "zed_under_gripper"

    return "unknown", ""

_TS_RE = re.compile(r"^\d+\.(png|jpe?g|npy)$", re.IGNORECASE)

def _infer_sensor_module(stream: Path) -> str:
    """
    Heuristic to name the sensor module from path segments.
    Extend as needed for your tree.
    """
    parts = [p.lower() for p in stream.parts]
    joined = "/".join(parts)

    # explicit sensor topics first
    if "wrench" in parts:
        return "wrench"
    if "imu" in parts:
        return "imu"
    if "joint_states" in parts:
        return "joint_states"
    if "telemetry" in parts:
        return "telemetry"
    if "digit" in parts:
        return "haptics"

    # depth directories
    if any(seg in ("camera_depth", "depth_registered") for seg in parts):
        return "depth"

    # zed depth hint
    if "zedm" in parts and "depth" in parts:
        return "depth"

    # fallback to last directory name or file stem
    if stream.is_file():
        return stream.parent.name.lower()
    return stream.name.lower() or "unknown"


def _duration_from_frame_dir(frame_dir: Path) -> Tuple[int, float]:
    """
    Returns (duration_ns, duration_sec) from numeric-stem files in a directory.
    """
    images: List[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.npy"):
        images.extend(frame_dir.glob(ext))
    if not images:
        return 0, 0.0

    nums = [int(p.stem) for p in images if p.stem.isdigit()]
    if not nums:
        return 0, 0.0

    dur_ns = max(nums) - min(nums)
    return dur_ns, dur_ns / 1e9


def _duration_from_csv(csv_path: Path, ts_column_candidates: List[str] = None) -> Tuple[int, float]:
    """
    Returns (duration_ns, duration_sec) by scanning a CSV 'timestamp' column (ns).
    Efficient: single pass, no pandas dependency.
    """
    if ts_column_candidates is None:
        ts_column_candidates = ["timestamp"]  # add more if needed

    if not csv_path.is_file():
        return 0, 0.0

    try:
        with csv_path.open("r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return 0, 0.0

            header_l = [h.strip().lower() for h in header]
            # find the first present candidate
            try:
                ts_idx = next(header_l.index(c) for c in ts_column_candidates if c in header_l)
            except StopIteration:
                return 0, 0.0

            min_ts = None
            max_ts = None
            for row in reader:
                if ts_idx >= len(row):
                    continue
                val = row[ts_idx].strip()
                if not val or not val.isdigit():
                    continue
                t = int(val)
                if min_ts is None or t < min_ts:
                    min_ts = t
                if max_ts is None or t > max_ts:
                    max_ts = t

            if min_ts is None or max_ts is None:
                return 0, 0.0

            dur_ns = max_ts - min_ts
            return dur_ns, dur_ns / 1e9
    except Exception:
        # robust fallback
        return 0, 0.0
    
def get_image_duration_of_location(rec_location: str, base_path: Path) -> float:
    data_indexer = RecordingIndex(os.path.join(str(base_path), "extracted"))

    data_streams = data_indexer.get_all_extracted_data_streams(
        extraction_path=base_path / "extracted" / rec_location,
        image_streams=True, 
        sensor_streams=False
    )

    duration_data: Dict[Path, Dict[str, Any]] = {}
    total_duration_data = {}

    for stream in data_streams:
        stream = Path(stream)

        # classify viewpoint/module using robust segment-based rules
        viewpoint, module_key = _classify_viewpoint_by_parts(stream)

        # collect image filenames
        images: List[Path] = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            images.extend(stream.glob(ext))
        if not images:
            continue

        # filenames are numeric timestamps (ns)
        nums = [int(img.stem) for img in images if img.stem.isdigit()]
        if not nums:
            continue

        min_num, max_num = min(nums), max(nums)
        dur_ns = max_num - min_num
        dur_sec = dur_ns / 1e9
        dur_min = dur_sec / 60.0
        dur_hr  = dur_min / 60.0

        duration_data[stream] = {
            "min_num": min_num,
            "max_num": max_num,
            "duration_ns": dur_ns,
            "duration_sec": dur_sec,
            "duration_min": dur_min,
            "duration_hours": dur_hr,
            "viewpoint": viewpoint,
            "module": module_key,
        }

    # aggreagte statistics
    total_duration_data["num_streams"] = len(duration_data)
    total_duration_data["total_duration_ns"] = sum(d["duration_ns"] for d in duration_data.values())
    total_duration_data["total_duration_sec"] = sum(d["duration_sec"] for d in duration_data.values())
    total_duration_data["total_duration_min"] = total_duration_data["total_duration_sec"] / 60.0
    total_duration_data["total_duration_hours"] = total_duration_data["total_duration_min"] / 60.0

    # get statistics by viewpoint
    total_duration_data["ego"] = sum(d["duration_sec"] for d in duration_data.values() if d["viewpoint"] == "ego")
    total_duration_data["wrist"] = sum(d["duration_sec"] for d in duration_data.values() if d["viewpoint"] == "wrist")
    total_duration_data["gripper"] = sum(d["duration_sec"] for d in duration_data.values() if d["viewpoint"] == "gripper")
    total_duration_data["third_person"] = sum(d["duration_sec"] for d in duration_data.values() if d["viewpoint"] == "third_person")    

    # get statistics by module hint
    total_duration_data["aria_human"] = sum(d["duration_sec"] for d in duration_data.values() if d["module"] == "aria_human")
    total_duration_data["aria_wrist"] = sum(d["duration_sec"] for d in duration_data.values() if d["module"] == "aria_wrist")
    total_duration_data["aria_gripper"] = sum(d["duration_sec"] for d in duration_data.values() if d["module"] == "aria_gripper")
    total_duration_data["umi_gripper"] = sum(d["duration_sec"] for d in duration_data.values() if d["module"] == "umi_gripper")
    total_duration_data["gripper"] = sum(d["duration_sec"] for d in duration_data.values() if d["module"] == "gripper/gripper")
    total_duration_data["iphone"] = sum(d["duration_sec"] for d in duration_data.values() if d["module"] == "iphone") 
    
    # Example aggregate: total duration in seconds across streams
    return total_duration_data
            
    
def get_sensor_duration_of_location(rec_location: str, base_path: Path) -> Dict[str, Any]:
    """
    Aggregates durations for sensor streams (depth frames OR CSV streams with 'timestamp' column).

    Returns a dict with totals and per-class breakdowns.
    """
    data_indexer = RecordingIndex(os.path.join(str(base_path), "extracted"))

    data_streams = data_indexer.get_all_extracted_data_streams(
        extraction_path=base_path / "extracted" / rec_location,
        image_streams=False,
        sensor_streams=True,
    )

    duration_data: Dict[Path, Dict[str, Any]] = {}
    total_duration_data: Dict[str, Any] = {}

    for stream in data_streams:
        stream = Path(stream)

        # classify viewpoint/module
        viewpoint, _module_hint = _classify_viewpoint_by_parts(stream)
        module_key = _infer_sensor_module(stream if stream.is_dir() else stream.parent)

        # duration from dir-of-frames or CSV with timestamps
        if stream.is_dir():
            dur_ns, dur_sec = _duration_from_frame_dir(stream)
        else:
            # expect a CSV path
            dur_ns, dur_sec = _duration_from_csv(stream)

        if dur_ns <= 0:
            continue

        dur_min = dur_sec / 60.0
        dur_hr  = dur_min / 60.0

        duration_data[stream] = {
            "duration_ns": dur_ns,
            "duration_sec": dur_sec,
            "duration_min": dur_min,
            "duration_hours": dur_hr,
            "viewpoint": viewpoint,
            "module": module_key,
            "is_csv": stream.is_file(),
            "path": str(stream),
        }

    # --- aggregate statistics
    total_duration_data["num_streams"] = len(duration_data)
    total_duration_data["total_duration_ns"] = sum(d["duration_ns"] for d in duration_data.values())
    total_duration_data["total_duration_sec"] = sum(d["duration_sec"] for d in duration_data.values())
    total_duration_data["total_duration_min"] = total_duration_data["total_duration_sec"] / 60.0
    total_duration_data["total_duration_hours"] = total_duration_data["total_duration_min"] / 60.0

    # by viewpoint
    for vp in ("ego", "wrist", "gripper", "third_person", "unknown"):
        total_duration_data[vp] = sum(d["duration_sec"] for d in duration_data.values() if d["viewpoint"] == vp)

    # by sensor module (common keys; extend as needed)
    for mod in ("wrench", "imu", "joint_states", "telemetry", "depth", "haptics", "unknown"):
        total_duration_data[mod] = sum(d["duration_sec"] for d in duration_data.values() if d["module"] == mod)

    # # raw per-stream entries if you want to inspect
    # total_duration_data["streams"] = duration_data

    return total_duration_data

def _stringify_keys(obj):
    """Recursively convert Path keys to strings for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): _stringify_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_stringify_keys(v) for v in obj]
    else:
        return obj


def get_all_durations_from_list_of_locations(
    locations: List[str],
    base_path: Path,
    save_path: str | Path = None
) -> Dict[str, Any]:
    all_durations: Dict[str, Dict[str, Any]] = {}

    for loc in locations:
        try:
            sensor_dur = get_sensor_duration_of_location(loc, base_path)
            image_dur = get_image_duration_of_location(loc, base_path)
            all_durations[loc] = {
                "sensor": sensor_dur,
                "image": image_dur,
            }
            print(f"Processed location: {loc}")
        except Exception as e:
            print(f"⚠️Failed to process {loc}: {e}")
            all_durations[loc] = {"sensor": None, "image": None, "error": str(e)}

    if save_path is not None:
        save_path = Path(save_path).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert Path keys to strings recursively
        json_safe = _stringify_keys(all_durations)

        with open(save_path, "w") as f:
            json.dump(json_safe, f, indent=2)
        print(f"Saved results to {save_path}")

    return all_durations

if __name__ == "__main__":

    rec_location = "kitchen_7"
    base_path = Path(f"/data/ikea_recordings")
    data_indexer = RecordingIndex(
        os.path.join(str(base_path), "raw") 
    )

    # get_image_duration_of_location(rec_location, base_path)
    # get_sensor_duration_of_location(rec_location, base_path)

    locations = ["kitchen_7", "livingroom_1", "office_1", "bedroom_4", "bathroom_2", "bedroom_6"]

    all_durations = get_all_durations_from_list_of_locations(
        locations=locations,
        base_path=base_path,
        save_path=f"/data/evaluations/stats/dataset_durations_evaluation_scenes.json"
    )

    a = 2

