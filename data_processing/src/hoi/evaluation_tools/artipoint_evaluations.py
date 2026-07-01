import os
from pathlib import Path
import json
import csv


def load_ground_truth_articulations(file_path: str | Path):
    """
    read json file containing ground truth articulation data
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def load_artipoint_estimation_results_from_base(base_path: str | Path):
    """
    read segment_x folders containing the per articulation estimation results
    and return as a list of dictionaries
    """

    base_path = Path(base_path)

    results = []
    shared = {}

    if not base_path.exists():
        return results

    for child in sorted(base_path.iterdir()):
        if not child.is_dir():
            continue
        if not child.name.startswith("segment_"):
            continue

        axis_info_path = child / "axis_info.json"
        if not axis_info_path.exists():
            continue

        try:
            with open(axis_info_path, "r") as f:
                axis_info = json.load(f)
        except (OSError, json.JSONDecodeError):
            # skip unreadable or invalid json files
            continue

        # normalize segment key: segment_3 -> 3 (int) if possible, otherwise keep name
        seg_key = child.name
        try:
            if seg_key.startswith("segment_"):
                seg_key_num = int(seg_key.split("_", 1)[1])
                seg_key = seg_key_num
        except (ValueError, IndexError):
            pass

        shared[seg_key] = axis_info

        results.append(
            {
                "segment": seg_key,
                "segment_name": child.name,
                "axis_info_path": str(axis_info_path),
                "axis_info": axis_info,
            }
        )

    return results

def load_matched_cues(file_path: str | Path):
    """
    To get the mapping from articulation estimation results to ground truth data.
    read csv file containing matched cues and return as dictionary
    """
    file_path = Path(file_path)

    with open(file_path, newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.get_dialect("excel")
        reader = csv.DictReader(f, dialect=dialect)

        matched = {}
        for row in reader:
            # normalize keys and values
            row_norm = { (k.strip().lower() if k else ""): (v.strip() if v is not None else "") for k, v in row.items() }

            axis_key = row_norm.get("axis_name") or row_norm.get("axis")
            if not axis_key:
                # skip rows without an axis identifier
                continue

            try:
                axis = int(axis_key)
            except ValueError:
                axis = axis_key  # keep as string if not integer

            def to_int_or_none(val):
                try:
                    return int(val)
                except (TypeError, ValueError):
                    return None

            cue_start = to_int_or_none(row_norm.get("cue_start"))
            cue_end = to_int_or_none(row_norm.get("cue_end"))
            verification = row_norm.get("verification", "")

            matched[axis] = {
                "cue_start": cue_start,
                "cue_end": cue_end,
                "verification": verification,
            }

    return matched

def match_estimation_to_ground_truth(
    estimated_articulations: list[dict],
    ground_truth_articulations: dict,
    matched_cues: dict,
):
    """
    Match estimated articulations to ground truth data using matched cues.
    matched cues and the estimated articulation segment can be uniquely matched NOT by index, but by the start and end frame indices.

    get index of ground truth articulation, as well as the groudn truth articulation data, then use the matched cues to find the corresponding estimated articulation segment.

    """

    matched_articulations = {}
    for est_articulation in estimated_articulations:
        # get start and end frame for matching with estimated articulation
        start_index_est = est_articulation["axis_info"].get("start_frame")
        end_index_est = est_articulation["axis_info"].get("end_frame")

        # get id of ground truth articulation
        articulation_id = None
        for gt_key, gt_info in ground_truth_articulations.items():
            # get start and end frame for matching with estimated articulation
            start_index_gt = matched_cues.get(int(gt_key), {}).get("cue_start")
            end_index_gt = matched_cues.get(int(gt_key), {}).get("cue_end")

            if start_index_est == start_index_gt and end_index_est == end_index_gt:
                articulation_id = int(gt_key)
                axis_info_gt = gt_info
                break

        if articulation_id is None:
            continue

        axis_info_est = {}
        axis_info_est["position"] = est_articulation["axis_info"].get("center")
        axis_info_est["axis"] = est_articulation["axis_info"].get("axis")
        axis_info_est["type"] = est_articulation["axis_info"].get("joint_type")

        # process matched estimated articulation
        if axis_info_est:
            matched_articulations[articulation_id] = {
                "ground_truth": axis_info_gt,
                "estimation": axis_info_est,
            }

    # sort matched articulations by articulation id
    matched_articulations = dict(sorted(matched_articulations.items())) 
        

    a = 2





    a = 2

def evaluate_articulated_object_estimation_for_recording(
    data_root: str | Path,
    rec_location: str,
    interaction_indices: str,
):
    """
    Evaluate articulated object estimation results against ground truth data.
    """
    
    # get paths
    data_root = Path(data_root)

    ground_truth_base_path = data_root / "ground_truth" / "hej" / "raw" / "ikea" / f"{rec_location}_{interaction_indices}"
    results_base_path = data_root / "results" / "hej" / "ikea" / f"{rec_location}_{interaction_indices}" / "results"

    ground_truth_path = ground_truth_base_path / f"{rec_location}.json"
    matched_cues_path = ground_truth_base_path / "matched_cues.csv"

    # load data
    ground_truth_articulations = load_ground_truth_articulations(ground_truth_path)
    matched_cues = load_matched_cues(matched_cues_path)
    estimated_articulations = load_artipoint_estimation_results_from_base(results_base_path)

    # match estimated to ground truth
    matched_results = match_estimation_to_ground_truth(
        estimated_articulations,
        ground_truth_articulations,
        matched_cues,
    )

    a = 2


if __name__ == "__main__":
    rec_location = "bathroom_2"
    base_path = Path(f"/data/ikea_recordings")
    rec_type = "hand"

    # interaction indices "666" for mocap tests
    interaction_indices = "1-5"

    data_root = f"/data/evaluations/articulated_object_estimation"

    evaluate_articulated_object_estimation_for_recording(
        data_root=data_root,
        rec_location=rec_location,
        interaction_indices=interaction_indices,
    )