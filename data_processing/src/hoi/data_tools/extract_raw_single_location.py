from .data_loader_aria import AriaData
from .data_loader_iphone import IPhoneData
from .data_loader_gripper import GripperData
from pathlib import Path
from .data_indexer import RecordingIndex
import os

if __name__ == "__main__":

    """Extracts all raw data from a single location in the IKEA dataset.
    TODO: Leica data extraction"""

    rec_location = "bedroom_1"
    base_path = Path(f"/data/ikea_recordings")

    
    data_indexer = RecordingIndex(
        os.path.join(str(base_path), "raw") 
    )

    # Query for all gripper recorders at the specified location
    gripper_queries_at_loc = data_indexer.query(
        location=rec_location, 
        interaction=None, 
        recorder="gripper",
        interaction_index=None
    )

    # extract gripper data for each recorder found at the location
    for loc, inter, rec, ii, path in gripper_queries_at_loc:
        print(f"Found recorder: {rec} at {path}")

        rec_type = inter
        rec_module = rec
        interaction_indices = ii

        gripper_data = GripperData(base_path, rec_location, rec_type, rec_module, interaction_indices, data_indexer)

        gripper_data.extract_bag_full()

    # Query for all iPhone recorders at the specified location
    iphone_queries_at_loc = data_indexer.query(
        location=rec_location, 
        interaction=None, 
        recorder="iphone*",
        interaction_index=None
    )

    for loc, inter, rec, ii, path in iphone_queries_at_loc:
        print(f"Found recorder: {rec} at {path}")

        rec_type = inter
        rec_module = rec
        interaction_indices = ii

        iphone_data = IPhoneData(base_path, rec_location, rec_type, rec_module, interaction_indices, data_indexer)

        iphone_data.extract_rgbd()
        iphone_data.extract_poses()

    # Query for all Aria recorders at the specified location
    aria_queries_at_loc = data_indexer.query(
        location=rec_location, 
        interaction=None, 
        recorder="aria*",
        interaction_index=None
    )

    for loc, inter, rec, ii, path in aria_queries_at_loc:
        print(f"Found recorder: {rec} at {path}")

        rec_type = inter
        rec_module = rec
        interaction_indices = ii

        aria_data = AriaData(base_path, rec_location, rec_type, rec_module, interaction_indices, data_indexer)

        aria_data.request_mps(force=False)
        aria_data.extract_vrs(undistort=True)

        aria_data.extract_mps()
        aria_data.extract_mps_multi()