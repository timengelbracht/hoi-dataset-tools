
from hoi.data_tools.data_loader_aria import AriaData
from hoi.data_tools.data_indexer import RecordingIndex
from pathlib import Path
import os

if __name__ == "__main__":
    test = True
    
    rec_location = "bedroom_6"
    base_path = Path(f"/data/ikea_recordings")
    data_indexer = RecordingIndex(
        os.path.join(str(base_path), "raw") 
    )
    
    rec_type = "hand"
    rec_module = "aria_human"
    interaction_indices = "1-5"

    aria_data = AriaData(base_path, rec_location, rec_type, rec_module, interaction_indices, data_indexer)
    aria_data.visualize_camera_transforms()