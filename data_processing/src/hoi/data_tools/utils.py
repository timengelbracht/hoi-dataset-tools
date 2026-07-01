from pathlib import Path
import numpy as np
import cv2
import re
import pandas as pd

def ensure_dir(path: Path) -> None:
    """Create directory and parents if not exists."""
    path.mkdir(parents=True, exist_ok=True)

def save_image(img_array: np.ndarray, out_path: Path) -> None:
    """Save image as PNG to disk."""
    out_path.write_bytes(cv2.imencode('.png', img_array)[1])

def clean_label(label: str) -> str:
    """Clean label string to strip leading/trailing slashes."""
    return label.strip("/")

def estimate_fps(timestamps: list[int]) -> float:
    """Estimate FPS from nanosecond timestamps."""
    time_diffs = np.diff(timestamps)
    avg_dt = np.mean(time_diffs)
    return 1e9 / avg_dt if avg_dt > 0 else 30.0

def load_sorted_images(directory: Path) -> list[Path]:
    """Return list of image paths sorted by integer timestamp name."""
    return sorted(directory.glob("*.png"), key=lambda x: int(x.stem))

def is_valid_image(img_path: Path) -> bool:
    """Check if image is readable."""
    try:
        return cv2.imread(str(img_path)) is not None
    except Exception:
        return False
    

# csv utils
def load_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    Load a CSV with timestamp as a column, keeping the default integer index.
    """
    return pd.read_csv(csv_path)


def get_df_row(df: pd.DataFrame, id: int | str, timestamp: bool = False) -> pd.Series:
    """
    Retrieve a row from the DataFrame.

    Args:
        df: The pandas DataFrame.
        id: The index (int) or timestamp (int or str).
        timestamp: If True, search by timestamp column; otherwise, use row index.

    Returns:
        A pandas Series representing one row.
    """
    if timestamp:
        ts = int(id)
        row = df[df["timestamp"] == ts]
        if row.empty:
            raise ValueError(f"No row found with timestamp {ts}")
        return row.iloc[0]
    else:
        return df.iloc[int(id)]


# ros utils
def parse_str_ros_geoemtry_msgs_pose(pose_str: str):
    """Helper function to parse position and quaternion from ROS pose string."""

    position_match = re.search(r"position=.*?x=([-\d.e+]+), y=([-\d.e+]+), z=([-\d.e+]+)", pose_str)
    orientation_match = re.search(r"orientation=.*?x=([-\d.e+]+), y=([-\d.e+]+), z=([-\d.e+]+), w=([-\d.e+]+)", pose_str)

    if position_match and orientation_match:
        tx, ty, tz = map(float, position_match.groups())
        qx, qy, qz, qw = map(float, orientation_match.groups())
        return pd.Series([tx, ty, tz, qx, qy, qz, qw])
    else:
        return pd.Series([None]*7)