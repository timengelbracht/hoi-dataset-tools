import re
from pathlib import Path
import pandas as pd
import ast 
import numpy as np

from .utils import load_csv, get_df_row


import pandas as pd
import re
from pathlib import Path
from typing import Dict, Any, Callable, Union, List

def ros_to_dict(obj: Any) -> Any:
    """
    Recursively turn a ROS-2 message (or field) into plain Python primitives.
    Works for scalars, lists, nested messages, arrays, etc.
    """
    # primitive leaf?
    if isinstance(obj, (int, float, bool, str)):
        return obj
    # numpy array or list of primitives
    if isinstance(obj, (list, tuple)):
        return [ros_to_dict(x) for x in obj]
    # ROS 2 messages expose __slots__
    if hasattr(obj, "__slots__"):
        out = {}
        for slot in obj.__slots__:
            out[slot] = ros_to_dict(getattr(obj, slot))
        return out
    # anything else (e.g. builtin time) – cast to str
    return str(obj)

def flatten_dict(d: Dict[str, Any], parent: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Flatten nested dicts/lists:
       {'a': {'b': 1}, 'c': [2,3]}  →  {'a.b':1, 'c.0':2, 'c.1':3}
    """
    items: List[tuple] = []
    for k, v in d.items():
        new_key = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, item in enumerate(v):
                items.extend(flatten_dict({str(i): item}, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def _parse_time_string(time_str: str) -> Dict[str, Any]:
    """
    Parses a string like "builtin_interfaces__msg__Time(sec=123, nanosec=456, __msgtype__='builtin_interfaces/msg/Time')"
    into a dictionary.
    """
    match = re.match(r"builtin_interfaces__msg__Time\(sec=(\d+), nanosec=(\d+), __msgtype__='([^']+)'\)", time_str)
    if match:
        return {
            'sec': int(match.group(1)),
            'nanosec': int(match.group(2)),
            '__msgtype__': match.group(3)
        }
    raise ValueError(f"Could not parse Time string: {time_str}")

def _parse_vector3_string(vector_str: str) -> Dict[str, Any]:
    """
    Parses a string like "geometry_msgs__msg__Vector3(x=1.0, y=2.0, z=3.0, __msgtype__='abc')"
    into a dictionary. Handles scientific notation.
    """
    match = re.match(
        r"geometry_msgs__msg__Vector3\(x=(-?\d+\.?\d*(?:[eE][-+]?\d*)?), y=(-?\d+\.?\d*(?:[eE][-+]?\d*)?), z=(-?\d+\.?\d*(?:[eE][-+]?\d*)?), __msgtype__='([^']+)'\)",
        vector_str
    )
    if match:
        return {
            'x': float(match.group(1)),
            'y': float(match.group(2)),
            'z': float(match.group(3)),
            '__msgtype__': match.group(4)
        }
    raise ValueError(f"Could not parse Vector3 string: {vector_str}")

def _parse_quaternion_string(quat_str: str) -> Dict[str, Any]:
    """
    Parses a string like "geometry_msgs__msg__Quaternion(x=0.0, y=0.0, z=0.0, w=1.0, __msgtype__='abc')"
    into a dictionary.
    """
    match = re.match(
        r"geometry_msgs__msg__Quaternion\(x=(-?\d+\.?\d*(?:[eE][-+]?\d*)?), y=(-?\d+\.?\d*(?:[eE][-+]?\d*)?), z=(-?\d+\.?\d*(?:[eE][-+]?\d*)?), w=(-?\d+\.?\d*(?:[eE][-+]?\d*)?), __msgtype__='([^']+)'\)",
        quat_str
    )
    if match:
        return {
            'x': float(match.group(1)),
            'y': float(match.group(2)),
            'z': float(match.group(3)),
            'w': float(match.group(4)),
            '__msgtype__': match.group(5)
        }
    raise ValueError(f"Could not parse Quaternion string: {quat_str}")

def _parse_point_string(point_str: str) -> Dict[str, Any]:
    """
    Parses a string like "geometry_msgs__msg__Point(x=1.0, y=2.0, z=3.0, __msgtype__='abc')"
    into a dictionary.
    """
    match = re.match(
        r"geometry_msgs__msg__Point\(x=(-?\d+\.?\d*(?:[eE][-+]?\d*)?), y=(-?\d+\.?\d*(?:[eE][-+]?\d*)?), z=(-?\d+\.?\d*(?:[eE][-+]?\d*)?), __msgtype__='([^']+)'\)",
        point_str
    )
    if match:
        return {
            'x': float(match.group(1)),
            'y': float(match.group(2)),
            'z': float(match.group(3)),
            '__msgtype__': match.group(4)
        }
    raise ValueError(f"Could not parse Point string: {point_str}")

def _parse_header_string(header_str: str) -> Dict[str, Any]:
    """
    Parses a string like "std_msgs__msg__Header(seq=123, stamp=builtin_interfaces__msg__Time(...), frame_id='abc', __msgtype__='xyz')"
    into a dictionary. Recursively calls _parse_time_string.
    """
    # This regex is more complex as it needs to capture the stamp sub-string accurately
    match = re.match(
        r"std_msgs__msg__Header\(seq=(\d+), stamp=(builtin_interfaces__msg__Time\(.+?\)), frame_id='([^']*)', __msgtype__='([^']+)'\)",
        header_str
    )
    if match:
        return {
            'seq': int(match.group(1)),
            'stamp': _parse_time_string(match.group(2)),
            'frame_id': match.group(3),
            '__msgtype__': match.group(4)
        }
    raise ValueError(f"Could not parse Header string: {header_str}")

def _parse_transform_string(transform_str: str) -> Dict[str, Any]:
    """
    Parses a string like "geometry_msgs__msg__Transform(translation=Vector3(...), rotation=Quaternion(...), __msgtype__='abc')"
    into a dictionary. Recursively calls _parse_vector3_string and _parse_quaternion_string.
    """
    # This regex needs to capture the translation and rotation sub-strings accurately
    match = re.match(
        r"geometry_msgs__msg__Transform\(translation=(geometry_msgs__msg__Vector3\(.+?\)), rotation=(geometry_msgs__msg__Quaternion\(.+?\)), __msgtype__='([^']+)'\)",
        transform_str
    )
    if match:
        return {
            'translation': _parse_vector3_string(match.group(1)),
            'rotation': _parse_quaternion_string(match.group(2)),
            '__msgtype__': match.group(3)
        }
    raise ValueError(f"Could not parse Transform string: {transform_str}")

def _parse_transform_stamped_string(ts_str: str) -> Dict[str, Any]:
    """
    Parses a string like "geometry_msgs__msg__TransformStamped(header=Header(...), child_frame_id='abc', transform=Transform(...), __msgtype__='xyz')"
    into a dictionary. Recursively calls _parse_header_string and _parse_transform_string.
    """
    match = re.match(
        r"geometry_msgs__msg__TransformStamped\(header=(std_msgs__msg__Header\(.+?\)), child_frame_id='([^']*)', transform=(geometry_msgs__msg__Transform\(.+?\)), __msgtype__='([^']+)'\)",
        ts_str
    )
    if match:
        return {
            'header': _parse_header_string(match.group(1)),
            'child_frame_id': match.group(2),
            'transform': _parse_transform_string(match.group(3)),
            '__msgtype__': match.group(4)
        }
    raise ValueError(f"Could not parse TransformStamped string: {ts_str}")

def _parse_pose_string(pose_str: str) -> Dict[str, Any]:
    """
    Parses a string like "geometry_msgs__msg__Pose(position=Point(...), orientation=Quaternion(...), __msgtype__='abc')"
    into a dictionary. Recursively calls _parse_point_string and _parse_quaternion_string.
    """
    match = re.match(
        r"geometry_msgs__msg__Pose\(position=(geometry_msgs__msg__Point\(.+?\)), orientation=(geometry_msgs__msg__Quaternion\(.+?\)), __msgtype__='([^']+)'\)",
        pose_str
    )
    if match:
        return {
            'position': _parse_point_string(match.group(1)),
            'orientation': _parse_quaternion_string(match.group(2)),
            '__msgtype__': match.group(3)
        }
    raise ValueError(f"Could not parse Pose string: {pose_str}")

def _parse_float_list_string(list_str: str) -> List[float]:
    """
    Parses a string representation of a float list (e.g., "[0. 0. 0.]" or "[1.2, 3.4e-5]")
    into a Python list of floats.
    """
    # Remove brackets and split by space or comma, then convert to float
    cleaned_str = list_str.strip('[] ').replace(',', ' ')
    if not cleaned_str:
        return []
    try:
        return [float(x) for x in cleaned_str.split()]
    except ValueError as e:
        raise ValueError(f"Could not parse float list string '{list_str}': {e}")


# --- 2. Configuration Mapping for ROS Message Types ---

# This dictionary stores the parsing rules for each ROS message type.
# Each entry specifies:
#   - 'mapping': How CSV columns map to nested dictionary paths.
#                Value is a tuple (csv_column_name, parser_function or None).
#                For combined fields (like Imu angular_velocity), the value
#                is a tuple (list of csv_column_names, combiner_function).
ROS_MESSAGE_PARSING_CONFIG: Dict[str, Dict[str, Any]] = {
    "geometry_msgs/WrenchStamped": {
        "mapping": {
            "timestamp": ("timestamp", None),
            "header.seq": ("header.seq", None),
            "header.stamp": ("header.stamp", _parse_time_string),
            "header.frame_id": ("header.frame_id", None),
            "header.__msgtype__": ("header.__msgtype__", None),
            "wrench.force": ("wrench.force", _parse_vector3_string),
            "wrench.torque": ("wrench.torque", _parse_vector3_string),
            "wrench.__msgtype__": ("wrench.__msgtype__", None),
        }
    },
    "sensor_msgs/JointState": {
        "mapping": {
            "timestamp": ("timestamp", None),
            "header.seq": ("header.seq", None),
            "header.stamp": ("header.stamp", _parse_time_string),
            "header.frame_id": ("header.frame_id", None),
            "header.__msgtype__": ("header.__msgtype__", None),
            "name.0": ("name.0", None), # Assuming single name for simplicity based on 'name.0'
            "position": ("position", _parse_float_list_string),
            "velocity": ("velocity", _parse_float_list_string),
            "effort": ("effort", _parse_float_list_string),
        }
    },
    "sensor_msgs/Temperature": {
        "mapping": {
            "timestamp": ("timestamp", None),
            "header.seq": ("header.seq", None),
            "header.stamp": ("header.stamp", _parse_time_string),
            "header.frame_id": ("header.frame_id", None),
            "header.__msgtype__": ("header.__msgtype__", None),
            "temperature": ("temperature", None), # Direct float
            "variance": ("variance", None),     # Direct float
        }
    },
    "sensor_msgs/Imu": {
        "mapping": {
            "timestamp": ("timestamp", None),
            "header.seq": ("header.seq", None),
            "header.stamp": ("header.stamp", _parse_time_string),
            "header.frame_id": ("header.frame_id", None),
            "header.__msgtype__": ("header.__msgtype__", None),
            "angular_velocity": (
                ["angular_velocity.x", "angular_velocity.y", "angular_velocity.z", "angular_velocity.__msgtype__"],
                lambda x, y, z, msgtype: {'x': float(x), 'y': float(y), 'z': float(z), '__msgtype__': str(msgtype)}
            ),
            "angular_velocity_covariance": ("angular_velocity_covariance", _parse_float_list_string),
            "linear_acceleration": (
                ["linear_acceleration.x", "linear_acceleration.y", "linear_acceleration.z", "linear_acceleration.__msgtype__"],
                lambda x, y, z, msgtype: {'x': float(x), 'y': float(y), 'z': float(z), '__msgtype__': str(msgtype)}
            ),
            "linear_acceleration_covariance": ("linear_acceleration_covariance", _parse_float_list_string),
            "orientation": (
                ["orientation.x", "orientation.y", "orientation.z", "orientation.w", "orientation.__msgtype__"],
                lambda x, y, z, w, msgtype: {'x': float(x), 'y': float(y), 'z': float(z), 'w': float(w), '__msgtype__': str(msgtype)}
            ),
            "orientation_covariance": ("orientation_covariance", _parse_float_list_string),
        }
    },
    "std_msgs/Float32": {
        "mapping": {
            "timestamp": ("timestamp", None),
            "data": ("data", lambda x: float(x)) # Convert to float
        }
    },
    "tf2_msgs/TFMessage": {
        "mapping": {
            "timestamp": ("timestamp", None),
            # Dynamic fields like transforms.0, transforms.1, ... will be handled separately
            # The 'transforms' key below is a placeholder to indicate it's a list.
            # Actual columns will be iterated over in parse_ros_message_row_to_dict.
            "transforms": (None, None) # Special handling in the main parsing function
        }
    },
    "geometry_msgs/PoseWithCovarianceStamped": {
        "mapping": {
            "timestamp": ("timestamp", None),
            "header.seq": ("header.seq", None),
            "header.stamp": ("header.stamp", _parse_time_string),
            "header.frame_id": ("header.frame_id", None),
            "header.__msgtype__": ("header.__msgtype__", None),
            "pose.pose": ("pose.pose", _parse_pose_string), # This contains Point and Quaternion
            "pose.covariance": ("pose.covariance", _parse_float_list_string), # 36-element covariance matrix
            "pose.__msgtype__": ("pose.__msgtype__", None),
        }
    }
}

# --- 3. Main Parsing Function (Revised) ---

def _set_nested_value(d: Dict[str, Any], keys: str, value: Any):
    """Sets a value in a nested dictionary using dot-separated keys."""
    parts = keys.split('.')
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value

def parse_ros_message_row_to_dict(row: pd.Series, ros_msg_type: str) -> Dict[str, Any]:
    """
    Parses a single pandas DataFrame row (representing a ROS message) into a
    nested dictionary based on the predefined configuration.

    Args:
        row: A pandas Series representing one row of the CSV.
        ros_msg_type: The full ROS message type string (e.g., 'geometry_msgs/WrenchStamped').

    Returns:
        A nested dictionary representing the parsed ROS message.
    """
    config = ROS_MESSAGE_PARSING_CONFIG.get(ros_msg_type)
    if not config:
        raise ValueError(f"No parsing configuration found for ROS message type: {ros_msg_type}")

    parsed_data = {}
    mapping = config["mapping"]

    for target_path, definition in mapping.items():
        if target_path == "transforms" and ros_msg_type == "tf2_msgs/TFMessage":
            # Special handling for tf2_msgs/TFMessage's dynamic 'transforms.X' columns
            transforms_list = []
            transform_col_pattern = re.compile(r"transforms\.(\d+)")
            for col_name in row.index:
                match = transform_col_pattern.match(col_name)
                if match:
                    transform_str = row[col_name]
                    if pd.isna(transform_str):
                        transforms_list.append(None)
                    else:
                        try:
                            parsed_transform = _parse_transform_stamped_string(str(transform_str))
                            transforms_list.append(parsed_transform)
                        except ValueError as e:
                            print(f"Error parsing TFMessage transform string '{transform_str}': {e}. Setting to None.")
                            transforms_list.append(None)
            _set_nested_value(parsed_data, "transforms", transforms_list)
            continue # Skip normal processing for this entry

        if isinstance(definition[0], list): # This handles combined fields (like Imu's Vector3/Quaternion components)
            csv_columns = definition[0]
            combiner_func = definition[1]
            try:
                # Get values from row for the specified CSV columns
                values = [row[col] for col in csv_columns if col in row.index and not pd.isna(row[col])]
                # If some columns are missing or NaN, set the target to None or handle as appropriate
                if len(values) != len(csv_columns):
                    _set_nested_value(parsed_data, target_path, None)
                    continue
                
                # Apply the combiner function to the values
                parsed_value = combiner_func(*values)
                _set_nested_value(parsed_data, target_path, parsed_value)
            except KeyError as e:
                print(f"Warning: Missing column for combined field '{target_path}': {e}. Setting to None.")
                _set_nested_value(parsed_data, target_path, None)
            except ValueError as e:
                print(f"Error combining values for '{target_path}': {e}. Setting to None.")
                _set_nested_value(parsed_data, target_path, None)
            continue

        # Normal single column processing
        csv_column_name, parser = definition

        if csv_column_name not in row.index:
            print(f"Warning: CSV column '{csv_column_name}' not found in row for {ros_msg_type}. Skipping '{target_path}'.")
            _set_nested_value(parsed_data, target_path, None) # Set to None if column is missing
            continue

        value = row[csv_column_name]
        if pd.isna(value):
            _set_nested_value(parsed_data, target_path, None)
            continue

        if parser:
            try:
                # Ensure value is string for regex parsing or other string-based parsers
                parsed_value = parser(str(value))
            except ValueError as e:
                print(f"Error parsing '{csv_column_name}' with value '{value}' for '{target_path}': {e}. Setting to None.")
                parsed_value = None
            _set_nested_value(parsed_data, target_path, parsed_value)
        else:
            _set_nested_value(parsed_data, target_path, value)

    return parsed_data

def ros_message_to_dict_recursive(msg_obj: Any) -> Dict[str, Any]:
    """
    Recursively converts a ROS message object (or any object with public attributes
    like rosbags.usertypes) into a nested Python dictionary, making it JSON-serializable.
    Handles standard ROS message fields, builtin types, lists, and numpy arrays.
    Specifically deals with builtin_interfaces__msg__Time and other custom types.
    """

    if isinstance(msg_obj, (str, int, float, bool, bytes, type(None))):
        return msg_obj

    if isinstance(msg_obj, np.ndarray):
        return msg_obj.tolist()

    if isinstance(msg_obj, dict):
        return {k: ros_message_to_dict_recursive(v) for k, v in msg_obj.items()}
    if isinstance(msg_obj, list):
        return [ros_message_to_dict_recursive(item) for item in msg_obj]

    data = {}

    # Detect and annotate message type
    msg_type_found = False
    if hasattr(msg_obj, '_TYPE'):
        data['__msgtype__'] = msg_obj._TYPE
        msg_type_found = True
    elif hasattr(msg_obj, '_type'):
        data['__msgtype__'] = msg_obj._type
        msg_type_found = True
    else:
        module_name = msg_obj.__class__.__module__
        class_name = msg_obj.__class__.__name__
        if 'rosbags.usertypes.' in module_name:
            parts = class_name.split('__')
            if len(parts) >= 3 and parts[1] == 'msg':
                data['__msgtype__'] = f"{parts[0]}/{parts[2]}"
            else:
                data['__msgtype__'] = f"{module_name.split('.')[-1]}/{class_name}"
        elif module_name.startswith(('std_msgs', 'geometry_msgs', 'sensor_msgs', 'tf2_msgs', 'builtin_interfaces')):
            if 'builtin_interfaces' in module_name and 'Time' in class_name:
                data['__msgtype__'] = 'builtin_interfaces/Time'
            else:
                data['__msgtype__'] = f"{module_name.split('.')[-1]}/{class_name}"
        else:
            data['__msgtype__'] = f"{class_name}"

    # Determine fields to process
    if hasattr(msg_obj, '__slots__'):
        fields_to_process = msg_obj.__slots__
    elif hasattr(msg_obj, '_fields_and_field_types'):
        fields_to_process = msg_obj._fields_and_field_types.keys()
    else:
        # Fallback to public attributes
        fields_to_process = [
            attr for attr in dir(msg_obj)
            if not attr.startswith('_') and not callable(getattr(msg_obj, attr, None))
        ]

    for field_name in fields_to_process:
        try:
            field_value = getattr(msg_obj, field_name)
            data[field_name] = ros_message_to_dict_recursive(field_value)
        except Exception as e:
            data[field_name] = f"<unserializable: {type(field_name).__name__}>"

    # # Final pass: ensure all values are JSON serializable
    # for k, v in data.items():
    #     try:
    #         json.dumps(v)
    #     except TypeError:
    #         data[k] = str(v)

    return data


if __name__ == "__main__":  

    topic = "/zedm/zed_node/pose_with_covariance"
    NON_IMAGE_TOPICS = {
        "/gripper_force_trigger": "std_msgs/Float32",
        "/joint_states": "sensor_msgs/JointState",
        "/tf": "tf2_msgs/TFMessage",
        "/tf_static": "tf2_msgs/TFMessage",
        "/zedm/zed_node/imu/data": "sensor_msgs/Imu",
        "/zedm/zed_node/imu/data_raw": "sensor_msgs/Imu",
        "/zedm/zed_node/pose_with_covariance": "geometry_msgs/PoseWithCovarianceStamped",
        "/force_torque/ft_sensor0/ft_sensor_readings/imu": "sensor_msgs/Imu",
        "/force_torque/ft_sensor0/ft_sensor_readings/temperature": "sensor_msgs/Temperature",
        "/force_torque/ft_sensor0/ft_sensor_readings/wrench": "geometry_msgs/WrenchStamped"
    }
    msg_type = NON_IMAGE_TOPICS[topic]

    path = Path(f"/data/dlab_recordings/extracted/force_torque_test/gripper_right/{topic.strip('/')}/data.csv")
    df = load_csv(path)
    message = get_df_row(df, 0, timestamp=False)

    message_dict = parse_ros_message_row_to_dict(message, msg_type)

    a = 2
