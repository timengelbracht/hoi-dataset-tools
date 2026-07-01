# cam_calib_loader.py
from pathlib import Path
from dataclasses import dataclass
import yaml   # pip install pyyaml

@dataclass
class CameraCalib:
    model: str                
    distortion_model: str    
    intrinsics: list          
    distortion: list         
    resolution: tuple      
    rostopic: str | None = None,
    T_cn_cnm1: list | None = None

@dataclass
class IMUCamCalib:
    T_cam_imu: list | None = None
    timeshift_cam_imu: float | None = None

@dataclass
class IMUCalib:
    T_imu_body: list | None = None
    T_imu_tool: list | None = None
    T_imu_sensor: list | None = None

def load_camchain(path: str | Path, cam_name: str = "cam0") -> CameraCalib:
    """Load *cam_name* (default cam0) from a Kalibr camchain YAML."""
    data = yaml.safe_load(Path(path).read_text())
    if cam_name not in data:
        raise KeyError(f"{cam_name} not found in {path}")

    block = data[cam_name]
    return CameraCalib(
        model             = block.get("camera_model"),
        distortion_model  = block.get("distortion_model"),
        intrinsics        = block.get("intrinsics"),
        distortion        = block.get("distortion_coeffs"),
        resolution        = tuple(block.get("resolution")),
        rostopic          = block.get("rostopic"),
        T_cn_cnm1         = block.get("T_cn_cnm1")
    )

def load_imucam(path: str | Path, imu_name: str = "cam0") -> dict:
    """Load *imu_name* (default imu0) from a Kalibr IMU YAML."""
    data = yaml.safe_load(Path(path).read_text())
    if imu_name not in data:
        raise KeyError(f"{imu_name} not found in {path}")

    block = data[imu_name]
    return IMUCamCalib(
        T_cam_imu         = block.get("T_cam_imu"),
        timeshift_cam_imu = block.get("timeshift_cam_imu")
    )

def load_imu(path: str | Path, imu_name: str = "imu0") -> dict:
    """Load *imu_name* (default imu0) from a Kalibr IMU YAML."""
    data = yaml.safe_load(Path(path).read_text())
    if imu_name not in data:
        raise KeyError(f"{imu_name} not found in {path}")

    block = data[imu_name]
    return IMUCalib(
        T_imu_body         = block.get("T_i_b"),
        T_imu_tool = block.get("T_i_tool"),
        T_imu_sensor = block.get("T_i_s")
    )
    

# ----------------------------------------------------------------------
if __name__ == "__main__":
    calib = load_camchain("calib_yellow-camchain.yaml")
    print(calib)