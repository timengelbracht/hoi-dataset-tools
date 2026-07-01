from pathlib import Path
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import RecordableTypeId, StreamId
from tqdm import tqdm
import numpy as np
from typing import Optional
import cv2


class VRSUtils:

    def __init__(self, vrs_path: str | Path, undistort: bool):
        self.vrs_file = Path(vrs_path)
        self.undistort = undistort
        self._load_provider_and_calib_from_vrs()
        

    def _load_provider_and_calib_from_vrs(self):
        if not self.vrs_file.exists():
            raise FileNotFoundError(f"VRS file not found: {self.vrs_file}")
        
        self.provider = data_provider.create_vrs_data_provider(str(self.vrs_file))
        if not self.provider:
            raise RuntimeError(f"Failed to create data provider for {self.vrs_file}")

        self.device_calib = self.provider.get_device_calibration()

        
    def get_frames_from_vrs(self, out_dir: str | Path | None = None) -> None:
        
        calib = self.device_calib.get_camera_calib("camera-rgb")
        f = self.device_calib.get_camera_calib("camera-rgb").get_focal_lengths()[0]
        h = self.device_calib.get_camera_calib("camera-rgb").get_image_size()[0]
        w = self.device_calib.get_camera_calib("camera-rgb").get_image_size()[1]

        pinhole = calibration.get_linear_camera_calibration(w, h, f)

        stream_id = self.provider.get_stream_id_from_label("camera-rgb")

        frames = []
        timestamps = []
        for i in tqdm(range(0, self.provider.get_num_data(stream_id)), total=self.provider.get_num_data(stream_id)):
            image_data =  self.provider.get_image_data_by_index(stream_id, i)
            sensor_data = self.provider.get_sensor_data_by_index(stream_id, i)
            ts = sensor_data.get_time_ns(TimeDomain.DEVICE_TIME)
            image_array = image_data[0].to_numpy_array()
            if self.undistort:
                image_array = calibration.distort_by_calibration(image_array, pinhole, calib)
                image_array = np.rot90(image_array, k=3)
            if out_dir is not None:
                out_file = out_dir / f"{ts}.png"
                image_array= cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)            
                cv2.imwrite(str(out_file), image_array)
            frames.append(image_array)
            timestamps.append(ts)

        return frames, timestamps


    def get_calibration_from_vrs(self) -> np.ndarray:

        """
        Returns the intrinsic matrix for the RGB camera.
        If undistort is True, it returns the undistorted intrinsic matrix.
        """

        if not self.device_calib:
            raise RuntimeError("Device calibration not loaded")
        
        clb = {}

        if self.undistort:
            f = self.device_calib.get_camera_calib("camera-rgb").get_focal_lengths()[0]
            h = self.device_calib.get_camera_calib("camera-rgb").get_image_size()[0]
            w = self.device_calib.get_camera_calib("camera-rgb").get_image_size()[1]

            pinhole = calibration.get_linear_camera_calibration(w, h, f)
            pinhole_rot = calibration.rotate_camera_calib_cw90deg(pinhole)
            f_x = pinhole_rot.get_projection_params()[0]
            f_y = pinhole_rot.get_projection_params()[1]
            c_x = pinhole_rot.get_projection_params()[2]
            c_y = pinhole_rot.get_projection_params()[3]
            K = np.array([f_x, 0, c_x, 0, f_y, c_y, 0, 0, 1]).reshape(3, 3)

            clb["K"] = K
            clb["h"] = h
            clb["w"] = w
            clb["model"] = "PINHOLE"
            clb["distortion"] = np.zeros(5, dtype=np.float32)
            clb["focal_length"] = np.array([f_x, f_y], dtype=np.float32)
            clb["principal_point"] = np.array([c_x, c_y], dtype=np.float32)
            clb["pinhole_T_device_camera"] = pinhole_rot.get_transform_device_camera().to_matrix()
            clb["T_device_camera"] = self.device_calib.get_camera_calib("camera-rgb").get_transform_device_camera().to_matrix()

            return clb
        else:
            calib = self.device_calib.get_camera_calib("camera-rgb")

            h, w = calib.get_image_size()
            f_x = calib.get_focal_lengths()[0]
            f_y = calib.get_focal_lengths()[1]
            c_x = calib.get_principal_point()[0]
            c_y = calib.get_principal_point()[1]
            K = np.array([f_x, 0, c_x, 0, f_y, c_y, 0, 0, 1]).reshape(3, 3)

            clb["K"] = K
            clb["h"] = h
            clb["w"] = w
            clb["model"] = "FISHEYE"
            clb["distortion"] = calib.get_projection_params()[3:7]
            clb["focal_length"] = np.array([f_x, f_y], dtype=np.float32)
            clb["principal_point"] = np.array([c_x, c_y], dtype=np.float32)
            clb["T_device_camera"] = calib.get_transform_device_camera().to_matrix()

            return clb

    def get_imu_from_vrs():
        pass