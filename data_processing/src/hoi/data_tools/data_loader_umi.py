import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
from pathlib import Path
import json
from typing import Optional, Union
import numpy as np
import cv2
from tqdm import tqdm
import zipfile
import open3d as o3d
import pandas as pd
from scipy.spatial.transform import Rotation as R
from .data_indexer import RecordingIndex
import telemetry_parser
import subprocess
from .utils_keyframing import KeyframeExtractor
from .utils import ensure_dir
import shutil
import av
from .utils_mp4 import get_frames_from_mp4, get_imu_from_mp4
from .utils_yaml import load_camchain, load_imucam
import math
from PIL import Image
import torch
from transformers import pipeline
from accelerate.test_utils.testing import get_backend
import subprocess
import os
import sys
import shlex
from .utils_anonymization import EgoBlurFaceAnonymizer


class UmiData:

        # monodepth model setup
    DEVICE, _, _ = get_backend()
    MONO_DEPTH_CHECKPOINT = "depth-anything/Depth-Anything-V2-base-hf"
    PIPE_MONO_DEPTH = pipeline("depth-estimation", model=MONO_DEPTH_CHECKPOINT, device=DEVICE)

    def __init__(self, base_path: Path, 
                 rec_loc: str, 
                 rec_type: str,
                 rec_module: str,
                 interaction_indices: str,
                 data_indexer: Optional[RecordingIndex] = None,
                 color: str = "yellow") -> None:

        self.rec_loc = rec_loc
        self.base_path = base_path
        self.rec_type = rec_type
        self.rec_module = rec_module
        self.interaction_indices = interaction_indices
        if data_indexer is not None:
            self.data_indexer = data_indexer


        self.mp4_path_raw = Path(self.base_path) / "raw" / self.rec_loc / self.rec_type /  "umi_gripper" / f"{self.rec_loc}_{self.interaction_indices}_{self.rec_type}.MP4"
        
        self.extraction_path_base = self.base_path / "extracted" / self.rec_loc / self.rec_type
        self.extraction_path = self.base_path / "extracted" / self.rec_loc / self.rec_type / self.rec_module / f"{self.rec_loc}_{self.interaction_indices}_{self.rec_type}"


        self.mask_path = Path(self.base_path) / "raw" / "umi_meta" / "umi_mask.png"
        self.calibration_path = Path(self.base_path) / "raw" / "umi_meta" / "calib" / color

        self.label_rgb = f"/camera_rgb"
        self.label_depth = f"/camera_depth"
        self.label_imu = f"/telemetry"
        self.label_keyframes = f"visual_registration/keyframes/rgb"
        self.label_bag = f"odometry/data.bag"
        self.label_poses = f"slam/poses"
        self.label_odometry = f"odometry/poses"

        self.visual_registration_output_path = self.extraction_path / "visual_registration"

        self.K = None
        self.fps = None
        self.timestamps = None
        self.logging_tag = f"{self.rec_loc}_{self.rec_type}_{self.rec_module}".upper()

        self.extract_umi_meta_data()
        self.calibration = self.get_calibration()

        self.t_ns_init = 0        
        self.rgb_extension = "jpg"
        self.statistics = {}
        self.anonym_info: dict = {}

    def extract_umi_meta_data(self):
        """Seed the per-recording calib + mask from raw/umi_meta.

        Each extracted UMI recording carries its own calib/mask copy, so this
        skips entirely when they already exist — letting the loader run on
        extracted-only datasets (no raw/umi_meta required). raw is only read
        when the extracted copies are missing (i.e. during a fresh extraction).
        """
        calib_dst = self.extraction_path / "calib" / "calib_pinhole-equi"
        mask_dst = self.extraction_path / "calib" / "umi_mask.png"

        # already seeded (extracted-only mode) → nothing to copy from raw
        if calib_dst.is_dir() and any(calib_dst.iterdir()) and mask_dst.exists():
            return

        # copy calib
        calibration_path = self.calibration_path
        if not calibration_path.exists():
            raise FileNotFoundError(f"[{self.logging_tag}] calib folder not found at {self.calibration_path}")

        ensure_dir(calib_dst)

        # copy all files from mask_path to mask_dst
        for file in calibration_path.glob("*"):
            if file.is_file():
                shutil.copy2(file, calib_dst / file.name)

        # copy mask
        if not self.mask_path.exists():
            raise FileNotFoundError(f"[{self.logging_tag}] mask file not found at {self.mask_path}")

        if not mask_dst.exists():
            shutil.copy2(self.mask_path, mask_dst)


    def get_calibration(self):
        """
        load metadata from camera calibration file
        """
        # types of calibration models taht we calibrated beforehand
        # pinhole-equi for pycolmap/ hloc
        # omni-radtan for openvins
        calibration = {}
        calibration_models = ["pinhole-equi"]

        for model in calibration_models:
            calib_path = self.extraction_path / "calib" / f"calib_{model}"

            if not calib_path.exists():
                print(f"[{self.logging_tag}] No calibration file found at {calib_path}")
                continue

            file = calib_path.glob("*camchain.yaml")
            file_imucam = calib_path.glob("*imucam.yaml")

            calib_file = next(file, None)
            if calib_file is None:
                print(f"[{self.logging_tag}] No calibration file found at {calib_path}")
                continue

            calib_file_imucam = next(file_imucam, None)
            if calib_file_imucam is None:
                print(f"[{self.logging_tag}] No IMU calibration file found at {calib_path}")
                continue

            print(f"[{self.logging_tag}] Loading calibration file {calib_file}")
            print(f"[{self.logging_tag}] Loading IMU calibration file {calib_file_imucam}")

            calib_data = load_camchain(calib_file, cam_name="cam0")
            calib_data_imu = load_imucam(calib_file_imucam, imu_name="cam0")

            clb = {}
            h = calib_data.resolution[1]
            w = calib_data.resolution[0]
            f_x, f_y = calib_data.intrinsics[0], calib_data.intrinsics[1]
            c_x, c_y = calib_data.intrinsics[2], calib_data.intrinsics[3]
            disortion = calib_data.distortion
            timeshift_cam_imu = calib_data_imu.timeshift_cam_imu
            T_cam_imu = calib_data_imu.T_cam_imu

            K = np.array([
                [f_x, 0, c_x],
                [0, f_y, c_y],
                [0, 0, 1]
            ], dtype=np.float32)

            # convert to dictionary
            if calib_data.model == "pinhole" and calib_data.distortion_model == "equidistant":
                model = "OPENCV_FISHEYE"
                type = "PINHOLE"
                colmap_camera_cfg = {
                "model":  model,
                "width":   w,
                "height":  h,
                "params": [f_x, f_y, c_x, c_y] + disortion,
                }

            elif calib_data.model == "omni" and calib_data.distortion_model == "radtan":
                model = "OMNI_RADTAN"
                type = "NON_PINHOLE"
                colmap_camera_cfg = {}

            clb["K"] = K
            clb["model"] = model
            clb["w"] = w
            clb["h"] = h
            clb["focal_length"] = np.array([f_x, f_y], dtype=np.float32)
            clb["principal_point"] = np.array([c_x, c_y], dtype=np.float32)
            clb["distortion"] = np.array(disortion, dtype=np.float32)
            clb["colmap_camera_cfg"] = colmap_camera_cfg
            clb["T_cam_imu"] = np.array(T_cam_imu, dtype=np.float32) 
            clb["timeshift_cam_imu"] = timeshift_cam_imu

            calibration[type] = clb


        return calibration

    def extract_mp4(self):
        """
        Extracts RGB frames and IMU data from the MP4 file.
        """
        
        if not self.mp4_path_raw.exists():
            raise FileNotFoundError(
                f"[{self.logging_tag}] No MP4 file found at {self.mp4_path_raw}"
            )
    
        out_dir_rgb = self.extraction_path / self.label_rgb.strip("/")
        if not self._extracted(self.label_rgb):
            ensure_dir(out_dir_rgb)
            get_frames_from_mp4(
                mp4_path=self.mp4_path_raw,
                outdir=out_dir_rgb, 
                ext=self.rgb_extension,
            )
        else:
            print(f"[{self.logging_tag}] RGB frames already extracted to {out_dir_rgb}")

        outdir_telemetry = self.extraction_path / self.label_imu.strip("/")
        if not self._extracted(self.label_imu):
            ensure_dir(outdir_telemetry)
            imu_df = get_imu_from_mp4(
                mp4_file=self.mp4_path_raw,
                outdir=outdir_telemetry
            )
        else:
            print(f"[{self.logging_tag}] IMU data already extracted to {outdir_telemetry / self.label_imu.strip('/')}")

    def extract_keyframes(self, stride: int = 2, force: bool = False, n_keyframes: int = 20) -> None:

        if self._extracted(self.label_keyframes) and not force:
            print(f"[{self.logging_tag}] Keyframes already extracted to {self.visual_registration_output_path / self.label_keyframes.strip('/')}")
            return
        
        if not (
            self._extracted(self.label_rgb)
            or self._extracted(self.label_depth)
        ):
            raise FileNotFoundError(
            f"[{self.logging_tag}] No RGB, depth data found in "
            f"{self.extraction_path / self.label_rgb.strip('/')} or "
            f"{self.extraction_path / self.label_depth.strip('/')}"
            )
        
        out_dir = self.extraction_path / self.label_keyframes.strip("/")
        ensure_dir(out_dir)

        rgb_files = sorted(
            self.extraction_path.glob(f"{self.label_rgb.strip('/')}/**/*.jpg"),
            key=lambda x: int(x.stem)
        )

        depth_files = sorted(
            self.extraction_path.glob(f"{self.label_depth.strip('/')}/**/*.npy"),
            key=lambda x: int(x.stem)
        )
    
        # setup the keyframe extractor
        keyframe_extractor = KeyframeExtractor(
            rgb_files=rgb_files,
            depth_files=depth_files,
            n_keyframes=n_keyframes,
            stride=stride)

        # if not any(self.statistics.values()):
        #     print(f"[{self.logging_tag}] No statistics found, computing...")
        #     self.get_statistics(stride=stride)
        
        # keyframe_extractor.statistics = self.statistics
        # keyframe_extractor.visualize_statistics()
        self.selected_keyframes = keyframe_extractor.extract_keyframes_evenly_spaced()
        out_dir.mkdir(parents=True, exist_ok=True)

        # write the list
        with open(out_dir / "keyframes.txt", "w") as f:
            for p in self.selected_keyframes:
                f.write(p.name + "\n")

        # copy the actual image files
        for p in self.selected_keyframes:
            dst = out_dir / p.name
            shutil.copy2(p, dst)  # preserves metadata

        print(f"[{self.logging_tag}] Wrote {len(self.selected_keyframes)} keyframes to {out_dir}")

    def anonymize_rgb(
        self,
        model_dir: Optional[Path] = None,
        tmp_dir: Optional[Path] = None,
        max_face_image_size: int = 640,
        force: bool = False,
    ) -> dict:
        """
        Run EgoBlur face anonymization *in place* on the UMI RGB frames
        extracted from the MP4.

        - Operates on /camera_rgb (self.label_rgb).
        - Overwrites original JPGs only if faces are detected.
        - Tracks status in anonymization.json to avoid repeated work.
        """

        # ensure RGB frames exist
        if not self._extracted(self.label_rgb):
            raise FileNotFoundError(
                f"[{self.logging_tag}] No RGB data found in "
                f"{self.extraction_path / self.label_rgb.strip('/')}. "
                "Run extract_mp4() first."
            )

        rgb_dir = self.extraction_path / self.label_rgb.strip("/")
        rgb_files = sorted(
            rgb_dir.glob(f"*.{self.rgb_extension}"),
            key=lambda p: int(p.stem),
        )

        if len(rgb_files) == 0:
            print(f"[{self.logging_tag}] No RGB frames to anonymize in {rgb_dir}.")
            return {"faces": 0}

        # load anonymization metadata
        anonym_info_path = self.extraction_path / "anonymization.json"
        self.load_anonym_info(anonym_info_path if anonym_info_path.exists() else None)

        stream_key = self.label_rgb  # "/camera_rgb"

        # skip if already anonymized and not forced
        if not force and stream_key in self.anonym_info:
            entry = self.anonym_info[stream_key]
            if entry.get("faces_anonymized", False):
                print(
                    f"[{self.logging_tag}] Stream {stream_key} already anonymized "
                    f"(faces_anonymized=True). Skipping."
                )
                return {"faces": entry.get("num_faces", 0)}

        # defaults for model + tmp dir
        if model_dir is None:
            # wherever you keep the EgoBlur weights
            model_dir = self.base_path / "ego_blur_weights"

        if tmp_dir is None:
            tmp_dir = self.extraction_path / "anonymization_cache" / "egoblur_umi_rgb"

        print(
            f"[{self.logging_tag}] Running EgoBlur face anonymization in-place on "
            f"{len(rgb_files)} UMI RGB frames.\n"
            f"  stream={stream_key}\n"
            f"  model_dir={model_dir}\n"
            f"  tmp_dir={tmp_dir}"
        )

        anonymizer = EgoBlurFaceAnonymizer(anonymization_dir=model_dir)

        counts = anonymizer.run_anonymization(
            image_paths=rgb_files,
            tmp_dir=tmp_dir,
            outdir=None,             # ignored when inplace=True
            inplace=True,            # overwrite original JPGs
            overwrite=True,          # explicit, though irrelevant for inplace
            max_face_image_size=max_face_image_size,
        )

        # update metadata
        self.anonym_info[stream_key] = {
            "faces_anonymized": True,
            "num_faces": counts.get("faces", 0),
            "num_images": len(rgb_files),
            "method": "EgoBlurFaceAnonymizer",
            "inplace": True,
        }
        self.save_anonym_info(anonym_info_path)

        print(
            f"[{self.logging_tag}] EgoBlur (UMI RGB) finished: "
            f"{counts['faces']} faces blurred across {len(rgb_files)} frames (in-place)."
        )
        return counts
    
    def extract_mono_depth(
        self,
        downsampling_factor: int = 2,
        batch_size: int = 8,
        force: bool = False,
    ) -> None:
        """Batch the images in pure Python, call PIPE once per batch, save each depth map."""

        out_dir = self.extraction_path / self.label_depth.strip("/")
        if out_dir.exists() and any(out_dir.glob("*.npy")) and not force:
            print(f"[{self.logging_tag}] already extracted → {out_dir}")
            return
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) Gather & sort all the image paths
        img_paths = sorted(
            self.extraction_path.glob(f"{self.label_rgb.strip('/')}/**/*.jpg")
        )
        total = len(img_paths)
        if total == 0:
            print(f"[{self.logging_tag}] no images found")
            return

        # 2) Process in Python chunks of batch_size
        n_batches = math.ceil(total / batch_size)
        pbar = tqdm(total=total, desc=f"[{self.logging_tag}] Processing monodepth batches", unit="batch")
        for i in range(n_batches):
            start = i * batch_size
            end   = min(start + batch_size, total)
            batch_paths = img_paths[start:end]

            # load + preprocess into PIL list
            batch_pils = []
            stems      = []
            for p in batch_paths:
                img = cv2.imread(str(p))
                if img is None:
                    print(f"[{self.logging_tag}] skipping {p}")
                    continue
                h, w = img.shape[:2]
                img = cv2.resize(img, (w // downsampling_factor, h // downsampling_factor))
                batch_pils.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                stems.append(p.stem)

            # 3) Single batched call
            with torch.no_grad():
                preds = self.PIPE_MONO_DEPTH(batch_pils)  # returns list of dicts

            # 4) Save out each result
            for pred, stem in zip(preds, stems):
                depth_map = pred["predicted_depth"].squeeze().cpu().numpy()
                np.save(out_dir / f"{stem}.npy", depth_map)

            pbar.update(len(batch_pils))

        print(f"[{self.logging_tag}] done! depth maps in {out_dir}")

    def extract_euroc_format_for_orbslam(self, force: bool = False, mask: bool = False, image_scale: int = 3, stride: int = 2) -> None:
        """
        Convert this recording into EuRoC-style folder structure compatible with
        ORB-SLAM3.  If the output already exists, nothing is done unless
        `force=True`.
        """

        root_out   = self.extraction_path / "odometry" / "euroc_style" / "mav0"
        cam_data   = root_out / "cam0" / "data"
        imu_csv    = root_out / "imu0" / "data.csv"
        ts_txt     = self.extraction_path / "odometry" / "euroc_style" / "timestamps.txt"

        if cam_data.is_dir() and imu_csv.is_file() and ts_txt.is_file() and not force:
            print("[extract] EuroC folder already exists — skip.   (use force=True to rebuild)")
            return
        
        if mask:
            mask = Path(self.extraction_path / "calib" / "umi_mask.png")
            if not mask.is_file():
                raise FileNotFoundError(f"Mask file not found: {mask}")
                
            # load mask
            mk = cv2.imread(str(mask), cv2.IMREAD_UNCHANGED)   # or cv2.IMREAD_UNCHANGED == -1
            if mk is None:
                raise FileNotFoundError(mask)
            
            # rescale mask to the resolution of the camera
            h = self.calibration["PINHOLE"]["h"]
            w = self.calibration["PINHOLE"]["w"]
            mk = cv2.resize(mk, (w//image_scale, h//image_scale), interpolation=cv2.INTER_NEAREST)

        # fresh tree
        (cam_data).mkdir(parents=True, exist_ok=True)
        (imu_csv.parent).mkdir(parents=True, exist_ok=True)

        # imu
        calib   = self.calibration
        offset  = calib["PINHOLE"]["timeshift_cam_imu"]      # [s]
        imu_in  = self.extraction_path / self.label_imu.strip("/") / "data.csv"

        imu_df = pd.read_csv(imu_in)
        imu_df["timestamp"] += int(offset * 1e9)             
        imu_df.to_csv(imu_csv, index=False, header=False)

        # frames
        w_full  = calib["PINHOLE"]["w"]
        h_full  = calib["PINHOLE"]["h"]
        w_ds, h_ds = w_full // image_scale, h_full // image_scale

        rgb_src = self.extraction_path / self.label_rgb.strip("/")
        jpgs    = sorted(rgb_src.glob("*.jpg"), key=lambda p: int(p.stem))

        timestamps = []
        for i, jpg_path in enumerate(tqdm(jpgs, desc="[extract] downsample+mask")):
            if i % stride:                          # temporal stride 2  (30 → 15 fps)
                continue

            img = cv2.imread(str(jpg_path), cv2.IMREAD_COLOR)
            if img is None:
                print("  could not read", jpg_path)
                continue

            # resize (spatial stride 3)
            img = cv2.resize(img, (w_ds, h_ds), interpolation=cv2.INTER_LINEAR)

            if mask:
                # apply mask
                img = cv2.bitwise_and(img, img, mask=mk[:,:,-1])

            stamp = int(jpg_path.stem)
            cv2.imwrite(str(cam_data / f"{stamp}.png"), img)
            timestamps.append(stamp)

        # ───────── 3.  write timestamp list ─────────────────────────────────────
        with open(ts_txt, "w") as f:
            f.write("\n".join(map(str, timestamps)))

        print(f"[extract] wrote {len(timestamps)} frames  →  {cam_data}")
        print(f"[extract] IMU csv saved                →  {imu_csv}")

    def run_orbslam(self, docker_root_path: str | Path) -> None:
        """
        Run ORB-SLAM3 on the extracted data.
        """

        # Paths
        compose_dir = os.path.expanduser(docker_root_path)
        dockerfile_path = os.path.join(compose_dir, "Dockerfile-orbslam3-melodic")
        compose_file_path = os.path.join(compose_dir, "docker-compose.yaml")
        container_name = "orbslam3-melodic"
        base_path = Path(f"/bags/{self.rec_loc}/{self.rec_type}/{self.rec_module}/{self.rec_loc}_{self.interaction_indices}_umi")
        odom_file = f"{self.rec_loc}_{self.interaction_indices}_umi"
        src_in_container = f"/root/f_{odom_file}.txt"

        if Path(f"{self.extraction_path}/{self.label_odometry.strip('/')}/data.txt").exists():
            print(f"[ORB-SLAM3] odometry file already exists.")
            return

        # Check if Dockerfile exists
        if not os.path.isfile(dockerfile_path):
            raise FileNotFoundError(f"Dockerfile not found at: {dockerfile_path}")

        # Check if docker-compose.yml exists
        if not os.path.isfile(compose_file_path):
            raise FileNotFoundError(f"docker-compose.yml not found at: {compose_file_path}")

        # ORB-SLAM3 command inside the container
        orbslam_cmd = (
            "./ORB_SLAM3/Examples/Monocular/mono_euroc "
            "./ORB_SLAM3/Vocabulary/ORBvoc.txt "
            "./ORB_SLAM3/orbslam_config.yaml "
            f"{str(base_path)}/odometry/euroc_style/ "
            f"{str(base_path)}/odometry/euroc_style/timestamps.txt "
            f"{odom_file}"
        )

        out_dir = base_path / self.label_odometry.strip('/')
        copy_cmd = (f"mkdir -p {str(base_path)}/{self.label_odometry.strip('/')} && "
                    f"cp {src_in_container} {str(base_path)}/{self.label_odometry.strip('/')}/data.txt")

        # Step 1: Go to Docker Compose directory
        os.chdir(compose_dir)

        # Step 2: Start from a clean state. `up -d` is a no-op on an already
        # running container (so a leftover from a previous run is NOT restarted
        # and breaks this run / reuses stale results). Tearing it down first
        # guarantees a fresh container even if the previous run crashed before
        # the teardown at the end.
        subprocess.run(["docker", "compose", "down"], check=False)
        subprocess.run(["docker", "compose", "up", "-d", container_name], check=True)

        # No try/finally on purpose: if ORB-SLAM fails we want a loud crash.
        # The `down` above already guarantees the next run starts clean.

        # check if source file is present in the container
        check = subprocess.run(
            ["docker", "exec", container_name, "bash", "-lc", f"test -f {shlex.quote(src_in_container)}"],
        )
        if check.returncode == 0:
            print(f"[ORB-SLAM3] Source file {src_in_container} already exists in the container.")
            out_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["docker", "exec", container_name, "bash", "-c", copy_cmd],
                stdout=sys.stdout,
                stderr=sys.stderr
            )
        else:
            # Step 3: Run ORB-SLAM3 command inside the container
            process = subprocess.Popen(
                ["docker", "exec", "-it", container_name, "bash", "-c", orbslam_cmd],
                stdout=sys.stdout,  # Live output
                stderr=sys.stderr
            )
            process.wait()

            # move file out of the container
            subprocess.run(
                ["docker", "exec", container_name, "bash", "-c", copy_cmd],
                stdout=sys.stdout,
                stderr=sys.stderr
            )

        # Step 4: Stop the container so a subsequent run starts clean.
        subprocess.run(["docker", "compose", "down"], check=True)

        return



    def get_statistics(self, stride: int = 1, visualize: bool = False, force: bool = False) -> None:
        """
        Computes and visualizes statistics from the RGB and depth data.
        Strides the RGB and depth files by the given stride.
        """

        if not self._extracted(self.label_depth) or not self._extracted(self.label_rgb):
            raise FileNotFoundError(f"[{self.logging_tag}] RGB or depth data not extracted to {self.extraction_path}")

        statistics_file = self.extraction_path / "statistics.json"
        if statistics_file.exists() and not force:
            print(f"[{self.logging_tag}] Statistics already computed and saved to {statistics_file}")
            self.load_statistics()
            return

        # get RGB files sorted by timestamp
        rgb_files = sorted(
            self.extraction_path.glob(f"{self.label_rgb.strip('/')}/**/*{self.rgb_extension}"),
            key=lambda x: int(x.stem)
        )

        # get depth files sorted by timestamp
        depth_files = sorted(
            self.extraction_path.glob(f"{self.label_depth.strip('/')}/**/*.npy"),
            key=lambda x: int(x.stem)
        )

        keyframe_extractor = KeyframeExtractor(
            rgb_files=rgb_files,
            depth_files=depth_files,
            n_keyframes=20,
            stride=stride
        )

        self.statistics = keyframe_extractor.get_statistics(
            force=force
        )
        
        # Save the statistics to a JSON file
        self.save_statistics(statistics_file)


    def save_statistics(self, out_path: str | Path | None = None) -> None:
        """
        Saves the computed statistics to a JSON file.
        """
        if not self.statistics:
            raise ValueError(f"[{self.logging_tag}] No statistics computed yet. Call get_statistics() first.")

        if out_path is None:
            out_path = self.extraction_path / "statistics.json"

        with open(out_path, 'w') as f:
            json.dump(self.statistics, f, indent=4)
        print(f"[{self.logging_tag}] Statistics saved to {out_path}")

    def load_statistics(self, in_path: str | Path | None = None) -> None:
        """
        Loads the statistics from a JSON file.
        """
        if in_path is None:
            in_path = self.extraction_path / "statistics.json"

        if not Path(in_path).exists():
            raise FileNotFoundError(f"[{self.logging_tag}] Statistics file not found: {in_path}")

        with open(in_path, 'r') as f:
            self.statistics = json.load(f)
        print(f"[{self.logging_tag}] Statistics loaded from {in_path}")

    # ------------------------------------------------------------------
    # Anonymization metadata helpers
    # ------------------------------------------------------------------
    def save_anonym_info(self, out_path: str | Path | None = None) -> None:
        """
        Saves anonymization metadata (what streams were anonymized) to a JSON file.
        """
        if out_path is None:
            out_path = self.extraction_path / "anonymization.json"

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w") as f:
            json.dump(self.anonym_info, f, indent=4)
        print(f"[{self.logging_tag}] Anonymization info saved to {out_path}")

    def load_anonym_info(self, in_path: str | Path | None = None) -> None:
        """
        Loads anonymization metadata from JSON into self.anonym_info.
        """
        if in_path is None:
            in_path = self.extraction_path / "anonymization.json"

        in_path = Path(in_path)
        if not in_path.exists():
            self.anonym_info = {}
            print(f"[{self.logging_tag}] No anonymization info found at {in_path}, starting fresh.")
            return

        with open(in_path, "r") as f:
            self.anonym_info = json.load(f)
        print(f"[{self.logging_tag}] Anonymization info loaded from {in_path}")

    def get_odometry_orbslam(self, only_keyframes: bool = False):

        if only_keyframes:
            fn = "data_kf.txt"
        else:
            fn = "data.txt"

        txt_path = self.extraction_path / self.label_odometry.strip("/") / fn

        if not txt_path.exists():
            raise FileNotFoundError(f"[{self.logging_tag}] Trajectory file not found: {txt_path}, run orbslam")

        col_names = ["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
        df = pd.read_csv(txt_path, sep=r"\s+", header=None, names=col_names)
        df["timestamp"] = df["timestamp"].astype(int)

        return df

    def get_closed_loop_trajectory_aligned(self):
        """
        Returns the closed loop trajectory aligned as a pandas DataFrame.
        """
        csv_dir = self.extraction_path / self.label_poses.strip("/") / "data.csv"
        if not csv_dir.exists():
            raise FileNotFoundError(f"[{self.logging_tag}] Closed loop trajectory aligned CSV not found: {csv_dir}")
        
        df = pd.read_csv(csv_dir)
        return df

    def _extracted(self, label: str) -> bool:
        """
        Check if the data for the given label has been extracted.
        """
        label_path = self.extraction_path / label.strip("/")
        return label_path.exists() and any(label_path.iterdir())
    

if __name__ == "__main__":

    location = True
    test = False
    if location:
        rec_location = "bedroom_1"
        base_path = Path(f"/data/ikea_recordings")

        
        data_indexer = RecordingIndex(
            os.path.join(str(base_path), "raw") 
        )

        umi_queries_at_loc = data_indexer.query(
            location=rec_location, 
            interaction=None, 
            recorder="umi*"
        )

        
        for loc, inter, rec, ii, path in umi_queries_at_loc:
            print(f"Found recorder: {rec} at {path}")

            rec_type = inter
            rec_module = rec
            interaction_indices = ii

            umi_data = UmiData(base_path, rec_location, rec_type, rec_module, interaction_indices, data_indexer)
            # umi_data.extract_mono_depth(downsampling_factor=4, batch_size=8, force=False)
            umi_data.extract_keyframes(stride=2, n_keyframes=400)
            # umi_data.get_poses_orbslam(only_keyframes=False)

            umi_data.extract_euroc_format_for_orbslam(mask=True, image_scale=2, stride=1)
            umi_data.run_orbslam(docker_root_path=Path("~/tim_ws/hoi-dataset-tools/data_processing/docker/odometry"))

        a = 2