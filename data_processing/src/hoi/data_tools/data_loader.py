from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List
from qrcode_detector_decoder import QRCodeDetectorDecoder
from time_aligner import TimeAligner

from data_loader_aria import AriaData
from data_loader_iphone import IPhoneData
from data_loader_gripper import GripperData
from pathlib import Path

class DataLoader:
    def __init__(self, rec_name: str, base_path: Path, modalities: List[str], extractors: Dict[str, object], labels: Dict[str, str], extensions: Dict[str, str]):
        self.rec_name = rec_name
        self.base_path = base_path
        self.modalities = modalities
        self.extractors = extractors
        self.labels = labels
        self.extensions = extensions
        self.qr_info_by_modality = {}
        self.deltas = {}

    def extract_all(self):
        for modality in self.modalities:
            kind = modality.split("_")[0]
            print(f"\n[⇨] Extracting {modality}...")
            extractor = self.extractors[kind](self.base_path, self.rec_name, modality)
            if kind == "aria":
                extractor.extract_vrs()
                extractor.extract_mps()
            elif kind == "gripper":
                extractor.extract_bag()
            elif kind == "iphone":
                extractor.extract_rgbd()
                extractor.extract_plys()
                extractor.extract_poses()

    def detect_qr_all(self):
        for modality in self.modalities:
            kind = modality.split("_")[0]
            label_rgb = self.labels[kind]
            if kind == "iphone":
                ext = ".jpg"
            else:
                ext = self.extensions[kind]
            frame_dir = self.base_path / "extracted" / self.rec_name / modality / label_rgb.strip("/")

            print(f"\n[⇨] Detecting QR in {modality}...")
            qr_detector = QRCodeDetectorDecoder(frame_dir, ext=ext)
            device_ts, qr_ts = qr_detector.find_first_valid_qr()
            self.qr_info_by_modality[modality] = (device_ts, qr_ts)

    def align_to_aria(self, aria_modality="aria_human_ego"):
        print("\n[✓] Alignment Offsets to Aria:")
        aria_pair = self.qr_info_by_modality[aria_modality]
        for modality, pair in self.qr_info_by_modality.items():
            if modality == aria_modality:
                continue
            aligner = TimeAligner(aria_pair, pair)
            delta = aligner.get_delta()
            self.deltas[modality] = delta
            print(f"    {modality.ljust(20)} : {delta:+} ns")

    def apply_delta_to_images(self, folder: Path, delta: int, ext: str):
        for img_path in folder.glob(f"*{ext}"):
            try:
                old_ts = int(img_path.stem)
                new_ts = old_ts + delta
                img_path.rename(img_path.with_name(f"{new_ts}{ext}"))
            except:
                continue

    def apply_delta_to_csv(self, csv_path: Path, delta: int):
        df = pd.read_csv(csv_path)
        if "timestamp" not in df.columns:
            return
        df["timestamp"] = df["timestamp"].astype(np.int64) + delta
        df.to_csv(csv_path, index=False)

    def update_timestamps(self):
        print("\n[⇨] Overwriting extracted timestamps to match Aria:")
        for modality, delta in self.deltas.items():
            kind = modality.split("_")[0]
            if kind == "iphone":
                ext = ".jpg"
            else:
                ext = self.extensions[kind]
            base_dir = self.base_path / "extracted" / self.rec_name / modality
            print(f"    Applying delta to {modality}: {delta:+} ns")

            for image_dir in base_dir.rglob("*"):
                if image_dir.is_dir():
                    self.apply_delta_to_images(image_dir, delta, ext)

            if kind == "iphone":
                depth_dir = base_dir / "camera_depth"
                if depth_dir.exists():
                    self.apply_delta_to_images(depth_dir, delta, ".exr")

                ply_dir = base_dir / "points"
                if ply_dir.exists():
                    self.apply_delta_to_images(ply_dir, delta, ".ply")

            for subdir in base_dir.rglob("*"):
                csv_file = subdir / "data.csv"
                if csv_file.exists():
                    try:
                        self.apply_delta_to_csv(csv_file, delta)
                    except Exception as e:
                        print(f"[!] Failed to update CSV {csv_file}: {e}")

    def get_min_max_timestamp(self, modality: str) -> tuple[int, int]:
        kind = modality.split("_")[0]
        if kind == "iphone":
            ext = ".jpg"
        else:
            ext = self.extensions[kind]
        base_dir = self.base_path / "extracted" / self.rec_name / modality
        all_ts = []

        for img_dir in base_dir.rglob("*"):
            if img_dir.is_dir():
                for f in img_dir.glob(f"*{ext}"):
                    try:
                        all_ts.append(int(f.stem))
                    except:
                        continue

        for csv_file in base_dir.rglob("data.csv"):
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                print(f"[!] Failed to read CSV {csv_file}: {e}")
                continue
            if "timestamp" in df.columns:
                all_ts += df["timestamp"].astype(np.int64).tolist()

        return min(all_ts), max(all_ts)

    def crop_to_shared_window(self):
        min_t = max(self.get_min_max_timestamp(m)[0] for m in self.modalities)
        max_t = min(self.get_min_max_timestamp(m)[1] for m in self.modalities)

        print(f"\n[✂️] Cropping data to overlap window: {min_t} → {max_t}")
        for modality in self.modalities:

            print(f"modality: {modality}")

            kind = modality.split("_")[0]
            ext = self.extensions[kind]
            base_dir = self.base_path / "extracted" / self.rec_name / modality

            for img_dir in base_dir.rglob("*"):
                if img_dir.is_dir():
                    for f in img_dir.glob("*"):
                        if f.suffix.lower() in ext:
                            try:
                                ts = int(f.stem)
                                if ts < min_t or ts > max_t:
                                    f.unlink()
                            except:
                                continue

            for csv_file in base_dir.rglob("data.csv"):
                try:
                    df = pd.read_csv(csv_file)
                except Exception as e:
                    print(f"[!] Failed to read CSV {csv_file}: {e}")
                    continue

                if "timestamp" not in df.columns:
                    continue
                df = df[(df["timestamp"] >= min_t) & (df["timestamp"] <= max_t)]
                df.to_csv(csv_file, index=False)

    def extract_video_all(self):
        for modality in self.modalities:
            kind = modality.split("_")[0]
            print(f"\n[⇨] Extracting video from {modality}...")
            extractor = self.extractors[kind](self.base_path, self.rec_name, modality)
            extractor.extract_video()



if __name__ == "__main__":
    
    manager = DataLoader(
        rec_loc="bedroom_1",
        base_path=Path("/data/ikea_recordings"),
        modalities=["aria_human_ego", "aria_gripper_ego", "gripper_right", "iphone_left"],
        extractors={"aria": AriaData, "gripper": GripperData, "iphone": IPhoneData},
        labels={"aria": "/camera_rgb", "gripper": "/zedm/zed_node/left/image_rect_color", "iphone": "/camera_rgb"},
        extensions={"aria": ".png", "gripper": ".png", "iphone": (".jpg", ".exr", "ply")},
    )

    # manager.extract_all()
    manager.detect_qr_all()
    manager.align_to_aria()
    # manager.update_timestamps()
    # manager.crop_to_shared_window()
    # manager.extract_video_all()
