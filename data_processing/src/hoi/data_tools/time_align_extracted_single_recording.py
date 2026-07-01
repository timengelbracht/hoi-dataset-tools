from .data_indexer import RecordingIndex
from .data_loader_aria import AriaData
from .data_loader_iphone import IPhoneData
from .data_loader_gripper import GripperData
from .data_loader_umi import UmiData
import os
from pathlib import Path
from .time_aligner import TimeAligner
from .qrcode_detector_decoder import QRCodeDetectorDecoder
from typing import Dict, List, Tuple, Any
import ast
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN



class Datasyncer:
    def __init__(self, base_path, rec_location, rec_type, interaction_indices, data_indexer):
        self.base_path = base_path
        self.rec_location = rec_location
        self.rec_type = rec_type
        self.interaction_indices = interaction_indices
        self.data_indexer = data_indexer

        self.data_manager = {}

        self.registered = False
        self.aligned = False

        self.shared_window = None

        self.extaction_path_base = self.base_path / "extracted" / self.rec_location / self.rec_type


    def register_all_data_loaders(self, save_time_sync_info: bool = True):
        """
        This method registers all data loaders for the specified recording modules.
        It iterates through the recording modules and initializes the appropriate data loader.
        """
        queries_at_loc = self.get_all_data_for_loc_and_interaction()

        for loc, inter, rec, ii, path in queries_at_loc:
            print(f"Found recorder: {rec} at {path}")
            rec_module = rec
            self.register_data_loader_for_rec_module(rec_module)

        self.registered = True

        # After registering all data loaders, tiem deltas of eacxh sensor stream 
        # with respect to aria_human is calculated.
        # get aria_human data loader
        aria_human_data = self.data_manager["aria_human"]
        aria_human_time_pair = aria_human_data["time_pair"]

        manuals = []
        for idx, (rec_module, data_info) in enumerate(self.data_manager.items()):
            data_loader = data_info["data_loader"]

            if data_info["delta"] is not None:
                print(f"[{data_loader.logging_tag}] delta already present."
                    f"({data_info['delta']} ns) skipping.")
                continue

            time_pair = data_info["time_pair"]
    
            # if qr code is not detected manual aligment is done
            if time_pair[0] is None or time_pair[1] is None:
                print(f"[{data_loader.logging_tag}] No QR code detected, using manual time alignment.")
                manuals.append((rec_module, data_loader))
                continue

            # if qr code is detected, time delta is calculated
            print(f"[{data_loader.logging_tag}] QR code detected, calculating time delta.")
            time_aligner = TimeAligner(aria_pair=aria_human_time_pair, sensor_pair=time_pair)
            self.data_manager[rec_module]["delta"] = time_aligner.get_delta()
            print(f"[{data_loader.logging_tag}] Time delta: {time_aligner.get_delta()} ns")

        # iterate over manual aligments (it can technicaly only be then gripper)
        for rec_module, data_loader in manuals:
            print(f"[{data_loader.logging_tag}] No QR code detected, using manual time alignment.")

            # read manual time alignment from file
            event_pairs = {}
            path = data_loader.extraction_path / "event_pairs_for_manual_time_alignment.txt"
            if not path.exists():
                raise FileNotFoundError(f"Manual time alignment file not found: {path}")
            with path.open("r", encoding="utf-8") as f:
                # --- 1 · header line --------------------------------------------------
                header_line = f.readline().strip()
                header: Tuple[str, str] = ast.literal_eval(header_line)
                event_pairs["sensor_1"] = header[0]
                event_pairs["sensor_2"] = header[1]

                # --- 2 · remaining lines ---------------------------------------------
                pairs: List[Tuple[int, int]] = []
                for line in f:
                    line = line.strip()
                    if not line:                       # skip blank lines
                        continue
                    pair = ast.literal_eval(line)
                    # ensure they’re ints
                    pairs.append((int(pair[0]), int(pair[1])))
                event_pairs["pairs"] = pairs

            # get sens1_to_aria_aligner
            sens1_to_aria_human = TimeAligner.from_delta(
                delta_ns=self.data_manager[event_pairs["sensor_1"]]["delta"]
            )
            sens2_to_sens1 = TimeAligner.from_event_pairs(event_pairs["pairs"])
            sens2_to_aria_human = TimeAligner.chain(
                sens1_to_aria_human, sens2_to_sens1
            )   
            self.data_manager[rec_module]["delta"] = sens2_to_aria_human.get_delta()
            self.data_manager[rec_module]["manual_alignment"] = True
            print(f"[{data_loader.logging_tag}] Manual time delta: {sens2_to_aria_human.get_delta()} ns")

        # save time sync info
        if save_time_sync_info:
            self.save_all_time_sync_info()
        else:
            print("Time sync info not saved. Set save_time_sync_info=True to save it.")

        a = 2

    def register_all_data_loaders_high_precision(
        self,
        save_time_sync_info: bool = True,
        stride: int = 1,
        min_qr_pairs: int = 2,
        deduplicate_by_qr_timestamp: bool = True,
        mad_multiplier: float = 3.5,
    ):
        """
        High-precision registration mode that scans the full RGB streams and
        estimates one robust offset per recorder from multiple QR detections.

        The default registration path remains unchanged. This method is an
        additive alternative that callers must opt into explicitly.
        """
        queries_at_loc = self.get_all_data_for_loc_and_interaction()

        for loc, inter, rec, ii, path in queries_at_loc:
            print(f"Found recorder: {rec} at {path}")
            rec_module = rec
            self.register_data_loader_for_rec_module(rec_module)

        self.registered = True

        if "aria_human" not in self.data_manager:
            raise KeyError("High-precision alignment requires 'aria_human' as reference stream.")

        aria_human_data = self.data_manager["aria_human"]
        aria_human_loader = aria_human_data["data_loader"]
        aria_human_pairs = self._get_time_pairs_for_single_recording_module_high_precision(
            aria_human_loader,
            stride=stride,
            deduplicate_by_qr_timestamp=deduplicate_by_qr_timestamp,
        )

        if aria_human_pairs:
            aria_human_offset = self._estimate_robust_stream_offset_ns(
                aria_human_pairs,
                aria_human_loader.logging_tag,
                mad_multiplier=mad_multiplier,
            )
        else:
            aria_human_time_pair = aria_human_data["time_pair"]
            if aria_human_time_pair[0] is None or aria_human_time_pair[1] is None:
                raise RuntimeError(
                    "[aria_human] No valid QR detections found for the high-precision reference stream."
                )
            aria_human_offset = int(aria_human_time_pair[1]) - int(aria_human_time_pair[0])
            print(
                f"[{aria_human_loader.logging_tag}] No multi-QR detections found, "
                f"falling back to the stored first valid QR pair for reference."
            )

        if aria_human_data["delta"] is None:
            aria_human_data["delta"] = 0
            aria_human_data["manual_alignment"] = False
            print(f"[{aria_human_loader.logging_tag}] High-precision reference delta: 0 ns")

        manuals = []
        for rec_module, data_info in self.data_manager.items():
            if rec_module == "aria_human":
                continue

            data_loader = data_info["data_loader"]

            if data_info["delta"] is not None:
                print(
                    f"[{data_loader.logging_tag}] delta already present."
                    f"({data_info['delta']} ns) skipping."
                )
                continue

            time_pairs = self._get_time_pairs_for_single_recording_module_high_precision(
                data_loader,
                stride=stride,
                deduplicate_by_qr_timestamp=deduplicate_by_qr_timestamp,
            )

            if len(time_pairs) >= min_qr_pairs:
                sensor_offset = self._estimate_robust_stream_offset_ns(
                    time_pairs,
                    data_loader.logging_tag,
                    mad_multiplier=mad_multiplier,
                )
                data_info["delta"] = sensor_offset - aria_human_offset
                data_info["manual_alignment"] = False
                print(
                    f"[{data_loader.logging_tag}] High-precision time delta from "
                    f"{len(time_pairs)} QR correspondences: {data_info['delta']} ns"
                )
                continue

            if len(time_pairs) == 1:
                sensor_offset = int(time_pairs[0][1]) - int(time_pairs[0][0])
                data_info["delta"] = sensor_offset - aria_human_offset
                data_info["manual_alignment"] = False
                print(
                    f"[{data_loader.logging_tag}] Only one valid time QR found during "
                    f"full-stream scan, using that single correspondence: {data_info['delta']} ns"
                )
                continue

            time_pair = data_info["time_pair"]
            if time_pair[0] is not None and time_pair[1] is not None:
                sensor_offset = int(time_pair[1]) - int(time_pair[0])
                data_info["delta"] = sensor_offset - aria_human_offset
                data_info["manual_alignment"] = False
                print(
                    f"[{data_loader.logging_tag}] No multi-QR detections found, "
                    f"falling back to the stored first valid QR pair: {data_info['delta']} ns"
                )
                continue

            print(f"[{data_loader.logging_tag}] No QR code detected, using manual time alignment.")
            manuals.append((rec_module, data_loader))

        for rec_module, data_loader in manuals:
            print(f"[{data_loader.logging_tag}] No QR code detected, using manual time alignment.")

            event_pairs = self._load_manual_event_pairs(data_loader)
            sens1_to_aria_human = TimeAligner.from_delta(
                delta_ns=self.data_manager[event_pairs["sensor_1"]]["delta"]
            )
            sens2_to_sens1 = TimeAligner.from_event_pairs(event_pairs["pairs"])
            sens2_to_aria_human = TimeAligner.chain(
                sens1_to_aria_human, sens2_to_sens1
            )
            self.data_manager[rec_module]["delta"] = sens2_to_aria_human.get_delta()
            self.data_manager[rec_module]["manual_alignment"] = True
            print(
                f"[{data_loader.logging_tag}] Manual time delta: "
                f"{sens2_to_aria_human.get_delta()} ns"
            )

        if save_time_sync_info:
            self.save_all_time_sync_info()
        else:
            print("Time sync info not saved. Set save_time_sync_info=True to save it.")

    def save_interaction_splitting_info(self, interaction_splitting_info: Dict[str, Any], overwrite: bool = False):
        """
        This method saves the interaction splitting information to a JSON file.
        """
        if not self.registered:
            raise RuntimeError("Data loaders must be registered before saving interaction splitting info.")
        
        file_path = self.extaction_path_base / f"interaction_splitting_info_{self.interaction_indices}.json"

        if file_path.exists() and not overwrite:
            print(f"[INFO] Interaction splitting info already exists at {file_path}.")
            return
        
        with file_path.open("w", encoding="utf-8") as f:
            import json
            json.dump(interaction_splitting_info, f, indent=4)

        print(f"[INFO] Interaction splitting info saved to {file_path}")


    def save_all_time_sync_info(self, overwrite: bool = False):    
        """
        This method saves the time synchronization information for all registered data loaders.
        """
        if not self.registered:
            raise RuntimeError("Data loaders must be registered before saving time sync info.")

        for rec_module, data_info in self.data_manager.items():
            data_loader = data_info["data_loader"]
            time_pair = data_info["time_pair"]
            delta = data_info["delta"]
            manual_alignment = data_info["manual_alignment"]
            timestamps_aligned = data_info["timestamps_aligned"]
            time_window_cropped = data_info["time_window_cropped"]
            shared_time_window = data_info["shared_time_window"]

            # Create directory for the recording module
            file_path = data_loader.extraction_path / "time_sync_info.json"

            if file_path.exists() and not overwrite:
                print(f"[{data_loader.logging_tag}] Time sync info already exists at {file_path}.")
                continue
            
            if file_path.exists() and overwrite:
                print(f"[{data_loader.logging_tag}] Overwriting existing time sync info at {file_path}.")

            # Save the time sync info to a JSON file
            time_sync_info = {
                "rec_module": rec_module,
                "time_pair": time_pair,
                "delta": delta,
                "manual_alignment": manual_alignment,
                "timestamps_aligned": timestamps_aligned,
                "time_window_cropped": time_window_cropped,
                "shared_time_window": shared_time_window,

            }
            with file_path.open("w", encoding="utf-8") as f:
                import json
                json.dump(time_sync_info, f, indent=4)
            
            print(f"[{data_loader.logging_tag}] Time sync info saved to {file_path}")

    def get_all_data_for_loc_and_interaction(self):
        """
        This method retrieves all data for a specified location and interaction type.
        It uses the RecordingIndex to query the data based on the provided parameters.
        """
        queries_at_loc = self.data_indexer.query(
            location=self.rec_location,
            interaction=self.rec_type,
            recorder=None,
            interaction_index=self.interaction_indices
        )
        
        return queries_at_loc
    
    def register_data_loader_for_rec_module(self, rec_module: str):
        """
        Retrieves and registers the appropriate data loader based on the recording module type.
        - For 'gripper' the module name must be **exactly** 'gripper'.
        - For all others (e.g. 'aria_human_ego', 'iphone_left') a substring match is enough.
        """
        data_loader_classes = {
            "gripper": GripperData,
            "iphone": IPhoneData,
            "aria": AriaData,
            "umi": UmiData
        }

        for key, loader_class in data_loader_classes.items():
            # Exact match required only for 'gripper'
            match = (rec_module == key) if key == "gripper" else (key in rec_module)
            if not match:
                continue

            data_loader = loader_class(
                self.base_path,
                self.rec_location,
                self.rec_type,
                rec_module,
                self.interaction_indices,
                self.data_indexer,
            )

            if isinstance(data_loader, IPhoneData) or isinstance(data_loader, AriaData):
                stride = 3
            else:
                stride = 2
            
            # check if time alignment is already done
            file_path = data_loader.extraction_path / "time_sync_info.json"
            if file_path.exists():
                print(f"[{data_loader.logging_tag}] Time sync info already exists at {file_path}.")
                with file_path.open("r", encoding="utf-8") as f:
                    time_sync_info = json.load(f)
                    data_loader.time_pair         = tuple(time_sync_info["time_pair"])
                    data_loader.delta             = time_sync_info["delta"]
                    data_loader.manual_alignment  = time_sync_info["manual_alignment"]
                    data_loader.timestamps_aligned = time_sync_info["timestamps_aligned"]
                    data_loader.time_window_cropped = time_sync_info["time_window_cropped"]
                    data_loader.shared_time_window = time_sync_info["shared_time_window"]
                    print(f"[{data_loader.logging_tag}] Loaded time sync info: {time_sync_info}")

                # ←-- register and stop looping
                self.data_manager[rec_module] = {
                    "data_loader":       data_loader,
                    "time_pair":         data_loader.time_pair,
                    "delta":             data_loader.delta,
                    "manual_alignment":  data_loader.manual_alignment,
                    "timestamps_aligned": data_loader.timestamps_aligned,
                    "time_window_cropped": data_loader.time_window_cropped,
                    "shared_time_window": data_loader.shared_time_window
                }
                break

            self.data_manager[rec_module] = {
                "data_loader": data_loader,
                "time_pair": self._get_time_pair_for_single_recording_module(
                    data_loader, stride=stride
                ),
                "delta": None,
                "manual_alignment": False,
                "timestamps_aligned": False,
                "time_window_cropped": False,
                "shared_time_window": (None, None)

            }
            break
        else:
            # Optional: warn if nothing matched
            raise ValueError(f"No data-loader rule matched rec_module='{rec_module}'.")

    def _get_time_pair_for_single_recording_module(self, data_loader, stride: int = 1):
        """
        This method retrieves time pairs (qr code timestamp, recording timestamp) for a single recording module.

        """
        print("###############################################")
        print(f"[{data_loader.logging_tag}] - Extracting time pairs...")
        print("###############################################")
        rgb_dir = Path(data_loader.extraction_path / data_loader.label_rgb.strip("/"))
        rgb_ext = data_loader.rgb_extension

        qr_detector = QRCodeDetectorDecoder(rgb_dir, ext=rgb_ext)
        device_ts, qr_ts = qr_detector.find_first_valid_qr(stride=stride)
        print(f"{data_loader.logging_tag} - Finished extracting time pairs.")

        return (device_ts, qr_ts)

    def _get_time_pairs_for_single_recording_module_high_precision(
        self,
        data_loader,
        stride: int = 1,
        deduplicate_by_qr_timestamp: bool = True,
    ) -> List[Tuple[int, int]]:
        """
        Retrieve many time-QR correspondences for one recorder by scanning the
        full RGB stream instead of stopping at the first hit.
        """
        print("###############################################")
        print(f"[{data_loader.logging_tag}] - Extracting high-precision time pairs...")
        print("###############################################")
        rgb_dir = Path(data_loader.extraction_path / data_loader.label_rgb.strip("/"))
        rgb_ext = data_loader.rgb_extension

        qr_detector = QRCodeDetectorDecoder(rgb_dir, ext=rgb_ext)
        time_pairs = qr_detector.find_all_valid_time_qrs(
            stride=stride,
            deduplicate_by_qr_timestamp=deduplicate_by_qr_timestamp,
        )
        print(
            f"{data_loader.logging_tag} - Finished extracting "
            f"{len(time_pairs)} high-precision time pairs."
        )

        return time_pairs

    def _estimate_robust_stream_offset_ns(
        self,
        time_pairs: List[Tuple[int, int]],
        logging_tag: str,
        mad_multiplier: float = 3.5,
    ) -> int:
        """
        Estimate one robust QR-to-device offset from many
        ``(device_timestamp_ns, qr_timestamp_ns)`` correspondences.
        """
        if not time_pairs:
            raise ValueError("Need at least one time pair to estimate a stream offset.")

        offsets = np.array(
            [int(qr_ts) - int(device_ts) for device_ts, qr_ts in time_pairs],
            dtype=np.int64,
        )
        median_offset = int(np.median(offsets))

        if len(offsets) < 3:
            print(
                f"[{logging_tag}] Using median offset from {len(offsets)} QR correspondences: "
                f"{median_offset} ns"
            )
            return median_offset

        absolute_deviation = np.abs(offsets - median_offset)
        mad = int(np.median(absolute_deviation))

        if mad == 0:
            print(
                f"[{logging_tag}] QR offsets are already perfectly consistent across "
                f"{len(offsets)} correspondences."
            )
            return median_offset

        inlier_mask = absolute_deviation <= mad_multiplier * mad
        inlier_offsets = offsets[inlier_mask]
        if inlier_offsets.size == 0:
            inlier_offsets = offsets

        robust_offset = int(np.median(inlier_offsets))
        print(
            f"[{logging_tag}] Robust QR offset estimate: {robust_offset} ns "
            f"from {int(inlier_offsets.size)}/{len(offsets)} inlier correspondences "
            f"(MAD={mad} ns)"
        )
        return robust_offset

    def _load_manual_event_pairs(self, data_loader) -> Dict[str, Any]:
        """
        Load manual event pairs from disk for chained alignment fallback.
        """
        event_pairs = {}
        path = data_loader.extraction_path / "event_pairs_for_manual_time_alignment.txt"
        if not path.exists():
            raise FileNotFoundError(f"Manual time alignment file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            header_line = f.readline().strip()
            header: Tuple[str, str] = ast.literal_eval(header_line)
            event_pairs["sensor_1"] = header[0]
            event_pairs["sensor_2"] = header[1]

            pairs: List[Tuple[int, int]] = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pair = ast.literal_eval(line)
                pairs.append((int(pair[0]), int(pair[1])))
            event_pairs["pairs"] = pairs

        return event_pairs

    def apply_time_deltas_to_all_data_streams(self) -> int:
        """
        This method applies the calculated time deltas to all data streams of the registered data loaders.
        It iterates over all registered data loaders and applies the time deltas to the respective data streams.
        """
        if not self.registered:
            raise RuntimeError("Data loaders must be registered before applying time deltas.")
        
        print("###############################################")
        print("[INFO] Applying time deltas to all data streams...")
        print("###############################################")

        # iterate over all registered data loaders and apply time deltas
        for rec_module, data_info in self.data_manager.items():
            data_loader = data_info["data_loader"]
            delta = data_info["delta"]
            timestamps_aligned = data_info["timestamps_aligned"]

            extraction_path = data_loader.extraction_path

            if timestamps_aligned:
                print(f"[{data_loader.logging_tag}] Timestamps already aligned, skipping.")
                continue

            data_streams = self.data_indexer.get_all_extracted_data_streams(extraction_path)

            # iterate over datastreams of the current recording module
            _TS_EXTS = [".png", ".jpg", ".jpeg", ".npy"]
            for data_stream in data_streams:
                data_stream = Path(data_stream)
                if data_stream.is_dir():
                    for ext in _TS_EXTS:
                        self.apply_delta_to_images(data_stream, delta, ext)
                    print(f"[{data_loader.logging_tag}][DIR ] shifted filenames in {data_stream}")
                elif data_stream.is_file() and data_stream.suffix.lower() == ".csv":
                    self.apply_delta_to_csv(data_stream, delta)
                    print(f"[{data_loader.logging_tag}][CSV ] patched timestamps in {data_stream}")
                else:
                    print(f"[{data_loader.logging_tag}][SKIP] {data_stream} (neither dir nor .csv)")

            # update timestamps_aligned flag
            data_loader.timestamps_aligned = True
            data_info["timestamps_aligned"] = True

        # save time sync info after applying deltas
        self.save_all_time_sync_info(overwrite=True)

        self.aligned = True

        print("################################################")
        print("[INFO] Finished applying time deltas to all data streams.")
        print("################################################")

        a = 2

    def apply_time_window_cropping_to_all_data_streams(self):
        
        """
        This method applies time window cropping to all data streams of the registered data loaders.
        It iterates over all registered data loaders and crops the data streams to the shared time window.
        """
        if not self.registered:
            raise RuntimeError("Data loaders must be registered before applying time window cropping.")
        
        print("###############################################")
        print("[INFO] Applying time window cropping to all data streams...")
        print("###############################################")

        if self.shared_window is None:
            # TODO wrong logic, should depoend on the config file
            print("[WARNING] Shared time window not computed yet, computing now...")
            self._get_shared_time_window_from_all_data_streams()

        # iterate over all registered data loaders and apply time window cropping
        for rec_module, data_info in self.data_manager.items():
            data_loader = data_info["data_loader"]
            time_window_cropped = data_info["time_window_cropped"]
            if time_window_cropped:
                print(f"[{data_loader.logging_tag}] Time window already cropped, skipping.")
                continue
            extraction_path = data_loader.extraction_path

            shared_time_window = data_info["shared_time_window"]
            start, end = shared_time_window

            data_streams = self.data_indexer.get_all_extracted_data_streams(extraction_path)

            # iterate over datastreams of the current recording module
            _TS_EXTS = [".png", ".jpg", ".jpeg", ".npy"]
            for data_stream in data_streams:
                data_stream = Path(data_stream)
                if data_stream.is_dir():
                    for ext in _TS_EXTS:
                        self.crop_images_to_time_window(data_stream, start, end, ext)
                    print(f"[{data_loader.logging_tag}][DIR ] cropped filenames in {data_stream}")
                elif data_stream.is_file() and data_stream.suffix.lower() == ".csv":
                    self.crop_csv_to_time_window(data_stream, start, end)
                    print(f"[{data_loader.logging_tag}][CSV ] cropped timestamps in {data_stream}")
                else:
                    print(f"[{data_loader.logging_tag}][SKIP] {data_stream} (neither dir nor .csv)")

            # update time_window_cropped flag
            data_loader.time_window_cropped = True
            data_info["time_window_cropped"] = True

        # save time sync info after applying cropping
        self.save_all_time_sync_info(overwrite=True)

        print("################################################")
        print("[INFO] Finished applying time window cropping to all data streams.")
        print("################################################")

    def _get_shared_time_window_from_all_data_streams(self):
        """
        This method computes the *intersection* of the time windows across all recording modules.
        It sets self.shared_window = (start, end) such that every module has data from `start` through `end`.
        """
        if not self.registered:
            raise RuntimeError("Data loaders must be registered before applying time deltas.")

        print("###############################################")
        print("[INFO] Computing shared time window for all data streams...")
        print("###############################################")

        _TS_EXTS = [".png", ".jpg", ".jpeg", ".npy"]
        module_windows = []

        # 1) collect each module's min/max as integers
        for rec_module, data_info in self.data_manager.items():
            data_loader     = data_info["data_loader"]
            extraction_path = data_loader.extraction_path
            data_streams    = self.data_indexer.get_all_extracted_data_streams(extraction_path)

            module_min = None
            module_max = None

            for data_stream in data_streams:
                p = Path(data_stream)
                if p.is_dir():
                    for ext in _TS_EXTS:
                        min_ts_img, max_ts_img = self.get_min_max_timestamp_from_images(p, ext)
                        if min_ts_img is not None:
                            ts = int(min_ts_img)
                            module_min = ts if module_min is None else min(module_min, ts)
                        if max_ts_img is not None:
                            ts = int(max_ts_img)
                            module_max = ts if module_max is None else max(module_max, ts)

                elif p.is_file() and p.suffix.lower() == ".csv":
                    min_ts_csv, max_ts_csv = self.get_min_max_timestamp_from_csv(p)
                    if min_ts_csv is not None:
                        ts = int(min_ts_csv)
                        module_min = ts if module_min is None else min(module_min, ts)
                    if max_ts_csv is not None:
                        ts = int(max_ts_csv)
                        module_max = ts if module_max is None else max(module_max, ts)

            # ensure this module actually had data
            if module_min is None or module_max is None:
                raise RuntimeError(f"[{rec_module}] No valid timestamps found under '{extraction_path}'")

            print(f"[INFO] Module '{rec_module}' window: [{module_min}, {module_max}]")
            module_windows.append((module_min, module_max))

        # 2) compute the intersection window (integers)
        start = max(mn for mn, _ in module_windows)
        end   = min(mx for _, mx in module_windows)

        if start >= end:
            raise ValueError(f"No shared time window: start ({start}) ≥ end ({end}).")

        print(f"[INFO] Shared intersection window: [{start}, {end}]")
        self.shared_window = (start, end)

        print("################################################")
        print("[INFO] Finished computing shared time window for all data streams.")
        print("################################################")

        # update shared_time_window for each module and save
        for rec_module, data_info in self.data_manager.items():
            data_info["shared_time_window"] = self.shared_window

        self.save_all_time_sync_info(overwrite=True)

    def get_interaction_time_windows_from_qr_codes(self) -> Dict[str, Tuple[int, int]]:
        """
        This method retrieves the interaction time windows from QR codes for each recording module.
        It returns a dictionary where keys are recording module names and values are tuples of (start, end) timestamps.
        """
        if not self.registered:
            raise RuntimeError("Data loaders must be registered before retrieving interaction time windows.")
        
        # return if file with interaction time windows already exists
        file_path = self.extaction_path_base / f"interaction_splitting_info_{self.interaction_indices}.json"
        if file_path.exists():
            print(f"[INFO] Interaction time windows already exist at {file_path}.")
            with file_path.open("r", encoding="utf-8") as f:
                interaction_time_windows = json.load(f)
            print(f"[INFO] Loaded interaction time windows: {interaction_time_windows}")
            return interaction_time_windows
        
        # sort to start with the iphones, since most times the interaction qrs are seen in the iphone data
        sources = list(self.data_manager.keys())
        iphones = [k for k in sources if k.lower().startswith("iphone")]
        #swap iphone order
        iphones = sorted(iphones, reverse=True)
        others  = [k for k in sources if k not in iphones]
        ordered_sources = iphones + others

        for rec_module in ordered_sources:
            data_info = self.data_manager[rec_module]
            data_loader = data_info["data_loader"]

            if not data_info["time_window_cropped"]:
                raise ValueError("Time window must be cropped before retrieving interaction time windows.")
            try:
                interaction_time_windows_path = data_loader.extraction_path_base / f"interaction_splitting_info_{self.interaction_indices}.json"
            except:
                print()
                return
            
            if interaction_time_windows_path.exists():
                print(f"[{data_loader.logging_tag}] Interaction time windows already exist at {interaction_time_windows_path}.")
                with interaction_time_windows_path.open("r", encoding="utf-8") as f:
                    interaction_time_windows = json.load(f)
                print(f"[{data_loader.logging_tag}] Loaded interaction time windows: {interaction_time_windows}")
                return interaction_time_windows

            rgb_dir = Path(data_loader.extraction_path / data_loader.label_rgb.strip("/"))
            rgb_ext = data_loader.rgb_extension

            qr_detector = QRCodeDetectorDecoder(rgb_dir, ext=rgb_ext)
            all_hits = qr_detector.find_all_valid_interaction_qrs()
            del qr_detector

            blocks, gaps = cluster_qr_blocks_dbscan(all_hits, eps_s=0.5, min_samples=2)

            if len(blocks) < 3: 
                print(f"[{data_loader.logging_tag}] Not enough QR blocks found for interaction time windows.")
                continue

            # add gap before first detection and after last detection
            t_start = int(sorted(rgb_dir.glob(f"*{rgb_ext}"))[0].stem)
            t_end = int(sorted(rgb_dir.glob(f"*{rgb_ext}"))[-1].stem)

            gaps.insert(0, {    
                "start_ns": t_start,
                "end_ns": blocks[0]["start_ns"],
                "duration_s": (blocks[0]["start_ns"] - t_start) / 1e9
            })

            gaps.append({
                "start_ns": blocks[-1]["end_ns"],
                "end_ns": t_end,
                "duration_s": (t_end - blocks[-1]["end_ns"]) / 1e9
            })

            # save the interaction time windows
            interaction_time_windows = {}
            for i, gap in enumerate(gaps):
                if gap["duration_s"] < 3.0:
                    continue
            
                interaction_time_windows[f"window_{i}"] = {
                    "start_ns": gap["start_ns"],
                    "end_ns": gap["end_ns"],
                    "duration_s": gap["duration_s"]
                }

            # save to file
            self.save_interaction_splitting_info(interaction_time_windows, overwrite=True)

            print(f"[{data_loader.logging_tag}] Interaction time windows saved to {interaction_time_windows_path}")
            return interaction_time_windows

    def apply_delta_to_images(self, folder: Path, delta: int, ext: str):
        """ This method applies a time delta to the filenames of images in a specified folder.
        It renames each image file by adding the delta to its timestamp.
        """
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

    def get_min_max_timestamp_from_images(self, folder: Path, ext: str) -> Tuple[int, int]:
        """
        This method retrieves the minimum and maximum timestamps from image filenames in a specified folder.
        It assumes the filenames are in the format of timestamps (e.g., 1634551234567.png).
        """
        min_ts = None
        max_ts = None

        for img_path in folder.glob(f"*{ext}"):
            try:
                ts = int(img_path.stem)
                if min_ts is None or ts < min_ts:
                    min_ts = ts
                if max_ts is None or ts > max_ts:
                    max_ts = ts
            except ValueError:
                continue

        return min_ts, max_ts

    def get_min_max_timestamp_from_csv(self, csv_path: Path) -> Tuple[int, int]:
        """
        This method retrieves the minimum and maximum non‑negative timestamps from a CSV file.
        It assumes the CSV file has a 'timestamp' column.
        """
        df = pd.read_csv(csv_path)
        if "timestamp" not in df.columns:
            return None, None

        # Keep only timestamps ≥ 0
        ts_series = df["timestamp"]
        nonneg = ts_series[ts_series >= 0]

        if nonneg.empty:
            # No valid (non‑negative) timestamps
            return None, None

        min_ts = int(nonneg.min())
        max_ts = int(nonneg.max())

        return min_ts, max_ts
    
    def crop_images_to_time_window(self, folder: Path, start: int, end: int, ext: str):
        """
        Remove every file in `folder` matching `*{ext}` whose stem (timestamp) is
        outside [start, end].
        """
        for img_path in folder.glob(f"*{ext}"):
            try:
                ts = int(img_path.stem)
            except ValueError:
                # not a timestamp‐named file
                continue
            if ts < start or ts > end:
                img_path.unlink()
        
    def crop_csv_to_time_window(self, csv_path: Path, start: int, end: int):
        """
        Read the CSV at `csv_path`, find any column whose name contains "timestamp",
        drop rows whose values in that column fall outside [start, end], and overwrite.
        """
        df = pd.read_csv(csv_path)
        if df.empty:
            return

        # Find the timestamp column by name (case‑insensitive)
        ts_cols = [c for c in df.columns if "timestamp" in c.lower()]
        if not ts_cols:
            # No timestamp column → nothing to crop
            return
        ts_col = ts_cols[0]

        # Filter rows within the window
        mask = (df[ts_col] >= start) & (df[ts_col] <= end)
        if not mask.all():
            df.loc[mask].to_csv(csv_path, index=False)

def plot_hits(all_hits, start_time=None, end_time=None):
    """
    Visualize QR detection hits as vertical tick markers.

    Parameters
    ----------
    all_hits : list of int
        List of timestamps in nanoseconds.
    start_time : int, optional
        Minimum timestamp to show (ns). Defaults to first hit.
    end_time : int, optional
        Maximum timestamp to show (ns). Defaults to last hit.
    """
    if not all_hits:
        print("No hits to plot.")
        return
    
    all_hits = np.array(all_hits, dtype=np.int64)

    if start_time is None:
        start_time = all_hits.min()
    if end_time is None:
        end_time = all_hits.max()

    # Convert ns → seconds for readability
    hits_sec = (all_hits - start_time) / 1e9
    duration_sec = (end_time - start_time) / 1e9

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.eventplot(hits_sec, orientation='horizontal', lineoffsets=1, linelengths=0.6)

    ax.set_xlim(0, duration_sec)
    ax.set_yticks([])
    ax.set_xlabel("Time [s]")
    ax.set_title("QR Code Detections")

    plt.show()

def cluster_qr_blocks_dbscan(
    hits_ns: List[int],
    eps_s: float = 0.5,
    min_samples: int = 2,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Cluster QR detections (timestamps in ns) into blocks using DBSCAN.
    Also compute the windows between consecutive blocks.

    Args:
        hits_ns: Sorted list of timestamps (ns).
        eps_s: Maximum time gap (in seconds) for points to be in the same cluster.
        min_samples: Minimum samples to form a cluster.

    Returns:
        blocks: [{start_ns, end_ns, duration_s, n}]
        between: [{start_ns, end_ns, duration_s}]
    """
    if not hits_ns:
        return [], []

    t_s = np.array(sorted(hits_ns), dtype=np.int64) / 1e9
    X = t_s.reshape(-1, 1)

    labels = DBSCAN(eps=eps_s, min_samples=min_samples, metric="euclidean").fit_predict(X)

    blocks: List[Dict[str, Any]] = []
    for lbl in sorted(set(labels)):
        if lbl == -1:  # noise
            continue
        members = t_s[labels == lbl]
        start_ns = int(members.min() * 1e9)
        end_ns   = int(members.max() * 1e9)
        blocks.append({
            "start_ns": start_ns,
            "end_ns": end_ns,
            "duration_s": float(members.max() - members.min()),
            "n": int(len(members)),
        })

    blocks.sort(key=lambda b: b["start_ns"])

    # compute windows between consecutive blocks
    between: List[Dict[str, Any]] = []
    for i in range(len(blocks) - 1):
        a = blocks[i]["end_ns"]
        b = blocks[i+1]["start_ns"]
        if b > a:
            between.append({
                "start_ns": a,
                "end_ns": b,
                "duration_s": float((b - a) / 1e9),
            })

    return blocks, between

if __name__ == "__main__":

    rec_location = "bedroom_1"
    rec_interaction = "gripper"
    interaction_indices = "1-8"

    base_path = Path(f"/data/ikea_recordings")

    data_indexer = RecordingIndex(
        os.path.join(str(base_path), "raw") 
    )

    # Initialize the Datasyncer with the base path, location, interaction type, and interaction indices
    data_syncer = Datasyncer(
        base_path=base_path,
        rec_location=rec_location,
        rec_type=rec_interaction,
        interaction_indices=interaction_indices,
        data_indexer=data_indexer
    )

    # Register all data loaders for the specified recording modules
    data_syncer.register_all_data_loaders()
    data_syncer.apply_time_deltas_to_all_data_streams()
    data_syncer.apply_time_window_cropping_to_all_data_streams()


    # # get all data for the specified location and interaction
    # queries_at_loc = data_indexer.query(
    #     location=rec_location, 
    #     interaction=rec_interaction, 
    #     recorder=None,
    #     interaction_index=interaction_indices
    # )

    # time_pairs = []
    # for loc, inter, rec, ii, path in queries_at_loc:
    #     print(f"Found recorder: {rec} at {path}")

    #     rec_type = inter
    #     rec_module = rec
    #     interaction_indices = ii

    #     if "gripper" in rec_module:
    #         gripper_data = GripperData(base_path, rec_location, rec_type, rec_module, interaction_indices, data_indexer)
            
    #         a = 2


    #     if "iphone" in rec_module:
    #         iphone_data = IPhoneData(base_path, rec_location, rec_type, rec_module, interaction_indices, data_indexer)


    #         a = 2
    #     if "aria" in rec_module:
    #         aria_data = AriaData(base_path, rec_location, rec_type, rec_module, interaction_indices, data_indexer)
            
    #         # get rgb frames
    #         rgb_dir = Path(aria_data.extraction_path / aria_data.label_rgb.strip("/"))
    #         rgb_ext = aria_data.rgb_extension

    #         qr_detector = QRCodeDetectorDecoder(rgb_dir, ext=rgb_ext)
    #         device_ts, qr_ts = qr_detector.find_first_valid_qr()

    #         time_pairs.append(((device_ts, qr_ts), rec_module))
            




    # get all distinct interaction indices
    # interaction_indices_at_loc = set()
    # for loc, inter, rec, ii, path in queries_at_loc:
    #         interaction_indices_at_loc.add(ii)

    

        



    a = 2
