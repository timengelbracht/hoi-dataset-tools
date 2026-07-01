from torchvision import transforms, models
import cv2
from transformers import pipeline
from accelerate.test_utils.testing import get_backend
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import shutil
from pathlib import Path
import matplotlib.pyplot as plt



class KeyframeExtractor:

    # Constants for keyframe extraction
    FEATURE_PERCENTILE = 10.0
    DEPTH_PERCENTILE = 50.0
    BLUR_PERCENTILE = 10.0

    # monodepth model setup
    DEVICE, _, _ = get_backend()
    MONO_DEPTH_CHECKPOINT = "depth-anything/Depth-Anything-V2-base-hf"
    PIPE_MONO_DEPTH = pipeline("depth-estimation", model=MONO_DEPTH_CHECKPOINT, device=DEVICE)

    # DINOv2 model setup
    DINO_CHECKPOINT = "facebook/dinov2-small-imagenet1k-1-layer"
    PIPE_DINO = pipeline("image-feature-extraction", model=DINO_CHECKPOINT, device=DEVICE, pool=True)

    def __init__(self, rgb_files: list[str | Path], depth_files: list[str | Path], n_keyframes: int = 20, stride: int = 2):
        
        # rgb and depth data lists
        self.rgb_files = rgb_files[::stride]  # stride the RGB images
        self.depth_files = depth_files[::stride]  # stride the depth maps
        self.stride = stride
        self.n_keyframes = n_keyframes

        self.statistics = {
            "keypoints_per_image": [],
            "motion_blur_per_image": [],
            "average_depth_per_image": [],
            "average_depth_range_per_image": [],
            "relative_depth_range_per_image": [],
            "depth_cv_per_image": []
        }
        
    def get_statistics(self, force: bool = False) -> None:
        """
        Computes and visualizes statistics from the RGB and depth data.
        Strides the RGB and depth files by the given stride.
        """
        
        # depth statistics
        average_depth_per_image = []
        average_depth_range_per_image = []
        relative_depth_range_per_image = []
        depth_cv_per_image = []
        for depth_file in tqdm(self.depth_files, desc=f"[KEYFRAMING] Processing depth statistics", unit="file"):
            depth_map = np.load(depth_file)
            depth_map[np.isnan(depth_map) | np.isinf(depth_map) | np.isneginf(depth_map)] = 0  # replace NaNs, infs, and -infs with 0
            if np.all(depth_map == 0):  # check if the array contains only zeros
                continue
            average_depth = np.nanmean(depth_map)
            
            d = depth_map.flatten()
            d = d[~np.isnan(d)]

            mean_d  = np.mean(d)
            std_d   = np.std(d)
            depth_cv = std_d / (mean_d + 1e-6)

            p5, p95 = np.percentile(d, [5, 95])
            depth_prange = (p95 - p5) / (p95 + 1e-6)

            average_depth_per_image.append(float(average_depth))
            relative_depth_range_per_image.append(float(depth_prange))
            depth_cv_per_image.append(float(depth_cv))
            average_depth_range = np.nanmax(depth_map) - np.nanmin(depth_map)
            average_depth_range_per_image.append(float(average_depth_range))

        # local feature statistics (ORB)
        orb = cv2.ORB_create()
        keypoints_per_image = []
        for rgb_file in tqdm(self.rgb_files, desc=f"[KEYFRAMING] Processing RGB statistics", unit="file"):
            img = cv2.imread(str(rgb_file))
            if img is None:
                continue
            keypoints, _ = orb.detectAndCompute(img, None)
            keypoints_per_image.append(int(len(keypoints)))  

        # motion blur statistics
        motion_blur_per_image = []
        for rgb_file in tqdm(self.rgb_files, desc=f"[KEYFRAMING] Processing RGB files for motion blur", unit="file"):
            img = cv2.imread(str(rgb_file))
            if img is None:
                continue
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Calculate Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            motion_blur_per_image.append(float(laplacian_var))

        self.statistics = {
            "keypoints_per_image": keypoints_per_image,
            "motion_blur_per_image": motion_blur_per_image,
            "average_depth_per_image": average_depth_per_image,
            "average_depth_range_per_image": average_depth_range_per_image,
            "relative_depth_range_per_image": relative_depth_range_per_image,
            "depth_cv_per_image": depth_cv_per_image
        }


        return self.statistics
    
    def visualize_statistics(self):
        """
        Visualizes the computed statistics.

        """

        if not any(self.statistics.values()):
            print(f"[KEYFRAMING] No statistics found, computing...")
            self.get_statistics()
        
        keypoints_per_image = self.statistics["keypoints_per_image"]
        motion_blur_per_image = self.statistics["motion_blur_per_image"]
        average_depth_per_image = self.statistics["average_depth_per_image"]
        average_depth_range_per_image = self.statistics["average_depth_range_per_image"]
        relative_depth_range_per_image = self.statistics["relative_depth_range_per_image"]
        depth_cv_per_image = self.statistics["depth_cv_per_image"]

        fig, axes = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)

        # Top‑left: Keypoints per Image
        axes[0, 0].hist(keypoints_per_image, bins=50, color='green', alpha=0.7)
        axes[0, 0].set_title(f"Keypoints per Image")
        axes[0, 0].set_xlabel("Number of Keypoints")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].grid(True)
        # Top‑right: Motion Blur per Image
        axes[0, 1].hist(motion_blur_per_image, bins=50, color='blue', alpha=0.7)
        axes[0, 1].set_title(f"Motion Blur (Laplacian Var)")
        axes[0, 1].set_xlabel("Laplacian Variance")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].grid(True)
        # Middle‑left: Avg Depth per Image
        axes[1, 0].hist(average_depth_per_image, bins=50, alpha=0.7, color='teal')
        axes[1, 0].set_title(f"Avg Depth per Image")
        axes[1, 0].set_xlabel("Depth (m)")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].grid(True)
        # Middle‑right: Avg Depth Range per Image
        axes[1, 1].hist(average_depth_range_per_image, bins=50, alpha=0.7, color='red')
        axes[1, 1].set_title(f"Avg Depth Range per Image")
        axes[1, 1].set_xlabel("Range (m)")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].grid(True)
        # Bottom‑left: Relative Depth Range
        axes[2, 0].hist(relative_depth_range_per_image, bins=50, alpha=0.7, color='orange')
        axes[2, 0].set_title(f"Relative Depth Range")
        axes[2, 0].set_xlabel("Relative Range")
        axes[2, 0].set_ylabel("Frequency")
        axes[2, 0].grid(True)
        # Bottom‑right: Depth CoV per Image
        axes[2, 1].hist(depth_cv_per_image, bins=50, alpha=0.7, color='purple')
        axes[2, 1].set_title(f"Depth Coefficient of Variation")
        axes[2, 1].set_xlabel("Coefficient (sigma/mu)")
        axes[2, 1].set_ylabel("Frequency")
        axes[2, 1].grid(True)
        plt.show()

    def extract_keyframes(self):

        if not any(self.statistics.values()):
            print(f"[KEYFRAMING] No statistics found, computing...")
            self.get_statistics()

        # 2) Compute thresholds
        feats = np.array(self.statistics["keypoints_per_image"])
        depths = np.array(self.statistics["average_depth_per_image"])
        blurs = np.array(self.statistics["motion_blur_per_image"])

        feat_thr  = np.percentile(feats, self.FEATURE_PERCENTILE)        
        depth_thr = np.percentile(depths, self.DEPTH_PERCENTILE)     
        blur_thr  = np.percentile(blurs, self.BLUR_PERCENTILE)

        # corresponding stats lists (already pure Python types)
        feats_list  = self.statistics["keypoints_per_image"]
        depths_list = self.statistics["average_depth_per_image"]
        blurs_list  = self.statistics["motion_blur_per_image"]

        candidates = []
        for p, kp_count, mean_d, blur_var in zip(self.rgb_files, feats_list, depths_list, blurs_list):
            # fast threshold checks
            if kp_count  < feat_thr:  continue
            if mean_d    < depth_thr: continue
            if blur_var  < blur_thr:  continue
            candidates.append(p)

        if not candidates:
            print(f"[KEYFRAMING] No frames passed filtering.")
            return

        descs = []
        batch_size = 16
        for i in tqdm(range(0, len(candidates), batch_size), desc="Extracting DINO features"):
            batch = candidates[i : i + batch_size]

            # load as PIL RGB
            pil_imgs = [ Image.open(str(p)).convert("RGB") for p in batch ]

            # now call the pipeline
            feats = self.PIPE_DINO(pil_imgs)  # returns list of [batch_size, 1, dim]

            for out in feats: # dim [1, D]
                descs.append(np.asarray(out[0]))

        descs = np.stack(descs)  # shape [N, D]

        # 3) Farthest‐point sampling
        N = len(descs)
        if N <= self.n_keyframes:
            selected = list(range(N))
        else:
            selected = [0]
            min_dists = np.full(N, np.inf)
            for _ in range(1, self.n_keyframes):
                last = selected[-1]
                dists = np.linalg.norm(descs - descs[last], axis=1)
                min_dists = np.minimum(min_dists, dists)
                sel = int(np.argmax(min_dists))
                selected.append(sel)

        
        self.selected_keyframes = [candidates[i] for i in selected]

        return self.selected_keyframes
    
    def extract_keyframes_evenly_spaced(self):
        
        even_stride = len(self.rgb_files) // self.n_keyframes
        self.selected_keyframes = []

        for i in range(0, len(self.rgb_files), even_stride):
            self.selected_keyframes.append(self.rgb_files[i])

        # if the last keyframe is not the last image, add it
        if self.selected_keyframes[-1] != self.rgb_files[-1]:
            self.selected_keyframes.append(self.rgb_files[-1])
            
        return self.selected_keyframes