import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import pandas as pd
import shutil
import cv2
from collections import defaultdict

from pathlib import Path
from hoi.data_tools.data_indexer import RecordingIndex
from hoi.data_tools.data_loader_gripper import GripperData

import matplotlib.pyplot as plt

import clip


# -------------------------
# Dataset
# -------------------------
class ForceDataset(Dataset):
    def __init__(self, root, img_size=224):
        self.root = root
        self.samples = []

        # Walk two levels: root/kitchen_xxx/yyy/
        for scene in sorted(os.listdir(root)):
            scene_path = os.path.join(root, scene)
            if not os.path.isdir(scene_path):
                continue

            for sample in sorted(os.listdir(scene_path)):
                sample_path = os.path.join(scene_path, sample)
                if not os.path.isdir(sample_path):
                    continue

                self.samples.append(sample_path)

        print(f"[Dataset] Found {len(self.samples)} samples")

        self.tf = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.samples)

    def _find_rgb(self, path):
        for name in ["rgb.jpg", "rgb.png", "rgb.jpeg"]:
            p = os.path.join(path, name)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"No RGB image in {path}")

    def __getitem__(self, idx):
        path = self.samples[idx]

        rgb_path = self._find_rgb(path)
        depth_path = os.path.join(path, "depth.npy")
        force_path = os.path.join(path, "force.txt")
        label_path = os.path.join(path, "label.txt")
        mask_path = os.path.join(path, "mask.png")  # optional

        # --- Load ---
        rgb = Image.open(rgb_path).convert("RGB")
        depth = np.load(depth_path).astype(np.float32)

        with open(force_path) as f:
            force = float(f.read().strip())

        with open(label_path) as f:
            label_str = f.read().strip().lower()

        # Optional mask
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
        else:
            mask = Image.new("L", rgb.size, color=255)

        # --- Transforms ---
        rgb = self.tf(rgb)

        mask = T.Resize(rgb.shape[1:])(T.ToTensor()(mask))
        mask = (mask > 0.5).float()

        # Masked RGB
        rgb = rgb * mask + 0.5 * (1 - mask)

        # --- Depth stats ---
        mask_np = mask.squeeze().numpy() > 0

        # Resize depth to match mask/RGB resolution
        depth_resized = cv2.resize(
            depth,
            (mask.shape[-1], mask.shape[-2]),
            interpolation=cv2.INTER_NEAREST
        )

    # Keep only valid depth values inside mask
        dvals = depth_resized[mask_np]
        valid = np.isfinite(dvals) & (dvals > 0.0)
        dvals = dvals[valid]

        # Fallbacks if empty
        if dvals.size == 0:
            d_mean = 0.0
            d_std = 0.0
        else:
            d_mean = float(np.mean(dvals))
            d_std = float(np.std(dvals))

        area = float(mask.mean().item())

        depth_feats = torch.tensor(
            [d_mean, d_std, area],
            dtype=torch.float32
        )

        # Log-scale target
        y = torch.tensor(np.log1p(force), dtype=torch.float32)

        return rgb, depth_feats, label_str, y


class DinoClipForceNet(nn.Module):
    def __init__(
        self,
        dino_name="dinov2_vits14",
        clip_model_name="ViT-B/32",
        alpha=1.0
    ):
        super().__init__()

        self.alpha = alpha

        # ------------------
        # DINOv2 backbone
        # ------------------
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            dino_name
        )
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.dino_dim = 384 if "vits" in dino_name else 768

        # ------------------
        # Head = VISUAL RESIDUAL
        # ------------------
        self.head = nn.Sequential(
            nn.Linear(self.dino_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def _device(self):
        return next(self.parameters()).device

    # SAME SIGNATURE
    def forward(self, rgb, depth_feats, label_strs, prior_mu_batch):
        device = self._device()

        with torch.no_grad():
            v = self.backbone(rgb.to(device))

        # Predict small correction
        delta = self.head(v).squeeze(1)

        # Anchor to class prior
        mu = prior_mu_batch.to(device) + self.alpha * delta

        return mu

# -------------------------
# Loss
# -------------------------
def loss_fn(pred, target):
    return torch.mean((pred - target) ** 2)


# -------------------------
# Training
# -------------------------
def train_force_prior(
    data_root,
    epochs=50,
    batch_size=8,
    lr=1e-3,
    img_size=224,
    dino_name="dinov2_vits14",
    clip_model_name="ViT-B/32",
    alpha=3.0,
    save_path="force_net.pt", PRIORS=None, GLOBAL_MEAN=None
):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = ForceDataset(data_root, img_size=img_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DinoClipForceNet(
        dino_name=dino_name,
        clip_model_name=clip_model_name,
        alpha=alpha
    ).to(DEVICE)

    optimizer = Adam(model.head.parameters(), lr=lr, weight_decay=1e-3)

    print(f"Training on {len(dataset)} samples using {DEVICE}")
    print(f"CLIP weight alpha = {alpha}")

    for epoch in range(epochs):
        total_loss = 0.0

        for rgb, dfeat, labels, y in loader:
            rgb = rgb.to(DEVICE)
            dfeat = dfeat.to(DEVICE)
            y = y.to(DEVICE)

            prior_vals = [PRIORS.get(lbl, GLOBAL_MEAN) for lbl in labels]
            prior_mu = torch.tensor(prior_vals, dtype=torch.float32, device=DEVICE)

            mu = model(rgb, dfeat, labels, prior_mu)
            loss = loss_fn(mu, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch:03d} | Loss {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def compute_class_priors(root):
    by_class = defaultdict(list)

    for scene in os.listdir(root):
        sp = os.path.join(root, scene)
        if not os.path.isdir(sp):
            continue
        for sample in os.listdir(sp):
            p = os.path.join(sp, sample)
            if not os.path.isdir(p):
                continue

            try:
                lbl = open(os.path.join(p, "label.txt")).read().strip().lower()
                F = float(open(os.path.join(p, "force.txt")).read().strip())
                by_class[lbl].append(np.log1p(F))
            except:
                pass

    priors = {k: float(np.mean(v)) for k, v in by_class.items()}
    global_mean = float(np.mean([x for v in by_class.values() for x in v]))
    return priors, global_mean

def predict_force_from_folder(
    model,
    folder,
    img_size=224,
    device=None,
    PRIORS=None,
    GLOBAL_MEAN=None,
):
    model.eval()
    device = device or next(model.parameters()).device

    # --- Find files ---
    def find(name_list):
        for n in name_list:
            p = os.path.join(folder, n)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"Missing one of {name_list} in {folder}")

    rgb_path = find(["rgb.png", "rgb.jpg", "rgb.jpeg"])
    depth_path = find(["depth.npy"])
    label_path = find(["label.txt"])
    mask_path = None
    for m in ["mask.png", "mask.jpg"]:
        p = os.path.join(folder, m)
        if os.path.exists(p):
            mask_path = p
            break

    # --- Load ---
    rgb = Image.open(rgb_path).convert("RGB")
    depth = np.load(depth_path).astype(np.float32)
    with open(label_path) as f:
        label_str = f.read().strip().lower()

    if mask_path:
        mask = Image.open(mask_path).convert("L")
    else:
        mask = Image.new("L", rgb.size, color=255)

    # --- Transforms ---
    tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    rgb = tf(rgb)

    mask = T.Resize(rgb.shape[1:])(T.ToTensor()(mask))
    mask = (mask > 0.5).float()

    # Masked RGB
    rgb = rgb * mask + 0.5 * (1 - mask)

    # --- Depth stats ---
    depth_resized = cv2.resize(
        depth,
        (mask.shape[-1], mask.shape[-2]),
        interpolation=cv2.INTER_NEAREST
    )

    mask_np = mask.squeeze().numpy() > 0
    dvals = depth_resized[mask_np]
    valid = np.isfinite(dvals) & (dvals > 0.0)
    dvals = dvals[valid]

    if dvals.size == 0:
        d_mean, d_std = 0.0, 0.0
    else:
        d_mean = float(np.mean(dvals))
        d_std = float(np.std(dvals))

    area = float(mask.mean().item())

    depth_feats = torch.tensor(
        [[d_mean, d_std, area]],
        dtype=torch.float32
    ).to(device)

    # --- Forward ---
    rgb = rgb.unsqueeze(0).to(device)

    prior_mu = torch.tensor(
        [PRIORS.get(label_str, GLOBAL_MEAN)],
        dtype=torch.float32,
        device=device
    )

    mu = model(rgb, depth_feats, [label_str], prior_mu)[0]

    force_pred = float(torch.expm1(mu).item())
    force_pred = float(np.clip(force_pred, 0.0, 80.0))  # safety clamp

    return force_pred


def load_interaction_splitting_info(interaction_splitting_info_path: Path):
    # Load interaction splitting info from the given path json file

    #load json
    import json
    with open(interaction_splitting_info_path, 'r') as f:
        interaction_splitting_info = json.load(f)

    return interaction_splitting_info

def plot_forces_over_time(df_forces):

    time = df_forces['timestamp'] - df_forces['timestamp'].iloc[0]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(time, df_forces['wrench_ext.force.x_filt'], label='Force X')
    plt.plot(time, df_forces['wrench_ext.force.y_filt'], label='Force Y')
    plt.plot(time, df_forces['wrench_ext.force.z_filt'], label='Force Z')
    plt.title('Forces over Time')
    plt.xlabel('Time (ns)')
    plt.ylabel('Force (N)')
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(time, df_forces['wrench_ext.torque.x_filt'], label='Torque X')
    plt.plot(time, df_forces['wrench_ext.torque.y_filt'], label='Torque Y')
    plt.plot(time, df_forces['wrench_ext.torque.z_filt'], label='Torque Z')
    plt.title('Torques over Time')
    plt.xlabel('Time (ns)')
    plt.ylabel('Torque (Nm)')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig("/exchange/forces.png")

    a = 2
    

def generate_force_ground_truth_for_single_location(
        gripper_data,
        output_path,
        interaction_splitting_info,
):
    
    pass

    output_path.mkdir(parents=True, exist_ok=True)

    # load time windows from interaction_splitting_info
    #todo start at index 1
    for idx in range(0, len(interaction_splitting_info.keys()) + 1):

        #if idx % 1 == 0:
        #    continue  # only odd indices (opening)

        window_key = f'window_{idx}'
        if window_key not in interaction_splitting_info:
            continue


        window_start = interaction_splitting_info[window_key]['start_ns']
        window_end = interaction_splitting_info[window_key]['end_ns']

        #load forces from gripper data

        df_ft = gripper_data.get_force_torque_measurements() # ca 100 Hz
        df_ft = df_ft[["timestamp", "wrench_ext.force.x_filt",
                            "wrench_ext.force.y_filt",
                            "wrench_ext.force.z_filt",
                            "wrench_ext.torque.x_filt",
                            "wrench_ext.torque.y_filt",
                            "wrench_ext.torque.z_filt"]]  
        
        #filter forces in the time window
        df_ft_window = df_ft[(df_ft['timestamp'] >= window_start) & (df_ft['timestamp'] <= window_end)]

        #plot forces over time
        plot_forces_over_time(df_ft_window)

        # get rgb and depth frames
        rgb_frames = gripper_data.get_frames_rgb(side="left")
        df_rgb_frames = pd.DataFrame({
            'frame_path_iphone': [str(p) for p in rgb_frames],
            'timestamp': [int(Path(p).stem) for p in rgb_frames],
        })

        depth_frames = gripper_data.get_frames_depth()
        df_depth_frames = pd.DataFrame({
            'frame_path_depth': [str(p) for p in depth_frames],
            'timestamp': [int(Path(p).stem) for p in depth_frames],
        })

        # get peak force in z direction and time of peak force (use magnitued value)
        peak_force_z = df_ft_window['wrench_ext.force.z_filt'].abs().max()
        peak_force_time = df_ft_window.loc[df_ft_window['wrench_ext.force.z_filt'].abs().idxmax()]['timestamp']

        # get x and y at peak force time and compyte total resultant force
        peak_force_x = df_ft_window.loc[df_ft_window['wrench_ext.force.z_filt'].abs().idxmax()]['wrench_ext.force.x_filt']
        peak_force_y = df_ft_window.loc[df_ft_window['wrench_ext.force.z_filt'].abs().idxmax()]['wrench_ext.force.y_filt']
        peak_force_z_signed = df_ft_window.loc[df_ft_window['wrench_ext.force.z_filt'].abs().idxmax()]['wrench_ext.force.z_filt']
        peak_force_z = np.sqrt(peak_force_x**2 + peak_force_y**2 + peak_force_z_signed**2)
        

        # peak_force_z = df_ft_window['wrench_ext.force.z_filt'].max()
        # peak_force_time = df_ft_window.loc[df_ft_window['wrench_ext.force.z_filt'].idxmax()]['timestamp']

        # get frame 4 secs before peak force
        frame_time = peak_force_time - 2_500_000_000  #
        df_rgb_closest = df_rgb_frames.iloc[(df_rgb_frames['timestamp'] - frame_time).abs().argsort()[:1]]
        df_depth_closest = df_depth_frames.iloc[(df_depth_frames['timestamp'] - frame_time).abs().argsort()[:1]] 

        #save rgb for debugging with plt.savefig
        # only using plt
        plt.imshow(plt.imread(df_rgb_closest['frame_path_iphone'].values[0]))
        plt.axis('off')
        plt.imsave("/exchange/rgb_debug.png", plt.imread(df_rgb_closest['frame_path_iphone'].values[0]))
    

        # save rgb, depth, abs force, label to output_path
        sample_output_path = output_path / f"{idx:03d}"
        sample_output_path.mkdir(parents=True, exist_ok=True)

        # --- RGB ---
        rgb_src_path = Path(df_rgb_closest["frame_path_iphone"].values[0])
        rgb_dest_path = sample_output_path / "rgb.jpg"

        shutil.copy2(str(rgb_src_path), str(rgb_dest_path))

        # Sanity check
        try:
            img = Image.open(rgb_dest_path)
            img.verify()
        except Exception as e:
            print(f"[ERROR] Invalid RGB file: {rgb_dest_path}")
            raise e

        # --- Depth ---
        depth_src_path = Path(df_depth_closest["frame_path_depth"].values[0])
        depth_dest_path = sample_output_path / "depth.npy"

        shutil.copy2(str(depth_src_path), str(depth_dest_path))

        # save force
        force_dest_path = sample_output_path / "force.txt"
        with open(force_dest_path, 'w') as f:
            f.write(f"{peak_force_z}")

        # save label
        label_dest_path = sample_output_path / "label.txt"
        with open(label_dest_path, 'w') as f:
            f.write("")  # placeholder for future use


    a = 2






if __name__ == "__main__":

    rec_location = "livingroom_1"
    base_path = Path(f"/data/ikea_recordings")
    data_indexer = RecordingIndex(
        os.path.join(str(base_path), "extracted") 
    )
    visualize = True
    rec_type = "gripper"
    rec_module = "gripper"
    interaction_indices = "1-7"
    color = 'blue'


    gripper_data = GripperData(base_path, 
                               rec_loc=rec_location, 
                               rec_type=rec_type, 
                               rec_module=rec_module, 
                               interaction_indices=interaction_indices,
                               data_indexer=data_indexer,
                               color=color,)

        
    interaction_splitting_query = data_indexer.query_splitting(
        location=rec_location,
        interaction=rec_type,
        interaction_index=interaction_indices,
        
    )

    interaction_splitting_info_path = interaction_splitting_query[0][-1]
    interaction_splitting_info = load_interaction_splitting_info(interaction_splitting_info_path)

    output_path = Path("/data/robot_tests/dataset") / f"{rec_location}_{interaction_indices}"

    generate_force_ground_truth_for_single_location(
        gripper_data=gripper_data,
        output_path=output_path,
        interaction_splitting_info=interaction_splitting_info,
    )





