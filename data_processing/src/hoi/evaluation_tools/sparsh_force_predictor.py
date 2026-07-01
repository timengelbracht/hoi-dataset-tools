# sparsh_force_predictor_hydra.py
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union, Dict

import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
import pandas as pd
import csv
import json, pickle


class SparshForcePredictor:
    """
    Hydra-native SPaRSH predictor:
      - Instantiates the official dataset via Hydra (cfg.data.dataset)
      - Builds ForceSLModule with the given encoder/probe
      - Predicts on dataset samples (image pairs are provided by the dataset)

    Usage:
      pred = SparshForcePredictorHydra.from_hydra(
          cfg_path="/.../config.yaml",
          dataset_root="/.../datasets/T1_force/digit",
          dataset_name="sparsh_format/digit_left",   # same as your SPLIT
          encoder_ckpt="/.../dinov2_vitbase.ckpt",
          decoder_ckpt="/.../last.ckpt",
          encoder_type="dinov2",                     # or "dino"
          vit="base",                                # or "small"
          device="cuda",
      )
      out = pred.predict_all(batch_size=256, return_denorm=True, return_gt=True)
      # out is a dict with keys: idxs, y_pred_norm, y_pred_N, y_gt_norm, y_gt_N, scale_xyz
    """

    def __init__(
        self,
        ds,                            # hydra dataset object
        model,                         # ForceSLModule (eval)
        device: Union[str, torch.device],
        scale_xyz: Tuple[float, float, float],   # ds.max_abs_forceXYZ
    ):
        self.ds = ds
        self.model = model.eval().to(device)
        self.device = torch.device(device)
        self.scale_xyz = np.array(scale_xyz, dtype=np.float32)

    # ---------- construction ----------

    @staticmethod
    def _assert_file(p: str, label: str):
        if not Path(p).is_file():
            raise FileNotFoundError(f"[checkpoints] {label} not found: {p}")

    @classmethod
    def from_hydra(
        cls,
        cfg_path: Union[str, Path],
        dataset_root: Union[str, Path],
        dataset_name: str,                      # e.g., "sparsh_format/digit_left"
        encoder_ckpt: Union[str, Path],
        decoder_ckpt: Union[str, Path],
        encoder_type: str = "dinov2",          # "dino" or "dinov2"
        vit: str = "base",                     # "base" or "small"
        device: Optional[Union[str, torch.device]] = None,
        strict_manual_head_load: bool = True,  # re-load head tensors strictly (like your script)
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1) Load hydra config and override dataset root
        cfg = OmegaConf.load(str(cfg_path))
        cfg.data.dataset.config.path_dataset = str(dataset_root)

        # 2) Instantiate the official dataset exactly like the repo
        ds = hydra.utils.instantiate(cfg.data.dataset, dataset_name=dataset_name)

        # 3) Build model; let module auto-load the head via checkpoint_task (like your script)
        img_size = cfg.data.dataset.config.transforms.resize  # [W,H]
        if not len(img_size) == 2:
            raise ValueError(f"Unexpected resize in cfg: {img_size}")
        W, H = int(img_size[0]), int(img_size[1])

        model_cfg = OmegaConf.create({
            "_target_": "tactile_ssl.downstream_task.ForceSLModule",
            "checkpoint_encoder": str(encoder_ckpt),
            "checkpoint_task": str(decoder_ckpt),     # auto-load probe head
            "train_encoder": False,
            "encoder_type": encoder_type,
            "model_encoder": {
                "_target_": f"tactile_ssl.model.vit_{vit}",
                "img_size": [W, H],
                "in_chans": 6,
                "pos_embed_fn": "sinusoidal",
                "num_register_tokens": 1,
            },
            "model_task": {
                "_target_": "tactile_ssl.downstream_task.ForceLinearProbe",
                "embed_dim": "base",
                "num_heads": 12,
                "depth": 1,
                "with_last_activations": True,
            },
            "optim_cfg": {"_partial_": True, "_target_": "torch.optim.Adam", "lr": 1e-4},
            "scheduler_cfg": None,
        })
        model = hydra.utils.instantiate(model_cfg).to(device).eval()

        # Optional: manually re-load the head strictly (mirrors your force_load_probe_from_ckpt)
        if strict_manual_head_load:
            cls._manual_load_head(model, decoder_ckpt)

        pred = cls(ds=ds, model=model, device=device, scale_xyz=tuple(ds.max_abs_forceXYZ))        
        frame_stride = int(cfg.data.dataset.config.frame_stride)
        pred.side_dir = Path(dataset_root) / dataset_name
        expected_len = len(ds)
        pred.idx2name = pred.load_idx2name_from_cfg(
            cfg=cfg,
            side_dir=os.path.join(dataset_root, dataset_name),
            frame_stride=cfg.data.dataset.config.frame_stride,
            stamp="earlier",   # <— recommended to remove lag vs GT
            hop=1,
        )

        return pred

    @staticmethod
    def _manual_load_head(model, ckpt_path: Union[str, Path]):
        """Strictly load only 'model_task.' params from checkpoint (your script's behavior)."""
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        sd = ckpt.get("state_dict")
        if sd is None:
            sd = ckpt.get("model")
        if not isinstance(sd, dict):
            raise RuntimeError(f"Checkpoint has no tensor dict under 'state_dict' or 'model'. Keys: {list(ckpt.keys())[:10]}")

        head_sd = {k.split("model_task.", 1)[1]: v
                   for k, v in sd.items()
                   if isinstance(v, torch.Tensor) and "model_task." in k}
        if not head_sd:
            raise RuntimeError(f"No 'model_task.' keys found in ckpt. Sample keys: {list(sd.keys())[:20]}")
        missing, unexpected = model.model_task.load_state_dict(head_sd, strict=False)
        if missing or unexpected:
            model.model_task.load_state_dict(head_sd, strict=True)

    # ---------- prediction ----------

    @torch.inference_mode()
    def predict_indices(
        self,
        idxs: Optional[Sequence[int]] = None,
        batch_size: int = 256,
        return_denorm: bool = False,
        return_gt: bool = False,
    ) -> Dict[str, np.ndarray]:
        n = len(self.ds)
        if n == 0:
            raise RuntimeError("Dataset has length 0.")
        if idxs is None:
            idxs = list(range(n))

        used_idxs: list[int] = []
        used_names: list[str] = []
        y_pred_norm_chunks: list[np.ndarray] = []
        y_gt_norm_chunks: list[np.ndarray] = []

        for s in range(0, len(idxs), batch_size):
            chunk = idxs[s:s + batch_size]
            xs = []
            ys = []
            chunk_ok_idxs = []
            chunk_ok_names = []

            for i in chunk:
                # try get sample; skip on failure/missing image
                try:
                    sample = self.ds[i]
                except Exception as e:
                    print(f"[warn] skipping idx {i}: dataset error: {e}")
                    continue
                if not isinstance(sample, dict) or "image" not in sample or sample["image"] is None:
                    print(f"[warn] skipping idx {i}: sample None or missing 'image'")
                    continue

                x6 = sample["image"]
                if isinstance(x6, torch.Tensor):
                    x6 = x6.detach().cpu().numpy()
                else:
                    x6 = np.asarray(x6)
                xs.append(x6)
                chunk_ok_idxs.append(i)
                chunk_ok_names.append(str(self.idx2name[i]) if hasattr(self, "idx2name") and i in self.idx2name else str(i))

                if return_gt and "force" in sample and sample["force"] is not None:
                    f_norm = sample["force"]
                    if isinstance(f_norm, torch.Tensor):
                        f_norm = f_norm.detach().cpu().numpy()
                    else:
                        f_norm = np.asarray(f_norm)
                    ys.append(f_norm)

            if not xs:
                continue  # entire batch skipped

            xb = torch.from_numpy(np.stack(xs, axis=0)).to(self.device)  # (B,6,H,W)
            Fb = self.model(xb).detach().cpu().numpy().astype(np.float32)  # (B,3)

            y_pred_norm_chunks.append(Fb)
            used_idxs.extend(chunk_ok_idxs)
            used_names.extend(chunk_ok_names)

            if return_gt and ys:
                y_gt_norm_chunks.append(np.stack(ys, axis=0).astype(np.float32))

        # if everything was skipped, return empty but valid shapes
        if not used_idxs:
            result = {
                "idxs": np.empty((0,), dtype=np.int64),
                "names": [],
                "y_pred_norm": np.empty((0, 3), dtype=np.float32),
                "scale_xyz": self.scale_xyz.copy(),
            }
            if return_denorm:
                result["y_pred_N"] = np.empty((0, 3), dtype=np.float32)
            if return_gt:
                result["y_gt_norm"] = np.empty((0, 3), dtype=np.float32)
                if return_denorm:
                    result["y_gt_N"] = np.empty((0, 3), dtype=np.float32)
            return result

        y_pred_norm = np.concatenate(y_pred_norm_chunks, axis=0)
        result = {
            "idxs": np.asarray(used_idxs, dtype=np.int64),
            "names": used_names,
            "y_pred_norm": y_pred_norm,
            "scale_xyz": self.scale_xyz.copy(),
        }

        if return_denorm:
            result["y_pred_N"] = y_pred_norm * self.scale_xyz[None, :]

        if return_gt and y_gt_norm_chunks:
            y_gt_norm = np.concatenate(y_gt_norm_chunks, axis=0)
            result["y_gt_norm"] = y_gt_norm
            if return_denorm:
                result["y_gt_N"] = y_gt_norm * self.scale_xyz[None, :]

        return result


    @torch.inference_mode()
    def predict_all(
        self,
        batch_size: int = 256,
        return_denorm: bool = False,
        return_gt: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Convenience to run over the entire dataset (0..len-1)."""
        return self.predict_indices(
            idxs=None,
            batch_size=batch_size,
            return_denorm=return_denorm,
            return_gt=return_gt,
        )

    @classmethod
    def load_idx2name_from_cfg(self, *, cfg=None, side_dir=None, frame_stride=None,
                            stamp="earlier", hop=1, strict=False):
        """
        Build sample_idx -> timestamp mapping to mirror the dataset's (t, t+frame_stride) pairing.

        If a per-sequence pickle or index.json is missing (e.g., you manually deleted a bad seq),
        the sequence is skipped unless strict=True.
        """
        import os, json, pickle
        from pathlib import Path

        # allow deriving from cfg if not explicitly provided
        if side_dir is None:
            if cfg is None:
                raise ValueError("side_dir not given and cfg is None")
            side_dir = os.path.join(cfg.data.dataset.config.path_dataset, cfg.data.dataset_name)
        if frame_stride is None:
            if cfg is None:
                raise ValueError("frame_stride not given and cfg is None")
            frame_stride = int(cfg.data.dataset.config.frame_stride)

        side_dir = Path(side_dir)
        frame_stride = int(frame_stride)

        # 1) load trajectories (defines per-sequence ids)
        with open(side_dir / "dataset_slip_forces.pkl", "rb") as f:
            bundle = pickle.load(f)
        trajectories = bundle["trajectories"]  # {traj_id: {...}}

        def seq_base_of(traj_id: int) -> str:
            return f"dataset_digit_{int(traj_id):03d}"

        idx2name: dict[int, str] = {}
        sample_idx = 0

        for traj_id in sorted(trajectories, key=lambda x: int(x)):
            seq_base = seq_base_of(int(traj_id))
            pkl_path   = side_dir / f"{seq_base}.pkl"
            index_json = side_dir / f"{seq_base}_index.json"

            # Skip sequences whose artifacts were deleted or never written
            missing = []
            if not pkl_path.is_file():
                missing.append(str(pkl_path.name))
            if not index_json.is_file():
                missing.append(str(index_json.name))

            if missing:
                msg = f"[warn] skipping seq {seq_base} — missing: {', '.join(missing)}"
                if strict:
                    raise FileNotFoundError(msg)
                else:
                    print(msg)
                    continue

            # names aligned to every frame (including ref at local 0)
            with open(index_json, "r") as jf:
                names = json.load(jf).get("names", [])

            T = len(names)
            if T <= frame_stride:
                # not enough frames to form a single (t, t+stride) pair — skip safely
                print(f"[warn] skipping seq {seq_base} — T={T} <= frame_stride={frame_stride}")
                continue

            # pairing loop
            for t in range(0, T - frame_stride, hop):
                tp = t + frame_stride
                if stamp == "earlier":
                    label = names[t]
                elif stamp == "midpoint":
                    try:
                        t0 = int(str(names[t])); t1 = int(str(names[tp]))
                        label = str((t0 + t1) // 2)
                    except Exception:
                        label = names[tp]
                else:  # "later"
                    label = names[tp]

                idx2name[sample_idx] = str(label)
                sample_idx += 1

        # Optional: verify mapping size matches dataset length if dataset attached
        try:
            n_ds = len(self.dataset)
            if len(idx2name) != n_ds:
                have = set(idx2name.keys())
                missing = [i for i in range(n_ds) if i not in have][:10]
                print(f"[warn] idx2name size mismatch: built {len(idx2name)} vs dataset {n_ds}. "
                    f"First missing indices: {missing}")
        except Exception:
            pass

        return idx2name


    @staticmethod
    def _build_idx2name_from_bundle(
        side_dir: str,
        frame_stride: int,
        hop: int = 1,
        expected_len: int = None,   # pass len(dataset) to align exactly
    ) -> dict[int, str]:
        """
        Build sample_idx -> timestamp mapping from SPaRSH bundles.
        Uses the *later* frame (t + frame_stride) as the timestamp.
        Guarantees mapping length == expected_len (if provided).
        """
        import json, pickle
        from pathlib import Path

        side_dir = Path(side_dir)
        with open(side_dir / "dataset_slip_forces.pkl", "rb") as f:
            bundle = pickle.load(f)
        trajectories = bundle["trajectories"]  # dict[int]->{'indexes':[...]}

        idx2name: dict[int, str] = {}
        sample_idx = 0

        for traj_id in sorted(trajectories, key=lambda x: int(x)):
            seq_base = f"dataset_digit_{int(traj_id):03d}"
            with open(side_dir / f"{seq_base}_index.json", "r") as jf:
                names = json.load(jf)["names"]  # aligned 1:1 to frames (includes ref at 0)

            idxs = list(trajectories[traj_id]["indexes"])
            T = min(len(idxs), len(names))
            if T <= frame_stride:
                continue

            # --- FIXED LOOP: never produce out-of-bounds tp ---
            for t in range(1, T, hop):         # skip reference at 0
                tp = t + frame_stride
                if tp >= T:
                    break                      # stop before overflow
                idx2name[sample_idx] = str(names[tp])
                sample_idx += 1

        # If the caller gives expected_len, align exactly.
        if expected_len is not None:
            n = len(idx2name)
            if n > expected_len:
                # trim extras deterministically
                keep_keys = list(range(expected_len))
                idx2name = {k: idx2name[k] for k in keep_keys}
            elif n < expected_len:
                # pad with placeholders (won’t crash your code)
                for k in range(n, expected_len):
                    idx2name[k] = f"missing_{k}"

        return idx2name


    

def load_idx2name(sample_index_csv: Path) -> dict[int, str]:
    idx2name = {}
    with open(sample_index_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["sample_idx"])
            name = row["name"]  # this is your timestamp string
            idx2name[idx] = name
    return idx2name