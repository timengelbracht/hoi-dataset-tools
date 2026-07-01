# debug_test_force.py
import os, sys, json, time
from pathlib import Path
os.environ["XFORMERS_DISABLED"] = "1"
os.environ["HYDRA_FULL_ERROR"] = "1"

sys.path.insert(0, "/exchange/hoi-dataset-tools/evaluations/contact_force_estimation/sparsh/sparsh")

import torch, hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# --- config you can tweak quickly ---
DATASET_ROOT = Path(os.environ.get("DATASET_PATH", "/root/outputs_sparsh/datasets/T1_force/digit"))
SPLIT        = os.environ.get("TEST_SPLIT", "sparsh_format")     # e.g. sphere/batch_1 or your custom folder
ENCODER_CKPT = os.environ.get("ENCODER_CKPT", "/root/outputs_sparsh/checkpoints/sparsh_dinov2_base/dinov2_vitbase.ckpt")
PROBE_CKPT   = os.environ.get("PROBE_CKPT",   "/root/outputs_sparsh/checkpoints/digit_t1_force_eval/last.ckpt")
OUT_ROOT     = Path(os.environ.get("OUT_ROOT", "/root/outputs_sparsh/tacbench"))
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------

def main():
    cfg_path = Path("/root/outputs_sparsh/config.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing Hydra config at {cfg_path} (train/test wrote this).")

    cfg = OmegaConf.load(cfg_path)
    # force our paths/split/overrides
    cfg.paths.output_dir      = str(Path("/root/outputs_sparsh"))
    cfg.paths.log_dir         = str(Path("/root/outputs_sparsh/logs"))
    cfg.paths.checkpoints_dir = str(Path("/root/outputs_sparsh/checkpoints"))
    cfg.paths.data_root       = str(Path("/root/outputs_sparsh"))
    cfg.paths.tacbench_dir    = str(OUT_ROOT)

    cfg.data.dataset.config.path_dataset = str(DATASET_ROOT)
    cfg.data.dataset.config.look_in_folder = True          # <— important if you don’t have pickles
    cfg.data.dataset.config.remove_bg = True               # must match training
    # keep resize (320,240), num_frames=2, frame_stride=5 from config.yaml

    # point to your checkpoints
    cfg.task.checkpoint_encoder = ENCODER_CKPT
    cfg.task.checkpoint_task    = PROBE_CKPT
    cfg.task.train_encoder      = False
    cfg.task.encoder_type       = cfg.ssl_name  # e.g., "dinov2" from your config.yaml

    print("========== DEBUG TEST FORCE ==========")
    print(f"[cfg] dataset root: {cfg.data.dataset.config.path_dataset}")
    print(f"[cfg] split:        {SPLIT}")
    print(f"[cfg] look_in_folder={cfg.data.dataset.config.get('look_in_folder', None)}")
    print(f"[cfg] remove_bg:    {cfg.data.dataset.config.get('remove_bg', None)}")
    print(f"[cfg] resize:       {cfg.data.dataset.config.transforms.resize}")
    print(f"[ckpt] encoder:     {cfg.task.checkpoint_encoder}")
    print(f"[ckpt] probe:       {cfg.task.checkpoint_task}")
    print(f"[paths] tacbench:   {cfg.paths.tacbench_dir}")
    print("======================================")

    # 1) dataset
    ds = hydra.utils.instantiate(cfg.data.dataset, dataset_name=SPLIT)
    print(f"[dataset] len={len(ds)}  (each sample is a 2-frame pair)")

    if len(ds) == 0:
        print("[dataset] EMPTY! Check folder structure:")
        print("  <root>/<split>/digit_left/no_force_reference.png")
        print("  <root>/<split>/digit_left/seq_000/frames/<numeric>.png")
        print("  <root>/<split>/digit_right/seq_000/frames/<numeric>.png")
        return

    # peek first few files (if dataset exposes paths)
    try:
        s0 = ds[0]
        print("[sample0] keys:", list(s0.keys()))
        print("[sample0] image shape:", getattr(s0["image"], "shape", None))
        print("[sample0] force (normalized):", s0.get("force", None))
    except Exception as e:
        print(f"[warn] could not inspect sample 0: {e}")

    # 2) dataloader
    bs = 64
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    print(f"[loader] batch_size={bs}  num_workers=2")

    # 3) model (ForceSLModule will load encoder+probe itself)
    print("[model] instantiating ForceSLModule …")
    model = hydra.utils.instantiate(cfg.task).to(DEVICE).eval()
    with torch.no_grad():
        n_params = sum(p.numel() for p in model.model_task.parameters())
        mean_abs = float(torch.stack([p.detach().float().abs().mean() for p in model.model_task.parameters()]).mean())
    print(f"[model] probe params={n_params}  mean|param|={mean_abs:.6f}")

    # 4) run a few batches & collect predictions
    all_preds = []
    t0 = time.time()
    with torch.inference_mode():
        for bi, batch in enumerate(dl):
            x = batch["image"].to(DEVICE, non_blocking=True)
            y = model(x)  # normalized prediction in [-1,1] scale
            y = y.detach().cpu()

            # stash timestamp if present; else fabricate indices
            # many loaders return "timestamp" or "ts", otherwise we keep row index
            ts = batch.get("timestamp", None)
            if ts is None:  # try something else
                ts = batch.get("ts", None)
            if ts is None:
                # fabricate sequential ids within this batch
                ts = [f"b{bi:04d}_i{k:03d}" for k in range(y.shape[0])]

            for tstamp, vec in zip(ts, y):
                all_preds.append((str(tstamp), float(vec[0]), float(vec[1]), float(vec[2])))

            # be verbose but light
            if bi < 3:
                m = y.mean(0).numpy()
                s = y.std(0).numpy()
                print(f"[batch {bi}] x={tuple(x.shape)} y_mean={m} y_std={s}")

    dt = time.time() - t0
    print(f"[done] forwarded {len(all_preds)} samples in {dt:.2f}s")

    # 5) save a CSV to tacbench-like location (so you can find *something* immediately)
    out_dir = Path(cfg.paths.tacbench_dir) / f"{cfg.task_name}_{cfg.sensor}" / SPLIT
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "debug_preds.csv"
    import csv
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "Fx_norm", "Fy_norm", "Fz_norm"])
        w.writerows(all_preds)
    print(f"[write] predictions → {out_csv}")

    # 6) quick summary stats (normalized space)
    import numpy as np
    arr = np.array([[p[1], p[2], p[3]] for p in all_preds], dtype=float)
    print(f"[stats] pred mean = {arr.mean(0)}")
    print(f"[stats] pred std  = {arr.std(0)}")

if __name__ == "__main__":
    main()

    