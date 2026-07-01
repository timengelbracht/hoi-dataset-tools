# peek_and_compare_sparsh.py
import sys, os
sys.path.insert(0, "/exchange/hoi-dataset-tools/evaluations/contact_force_estimation/sparsh/sparsh")
os.environ["XFORMERS_DISABLED"] = "1"
import sys, numpy
import numpy.core.numeric as numeric

from pathlib import Path
import numpy as np
from PIL import Image
import torch, hydra
from omegaconf import OmegaConf
import random

# ---------- settings ----------
CFG_PATH       = "/data/evaluations/contact_force_estimation/outputs_sparsh/config.yaml"
DATASET_ROOT   = "/data/evaluations/contact_force_estimation/outputs_sparsh/datasets/T1_force/digit"
SPLIT          = "sparsh_format/digit_right"  # e.g. sphere/batch_1 or your custom folder
N_SAMPLES      = 8
OUT_DIR        = Path("/exchange/tmp/sparsh_peek_compare")
IMG_SIZE_WH    = (320, 240)        # (W,H)
ENCODER_TYPE   = "dinov2"          # "dino" or "dinov2"  <<< must match training
VIT_SIZE       = "base"            # "base"(768) or "small"(384)
ENCODER_CKPT   = ""                # leave empty to auto-resolve by family
DECODER_CKPT   = "/root/outputs_sparsh/checkpoints/digit_t1_force_eval/last.ckpt"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

def to_uint8_rgb(chw: np.ndarray) -> np.ndarray:
    x = np.asarray(chw, dtype=np.float32)
    vmin, vmax = float(x.min()), float(x.max())
    if vmax > vmin: x = (x - vmin) / (vmax - vmin)
    else:           x = np.zeros_like(x)
    return (x.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)

def to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

def resolve_encoder_ckpt(family: str) -> str:
    base = Path("/root/outputs_sparsh/checkpoints")
    expected = {
        "dino":   base / "sparsh_dino_base"   / "dino_vitbase.ckpt",
        "dinov2": base / "sparsh_dinov2_base" / "dinov2_vitbase.ckpt",
    }
    p = expected[family]
    if not p.is_file():
        raise FileNotFoundError(f"[checkpoints] Encoder ckpt not found for family='{family}': {p}")
    return str(p)

def pick_encoder_ckpt(family: str, provided: str) -> str:
    p = Path(provided) if provided else None
    if p and p.is_file():
        return str(p)
    return resolve_encoder_ckpt(family)

def assert_file(p: str, label: str):
    if not Path(p).is_file():
        raise FileNotFoundError(f"[checkpoints] {label} not found: {p}")

def force_load_probe_from_ckpt(model, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # Prefer 'state_dict' if present; else the repo seems to save tensors under 'model'
    sd = ckpt.get("state_dict")
    if sd is None:
        sd = ckpt.get("model")
    if not isinstance(sd, dict):
        raise RuntimeError(f"Checkpoint has no tensor dict under 'state_dict' or 'model'. Keys: {list(ckpt.keys())[:10]}")
    # Extract head weights and strip the 'model_task.' prefix
    head_sd = {k.split("model_task.", 1)[1]: v
               for k, v in sd.items()
               if isinstance(v, torch.Tensor) and "model_task." in k}
    if not head_sd:
        raise RuntimeError(f"No 'model_task.' keys found in ckpt. Sample keys: {list(sd.keys())[:20]}")
    missing, unexpected = model.model_task.load_state_dict(head_sd, strict=False)
    print(f"[manual load] loaded={len(head_sd)} missing={len(missing)} unexpected={len(unexpected)}")
    # Optional: enforce strictness to catch shape/name mismatches
    if missing or unexpected:
        model.model_task.load_state_dict(head_sd, strict=True)

def audit_probe_equals_ckpt(model, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict") or ckpt.get("model") or {}
    head_sd = {k.split("model_task.", 1)[1]: v
               for k, v in sd.items()
               if isinstance(v, torch.Tensor) and "model_task." in k}
    msd = model.model_task.state_dict()
    matched = [k for k in msd.keys() if k in head_sd]
    mismatched = [k for k in matched if not torch.equal(msd[k].cpu(), head_sd[k].cpu())]
    print(f"[audit after manual] matched={len(matched)} mismatched={len(mismatched)} (should be 0)")


def audit_probe_load(model, ckpt_path: str):
    """
    Non-invasive audit AFTER the module has loaded the probe via checkpoint_task.
    Tries to find a state-dict in the ckpt and compares it to model.model_task.
    If the ckpt doesn't expose a state-dict (e.g., has only 'model','optim',...),
    we still print head stats so you can sanity-check the load.
    """
    print(f"[audit] probing ckpt: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print(f"[audit] couldn't load ckpt ({e}); skipping ckpt-vs-model comparison.")
        ckpt = None

    head = getattr(model, "model_task", None)
    if head is None:
        print("[audit] model.model_task not found")
        return

    # quick numeric sanity: mean|weights|
    with torch.no_grad():
        mean_abs = float(torch.stack([p.detach().float().abs().mean()
                                      for p in head.parameters()]).mean().item())
        n_params = sum(p.numel() for p in head.parameters())
    print(f"[audit] head params={n_params}  mean|param|={mean_abs:.6f}")

    # If we can't find a state-dict inside the ckpt, just report keys and stop here
    if not isinstance(ckpt, dict):
        return
    top_keys = list(ckpt.keys())
    sd = None
    if isinstance(ckpt.get("state_dict"), dict):
        sd = ckpt["state_dict"]
        print("[audit] found 'state_dict' in ckpt")
    elif isinstance(ckpt.get("model"), dict):
        # sometimes ckpt['model'] is directly an OrderedDict of tensors
        maybe = ckpt["model"]
        if any(isinstance(v, torch.Tensor) for v in maybe.values()):
            sd = maybe
            print("[audit] using ckpt['model'] as state_dict (tensor dict)")
    if sd is None:
        print(f"[audit] ckpt has no obvious state-dict; top-level keys: {top_keys[:10]}")
        return

    # extract head-like keys from ckpt state_dict
    head_sd = {}
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        if "model_task." in k:
            head_sd[k.split("model_task.", 1)[1]] = v
        elif k.startswith(("probe.", "pooler.", "query_tokens", "cross_attention_block")):
            head_sd[k] = v

    if not head_sd:
        print(f"[audit] no head-like keys found in ckpt state_dict; sample keys: {list(sd.keys())[:20]}")
        return

    msd = head.state_dict()
    msd_keys = set(msd.keys())
    ckpt_keys = set(head_sd.keys())
    missing_in_ckpt = sorted(list(msd_keys - ckpt_keys))
    extra_in_ckpt   = sorted(list(ckpt_keys - msd_keys))
    matched = sorted(list(msd_keys & ckpt_keys))

    mismatched = []
    for k in matched:
        if not torch.equal(msd[k].cpu(), head_sd[k].cpu()):
            mismatched.append(k)

    print(f"[audit] matched={len(matched)}  missing_in_ckpt={len(missing_in_ckpt)}  extra_in_ckpt={len(extra_in_ckpt)}  mismatched={len(mismatched)}")
    if missing_in_ckpt:
        print("        e.g. missing:", missing_in_ckpt[:10])
    if extra_in_ckpt:
        print("        e.g. extra:", extra_in_ckpt[:10])
    if mismatched:
        print("        e.g. mismatch:", mismatched[:10])

# 0) Resolve & validate checkpoints
ENCODER_CKPT = pick_encoder_ckpt(ENCODER_TYPE, ENCODER_CKPT)
assert_file(ENCODER_CKPT, "Encoder checkpoint")
assert_file(DECODER_CKPT, "Probe checkpoint")
print(f"[use] encoder_type={ENCODER_TYPE} encoder_ckpt={ENCODER_CKPT}")
print(f"[use] decoder_ckpt={DECODER_CKPT}")

# open pickle file
with open("/data/evaluations/contact_force_estimation/outputs_sparsh/datasets/T1_force/digit/sparsh_format/digit_left/dataset_slip_forces.pkl", "rb") as f:
    import pickle
    meta = pickle.load(f)
print(f"[metadata] {list(meta.keys())}")
a = 2

# 1) Load full Hydra config & override dataset root
cfg = OmegaConf.load(CFG_PATH)
cfg.data.dataset.config.path_dataset = str(DATASET_ROOT)

# 2) Instantiate the official dataset exactly like the repo
ds = hydra.utils.instantiate(cfg.data.dataset, dataset_name=SPLIT)
scale_xyz = np.array(ds.max_abs_forceXYZ, dtype=np.float32)  # typically [4,4,5]
print(f"[dataset] {SPLIT}  len={len(ds)}  max_abs_forceXYZ={scale_xyz.tolist()}  "
      f"frame_stride={cfg.data.dataset.config.frame_stride}  resize={cfg.data.dataset.config.transforms.resize}")

# 3) Build model (let module auto-load the probe via checkpoint_task)
embed_dim = 768 if VIT_SIZE == "base" else 384
model_cfg = OmegaConf.create({
    "_target_": "tactile_ssl.downstream_task.ForceSLModule",
    "checkpoint_encoder": ENCODER_CKPT,
    "checkpoint_task": DECODER_CKPT,     # <— auto-load head here
    "train_encoder": False,
    "encoder_type": ENCODER_TYPE,
    "model_encoder": {
        "_target_": f"tactile_ssl.model.vit_{VIT_SIZE}",
        "img_size": [IMG_SIZE_WH[0], IMG_SIZE_WH[1]],  # [W,H]
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
model = hydra.utils.instantiate(model_cfg).to(DEVICE).eval()
print(f"[model] encoder_type={ENCODER_TYPE} vit={VIT_SIZE}({embed_dim}) img_size={IMG_SIZE_WH} device={DEVICE}")



force_load_probe_from_ckpt(model, DECODER_CKPT)
audit_probe_equals_ckpt(model, DECODER_CKPT)


# # 3b) Audit what got loaded (no remapping; just report)
# audit_probe_load(model, DECODER_CKPT)

# 4) Random sample a few pairs, save images, forward, compare to GT
rng = random.Random(0)
idxs = sorted(rng.sample(range(len(ds)), k=min(N_SAMPLES, len(ds))))
xs = []
ys_norm = []
for i in idxs:
    sample = ds[i]
    x6 = to_numpy(sample["image"])         # (6,H,W)
    f_norm = to_numpy(sample["force"])     # (3,)
    img0 = to_uint8_rgb(x6[0:3])
    img1 = to_uint8_rgb(x6[3:6])
    base = f"{SPLIT.replace('/','_')}_idx{i}"
    Image.fromarray(img0).save(OUT_DIR / f"{base}_t.png")
    Image.fromarray(img1).save(OUT_DIR / f"{base}_t+1.png")

    xs.append(x6)
    ys_norm.append(f_norm)

xb = torch.from_numpy(np.stack(xs, axis=0)).to(DEVICE)  # (B,6,H,W)
with torch.inference_mode():
    y_pred_norm = model(xb).detach().cpu().numpy()      # (B,3)

# 5) Denormalize and print
ys_norm = np.stack(ys_norm, axis=0)
ys_N    = ys_norm * scale_xyz[None, :]
yhat_N  = y_pred_norm * scale_xyz[None, :]

def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
rmse_all = rmse(ys_N, yhat_N)
pred_std = yhat_N.std(axis=0)

print("\n[idx]      GT(N) -> pred(N)")
for j, i in enumerate(idxs):
    gt = ys_N[j]; pr = yhat_N[j]
    print(f"[{i:04d}] GT=({gt[0]:7.3f},{gt[1]:7.3f},{gt[2]:7.3f})  "
          f"PR=({pr[0]:7.3f},{pr[1]:7.3f},{pr[2]:7.3f})")

print(f"\n[summary] RMSE over {ys_N.shape[0]} pairs (N): {rmse_all:.3f}")
print(f"[summary] pred std across batch (N): Fx={pred_std[0]:.3f}, Fy={pred_std[1]:.3f}, Fz={pred_std[2]:.3f}")
print(f"images written to: {OUT_DIR}")
