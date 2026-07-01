import os
import sys, os
sys.path.insert(0, "/exchange/hoi-dataset-tools/evaluations/contact_force_estimation/sparsh/sparsh")
import pandas as pd
from pathlib import Path
import numpy as np
import json
import math
os.environ["XFORMERS_DISABLED"] = "1"
from sparsh_force_predictor import SparshForcePredictor
import matplotlib.pyplot as plt

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_per_sequence(df_out: pd.DataFrame, preds_csv_path: str, scales=(4.0, 4.0, 5.0)):
    """
    Create per-sequence plots comparing GT vs predicted loads (raw and clipped):
      • L_star_tangential_L2  vs  L_pred_tangential
      • L_star_z              vs  L_pred_normal
    and also overlay their *clipped* counterparts.

    Saves PNGs into <csv_dir>/<csv_stem>_plots/.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    preds_csv_path = Path(preds_csv_path)
    out_dir = preds_csv_path.parent / (preds_csv_path.stem + "_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # capacity from training scales (two digits)
    cap_x, cap_y, cap_z = 2 * scales[0], 2 * scales[1], 2 * scales[2]
    TAN_MAX = float(np.hypot(cap_x, cap_y))  # e.g. ~11.3137
    NORM_MAX = float(cap_z)                  # e.g. 10.0

    # detect optional per-sequence column
    cand_cols = ["seq", "sequence", "sequence_id", "dataset_seq", "batch", "batch_id"]
    seq_col = next((c for c in cand_cols if c in df_out.columns), None)

    # x-axis: prefer timestamp_img, else timestamp, else row index
    if "timestamp_img" in df_out.columns:
        x_series = df_out["timestamp_img"].astype(str)
        xlabel = "timestamp_img"
    elif "timestamp" in df_out.columns:
        x_series = df_out["timestamp"].astype(str)
        xlabel = "timestamp"
    else:
        x_series = pd.Series(np.arange(len(df_out)), name="index")
        xlabel = "index"

    # group rows
    groups = [("all", df_out.copy())] if seq_col is None else list(df_out.groupby(seq_col, sort=False))

    # helpers
    def _num(s, default=0.0):
        return pd.to_numeric(s, errors="coerce").fillna(default).to_numpy(dtype=float)

    # check if component preds exist so we can clip component-wise
    have_components = all(c in df_out.columns for c in
                          ["Fx_left_pred", "Fy_left_pred", "Fz_left_pred",
                           "Fx_right_pred", "Fy_right_pred", "Fz_right_pred"])

    for seq_name, df_g in groups:
        X = x_series.loc[df_g.index]
        # numeric sort if possible for prettier lines
        try:
            order = np.argsort(pd.to_numeric(X, errors="coerce").fillna(np.inf).to_numpy())
        except Exception:
            order = np.argsort(np.arange(len(X)))
        Xs = X.to_numpy()[order]

        # --- Ground truth (raw & clipped) ---
        gt_tan_raw  = _num(df_g.loc[X.index, "L_star_tangential_L2"])[order]
        gt_norm_raw = _num(df_g.loc[X.index, "L_star_z"])[order]

        if {"L_star_tangential_L2_clipped", "L_star_z_clipped"} <= set(df_g.columns):
            gt_tan_clip  = _num(df_g.loc[X.index, "L_star_tangential_L2_clipped"])[order]
            gt_norm_clip = _num(df_g.loc[X.index, "L_star_z_clipped"])[order]
        else:
            # fallback: clip raw GT magnitudes by capacity
            gt_tan_clip  = np.minimum(gt_tan_raw, TAN_MAX)
            gt_norm_clip = np.minimum(gt_norm_raw, NORM_MAX)

        # --- Predictions (raw) ---
        pr_tan_raw  = _num(df_g.loc[X.index, "L_pred_tangential"])[order]
        pr_norm_raw = _num(df_g.loc[X.index, "L_pred_normal"])[order]

        # --- Predictions (clipped) ---
        if have_components:
            FxL = _num(df_g.loc[X.index, "Fx_left_pred"])[order]
            FyL = _num(df_g.loc[X.index, "Fy_left_pred"])[order]
            FzL = _num(df_g.loc[X.index, "Fz_left_pred"])[order]
            FxR = _num(df_g.loc[X.index, "Fx_right_pred"])[order]
            FyR = _num(df_g.loc[X.index, "Fy_right_pred"])[order]
            FzR = _num(df_g.loc[X.index, "Fz_right_pred"])[order]

            fx_sum = np.abs(FxL) + np.abs(FxR)
            fy_sum = np.abs(FyL) + np.abs(FyR)
            fz_sum = np.abs(FzL) + np.abs(FzR)

            fx_c = np.minimum(fx_sum, cap_x)
            fy_c = np.minimum(fy_sum, cap_y)
            fz_c = np.minimum(fz_sum, cap_z)

            pr_tan_clip  = np.sqrt(fx_c**2 + fy_c**2)
            pr_norm_clip = fz_c
        else:
            # fallback: clip magnitudes
            pr_tan_clip  = np.minimum(pr_tan_raw, TAN_MAX)
            pr_norm_clip = np.minimum(pr_norm_raw, NORM_MAX)

        # ================= Plots =================
        # Tangential
        plt.figure(figsize=(11, 4))
        plt.plot(Xs, gt_tan_raw,  label="GT tangential (raw)")
        plt.plot(Xs, pr_tan_raw,  label="Pred tangential (raw)")
        plt.plot(Xs, gt_tan_clip, label=f"GT tangential (clipped ≤ {TAN_MAX:.2f}N)")
        plt.plot(Xs, pr_tan_clip, label=f"Pred tangential (clipped ≤ {TAN_MAX:.2f}N)")
        plt.title(f"Sequence {seq_name} — Tangential load")
        plt.xlabel(xlabel)
        plt.ylabel("N")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{seq_name}_tangential.png", dpi=150)
        plt.close()

        # Normal
        plt.figure(figsize=(11, 4))
        plt.plot(Xs, gt_norm_raw,  label="GT normal (raw)")
        plt.plot(Xs, pr_norm_raw,  label="Pred normal (raw)")
        plt.plot(Xs, gt_norm_clip, label=f"GT normal (clipped ≤ {NORM_MAX:.1f}N)")
        plt.plot(Xs, pr_norm_clip, label=f"Pred normal (clipped ≤ {NORM_MAX:.1f}N)")
        plt.title(f"Sequence {seq_name} — Normal load")
        plt.xlabel(xlabel)
        plt.ylabel("N")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{seq_name}_normal.png", dpi=150)
        plt.close()

    print(f"[ok] wrote per-sequence plots (raw + clipped) to {out_dir}")

def eval_l2_with_splits_and_calibration(
    df: pd.DataFrame,
    json_path,
    scales=(4.0, 4.0, 5.0),        # training per-digit scales -> per-axis caps = 2*scale
    preds_in_newtons: bool = True, 
    seed: int = 0,
):
    import math, json
    from pathlib import Path
    import numpy as np
    import pandas as pd

    D = df.copy()

    # --- required GT columns (raw) ---
    req_raw = {"L_star_tangential_L2", "L_star_z", "L_star_L2"}
    missing = sorted(req_raw - set(D.columns))
    if missing:
        raise KeyError(f"Missing GT columns: {missing}")

    # --- recommended GT clipped columns (produced in your pipeline) ---
    req_clip = {
        "L_star_tangential_L2_clipped", "L_star_z_clipped", "L_star_L2_clipped"
    }
    missing_clip = sorted(req_clip - set(D.columns))
    if missing_clip:
        raise KeyError(
            "Missing *clipped* GT columns (build them where you compute L*): "
            f"{missing_clip}"
        )

    def _col(name):
        return pd.to_numeric(D.get(name, 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)

    # predictions (per-digit components)
    FxL = _col("Fx_left_pred");  FyL = _col("Fy_left_pred");  FzL = _col("Fz_left_pred")
    FxR = _col("Fx_right_pred"); FyR = _col("Fy_right_pred"); FzR = _col("Fz_right_pred")

    # --- scale preds to Newtons if needed ---
    if preds_in_newtons:
        sx = sy = sz = 1.0
    else:
        sx, sy, sz = scales
    FxL *= sx; FyL *= sy; FzL *= sz
    FxR *= sx; FyR *= sy; FzR *= sz

    # --- combined per-axis magnitudes (antipodal: sum of absolutes) ---
    fx_sum = np.abs(FxL) + np.abs(FxR)
    fy_sum = np.abs(FyL) + np.abs(FyR)
    fz_sum = np.abs(FzL) + np.abs(FzR)

    # --- raw GT vectors (already Newtons) ---
    y_tan_raw   = _col("L_star_tangential_L2")
    y_norm_raw  = _col("L_star_z")
    y_total_raw = _col("L_star_L2")

    # --- raw predictions (combined magnitudes) ---
    yhat_tan_raw   = np.sqrt(fx_sum**2 + fy_sum**2)
    yhat_norm_raw  = fz_sum
    yhat_total_raw = np.sqrt(fx_sum**2 + fy_sum**2 + fz_sum**2)

    # --- clipping thresholds from training capacity (two digits) ---
    #   per-axis caps (sum of two digits): Fx<=8, Fy<=8, Fz<=10 when scales=[4,4,5]
    cap_x, cap_y, cap_z = 2*scales[0], 2*scales[1], 2*scales[2]
    TAN_MAX   = math.hypot(cap_x, cap_y)
    NORM_MAX  = cap_z
    TOTAL_MAX = math.sqrt(cap_x**2 + cap_y**2 + cap_z**2)

    # --- clipped predictions (component-wise, then recompute L2s) ---
    fx_sum_c = np.minimum(fx_sum, cap_x)
    fy_sum_c = np.minimum(fy_sum, cap_y)
    fz_sum_c = np.minimum(fz_sum, cap_z)
    yhat_tan_clip   = np.sqrt(fx_sum_c**2 + fy_sum_c**2)
    yhat_norm_clip  = fz_sum_c
    yhat_total_clip = np.sqrt(fx_sum_c**2 + fy_sum_c**2 + fz_sum_c**2)

    # --- clipped GT (from your dataframe) ---
    y_tan_clip   = _col("L_star_tangential_L2_clipped")
    y_norm_clip  = _col("L_star_z_clipped")
    y_total_clip = _col("L_star_L2_clipped")

    # --- finite masks (raw) ---
    mt_raw = np.isfinite(y_tan_raw)   & np.isfinite(yhat_tan_raw)
    mn_raw = np.isfinite(y_norm_raw)  & np.isfinite(yhat_norm_raw)
    mT_raw = np.isfinite(y_total_raw) & np.isfinite(yhat_total_raw)

    # --- finite masks (clipped) ---
    mt_c = np.isfinite(y_tan_clip)   & np.isfinite(yhat_tan_clip)
    mn_c = np.isfinite(y_norm_clip)  & np.isfinite(yhat_norm_clip)
    mT_c = np.isfinite(y_total_clip) & np.isfinite(yhat_total_clip)

    # --- in/out splits are based on RAW GT (so we can compare fairly) ---
    t_in  = (y_tan_raw  <= TAN_MAX)
    n_in  = (y_norm_raw <= NORM_MAX)
    T_in  = (y_total_raw<= TOTAL_MAX)
    t_out = ~t_in
    n_out = ~n_in
    T_out = ~T_in

    # --- metrics helpers ---
    def _rmse(y, yhat): return float(np.sqrt(np.mean((yhat - y) ** 2))) if y.size else np.nan
    def _mae(y, yhat):  return float(np.mean(np.abs(yhat - y)))         if y.size else np.nan
    def _bias(y, yhat): return float(np.mean(yhat - y))                 if y.size else np.nan
    def _pearson(y, yhat):
        if y.size < 2: return np.nan
        return float(np.corrcoef(y, yhat)[0, 1])
    def _rmspe_pct(y, yhat, eps=1e-6):
        if y.size == 0: return np.nan
        denom = np.maximum(np.abs(y), eps)
        return float(100.0 * np.sqrt(np.mean(((yhat - y) / denom) ** 2)))

    def _rmse_ci_bootstrap(y, yhat, n_boot=2000, seed=seed):
        n = y.shape[0]
        if n == 0:
            return dict(rmse=np.nan, ci_low=np.nan, ci_high=np.nan, ci_halfwidth=np.nan, n=0,
                        mae=np.nan, bias=np.nan, pearson_r=np.nan, y_mean=np.nan, yhat_mean=np.nan,
                        rmspe_pct=np.nan)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(n_boot, n))
        s = np.sqrt(np.mean((yhat[idx] - y[idx]) ** 2, axis=1))
        lo, hi = np.percentile(s, [2.5, 97.5])
        return dict(
            rmse=_rmse(y, yhat),
            ci_low=float(lo), ci_high=float(hi), ci_halfwidth=float((hi - lo) / 2),
            n=int(n), mae=_mae(y, yhat), bias=_bias(y, yhat),
            pearson_r=_pearson(y, yhat),
            y_mean=float(np.mean(y)), yhat_mean=float(np.mean(yhat)),
            rmspe_pct=_rmspe_pct(y, yhat),
        )

    def _block(y, yhat, mask_full, mask_in, mask_out):
        y_, yhat_ = y[mask_full], yhat[mask_full]
        return {
            "overall":  _rmse_ci_bootstrap(y_, yhat_),
            "in_<=max": _rmse_ci_bootstrap(y[mask_full & mask_in],  yhat[mask_full & mask_in]),
            "out_>max": _rmse_ci_bootstrap(y[mask_full & mask_out], yhat[mask_full & mask_out]),
        }

    metrics = {
        # RAW comparison
        "raw": {
            "tangential_L2": _block(y_tan_raw,  yhat_tan_raw,  mt_raw, t_in, t_out),
            "normal_L2":     _block(y_norm_raw, yhat_norm_raw, mn_raw, n_in, n_out),
            "combined_L2":   _block(y_total_raw,yhat_total_raw,mT_raw, T_in, T_out),
        },
        # CLIPPED comparison (both GT & preds clipped to capacity)
        "clipped": {
            "tangential_L2": _block(y_tan_clip,  yhat_tan_clip,  mt_c, t_in, t_out),
            "normal_L2":     _block(y_norm_clip, yhat_norm_clip, mn_c, n_in, n_out),
            "combined_L2":   _block(y_total_clip,yhat_total_clip,mT_c, T_in, T_out),
        },
    }

    counts = {
        "tangential": {"overall_n": int(np.isfinite(y_tan_raw).sum()),
                       f"in_<={TAN_MAX:.2f}N": int((np.isfinite(y_tan_raw) & t_in).sum()),
                       f"out_>{TAN_MAX:.2f}N": int((np.isfinite(y_tan_raw) & t_out).sum())},
        "normal":     {"overall_n": int(np.isfinite(y_norm_raw).sum()),
                       f"in_<={NORM_MAX:.2f}N": int((np.isfinite(y_norm_raw) & n_in).sum()),
                       f"out_>{NORM_MAX:.2f}N": int((np.isfinite(y_norm_raw) & n_out).sum())},
        "combined":   {"overall_n": int(np.isfinite(y_total_raw).sum()),
                       f"in_<={TOTAL_MAX:.2f}N": int((np.isfinite(y_total_raw) & T_in).sum()),
                       f"out_>{TOTAL_MAX:.2f}N": int((np.isfinite(y_total_raw) & T_out).sum())},
    }

    report = {
        "preds_in_newtons": preds_in_newtons,
        "scales": scales,
        "thresholds_N": {"tangential_max": TAN_MAX, "normal_max": NORM_MAX, "combined_max": TOTAL_MAX},
        "counts": counts,
        "metrics": metrics,   # only 'raw' and 'clipped' (no calibration)
    }

    jp = Path(json_path)
    jp.parent.mkdir(parents=True, exist_ok=True)
    with open(jp, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[ok] wrote L2 metrics (raw + clipped) → {jp}")

    return report

def evaluate_sparsh_on_single_recording(
    output_path_gt,
    rec_location,
    interaction_indices,
    encoder_ckpt,
    decoder_ckpt,
    encoder_type
):
    
    predictor_left = SparshForcePredictor.from_hydra(
        cfg_path=f"{output_path_gt}/config.yaml",
        dataset_root=f"{output_path_gt}/{rec_location}",
        dataset_name=f"{rec_location}_{interaction_indices}_sparsh_format/digit_left",
        encoder_ckpt=encoder_ckpt,
        decoder_ckpt=decoder_ckpt,
        encoder_type=encoder_type,
        vit="base",
        device="cuda",
        strict_manual_head_load=True,   # does the force-load+strict check like your script
    )
    preds_left = predictor_left.predict_all(batch_size=256, return_denorm=True, return_gt=False)

    predictor_right = SparshForcePredictor.from_hydra(
        cfg_path=f"{output_path_gt}/config.yaml",
        dataset_root=f"{output_path_gt}/{rec_location}",
        dataset_name=f"{rec_location}_{interaction_indices}_sparsh_format/digit_right",
        encoder_ckpt=encoder_ckpt,
        decoder_ckpt=decoder_ckpt,
        encoder_type=encoder_type,
        vit="base",
        device="cuda",
        strict_manual_head_load=True,   # does the force-load+strict check like your script
    )
    preds_right = predictor_right.predict_all(batch_size=256, return_denorm=True, return_gt=False)

    # --- LEFT ---
    idxs_left = preds_left["idxs"]
    forces_left = preds_left["y_pred_N"]
    names_left = [predictor_left.idx2name.get(int(i), str(i)) for i in idxs_left]

    df_left = pd.DataFrame({
        "timestamp": pd.Series(names_left, dtype=str),
        "index_left": idxs_left.astype(int),
        "Fx_left_pred": forces_left[:, 0],
        "Fy_left_pred": forces_left[:, 1],
        "Fz_left_pred": forces_left[:, 2],
    })
    # keep last occurrence per timestamp
    df_left = df_left.drop_duplicates(subset=["timestamp"], keep="last")

    # --- RIGHT ---
    idxs_right = preds_right["idxs"]
    forces_right = preds_right["y_pred_N"]
    names_right = [predictor_right.idx2name.get(int(i), str(i)) for i in idxs_right]

    df_right = pd.DataFrame({
        "timestamp": pd.Series(names_right, dtype=str),
        "Fx_right_pred": forces_right[:, 0],
        "Fy_right_pred": forces_right[:, 1],
        "Fz_right_pred": forces_right[:, 2],
    })
    df_right = df_right.drop_duplicates(subset=["timestamp"], keep="last")

    # --- COMBINE LEFT+RIGHT ON TIMESTAMP (outer join so we don’t lose either side) ---
    df_preds = pd.merge(df_left, df_right, on="timestamp", how="outer")

    # --- MERGE PREDICTIONS ONTO GT ---
    df_gt = pd.read_csv(f"{output_path_gt}/{rec_location}/{interaction_indices}_gt.csv")
    df_gt["timestamp_img"] = df_gt["timestamp_img"].astype(str)

    df_out = df_gt.merge(df_preds, left_on="timestamp_img", right_on="timestamp", how="left")

    # --- KEEP ONLY ROWS WITH AT LEAST ONE SIDE PREDICTED ---
    keep_mask = df_out[[
        "Fx_left_pred", "Fy_left_pred", "Fz_left_pred",
        "Fx_right_pred", "Fy_right_pred", "Fz_right_pred"
    ]].notna().any(axis=1)
    df_out = df_out[keep_mask].reset_index(drop=True)

    # compute total loads
    L_pred_tangential = np.sqrt(
        (df_out["Fx_left_pred"].fillna(0.0).abs() + df_out["Fx_right_pred"].fillna(0.0).abs())**2 +
        (df_out["Fy_left_pred"].fillna(0.0).abs() + df_out["Fy_right_pred"].fillna(0.0).abs())**2
    )
    L_pred_normal = df_out["Fz_left_pred"].fillna(0.0).abs() + df_out["Fz_right_pred"].fillna(0.0).abs()
    L_pred_total = np.sqrt(L_pred_tangential**2 + L_pred_normal**2)
    df_out["L_pred_tangential"] = L_pred_tangential
    df_out["L_pred_normal"] = L_pred_normal
    df_out["L_pred_total"] = L_pred_total

    # (optional) ensure numeric dtypes for metrics later
    for c in ["Fx_left_pred","Fy_left_pred","Fz_left_pred","Fx_right_pred","Fy_right_pred","Fz_right_pred"]:
        if c in df_out.columns:
            df_out[c] = pd.to_numeric(df_out[c], errors="coerce")

    # --- SAVE ---
    output_path = Path(f"{output_path_gt}/{rec_location}/{interaction_indices}_preds.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"[ok] wrote predictions to {output_path}")

    plot_per_sequence(df_out, output_path)
    metrics_path = output_path.with_name(output_path.stem.replace("_preds", "") + "_metrics.json")
    metrics_report = eval_l2_with_splits_and_calibration(
        df_out,
        json_path=metrics_path,
        preds_in_newtons=True,  
        scales=(4.0,4.0,5.0) ) # set to None if preds already scaled to N)

def _weighted_avg_blocks(blocks):
    """Weighted by 'n', sums n; non-finite/missing values are ignored."""
    total_n = int(sum(b.get("n", 0) or 0 for b in blocks))
    out = {"n": total_n}
    if total_n == 0:
        return {k: None for k in blocks[0].keys()} | {"n": 0}
    keys = set().union(*(b.keys() for b in blocks))
    for k in keys:
        if k == "n": 
            continue
        vals, ws = [], []
        for b in blocks:
            v, n = b.get(k, None), b.get("n", 0) or 0
            if isinstance(v, (int, float)) and np.isfinite(v) and n > 0:
                vals.append(float(v)); ws.append(n)
        out[k] = float(np.average(vals, weights=ws)) if ws else None
    return out

def _merge_metric_struct(metric_files, out_json_path):
    ms = []
    for p in metric_files:
        try:
            with open(p, "r") as f:
                ms.append(json.load(f))
        except Exception as e:
            print(f"[warn] skip {p}: {e}")
    if not ms:
        raise RuntimeError("No readable metrics files.")

    modes  = list(ms[0]["metrics"].keys())                       # ["raw","clipped"]
    heads  = list(ms[0]["metrics"][modes[0]].keys())             # ["tangential_L2","normal_L2","combined_L2"]
    splits = list(ms[0]["metrics"][modes[0]][heads[0]].keys())   # ["overall","in_<=max","out_>max"]

    agg = {
        "preds_in_newtons": all(m.get("preds_in_newtons", True) for m in ms),
        "scales": ms[0].get("scales", [4.0,4.0,5.0]),
        "thresholds_N": ms[0].get("thresholds_N", {}),
        "counts": {},
        "metrics": {},
        "sources": [str(p) for p in metric_files],
    }

    # sum counts
    for cat, d in ms[0]["counts"].items():
        agg["counts"][cat] = {}
        for k in d.keys():
            s = 0
            for m in ms:
                v = m["counts"][cat].get(k, 0)
                if isinstance(v, (int, float)): s += int(v)
            agg["counts"][cat][k] = int(s)

    # weighted metrics
    for mode in modes:
        agg["metrics"][mode] = {}
        for head in heads:
            agg["metrics"][mode][head] = {}
            for split in splits:
                blocks = [m["metrics"][mode][head][split] for m in ms]
                agg["metrics"][mode][head][split] = _weighted_avg_blocks(blocks)

    out_json_path = Path(out_json_path)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(agg, f, indent=2)

    # quick CSV of overall RMSE/MAE
    rows = []
    for mode in modes:
        for head in heads:
            b = agg["metrics"][mode][head]["overall"]
            rows.append({"mode": mode, "metric": head, "rmse": b.get("rmse"), "mae": b.get("mae"), "n": b.get("n")})
    csv_path = out_json_path.with_suffix(".csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[ok] wrote aggregate → {out_json_path}")
    print(f"[ok] wrote summary CSV → {csv_path}")
    return str(out_json_path), str(csv_path)

def aggregate_from_layout(base_dir: str, encoder: str, pairs: list[tuple[str, str]],
                          out_name: str = "aggregate_metrics.json"):
    """
    base/
      rec_loc/
        encoder/   <-- e.g. 'dinov2'
          interaction_metrics.json  <-- e.g. '1-5_metrics.json'
    pairs: [(rec_loc, interaction_indices), ...]
    """
    base = Path(base_dir)
    files = []
    for rec_loc, inter in pairs:
        p = base / rec_loc / encoder / f"{inter}_metrics.json"
        if p.is_file():
            files.append(p)
        else:
            print(f"[warn] missing: {p}")
    if not files:
        raise RuntimeError("No metrics found for the given pairs.")
    out_json = base / f"{encoder}_{out_name}"
    return _merge_metric_struct(files, out_json)

if __name__ == "__main__":
    # rec_location = "office_1"

    # color = "blue"
    # visualize = False

    # rec_type = "gripper"
    # rec_module = "gripper"
    # interaction_indices = "1-7"

    # output_path_gt=f"/data/evaluations/contact_force_estimation/outputs_sparsh/groundtruth"
    # encoder_ckpt="/data/evaluations/contact_force_estimation/outputs_sparsh/checkpoints/sparsh_dino_base/dino_vitbase.ckpt"
    # decoder_ckpt="/data/evaluations/contact_force_estimation/outputs_sparsh/checkpoints/digit_t1_force_eval_dino/last.ckpt"
    # encoder_type = "dino" # or "dinov2"

    # evaluate_sparsh_on_single_recording(
    #     output_path_gt=output_path_gt,
    #     rec_location=rec_location,
    #     interaction_indices=interaction_indices,
    #     encoder_ckpt=encoder_ckpt,
    #     decoder_ckpt=decoder_ckpt,
    #     encoder_type=encoder_type
    # )

    pairs = [
    # ("bathroom_2", "1-5"),
    ("bedroom_6", "1-5"),
    # ("livingroom_1", "1-7"),
    # ("livingroom_1", "8-14"),
    # ("office_1", "1-7"),
    # ("office_1", "8-14"),
    # ("kitchen_7", "1-3-5-7-9"),
    # ("kitchen_7", "2-4-6-8"),
    # ("bedroom_4", "1-11"),
    # ("bedroom_4", "12-14"),
    ]

    aggregate_from_layout(
        base_dir="/data/evaluations/contact_force_estimation/outputs_sparsh/groundtruth",
        encoder="dinov2",            # or "dino"
        pairs=pairs,
        out_name="bedroom_6_runs_weighted.json",
    )


