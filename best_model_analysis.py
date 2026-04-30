"""
Analyze and visualize the best model from model_comparison_batch_results.xlsx.

Outputs at least:
- Feature importance
- SHAP summary (dot + bar, if shap is installed)
- ROC curve
- PR curve
- Gain chart
- Lift chart
"""

import argparse
import ast
import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import lightgbm as lgb

MODEL_LABEL_TO_CODE = {
    "L1 Logistic": "LR",
    "Random Forest": "RF",
    "LightGBM": "LGB",
}
MODEL_CODE_TO_LABEL = {v: k for k, v in MODEL_LABEL_TO_CODE.items()}

LR_PARAM_GRID = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "class_weight": [None, "balanced"],
}
RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 7, 10],
    "min_samples_split": [5, 10],
    "class_weight": [None, "balanced"],
    "min_samples_leaf": [2],
}
LGB_PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05],
    "num_leaves": [8, 15, 31],
    "min_child_samples": [30, 40, 50],
    "scale_pos_weight": [15, 20, 25, 30],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}


def calc_zscore_252(series, min_periods=200):
    rolling_mean = series.rolling(window=252, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=252, min_periods=min_periods).std()
    return (series - rolling_mean) / rolling_std.clip(lower=1e-8)


def needs_zscore(feature_name):
    no_zscore_patterns = [
        "logret",
        "diff",
        "P_Calm",
        "P_Transitional",
        "P_Stress",
        "Delta_P_Stress",
        "INTERACT",
        "STRESS",
        "SPIKE",
        "RR_60d",
        "DET_60d",
        "CREDIT_FACTOR",
        "RATIO",
        "SPREAD_VIX_HIGH",
        "SPX_MAXDD_20D",
    ]
    for pattern in no_zscore_patterns:
        if pattern in feature_name:
            if feature_name == "Delta_P_Stress_5d_z252":
                return False
            return False

    zscore_patterns = ["_range", "SPX_VOL_20D", "RATE_SLOPE_10Y_3M", "VIX_price"]
    for pattern in zscore_patterns:
        if pattern in feature_name:
            if "_z252" in feature_name:
                return False
            return True
    return False


def get_zscore_name(feature_name):
    if "_z252" in feature_name:
        return feature_name
    return f"{feature_name}_z252"


def parse_params(param_str):
    if isinstance(param_str, dict):
        return param_str
    if pd.isna(param_str):
        return {}
    try:
        return ast.literal_eval(str(param_str))
    except Exception:
        return {}


def build_run_id(run_tag=None):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_tag:
        safe_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", str(run_tag)).strip("_")
        if safe_tag:
            return f"{ts}_{safe_tag}"
    return ts


def script_sha256():
    try:
        return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
    except Exception:
        return None


def archive_existing_outputs(outdir, run_id):
    existing = [p for p in outdir.iterdir() if p.name != "_history"]
    if not existing:
        return None

    archive_dir = outdir / "_history" / run_id
    archive_dir.mkdir(parents=True, exist_ok=True)
    for p in existing:
        shutil.move(str(p), str(archive_dir / p.name))
    return archive_dir


def write_run_metadata(outdir, run_meta):
    history_dir = outdir / "_history"
    history_dir.mkdir(parents=True, exist_ok=True)

    (outdir / "run_meta.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (history_dir / "run_index.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(run_meta, ensure_ascii=False) + "\n")


def _safe_date_str(ts):
    if ts is None or pd.isna(ts):
        return None
    return pd.Timestamp(ts).date().isoformat()


def _series_range(index_like):
    if len(index_like) == 0:
        return None, None
    idx = pd.DatetimeIndex(index_like)
    return _safe_date_str(idx.min()), _safe_date_str(idx.max())


def make_date_split_config(train_start, train_end, name=None, test_end=None):
    ts = pd.Timestamp(train_start)
    te = pd.Timestamp(train_end)
    if ts >= te:
        raise ValueError(f"Invalid date split: train_start >= train_end ({ts} >= {te})")
    if name is None:
        name = f"train_{ts.strftime('%Y%m%d')}_{te.strftime('%Y%m%d')}_test_after"
    cfg = {
        "mode": "date",
        "name": name,
        "train_start": ts,
        "train_end": te,
    }
    if test_end is not None:
        cfg["test_end"] = pd.Timestamp(test_end)
    return cfg


def parse_extra_date_windows(spec_text):
    """
    Parse extra windows from:
    'YYYY-MM-DD:YYYY-MM-DD;YYYY-MM-DD:YYYY-MM-DD'
    """
    windows = []
    if spec_text is None:
        return windows
    for raw in str(spec_text).split(";"):
        s = raw.strip()
        if not s:
            continue
        parts = [p.strip() for p in s.split(":")]
        if len(parts) != 2:
            raise ValueError(f"Invalid window '{s}'. Expected 'YYYY-MM-DD:YYYY-MM-DD'.")
        windows.append(make_date_split_config(parts[0], parts[1]))
    return windows


def safe_roc_auc(y_true, y_prob):
    if pd.Series(y_true).nunique() < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def topk_metrics(y_true, y_prob, top_ratio=0.10):
    y_true = np.asarray(y_true).astype(int)
    n = len(y_true)
    k = int(n * top_ratio)
    total_pos = int(y_true.sum())

    if k > 0:
        idx = np.argsort(y_prob)[-k:]
        tp = int(y_true[idx].sum())
        precision = tp / k
    else:
        tp = 0
        precision = 0.0

    recall = tp / total_pos if total_pos > 0 else 0.0
    return k, tp, recall, precision


def baseline_metrics(y_true, top_ratio=0.10):
    """Random-rank/class-prior baseline metrics on test set."""
    y_true = np.asarray(y_true).astype(int)
    n = len(y_true)
    pos_rate = float(np.mean(y_true)) if n > 0 else np.nan
    total_pos = int(y_true.sum())
    k = int(n * top_ratio)

    if k > 0:
        expected_tp = total_pos * top_ratio
        recall_at_k = top_ratio if total_pos > 0 else 0.0
        precision_at_k = pos_rate
    else:
        expected_tp = 0.0
        recall_at_k = 0.0
        precision_at_k = 0.0

    roc = 0.5 if pd.Series(y_true).nunique() >= 2 else np.nan
    pr = pos_rate
    return {
        "Test_Samples": n,
        "Test_Positive_Rate": pos_rate,
        "Test_ROC_AUC": roc,
        "Test_PR_AUC": pr,
        "Top10%_Count": k,
        "TP@10%": expected_tp,
        "Recall@10%": recall_at_k,
        "Precision@10%": precision_at_k,
    }


def gain_lift_series(y_true, y_prob):
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob)
    order = np.argsort(-p)
    y_sorted = y[order]

    cum_pos = np.cumsum(y_sorted)
    total_pos = max(int(y_sorted.sum()), 1)
    n = len(y_sorted)
    pop = np.arange(1, n + 1)

    pop_pct = pop / n
    gain = cum_pos / total_pos
    lift = gain / pop_pct
    return pop_pct, gain, lift


def build_model(model_name, params):
    if model_name == "L1 Logistic":
        return LogisticRegression(
            penalty="l1",
            solver="saga",
            max_iter=2000,
            random_state=42,
            **params,
        )
    if model_name == "Random Forest":
        return RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    if model_name == "LightGBM":
        return lgb.LGBMClassifier(random_state=42, verbose=-1, **params)
    raise ValueError(f"Unsupported model name: {model_name}")


def choose_best_candidate_by_test(summary_df, group_name=None, metric="PR"):
    metric = metric.upper()
    if metric not in {"PR", "ROC"}:
        raise ValueError("metric must be PR or ROC")

    metric_suffix = "PR_AUC" if metric == "PR" else "ROC_AUC"
    model_codes = ["LR", "RF", "LGB"]

    base_df = summary_df.copy()
    if group_name:
        base_df = base_df[base_df["Group"] == group_name]
        if base_df.empty:
            raise ValueError(f"Group not found: {group_name}")

    rows = []
    for _, r in base_df.iterrows():
        for code in model_codes:
            rows.append(
                {
                    "Group": r["Group"],
                    "ModelCode": code,
                    "ModelName": MODEL_CODE_TO_LABEL[code],
                    "Score": r[f"{code}_Test_{metric_suffix}"],
                    "Test_PR_AUC": r[f"{code}_Test_PR_AUC"],
                    "Test_ROC_AUC": r[f"{code}_Test_ROC_AUC"],
                }
            )
    cand_df = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
    return cand_df.iloc[0]


def tune_params_on_val(model_name, X_train, y_train, X_val, y_val):
    """Tune one model family on Train and select params by Validation PR-AUC."""
    best_score = -1
    best_params = None

    if model_name == "L1 Logistic":
        from sklearn.model_selection import ParameterGrid

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.values)
        X_val_scaled = scaler.transform(X_val.values)

        for params in ParameterGrid(LR_PARAM_GRID):
            model = LogisticRegression(
                penalty="l1", solver="saga", max_iter=2000, random_state=42, **params
            )
            model.fit(X_train_scaled, y_train)
            y_val_prob = model.predict_proba(X_val_scaled)[:, 1]
            score = average_precision_score(y_val, y_val_prob)
            if score > best_score:
                best_score = score
                best_params = params
        return best_params, best_score

    if model_name == "Random Forest":
        from sklearn.model_selection import ParameterGrid

        for params in ParameterGrid(RF_PARAM_GRID):
            model = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
            model.fit(X_train.values, y_train)
            y_val_prob = model.predict_proba(X_val.values)[:, 1]
            score = average_precision_score(y_val, y_val_prob)
            if score > best_score:
                best_score = score
                best_params = params
        return best_params, best_score

    if model_name == "LightGBM":
        from sklearn.model_selection import ParameterGrid

        for params in ParameterGrid(LGB_PARAM_GRID):
            model = lgb.LGBMClassifier(random_state=42, verbose=-1, **params)
            model.fit(X_train.values, y_train)
            y_val_prob = model.predict_proba(X_val.values)[:, 1]
            score = average_precision_score(y_val, y_val_prob)
            if score > best_score:
                best_score = score
                best_params = params
        return best_params, best_score

    raise ValueError(f"Unsupported model name: {model_name}")


def prepare_data(project_dir, group_name, feature_details_df, split_cfg=None):
    features = pd.read_csv(project_dir / "features.csv", index_col=0, parse_dates=True)
    target_labels = pd.read_csv(project_dir / "target_label.csv", index_col=0, parse_dates=True)
    regime_labels = pd.read_csv(project_dir / "regime_label.csv", index_col=0, parse_dates=True)

    for feat in ["P_Calm", "P_Transitional", "P_Stress"]:
        if feat in regime_labels.columns:
            features[feat] = regime_labels[feat]

    if "Delta_P_Stress" in regime_labels.columns:
        delta = regime_labels["Delta_P_Stress"].reindex(features.index)
        delta_5d = delta.rolling(window=5, min_periods=4).mean()
        m = delta_5d.rolling(window=252, min_periods=202).mean()
        s = delta_5d.rolling(window=252, min_periods=202).std()
        features["Delta_P_Stress_5d_z252"] = (delta_5d - m) / s.clip(lower=1e-8)

    # Dynamically create required z-score features
    for col in list(features.columns):
        if needs_zscore(col):
            z_col = get_zscore_name(col)
            if z_col not in features.columns:
                features[z_col] = calc_zscore_252(features[col])

    group_features = (
        feature_details_df[feature_details_df["Group"] == group_name]
        .sort_values("Feature_Index")["Feature"]
        .tolist()
    )
    if not group_features:
        raise ValueError(f"No features found in Feature_Details for group: {group_name}")

    available_features = [f for f in group_features if f in features.columns]
    missing_features = [f for f in group_features if f not in features.columns]
    if missing_features:
        print(f"[WARN] Missing features for {group_name}: {missing_features}")
    if not available_features:
        raise ValueError(f"No available features for group: {group_name}")

    X_all = features[available_features].copy()
    y_all = target_labels["SP500_Label"].copy()

    common_idx = X_all.index.intersection(y_all.dropna().index)
    X_all = X_all.loc[common_idx]
    y_all = y_all.loc[common_idx]

    valid_mask = X_all.notna().all(axis=1) & y_all.notna()
    X_all = X_all[valid_mask].sort_index()
    y_all = y_all[valid_mask].astype(int).loc[X_all.index]

    cfg = split_cfg or {"mode": "regime", "name": "regime_split"}
    split_mode = str(cfg.get("mode", "regime")).lower()
    split_name = str(cfg.get("name", split_mode))

    if split_mode == "regime":
        test_index = regime_labels[regime_labels["Data_Split"] == "Test"].index
        test_mask = X_all.index.isin(test_index)
        X_test = X_all[test_mask]
        y_test = y_all[test_mask]
        X_non_test = X_all[~test_mask]
        y_non_test = y_all[~test_mask]
        split_spec = {
            "mode": split_mode,
            "name": split_name,
            "train_start": None,
            "train_end": None,
            "test_end": None,
        }
    elif split_mode == "date":
        if "train_start" not in cfg or "train_end" not in cfg:
            raise ValueError("Date split requires train_start and train_end.")
        train_start = pd.Timestamp(cfg["train_start"])
        train_end = pd.Timestamp(cfg["train_end"])
        test_end = pd.Timestamp(cfg["test_end"]) if cfg.get("test_end") is not None else None
        if train_start >= train_end:
            raise ValueError(f"Invalid date split: train_start >= train_end ({train_start} >= {train_end})")

        train_val_mask = (X_all.index >= train_start) & (X_all.index <= train_end)
        if test_end is None:
            test_mask = X_all.index > train_end
        else:
            test_mask = (X_all.index > train_end) & (X_all.index <= test_end)

        X_non_test = X_all[train_val_mask]
        y_non_test = y_all[train_val_mask]
        X_test = X_all[test_mask]
        y_test = y_all[test_mask]

        split_spec = {
            "mode": split_mode,
            "name": split_name,
            "train_start": _safe_date_str(train_start),
            "train_end": _safe_date_str(train_end),
            "test_end": _safe_date_str(test_end),
        }
    else:
        raise ValueError(f"Unsupported split mode: {split_mode}")

    n_non_test = len(X_non_test)
    n_train = int(n_non_test * (2.0 / 3.0))
    n_train = max(1, min(n_train, n_non_test - 1)) if n_non_test >= 2 else 0
    if n_train <= 0 or len(X_test) == 0:
        raise ValueError("Insufficient train/val/test samples after alignment.")

    X_train = X_non_test.iloc[:n_train]
    y_train = y_non_test.iloc[:n_train]
    X_val = X_non_test.iloc[n_train:]
    y_val = y_non_test.iloc[n_train:]
    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)
    train_start, train_end = _series_range(X_train.index)
    val_start, val_end = _series_range(X_val.index)
    trainval_start, trainval_end = _series_range(X_train_val.index)
    test_start, test_end = _series_range(X_test.index)

    return {
        "features": available_features,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_train_val": X_train_val,
        "y_train_val": y_train_val,
        "X_test": X_test,
        "y_test": y_test,
        "split_meta": {
            **split_spec,
            "train_range": [train_start, train_end],
            "val_range": [val_start, val_end],
            "trainval_range": [trainval_start, trainval_end],
            "test_range": [test_start, test_end],
            "train_samples": int(len(X_train)),
            "val_samples": int(len(X_val)),
            "trainval_samples": int(len(X_train_val)),
            "test_samples": int(len(X_test)),
        },
    }


def plot_curves(y_test, y_prob, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    pos_rate = float(np.mean(y_test))

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = safe_roc_auc(y_test, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"Model ROC-AUC={roc_auc:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Baseline ROC-AUC=0.5", linewidth=1.2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "roc_curve.png", dpi=160)
    plt.close()

    # PR
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"Model PR-AUC={pr_auc:.4f}", linewidth=2)
    plt.axhline(y=pos_rate, color="k", linestyle="--", linewidth=1.2, label=f"Baseline={pos_rate:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "pr_curve.png", dpi=160)
    plt.close()

    # Gain / Lift
    pop_pct, gain, lift = gain_lift_series(y_test, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(pop_pct, gain, label="Model Gain", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Baseline", linewidth=1.2)
    plt.xlabel("Population Percentage")
    plt.ylabel("Cumulative Gain")
    plt.title("Gain Chart")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "gain_chart.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(pop_pct, lift, label="Model Lift", linewidth=2)
    plt.axhline(y=1.0, color="k", linestyle="--", linewidth=1.2, label="Baseline")
    plt.xlabel("Population Percentage")
    plt.ylabel("Lift")
    plt.title("Lift Chart")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "lift_chart.png", dpi=160)
    plt.close()


def plot_test_ranking_score_timeline(y_test, y_prob, outdir, top_ratio=0.10):
    """Plot model ranking scores on test set in time order."""
    score_s = pd.Series(np.asarray(y_prob), index=y_test.index, name="rank_score").sort_index()
    n = len(score_s)
    if n == 0:
        return

    k = int(n * top_ratio)
    top_idx = score_s.nlargest(k).index if k > 0 else pd.Index([])
    threshold = float(score_s.nlargest(k).min()) if k > 0 else np.nan

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(score_s.index, score_s.values, color="#1f77b4", linewidth=1.3, alpha=0.9, label="Model ranking score")

    if k > 0:
        ax.scatter(
            top_idx,
            score_s.loc[top_idx].values,
            color="#d62728",
            s=24,
            marker="o",
            facecolors="none",
            linewidths=1.0,
            label=f"Top10% days (n={k})",
            zorder=3,
        )
        ax.axhline(
            threshold,
            color="#d62728",
            linestyle="--",
            linewidth=1.1,
            alpha=0.85,
            label=f"Top10% threshold={threshold:.4f}",
        )

    ax.set_title("Test Set Ranking Score Timeline")
    ax.set_xlabel("Date")
    ax.set_ylabel("Ranking Score (Predicted Crash Probability)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(outdir / "test_ranking_score_timeline.png", dpi=170, bbox_inches="tight")
    plt.close()


def plot_crash_maxdd_comparison(project_dir, y_test, y_prob, outdir, horizon=20, top_ratio=0.10):
    """
    Compare crash days found by label vs model on TEST set, using forward 20d max drawdown.

    Model-found crash days are defined as top-10% predicted risk days in test,
    consistent with metrics_comparison.csv (Recall@10%, Precision@10%).
    """
    target_path = project_dir / "target_label.csv"
    if not target_path.exists():
        print("[WARN] target_label.csv not found. Skip crash/maxdd comparison plot.")
        return

    target_df = pd.read_csv(target_path, index_col=0, parse_dates=True)
    # Prefer max-drawdown-like columns if present; otherwise fallback to forward return column.
    metric_candidates = [
        f"SP500_forward_maxdd_{horizon}d",
        f"SP500_{horizon}d_forward_maxdd",
        "SP500_20d_forward_maxdd",
        "SP500_20d_maxdd",
        "SP500_20d_forward_return",
    ]
    metric_col = next((c for c in metric_candidates if c in target_df.columns), None)
    if metric_col is None:
        print("[WARN] No 20d crash metric column found in target_label.csv. Skip comparison plot.")
        return
    metric_s = target_df[metric_col]

    score_s = pd.Series(np.asarray(y_prob), index=y_test.index, name="y_prob")
    y_true_s = pd.Series(np.asarray(y_test).astype(int), index=y_test.index, name="y_true")

    test_df = pd.concat([y_true_s, score_s], axis=1)
    test_df["maxdd_20d"] = metric_s.reindex(test_df.index)

    label_idx = test_df.index[test_df["y_true"] == 1].sort_values()
    k_model = int(len(test_df) * top_ratio)
    if k_model > 0:
        model_idx = test_df["y_prob"].nlargest(k_model).index.sort_values()
    else:
        model_idx = pd.Index([])

    test_df["label_crash"] = test_df.index.isin(label_idx).astype(int)
    test_df["model_found_crash"] = test_df.index.isin(model_idx).astype(int)
    test_df["both"] = ((test_df["label_crash"] == 1) & (test_df["model_found_crash"] == 1)).astype(int)
    test_df.to_csv(outdir / "crash_maxdd20_comparison_data.csv", index_label="Date")

    label_dd = test_df.loc[label_idx, "maxdd_20d"].dropna()
    model_dd = test_df.loc[model_idx, "maxdd_20d"].dropna()
    both_idx = label_idx.intersection(model_idx).sort_values()
    both_dd = test_df.loc[both_idx, "maxdd_20d"].dropna()

    # Export and print debug day lists
    true_days = pd.DataFrame({"Date": label_idx, "y_prob": test_df.loc[label_idx, "y_prob"], "maxdd_20d": test_df.loc[label_idx, "maxdd_20d"]})
    model_days = pd.DataFrame({"Date": model_idx, "y_prob": test_df.loc[model_idx, "y_prob"], "maxdd_20d": test_df.loc[model_idx, "maxdd_20d"]})
    overlap_days = pd.DataFrame({"Date": both_idx, "y_prob": test_df.loc[both_idx, "y_prob"], "maxdd_20d": test_df.loc[both_idx, "maxdd_20d"]})

    true_days.to_csv(outdir / "test_true_crash_days.csv", index=False)
    model_days.to_csv(outdir / "test_model_found_crash_days_top10pct.csv", index=False)
    overlap_days.to_csv(outdir / "test_overlap_true_and_model_crash_days.csv", index=False)

    fmt = lambda idx: [d.strftime("%Y-%m-%d") for d in idx]
    print("\n[Crash Day Debug - Test Set]")
    print(f"True crash days (Label=1): {len(label_idx)}")
    print(", ".join(fmt(label_idx)))
    print(f"\nModel-found crash days (Top10% by score, n={len(model_idx)}):")
    print(", ".join(fmt(model_idx)))
    print(f"\nOverlap days (True & Model-found, n={len(both_idx)}):")
    print(", ".join(fmt(both_idx)))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: timeline scatter on max drawdown series
    ax0 = axes[0]
    ax0.plot(test_df.index, test_df["maxdd_20d"], color="lightgray", linewidth=1, alpha=0.9, label=f"Test metric ({metric_col})")
    if len(model_idx) > 0:
        ax0.scatter(
            model_idx,
            test_df.loc[model_idx, "maxdd_20d"],
            color="#1f77b4",
            s=26,
            marker="x",
            label=f"Model found crash (top10%, n={len(model_idx)})",
            zorder=3,
        )
    if len(label_idx) > 0:
        ax0.scatter(
            label_idx,
            test_df.loc[label_idx, "maxdd_20d"],
            color="#d62728",
            s=26,
            marker="o",
            facecolors="none",
            label=f"Label crash (n={len(label_idx)})",
            zorder=4,
        )
    ax0.set_title(f"Test Crash Day Timeline ({metric_col})")
    ax0.set_xlabel("Date")
    ax0.set_ylabel("20d Crash Metric Value")
    ax0.grid(alpha=0.3)
    ax0.legend(loc="best", fontsize=8)

    # Right: distribution comparison
    ax1 = axes[1]
    data = [label_dd.values, model_dd.values, both_dd.values]
    labels = [
        f"Label Crash\n(n={len(label_dd)})",
        f"Model Top10%\n(n={len(model_dd)})",
        f"Overlap\n(n={len(both_dd)})",
    ]
    bp = ax1.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)
    for patch, color in zip(bp["boxes"], ["#ff9896", "#9ecae1", "#98df8a"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax1.set_title(f"Distribution Comparison ({metric_col})")
    ax1.set_ylabel("20d Crash Metric Value")
    ax1.grid(alpha=0.3, axis="y")

    # Annotate means for quick comparison
    label_mean = float(label_dd.mean()) if len(label_dd) else np.nan
    model_mean = float(model_dd.mean()) if len(model_dd) else np.nan
    both_mean = float(both_dd.mean()) if len(both_dd) else np.nan
    text = (
        f"Label mean: {label_mean:.4f}\n"
        f"Model mean: {model_mean:.4f}\n"
        f"Overlap mean: {both_mean:.4f}\n"
        f"Overlap: {len(both_idx)} / {len(model_idx)} (model top10%)"
    )
    ax1.text(
        0.98,
        0.98,
        text,
        transform=ax1.transAxes,
        va="top",
        ha="right",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    plt.suptitle(
        "Crash Comparison on Test Set (Label vs Model Top10%, metric from target_label.csv)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(outdir / "crash_maxdd20_comparison.png", dpi=170, bbox_inches="tight")
    plt.close()

    # Combined panels: metric timeline + ranking timeline + boxplot
    fig2, (ax_metric, ax_score, ax_box) = plt.subplots(
        3,
        1,
        figsize=(15, 12),
        gridspec_kw={"height_ratios": [2.6, 2.2, 1.6]},
    )
    ax_metric.plot(
        test_df.index,
        test_df["maxdd_20d"],
        color="lightgray",
        linewidth=1.2,
        alpha=0.95,
        label=f"Test metric ({metric_col})",
    )
    if len(label_idx) > 0:
        ax_metric.scatter(
            label_idx,
            test_df.loc[label_idx, "maxdd_20d"],
            color="#d62728",
            s=24,
            marker="o",
            facecolors="none",
            linewidths=1.0,
            label=f"Label crash (n={len(label_idx)})",
            zorder=3,
        )
    if len(model_idx) > 0:
        ax_metric.scatter(
            model_idx,
            test_df.loc[model_idx, "maxdd_20d"],
            color="#1f77b4",
            s=24,
            marker="x",
            label=f"Model Top10% days (n={len(model_idx)})",
            zorder=4,
        )
    ax_metric.set_xlabel("Date")
    ax_metric.set_ylabel("20d Crash Metric Value")
    ax_metric.grid(alpha=0.3)
    ax_metric.legend(loc="best", fontsize=8)

    # Middle: ranking score timeline (not overlaid)
    score_t = test_df["y_prob"].sort_index()
    ax_score.plot(
        score_t.index,
        score_t.values,
        color="#1f77b4",
        linewidth=1.3,
        alpha=0.9,
        label="Ranking score",
    )
    if k_model > 0:
        threshold = float(test_df["y_prob"].nlargest(k_model).min())
        ax_score.axhline(
            threshold,
            color="#d62728",
            linestyle="--",
            linewidth=1.1,
            alpha=0.85,
            label=f"Top10% threshold={threshold:.4f}",
        )
        ax_score.scatter(
            model_idx,
            score_t.loc[model_idx].values,
            color="#d62728",
            s=24,
            marker="o",
            facecolors="none",
            linewidths=1.0,
            label=f"Top10% days (n={len(model_idx)})",
            zorder=3,
        )
    ax_score.set_xlabel("Date")
    ax_score.set_ylabel("Ranking Score (Predicted Crash Probability)")
    ax_score.grid(alpha=0.3)
    ax_score.legend(loc="best", fontsize=8)

    overlap_ratio = (len(both_idx) / len(model_idx)) if len(model_idx) > 0 else np.nan
    if not pd.isna(overlap_ratio):
        overlay_text = f"Overlap days: {len(both_idx)} / {len(model_idx)}\nOverlap ratio: {overlap_ratio:.4f}"
    else:
        overlay_text = f"Overlap days: {len(both_idx)} / {len(model_idx)}\nOverlap ratio: nan"
    ax_metric.text(
        0.01,
        0.98,
        overlay_text,
        transform=ax_metric.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    # Bottom: boxplot summary
    box_data = [label_dd.values, model_dd.values, both_dd.values]
    box_labels = [
        f"Label Crash\n(n={len(label_dd)})",
        f"Model Top10%\n(n={len(model_dd)})",
        f"Overlap\n(n={len(both_dd)})",
    ]
    bp2 = ax_box.boxplot(box_data, tick_labels=box_labels, patch_artist=True, showmeans=True)
    for patch, color in zip(bp2["boxes"], ["#ff9896", "#9ecae1", "#98df8a"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    ax_box.set_ylabel("20d Crash Metric Value")
    ax_box.set_title("Distribution Comparison")
    ax_box.grid(alpha=0.3, axis="y")

    fig2.suptitle("Test Panels: Crash Metric, Ranking Score, and Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(outdir / "crash_maxdd20_ranking_overlay.png", dpi=180, bbox_inches="tight")
    plt.close(fig2)


def feature_importance_df(model_name, model, feature_names):
    if model_name == "L1 Logistic":
        imp = np.abs(model.coef_[0])
    else:
        imp = np.asarray(model.feature_importances_)
    out = pd.DataFrame({"Feature": feature_names, "Importance": imp})
    return out.sort_values("Importance", ascending=False)


def plot_feature_importance(imp_df, outdir, topn=20):
    top_df = imp_df.head(topn).iloc[::-1]
    plt.figure(figsize=(8, max(5, topn * 0.3)))
    plt.barh(top_df["Feature"], top_df["Importance"])
    plt.title(f"Feature Importance (Top {topn})")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(outdir / "feature_importance.png", dpi=160)
    plt.close()


def _resolve_positive_class_index(model, positive_class=1):
    classes = getattr(model, "classes_", None)
    if classes is None:
        return 1, None

    classes = np.asarray(classes)
    n_classes = int(len(classes))
    if n_classes <= 1:
        return 0, n_classes

    matches = np.where(classes == positive_class)[0]
    if len(matches) > 0:
        return int(matches[0]), n_classes

    # Fallback for binary/multiclass when label `positive_class` is not present.
    return min(1, n_classes - 1), n_classes


def _extract_from_ndarray(arr, class_idx, n_classes=None, feature_count=None):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        class_axis = None
        if n_classes is not None:
            class_axes = [ax for ax, size in enumerate(arr.shape) if size == n_classes]
            if len(class_axes) == 1:
                class_axis = class_axes[0]
            elif len(class_axes) > 1:
                class_axis = class_axes[-1]

        if class_axis is None:
            # Common SHAP layout: (n_samples, n_features, n_classes)
            class_axis = 2

        use_idx = min(max(int(class_idx), 0), arr.shape[class_axis] - 1)
        selected = np.take(arr, indices=use_idx, axis=class_axis)

        if selected.ndim == 2:
            # Guard against accidental transpose-like orientation.
            if feature_count is not None and selected.shape[1] != feature_count and selected.shape[0] == feature_count:
                selected = selected.T
            return selected
        out = np.squeeze(selected)
        if out.ndim == 1 and feature_count is not None and out.size == feature_count:
            return out.reshape(1, -1)
        return out

    out = np.squeeze(arr)
    if out.ndim == 1 and feature_count is not None and out.size == feature_count:
        return out.reshape(1, -1)
    return out


def _extract_shap_array(shap_values, model=None, positive_class=1, feature_count=None):
    class_idx, n_classes = _resolve_positive_class_index(model, positive_class=positive_class)

    if hasattr(shap_values, "values"):
        vals = shap_values.values
        return _extract_from_ndarray(vals, class_idx=class_idx, n_classes=n_classes, feature_count=feature_count)

    if isinstance(shap_values, list):
        if len(shap_values) == 0:
            return np.array([])
        use_idx = min(max(int(class_idx), 0), len(shap_values) - 1)
        return np.asarray(shap_values[use_idx])

    arr = np.asarray(shap_values)
    return _extract_from_ndarray(arr, class_idx=class_idx, n_classes=n_classes, feature_count=feature_count)


def _safe_feature_filename(name):
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")
    return safe or "feature"


def run_shap(model_name, model, X_train_val, X_test, feature_names, outdir, max_samples=1000):
    try:
        import shap
    except Exception:
        print("[WARN] shap is not installed. Skip SHAP plots.")
        return

    n = min(len(X_test), max_samples)
    X_sample = X_test.iloc[:n].copy()

    if model_name == "L1 Logistic":
        scaler = StandardScaler()
        X_tv_scaled = scaler.fit_transform(X_train_val.values)
        X_sample_scaled = scaler.transform(X_sample.values)

        bg_n = min(500, len(X_tv_scaled))
        bg = X_tv_scaled[:bg_n]
        explainer = shap.LinearExplainer(model, bg)
        shap_values = explainer.shap_values(X_sample_scaled)
        shap_arr = _extract_shap_array(
            shap_values,
            model=model,
            positive_class=1,
            feature_count=len(feature_names),
        )
        X_for_plot = pd.DataFrame(X_sample_scaled, columns=feature_names)
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        shap_arr = _extract_shap_array(
            shap_values,
            model=model,
            positive_class=1,
            feature_count=len(feature_names),
        )
        X_for_plot = X_sample

    # Dot summary
    plt.figure()
    shap.summary_plot(shap_arr, X_for_plot, show=False, plot_type="dot")
    plt.tight_layout()
    plt.savefig(outdir / "shap_summary_dot.png", dpi=160, bbox_inches="tight")
    plt.close()

    # Bar summary
    plt.figure()
    shap.summary_plot(shap_arr, X_for_plot, show=False, plot_type="bar")
    plt.tight_layout()
    plt.savefig(outdir / "shap_summary_bar.png", dpi=160, bbox_inches="tight")
    plt.close()

    # Dependence plots in 2x2 panels (reduce number of files)
    dep_dir = outdir / "shap_dependence"
    dep_dir.mkdir(parents=True, exist_ok=True)
    dep_rows = []
    feature_to_idx = {f: i for i, f in enumerate(feature_names)}
    valid_features = [f for f in feature_names if f in X_for_plot.columns and f in feature_to_idx]
    if len(valid_features) == 0:
        return

    # Rank features by global SHAP impact so each page groups similar-importance features.
    mean_abs_shap = np.mean(np.abs(shap_arr), axis=0)
    shap_rank_df = pd.DataFrame(
        [
            {"Feature": f, "ColIdx": feature_to_idx[f], "SHAP_MeanAbs": float(mean_abs_shap[feature_to_idx[f]])}
            for f in valid_features
        ]
    ).sort_values("SHAP_MeanAbs", ascending=False, ignore_index=True)
    ordered_features = shap_rank_df["Feature"].tolist()
    feature_to_rank = {row["Feature"]: int(i + 1) for i, (_, row) in enumerate(shap_rank_df.iterrows())}
    feature_to_meanabs = {row["Feature"]: float(row["SHAP_MeanAbs"]) for _, row in shap_rank_df.iterrows()}
    n_per_page = 4

    for page_start in range(0, len(ordered_features), n_per_page):
        page_features = ordered_features[page_start: page_start + n_per_page]
        page_no = page_start // n_per_page + 1
        out_name = f"shap_dependence_page_{page_no:02d}.png"
        out_path = dep_dir / out_name

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = np.array(axes).reshape(-1)

        for ax_i, ax in enumerate(axes):
            if ax_i >= len(page_features):
                ax.axis("off")
                continue

            feature = page_features[ax_i]
            try:
                col_idx = feature_to_idx[feature]
                x = X_for_plot[feature].values
                y = shap_arr[:, col_idx]

                ax.scatter(x, y, s=12, alpha=0.65, color="#1f77b4", edgecolors="none")
                ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.8)
                ax.set_title(
                    f"#{feature_to_rank[feature]} {feature}\nmean|SHAP|={feature_to_meanabs[feature]:.4f}",
                    fontsize=10,
                )
                ax.set_xlabel(feature, fontsize=9)
                ax.set_ylabel("SHAP value", fontsize=9)
                ax.grid(alpha=0.25)

                dep_rows.append({
                    "Feature": feature,
                    "File": str((Path("shap_dependence") / out_name).as_posix()),
                    "Panel": ax_i + 1,
                    "Rank": feature_to_rank[feature],
                    "SHAP_MeanAbs": feature_to_meanabs[feature],
                })
            except Exception as e:
                ax.axis("off")
                print(f"[WARN] Failed SHAP dependence panel for {feature}: {e}")

        start_rank = page_start + 1
        end_rank = page_start + len(page_features)
        fig.suptitle(
            f"SHAP Dependence Panels ({model_name}) - Page {page_no} (Rank {start_rank}-{end_rank})",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    if dep_rows:
        pd.DataFrame(dep_rows).to_csv(outdir / "shap_dependence_index.csv", index=False)
        n_pages = (len(dep_rows) + n_per_page - 1) // n_per_page
        print(f"[Info] Saved {len(dep_rows)} SHAP dependence panels across {n_pages} figure(s): {dep_dir}")


def _plot_image_panel(ax, outdir, fname, title):
    if fname is None or not (outdir / fname).exists():
        ax.axis("off")
        return
    img = plt.imread(outdir / fname)
    ax.imshow(img)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")


def create_curves_dashboard(outdir, group_name, model_name):
    """Create one big figure for ROC/PR/Gain/Lift."""
    panels = [
        ("roc_curve.png", "ROC Curve"),
        ("pr_curve.png", "PR Curve"),
        ("gain_chart.png", "Gain Chart"),
        ("lift_chart.png", "Lift Chart"),
    ]
    if not any((outdir / f).exists() for f, _ in panels):
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = np.array(axes).reshape(2, 2)
    for i, (fname, title) in enumerate(panels):
        r, c = divmod(i, 2)
        _plot_image_panel(axes[r, c], outdir, fname, title)

    fig.suptitle(
        f"Curves Dashboard | {group_name} | {model_name}",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(outdir / "combined_curves_dashboard.png", dpi=180, bbox_inches="tight")
    plt.close()


def create_explainability_dashboard(outdir, group_name, model_name):
    """Create one big figure:
    top row = SHAP bar/dot, bottom row = feature importance (full width).
    """
    shap_bar = ("shap_summary_bar.png", "SHAP Summary (Bar)")
    shap_dot = ("shap_summary_dot.png", "SHAP Summary (Dot)")
    importance = ("feature_importance.png", "Feature Importance")

    if not any((outdir / f).exists() for f, _ in [shap_bar, shap_dot, importance]):
        return

    fig = plt.figure(figsize=(18, 11))
    # Three equal-size panels:
    # top-left SHAP bar, top-right SHAP dot, bottom-center feature importance.
    panel_w, panel_h = 0.34, 0.34
    top_y = 0.55
    bottom_y = 0.11
    left_x = 0.15
    right_x = 0.51
    center_x = 0.33

    ax_top_left = fig.add_axes([left_x, top_y, panel_w, panel_h])
    ax_top_right = fig.add_axes([right_x, top_y, panel_w, panel_h])
    ax_bottom = fig.add_axes([center_x, bottom_y, panel_w, panel_h])

    _plot_image_panel(ax_top_left, outdir, shap_bar[0], shap_bar[1])
    _plot_image_panel(ax_top_right, outdir, shap_dot[0], shap_dot[1])
    _plot_image_panel(ax_bottom, outdir, importance[0], importance[1])

    fig.suptitle(
        f"Explainability Dashboard | {group_name} | {model_name}",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    plt.savefig(outdir / "combined_explainability_dashboard.png", dpi=180, bbox_inches="tight")
    plt.close()


def run_single_period_analysis(
    project_dir,
    out_root,
    group_name,
    model_name,
    group_row,
    feature_details_df,
    split_cfg,
    max_shap_samples,
    run_tag=None,
    disable_output_versioning=False,
):
    run_started_at = datetime.now().isoformat(timespec="seconds")

    data = prepare_data(project_dir, group_name, feature_details_df, split_cfg=split_cfg)
    split_meta = data["split_meta"]
    X_train_val = data["X_train_val"]
    y_train_val = data["y_train_val"]
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    feature_names = data["features"]

    reuse_summary_params = (
        split_meta.get("mode") == "regime" and group_row["Best_Model"] == model_name
    )
    if reuse_summary_params:
        best_params = parse_params(group_row["Best_Params"])
        param_source = "Summary.Best_Params"
        tuned_val_score = float(group_row.get("Best_Val_PR_AUC", np.nan))
    else:
        best_params, tuned_val_score = tune_params_on_val(
            model_name, X_train, y_train, X_val, y_val
        )
        param_source = "RetuneOnCurrentSplit"

    model = build_model(model_name, best_params)
    if model_name == "L1 Logistic":
        scaler = StandardScaler()
        X_tv_fit = scaler.fit_transform(X_train_val.values)
        X_te_fit = scaler.transform(X_test.values)
        model.fit(X_tv_fit, y_train_val)
        y_prob = model.predict_proba(X_te_fit)[:, 1]
    else:
        model.fit(X_train_val.values, y_train_val)
        y_prob = model.predict_proba(X_test.values)[:, 1]

    test_roc = safe_roc_auc(y_test, y_prob)
    test_pr = average_precision_score(y_test, y_prob)
    top_n, tp, rec10, prec10 = topk_metrics(y_test, y_prob, top_ratio=0.10)
    base = baseline_metrics(y_test, top_ratio=0.10)

    split_name = split_meta.get("name", "split")
    if split_meta.get("mode") == "regime" and split_name == "regime_split":
        outdir = out_root / group_name
    else:
        outdir = out_root / group_name / split_name
    outdir.mkdir(parents=True, exist_ok=True)

    run_id_tag = split_name if run_tag is None else f"{split_name}_{run_tag}"
    run_id = build_run_id(run_id_tag)
    archived_dir = None
    if not disable_output_versioning:
        archived_dir = archive_existing_outputs(outdir, run_id)
        if archived_dir is not None:
            print(f"Archived previous outputs to: {archived_dir}")

    print("-" * 80)
    print(f"Split      : {split_name} ({split_meta.get('mode')})")
    print(
        f"Train/Val  : {split_meta['trainval_range'][0]} -> {split_meta['trainval_range'][1]} "
        f"({split_meta['trainval_samples']} rows)"
    )
    print(
        f"Train only : {split_meta['train_range'][0]} -> {split_meta['train_range'][1]} "
        f"({split_meta['train_samples']} rows)"
    )
    print(
        f"Val only   : {split_meta['val_range'][0]} -> {split_meta['val_range'][1]} "
        f"({split_meta['val_samples']} rows)"
    )
    print(
        f"Test       : {split_meta['test_range'][0]} -> {split_meta['test_range'][1]} "
        f"({split_meta['test_samples']} rows)"
    )
    print(f"Param source: {param_source}")
    print(f"Best params : {best_params}")
    print(f"Val PR-AUC of selected params: {tuned_val_score:.6f}")

    comparison = pd.DataFrame(
        [
            {
                "Group": group_name,
                "Model": model_name,
                "Test_Samples": len(y_test),
                "Test_Positive_Rate": float(np.mean(y_test)),
                "Test_ROC_AUC": test_roc,
                "Test_PR_AUC": test_pr,
                "Top10%_Count": top_n,
                "TP@10%": tp,
                "Recall@10%": rec10,
                "Precision@10%": prec10,
            },
            {
                "Group": group_name,
                "Model": "Baseline_RandomRank",
                **base,
            },
        ]
    )

    def _safe_ratio(num, den):
        if pd.isna(num) or pd.isna(den) or den == 0:
            return np.nan
        return num / den

    ratio = {
        "Group": group_name,
        "Model": "Ratio(Model/Baseline)",
        "Test_Samples": _safe_ratio(comparison.loc[0, "Test_Samples"], comparison.loc[1, "Test_Samples"]),
        "Test_Positive_Rate": _safe_ratio(comparison.loc[0, "Test_Positive_Rate"], comparison.loc[1, "Test_Positive_Rate"]),
        "Test_ROC_AUC": _safe_ratio(comparison.loc[0, "Test_ROC_AUC"], comparison.loc[1, "Test_ROC_AUC"]),
        "Test_PR_AUC": _safe_ratio(comparison.loc[0, "Test_PR_AUC"], comparison.loc[1, "Test_PR_AUC"]),
        "Top10%_Count": _safe_ratio(comparison.loc[0, "Top10%_Count"], comparison.loc[1, "Top10%_Count"]),
        "TP@10%": _safe_ratio(comparison.loc[0, "TP@10%"], comparison.loc[1, "TP@10%"]),
        "Recall@10%": _safe_ratio(comparison.loc[0, "Recall@10%"], comparison.loc[1, "Recall@10%"]),
        "Precision@10%": _safe_ratio(comparison.loc[0, "Precision@10%"], comparison.loc[1, "Precision@10%"]),
    }
    comparison = pd.concat([comparison, pd.DataFrame([ratio])], ignore_index=True)
    comparison.to_csv(outdir / "metrics_comparison.csv", index=False)
    comparison[comparison["Model"] == model_name].to_csv(outdir / "metrics.csv", index=False)

    plot_curves(y_test, y_prob, outdir)
    plot_test_ranking_score_timeline(y_test, y_prob, outdir, top_ratio=0.10)
    plot_crash_maxdd_comparison(project_dir, y_test, y_prob, outdir, horizon=20)

    imp_df = feature_importance_df(model_name, model, feature_names)
    imp_df.to_csv(outdir / "feature_importance.csv", index=False)
    plot_feature_importance(imp_df, outdir, topn=20)
    run_shap(
        model_name=model_name,
        model=model,
        X_train_val=X_train_val,
        X_test=X_test,
        feature_names=feature_names,
        outdir=outdir,
        max_samples=max_shap_samples,
    )
    create_curves_dashboard(outdir, group_name, model_name)
    create_explainability_dashboard(outdir, group_name, model_name)

    generated_files = sorted(
        [
            p.relative_to(outdir).as_posix()
            for p in outdir.rglob("*")
            if p.is_file() and "_history" not in p.parts
        ]
    )
    run_finished_at = datetime.now().isoformat(timespec="seconds")
    run_meta = {
        "RunID": run_id,
        "RunStartedAt": run_started_at,
        "RunFinishedAt": run_finished_at,
        "ProjectDir": str(project_dir),
        "OutDir": str(outdir),
        "Group": group_name,
        "Model": model_name,
        "SplitMeta": split_meta,
        "MaxSHAPSamples": int(max_shap_samples),
        "VersioningEnabled": not disable_output_versioning,
        "ArchivedPreviousTo": str(archived_dir) if archived_dir is not None else None,
        "ParamsSource": param_source,
        "BestParams": best_params,
        "ValPR_AUC_SelectedParams": float(tuned_val_score) if not pd.isna(tuned_val_score) else None,
        "TestMetrics": {
            "Test_ROC_AUC": float(test_roc) if not pd.isna(test_roc) else None,
            "Test_PR_AUC": float(test_pr) if not pd.isna(test_pr) else None,
            "Top10%_Count": int(top_n),
            "TP@10%": float(tp),
            "Recall@10%": float(rec10),
            "Precision@10%": float(prec10),
        },
        "ScriptSHA256": script_sha256(),
        "GeneratedFiles": generated_files,
    }
    write_run_metadata(outdir, run_meta)

    print(f"Split done : {split_name}")
    print(f"Outputs    : {outdir}")

    return {
        "Split_Name": split_name,
        "Split_Mode": split_meta.get("mode"),
        "Train_Start": split_meta["trainval_range"][0],
        "Train_End": split_meta["trainval_range"][1],
        "Test_Start": split_meta["test_range"][0],
        "Test_End": split_meta["test_range"][1],
        "TrainVal_Samples": split_meta["trainval_samples"],
        "Test_Samples": split_meta["test_samples"],
        "Model": model_name,
        "Param_Source": param_source,
        "Test_ROC_AUC": test_roc,
        "Test_PR_AUC": test_pr,
        "TP@10%": tp,
        "Recall@10%": rec10,
        "Precision@10%": prec10,
        "Output_Dir": str(outdir),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", type=str, default=".")
    parser.add_argument("--results_file", type=str, default="model_comparison_batch_results.xlsx")
    parser.add_argument("--group", type=str, default=None, help="Optional group name (e.g., Group2_RegimeProb)")
    parser.add_argument(
        "--selection_metric",
        type=str,
        default="PR",
        choices=["PR", "ROC", "pr", "roc"],
        help="Metric for selecting best candidate across models on test set.",
    )
    parser.add_argument("--outdir", type=str, default="best_model_analysis_output")
    parser.add_argument("--max_shap_samples", type=int, default=1000)
    parser.add_argument("--run_tag", type=str, default=None, help="Optional tag added to run id.")
    parser.add_argument(
        "--disable_default_period_comparison",
        action="store_true",
        help="Disable built-in extra period analysis (train 2008-01-29 to 2020-07-14, test to 2023-01-01).",
    )
    parser.add_argument(
        "--extra_date_windows",
        type=str,
        default=None,
        help="Extra windows, e.g. '2012-01-01:2022-01-01;2014-01-01:2023-06-30'.",
    )
    parser.add_argument(
        "--disable_output_versioning",
        action="store_true",
        help="Disable automatic archiving of old outputs before overwriting.",
    )
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    results_path = (project_dir / args.results_file).resolve()
    out_root = (project_dir / args.outdir).resolve()

    summary_df = pd.read_excel(results_path, sheet_name="Summary")
    feature_details_df = pd.read_excel(results_path, sheet_name="Feature_Details")
    selection_metric = args.selection_metric.upper()
    best_cand = choose_best_candidate_by_test(
        summary_df, group_name=args.group, metric=selection_metric
    )

    group_name = best_cand["Group"]
    model_name = best_cand["ModelName"]

    group_row = summary_df[summary_df["Group"] == group_name].iloc[0]

    print("=" * 80)
    print("BEST MODEL ANALYSIS")
    print("=" * 80)
    print(f"Project dir : {project_dir}")
    print(f"Results file: {results_path}")
    print(f"Group       : {group_name}")
    print(f"Selected by : Test {selection_metric} (across LR/RF/LGB)")
    print(f"Best model  : {model_name}")
    print(f"Reported test PR/ROC from Summary: {best_cand['Test_PR_AUC']:.6f} / {best_cand['Test_ROC_AUC']:.6f}")

    split_cfgs = [{"mode": "regime", "name": "regime_split"}]
    if not args.disable_default_period_comparison:
        split_cfgs.append(
            make_date_split_config(
                "2008-01-29",
                "2020-07-14",
                name="train_20080129_20200714_test_20230101",
                test_end="2023-01-01",
            )
        )
    split_cfgs.extend(parse_extra_date_windows(args.extra_date_windows))

    # Deduplicate identical window configs by signature
    unique_cfgs = []
    seen = set()
    for cfg in split_cfgs:
        key = (
            cfg.get("mode"),
            cfg.get("name"),
            _safe_date_str(cfg.get("train_start")),
            _safe_date_str(cfg.get("train_end")),
            _safe_date_str(cfg.get("test_end")),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_cfgs.append(cfg)

    print("\nPlanned split analyses:")
    for i, cfg in enumerate(unique_cfgs, 1):
        if cfg.get("mode") == "regime":
            print(f"  {i}. {cfg.get('name')} (regime Data_Split Test)")
        else:
            train_s = _safe_date_str(cfg.get("train_start"))
            train_e = _safe_date_str(cfg.get("train_end"))
            test_e = _safe_date_str(cfg.get("test_end"))
            test_part = "test after" if test_e is None else f"test to {test_e}"
            print(
                f"  {i}. {cfg.get('name')} "
                f"[train {train_s} -> {train_e}, {test_part}]"
            )

    period_rows = []
    for cfg in unique_cfgs:
        row = run_single_period_analysis(
            project_dir=project_dir,
            out_root=out_root,
            group_name=group_name,
            model_name=model_name,
            group_row=group_row,
            feature_details_df=feature_details_df,
            split_cfg=cfg,
            max_shap_samples=args.max_shap_samples,
            run_tag=args.run_tag,
            disable_output_versioning=args.disable_output_versioning,
        )
        row["Group"] = group_name
        row["SelectionMetric"] = selection_metric
        period_rows.append(row)

    summary_out_dir = out_root / group_name
    summary_out_dir.mkdir(parents=True, exist_ok=True)
    period_summary_df = pd.DataFrame(period_rows)
    period_summary_path = summary_out_dir / "period_batch_summary.csv"
    period_summary_df.to_csv(period_summary_path, index=False)

    print("\n[Batch Summary]")
    print(period_summary_df.to_string(index=False))
    print(f"\nSaved period summary to: {period_summary_path}")


if __name__ == "__main__":
    main()
