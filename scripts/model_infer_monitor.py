# scripts/model_infer_monitor.py
import argparse, os, json, tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix
)
from scipy.stats import ks_2samp

# --------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------
def _compute_inference_metrics(y_true, proba, y_pred):
    """Compute evaluation metrics consistent with training phase."""
    metrics = {
        "roc_auc_macro_ovr": roc_auc_score(y_true, proba, multi_class="ovr", average="macro"),
        "pr_auc_macro": np.mean([
            average_precision_score((y_true == i).astype(int), proba[:, i])
            for i in np.unique(y_true)
        ]),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
    }
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    return metrics, cm

def _psi(reference, current, bins=10):
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    if reference.size < 50 or current.size < 50:
        return None
    try:
        qs = np.linspace(0, 1, bins + 1)
        cuts = np.unique(np.quantile(reference, qs))
        if len(cuts) < 3:
            return None
        r_hist, _ = np.histogram(reference, bins=cuts)
        c_hist, _ = np.histogram(current, bins=cuts)
        r_prop = (r_hist + 1e-6) / (r_hist.sum() + 1e-6 * len(r_hist))
        c_prop = (c_hist + 1e-6) / (c_hist.sum() + 1e-6 * len(c_hist))
        return float(np.sum((c_prop - r_prop) * np.log(c_prop / r_prop)))
    except Exception:
        return None

def _safe_ks(ref, cur):
    if len(ref) < 2 or len(cur) < 2:
        return None
    try:
        return float(ks_2samp(ref, cur).statistic)
    except Exception:
        return None

def _p(stage: str, **kv):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    extras = ", ".join(f"{k}={v}" for k, v in kv.items())
    print(f"[{ts}] [model_infer_monitor] {stage}" + (f" | {extras}" if extras else ""))

def _load_gold_features(path) -> pd.DataFrame:
    path = Path(path)  # ensure it's a Path object

    if path.is_dir():
        parts = []
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(".parquet"):
                    parts.append(pd.read_parquet(os.path.join(root, f)))
                elif f.endswith(".csv"):
                    parts.append(pd.read_csv(os.path.join(root, f)))
        if not parts:
            raise FileNotFoundError(f"No data found under {path}")
        return pd.concat(parts, ignore_index=True)

    elif path.suffix == ".parquet":
        return pd.read_parquet(path)

    elif path.suffix == ".csv":
        return pd.read_csv(path)

    else:
        raise ValueError(f"Unsupported file format: {path}")

def _compute_multiclass_metrics(y_true, proba, y_pred):
    metrics = {
        "roc_auc_macro_ovr": roc_auc_score(y_true, proba, multi_class="ovr", average="macro"),
        "pr_auc_macro": np.mean([
            average_precision_score((y_true == i).astype(int), proba[:, i]) for i in np.unique(y_true)
        ]),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
    }
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    return metrics, cm

def _summarize_probs(proba):
    maxp = proba.max(axis=1)
    return {
        "sample_size": int(len(proba)),
        "prob_max_min": float(maxp.min()) if len(maxp) else None,
        "prob_max_mean": float(maxp.mean()) if len(maxp) else None,
        "prob_max_std": float(maxp.std()) if len(maxp) else None,
    }

# --------------------------------------------------------------------------
# Core inference logic
# --------------------------------------------------------------------------
def run_inference(args):
    """Run model inference and save daily predictions"""
    tracking_uri = args.tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    _p("mlflow_uri", uri=tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    client = MlflowClient()

    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    target_date = pd.to_datetime(args.snapshotdate).date()
    pattern = f"gold_combined_oot_{target_date.strftime('%Y_%m_%d')}.parquet"
    file_path = Path(args.gold_feature_dir) / pattern

    df = pd.read_parquet(file_path)
    df.columns = [c.strip() for c in df.columns]

    # --- Identify which date column to use ---
    if "snapshot_date" in df.columns:
        date_col = "snapshot_date"
    elif all(c in df.columns for c in ["year", "month", "day_of_month"]):
        # fallback: reconstruct from year/month/day
        df["snapshot_date"] = pd.to_datetime(
            df[["year", "month", "day_of_month"]].astype(str).agg("-".join, axis=1),
            errors="coerce"
        ).dt.date
        date_col = "snapshot_date"
    else:
        raise KeyError(f"No date column found! Available: {list(df.columns)[:12]}")

    # Parse date safely
    df[date_col] = pd.to_datetime(df[date_col].astype(str), errors="coerce").dt.date
    target_date = pd.to_datetime(args.snapshotdate).date()

    df = df[df["snapshot_date"] == target_date]
    if df.empty:
        _p("no_rows_for_date", snapshotdate=args.snapshotdate)
        return None

    _p("data_loaded", rows=len(df))

    # --- Identify label column (case-insensitive) ---
    possible_labels = [c for c in df.columns if c.lower() == "is_delayed"]
    if not possible_labels:
        raise KeyError(f"No label column 'IS_DELAYED' found! Available: {list(df.columns)[:12]}")
    label_col = possible_labels[0]
    _p("label_col_detected", label_col=label_col)

    y = df[label_col].astype(int).values

    y = df[label_col].astype(int).values
    drop_cols = [label_col, "FlightDate"] + [c for c in ["flight_id", "snapshot_date", "snapshotdate"] if c in df.columns]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

    excl = [c.strip() for c in args.exclude_cols.split(",") if c.strip()]
    X = X.drop(columns=[c for c in excl if c in X.columns], errors="ignore")

    model_uri = args.model_uri or f"models:/{args.model_name}@{args.model_alias}"
    import mlflow.sklearn as mls
    clf = mls.load_model(model_uri)
    _p("model_loaded", uri=model_uri)

    # --- Schema validation ---
    try:
        trained_features = clf.feature_names_in_
        missing = [c for c in trained_features if c not in X.columns]
        extra = [c for c in X.columns if c not in trained_features]
        if missing:
            _p("warn_missing_cols", count=len(missing), cols=",".join(missing[:10]))
        if extra:
            _p("warn_extra_cols", count=len(extra), cols=",".join(extra[:10]))
    except AttributeError:
        _p("feature_check_skipped", note="Pipeline missing feature_names_in_ attr")

    proba = clf.predict_proba(X)
    pred_class = proba.argmax(axis=1)
    _p("predictions_done", rows=len(pred_class))

    predicted_at = datetime.now(timezone.utc).astimezone().isoformat()
    base_name = f"pred_{args.snapshotdate.replace('-', '')}"
    parquet_path = Path(args.pred_dir) / f"{base_name}.parquet"
    csv_path = parquet_path.with_suffix(".csv")

    pred_df = pd.DataFrame({
        "pred_class": pred_class,
        "proba_0": proba[:, 0],
        "proba_1": proba[:, 1],
        "proba_2": proba[:, 2],
        "snapshot_date": args.snapshotdate,
        "predicted_at": predicted_at,
    })
    for k in ["FlightDate", "flight_id"]:
        if k in df.columns:
            pred_df[k] = df[k].values
    pred_df["IS_DELAYED"] = y

    if args.write_local:
        pred_df["predicted_at"] = pred_df["predicted_at"].astype(str)  # enforce string dtype
        pred_df.to_parquet(parquet_path, index=False)
        if args.write_csv:
            pred_df.to_csv(csv_path, index=False)
        _p("predictions_saved", parquet=str(parquet_path), csv=str(csv_path))

        # --- Log to MLflow as artifact ---
        mlflow.set_tracking_uri(args.tracking_uri or "http://mlflow:5000")
        mlflow.set_experiment(args.mlflow_experiment)

        with mlflow.start_run(run_name=f"inference_{args.snapshotdate.replace('-', '')}") as run:
            mlflow.log_artifact(parquet_path, artifact_path="predictions")
            mlflow.log_artifact(str(csv_path), artifact_path="predictions")
            mlflow.set_tags({
                "snapshot_date": args.snapshotdate,
                "purpose": "inference",
                "model_alias": args.model_alias,
            })
            _p("mlflow_prediction_logged", run_id=run.info.run_id)

    return pred_df

# --------------------------------------------------------------------------
# Monitoring logic
# --------------------------------------------------------------------------
def run_monitoring(args):
    """Compute PSI/KS drift metrics for same-day predictions"""
    snapshot = pd.to_datetime(args.snapshotdate)
    pred_dir = Path(args.pred_dir)
    cur_path = pred_dir / f"pred_{snapshot.strftime('%Y%m%d')}.parquet"
    if not cur_path.exists():
        _p("monitor_skip", reason=f"No predictions found for {snapshot.date()}")
        return None

    cur_df = pd.read_parquet(cur_path)
    cur_max = cur_df[["proba_0", "proba_1", "proba_2"]].max(axis=1)

    # Build reference window (past 14 days)
    ref_dfs = []
    for i in range(1, 15):
        d = snapshot - timedelta(days=i)
        f = pred_dir / f"pred_{d.strftime('%Y%m%d')}.parquet"
        if f.exists():
            ref_dfs.append(pd.read_parquet(f))
    if not ref_dfs:
        _p("monitor_skip", reason="No reference predictions available")
        return {"psi_max_proba": None, "ks_max_proba": None}

    ref_df = pd.concat(ref_dfs, ignore_index=True)
    ref_max = ref_df[["proba_0", "proba_1", "proba_2"]].max(axis=1)
    psi_val = _psi(ref_max, cur_max)
    ks_val = _safe_ks(ref_max, cur_max)
    _p("psi_ks_done", psi=psi_val, ks=ks_val)

    return {"psi_max_proba": psi_val, "ks_max_proba": ks_val}

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(args):
    mode = args.mode.lower()
    _p("mode_start", mode=mode, snapshotdate=args.snapshotdate)

    if mode in ["inference", "both"]:
        pred_df = run_inference(args)
        if pred_df is None:
            return 0

    if mode in ["monitor", "both"]:
        drift = run_monitoring(args)
        if drift is None:
            return 0

        # Load current day's prediction to compute model metrics
        snapshot = pd.to_datetime(args.snapshotdate)
        pred_path = Path(args.pred_dir) / f"pred_{snapshot.strftime('%Y%m%d')}.parquet"
        if not pred_path.exists():
            _p("monitor_skip", reason=f"No predictions found for {snapshot.date()}")
            return 0

        pred_df = pd.read_parquet(pred_path)
        y_true = pred_df["IS_DELAYED"].astype(int).values
        y_pred = pred_df["pred_class"].astype(int).values
        proba_cols = [c for c in pred_df.columns if c.startswith("proba_")]
        proba = pred_df[proba_cols].values

        # Compute metrics (same as training)
        metrics, cm = _compute_inference_metrics(y_true, proba, y_pred)
        _p("monitor_metrics_done", metrics_summary=json.dumps(metrics, indent=2))

        # Combine drift + performance metrics
        clean_drift = {
            k: float(v)
            for k, v in drift.items()
            if v is not None and not pd.isna(v)
        }
        all_metrics = {f"perf_{k}": float(v) for k, v in metrics.items()} | clean_drift

        # --- Log to MLflow ---
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        mlflow.set_experiment(args.mlflow_experiment)
        with mlflow.start_run(run_name=f"monitor_{args.snapshotdate.replace('-', '')}") as run:
            mlflow.set_tags({
                "snapshot_date": args.snapshotdate,
                "purpose": "monitoring",
                "model_alias": args.model_alias,
            })

            mlflow.log_metrics(all_metrics)

            # Save confusion matrix as artifact
            cm_path = Path(tempfile.gettempdir()) / f"confusion_matrix_{args.snapshotdate}.json"
            with open(cm_path, "w") as f:
                json.dump({"matrix": cm.tolist(), "labels": [0, 1, 2]}, f, indent=2)
            mlflow.log_artifact(str(cm_path), artifact_path="analysis")

            _p("mlflow_monitor_logged", run_id=run.info.run_id)

    _p("done")
    return 0

# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Unified inference + monitoring for flight delay prediction.")
    ap.add_argument("--mode", type=str, default="both", choices=["inference", "monitor", "both"], help="Run mode.")
    ap.add_argument("--snapshotdate", required=True, help="YYYY-MM-DD (FlightDate filter)")
    ap.add_argument("--gold-feature-dir", required=True, help="Path to Gold features parquet/CSV")
    ap.add_argument("--pred-dir", type=str, default="datamart/gold/model_predictions")
    ap.add_argument("--mlflow-experiment", type=str, default="flight_delay_inference_monitoring")
    ap.add_argument("--tracking-uri", type=str, default=None)
    ap.add_argument("--model-uri", default=None)
    ap.add_argument("--model-name", default="flight_delay_xgb")
    ap.add_argument("--model-alias", default="Production")
    ap.add_argument("--write-local", action="store_true")
    ap.add_argument("--write-csv", action="store_true")
    ap.add_argument(
        "--exclude-cols",
        type=str,
        default=(
            "DEP_DELAY_NEW,DEP_DELAY_GROUP,ARR_DELAY_NEW,ARR_DELAY_GROUP,DEP_TIME,ARR_TIME,"
            #"DEP_TIME_BLK,ARR_TIME_BLK,"
            "CARRIER_DELAY,WEATHER_DELAY,NAS_DELAY,SECURITY_DELAY,LATE_AIRCRAFT_DELAY,"
            "weather_ceiling_code,weather_visibility_var_code_idx,weather_wind_dir_deg,weather_visibility_var_code"
        ),
    )
    args = ap.parse_args()
    main(args)
