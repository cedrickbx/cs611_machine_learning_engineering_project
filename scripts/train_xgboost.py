# scripts/train_xgboost.py
import argparse
import json
import os
import tempfile
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# MLflow
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow.compute as pc

# Sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    cohen_kappa_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, label_binarize, StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# XGBoost
import xgboost as xgb
from xgboost.callback import EarlyStopping

# >>> PROGRESS: small helper (adds no dependency/logic changes)
from datetime import datetime as _dt
def _p(stage: str, **kv):
    ts = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
    extras = ", ".join(f"{k}={v}" for k, v in kv.items())
    print(f"[{ts}] [train_xgboost] {stage}" + (f" | {extras}" if extras else ""))

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def _load_parquet_range_fast(path: str,
                             date_col: str,
                             start: str,
                             end: str,
                             columns: list[str] | None = None) -> pd.DataFrame:
    """
    Fast parquet loader for file OR directory with pushdown:
      - path: parquet file or directory root
      - date_col: e.g., "snapshot_date" or "FlightDate"
      - start/end: ISO 'YYYY-MM-DD' inclusive (string OK)
      - columns: optional column subset to read
    """
    _p("fastload_begin", path=path, date_col=date_col, start=start, end=end)

    # Try Arrow dataset first (handles dir or file)
    try:
        dataset = ds.dataset(path, format="parquet")
        # dataset = ds.dataset("/opt/airflow/datamart/gold/gold_combined_historical.parquet", format="parquet")
        print("Partitioning:", dataset.partitioning)
        print("Num files:", len(dataset.files))
        print("Sample files:", dataset.files[:10])

        # Build filter (cast to timestamp if needed)
        fcol = ds.field(date_col)
        # Allow string, date, or timestamp columns
        filt = (pc.cast(fcol, "timestamp[ns]") >= pd.Timestamp(start)) & \
               (pc.cast(fcol, "timestamp[ns]") <= pd.Timestamp(end))

        table = dataset.to_table(filter=filt, columns=columns, use_threads=True)
        df = table.to_pandas()
        _p("fastload_arrow_ok", rows=len(df), cols=df.shape[1])
        return df
    except Exception as e:
        _p("fastload_arrow_fallback", error=str(e))

    # Fallback 1: pandas with filters arg (works for *file* paths with pyarrow engine)
    try:
        df = pd.read_parquet(path, filters=[(date_col, ">=", start), (date_col, "<=", end)], columns=columns)
        _p("fastload_pd_filters_ok", rows=len(df), cols=df.shape[1])
        return df
    except Exception as e:
        _p("fastload_pd_filters_fallback", error=str(e))

    # Fallback 2: full read + pandas filter (last resort)
    df = pd.read_parquet(path, columns=columns)
    if date_col not in df.columns:
        raise ValueError(f"date col '{date_col}' not found in data")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df[date_col].between(pd.to_datetime(start), pd.to_datetime(end))].copy()
    _p("fastload_full_read_filtered", rows=len(df), cols=df.shape[1])
    return df.reset_index(drop=True)

def _load_df(path: str) -> pd.DataFrame:
    # Accept a single parquet file, a directory of parquet files, or a csv/csv.gz
    if os.path.isdir(path):
        # directory of parquet files
        try:
            dataset = ds.dataset(path, format="parquet")
            return dataset.to_table().to_pandas()
        except Exception:
            # fallback: concat all parquet under dir
            parts = []
            for root, _, files in os.walk(path):
                for f in files:
                    if f.endswith(".parquet"):
                        parts.append(pd.read_parquet(os.path.join(root, f)))
            if not parts:
                raise FileNotFoundError(f"No parquet files under directory: {path}")
            return pd.concat(parts, ignore_index=True)
    else:
        lower = path.lower()
        if lower.endswith((".parquet", ".parq", ".pq")):
            return pd.read_parquet(path)
        if lower.endswith((".csv", ".csv.gz")):
            return pd.read_csv(path)
        # last resort: try parquet then csv
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_csv(path)

def parse_comma_list(s: str) -> List[str]:
    if not s:
        return []
    return [c.strip() for c in s.split(",") if c.strip()]

def month_floor(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.to_period("M").dt.to_timestamp()

def compute_time_splits(
    df: pd.DataFrame,
    date_col: str,
    total_months: int,
    val_months: int,
    test_months: int,
) -> Tuple[pd.Index, pd.Index, pd.Index, Dict[str, str]]:
    """
    Returns (train_idx, val_idx, test_idx, spans)
    """
    ym = month_floor(df[date_col])
    max_month = ym.max()
    start_month = (max_month.to_period("M") - (total_months - 1)).to_timestamp()

    window_mask = (ym >= start_month) & (ym <= max_month)
    dfw = df.loc[window_mask].copy()
    ym_w = month_floor(dfw[date_col])

    # windows from the end
    test_start = (max_month.to_period("M") - (test_months - 1)).to_timestamp()
    val_end = (test_start.to_period("M") - 1).to_timestamp()
    val_start = (val_end.to_period("M") - (val_months - 1)).to_timestamp()

    train_mask = ym_w < val_start
    val_mask = (ym_w >= val_start) & (ym_w <= val_end)
    test_mask = (ym_w >= test_start) & (ym_w <= max_month)

    spans = {
        "window_start": start_month.strftime("%Y-%m-01"),
        "window_end": max_month.strftime("%Y-%m-01"),
        "val_start": val_start.strftime("%Y-%m-01"),
        "val_end": val_end.strftime("%Y-%m-01"),
        "test_start": test_start.strftime("%Y-%m-01"),
        "test_end": max_month.strftime("%Y-%m-01"),
    }

    return dfw.index[train_mask], dfw.index[val_mask], dfw.index[test_mask], spans

def build_preprocessor(
    X: pd.DataFrame, numeric_only: bool
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Build preprocessing pipeline:
    - Impute + cap outliers + normalize numeric features
    - Impute + one-hot encode categorical features
    - Automatically exclude high-cardinality or ID-like columns (e.g., flight_id)
    """

    # ------------------------------------------------------------------
    # Explicit exclusions for safety
    # ------------------------------------------------------------------
    id_like_cols = ['flight_id', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID', 'IsPublicHoliday', 'is_extra_candidate']

    # ------------------------------------------------------------------
    # Detect types
    # ------------------------------------------------------------------
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = []
    if not numeric_only:
        cat_cols = [
            c for c in X.columns
            if (X[c].dtype == "object") or pd.api.types.is_categorical_dtype(X[c])
        ]
        cat_cols = [c for c in cat_cols if c not in num_cols]

    # Remove ID-like columns from both lists
    num_cols = [c for c in num_cols if c not in id_like_cols]
    cat_cols = [c for c in cat_cols if c not in id_like_cols]

    # ------------------------------------------------------------------
    # Outlier capping (IQR-based)
    # ------------------------------------------------------------------
    def cap_outliers(X, columns=None, lower_q=0.01, upper_q=0.99):
        """
        Caps values to [lower_q, upper_q] quantiles per column.
        Works with both pandas DataFrame and numpy ndarray.
        If X is ndarray, `columns` must be provided (or will be auto-generated).
        """
        is_array = isinstance(X, np.ndarray)
        if is_array:
            # X is the slice for your selected columns (e.g., numeric block)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if columns is None:
                columns = [f"col_{i}" for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=columns)
        else:
            # X is already a DataFrame
            df = X.copy()
            if columns is None:
                columns = list(df.columns)

        for col in columns:
            s = pd.to_numeric(df[col], errors="coerce")
            q_low, q_hi = s.quantile([lower_q, upper_q])
            df[col] = s.clip(q_low, q_hi)

        return df.values if is_array else df

    capper = FunctionTransformer(
        func=cap_outliers,
        kw_args={"columns": num_cols, "lower_q": 0.01, "upper_q": 0.99},
        validate=False,                   # allow pandas/ndarray flexibly
        feature_names_out="one-to-one",   # keeps same column count
    )

    # ------------------------------------------------------------------
    # Pipelines
    # ------------------------------------------------------------------
    num_pipeline = Pipeline(steps=[
        #("cap_outliers", capper),
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
        ]
    )

    transformers = [("num", num_pipeline, num_cols)]
    if cat_cols and not numeric_only:
        transformers.append(("cat", cat_pipeline, cat_cols))

    pre = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Optional logging
    print(f"[Preprocessor] Numeric: {len(num_cols)}, Categorical: {len(cat_cols)}, Ignored ID-like: {id_like_cols}")

    return pre, num_cols, cat_cols

def ovr_macro_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    # proba shape: (N, 3)
    return roc_auc_score(y_true, proba, multi_class="ovr", average="macro")

def macro_prauc(y_true: np.ndarray, proba: np.ndarray) -> float:
    # One-vs-rest PR-AUC averaged across classes
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    scores = []
    for k in range(y_bin.shape[1]):
        scores.append(average_precision_score(y_bin[:, k], proba[:, k]))
    return float(np.mean(scores))

def class_weights_inverse_freq(y: np.ndarray) -> np.ndarray:
    vals, counts = np.unique(y, return_counts=True)
    freq = {v: c for v, c in zip(vals, counts)}
    inv = {v: 1.0 / c for v, c in freq.items()}
    w = np.array([inv[v] for v in y], dtype=float)
    # normalize weights to mean=1 (optional; keeps scale stable)
    return w * (len(w) / w.sum())

def split_Xy(
    df: pd.DataFrame,
    label_col: str,
    date_col: str,
    exclude_cols: List[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    drop_cols = set([label_col, date_col] + exclude_cols)
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    y = df[label_col].astype(int).values
    return X, y

# ------------------------------------------------------------------------------
# Optuna Objective (uses pre-fitted preprocessor)
# ------------------------------------------------------------------------------
def make_objective(
    X_tr_t: np.ndarray,
    y_tr: np.ndarray,
    sample_w_tr: np.ndarray,
    n_splits: int,
    allow_max_bin: bool,
):
    """
    Optuna objective combining macro AUC (ranking quality) and macro F1 (minority balance)
    for a more balanced optimization under strong class imbalance.
    """
    def objective(trial):
        # Core model parameters — narrow range for faster tuning
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "tree_method": "hist",
            "eval_metric": "mlogloss",

            # tune only 5 knobs, with tight ranges
            "n_estimators":     trial.suggest_int("n_estimators", 300, 600),
            "learning_rate":    trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 6),
            "subsample":        trial.suggest_float("subsample", 0.7, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.95),

            # fixed, sensible defaults
            "min_child_weight": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "n_jobs": -1,  # parallel CPU
            # "device": "cuda",  # uncomment if GPU available; much faster
        }
        if allow_max_bin:
            params["max_bin"] = trial.suggest_int("max_bin", 128, 512)

        skf = StratifiedKFold(n_splits=min(3, n_splits), shuffle=True)
        scores = []

        for tr_idx, va_idx in skf.split(X_tr_t, y_tr):
            Xtr, Xva = X_tr_t[tr_idx], X_tr_t[va_idx]
            ytr, yva = y_tr[tr_idx], y_tr[va_idx]
            wtr = sample_w_tr[tr_idx]

            clf = xgb.XGBClassifier(**params)
            clf.fit(Xtr, ytr, sample_weight=wtr, verbose=False)

            proba = clf.predict_proba(Xva)
            y_pred = np.argmax(proba, axis=1)

            # ---- metrics per fold ----
            auc_macro = ovr_macro_auc(yva, proba)
            f1_macro = f1_score(yva, y_pred, average="macro", zero_division=0)

            # Combined score (tunable weights)
            blended = 0.5 * auc_macro + 0.5 * f1_macro
            scores.append(blended)

        # Mean blended score across folds
        return float(np.mean(scores))

    return objective

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    import optuna

    ap = argparse.ArgumentParser("Train XGBoost (multiclass ordinal 0/1/2) on Gold features with Optuna + MLflow")
    ap.add_argument("--data-path", type=str, required=True, help="Parquet/CSV file of Gold features")
    ap.add_argument("--date-col", type=str, default="FlightDate")
    ap.add_argument("--label-col", type=str, default="IS_DELAYED")
    ap.add_argument(
        "--exclude-cols",
        type=str,
        default=(
            # ---- delay columns ----
            "DEP_DELAY_NEW,DEP_DELAY_GROUP,ARR_DELAY_NEW,ARR_DELAY_GROUP,DEP_TIME,ARR_TIME,"
            #"DEP_TIME_BLK,ARR_TIME_BLK,"
            "CARRIER_DELAY,WEATHER_DELAY,NAS_DELAY,SECURITY_DELAY,LATE_AIRCRAFT_DELAY,"
            # ---- weather cleanup ----
            "weather_ceiling_code,weather_visibility_var_code_idx,"
            "weather_wind_dir_deg,weather_visibility_var_code"
        ),
        help="Comma-separated columns to exclude from X (leakage removal + weather cleanup)",
    )
    ap.add_argument("--total-months", type=int, default=24)
    ap.add_argument("--val-months", type=int, default=3)
    ap.add_argument("--test-months", type=int, default=3)
    ap.add_argument("--numeric-only", action="store_true", help="Drop categoricals instead of OHE")
    ap.add_argument("--n-splits", type=int, default=3, help="StratifiedKFold splits for CV")
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--allow-max-bin", action="store_true")
    ap.add_argument("--mlflow-experiment", type=str, default="flight_delay_xgb_optuna_ord3")
    ap.add_argument("--outdir", type=str, default="/tmp", help="Temporary only; nothing kept locally")
    ap.add_argument("--hardcode-q1-2023", action="store_true",
                help="If set, ignore --data-path and load the three fixed partitions "
                     "../../datamart/gold/gold_combined_historical.parquet/snapshot_date=2023-01-01, "
                     "../../datamart/gold/gold_combined_historical.parquet/snapshot_date=2023-02-01, "
                     "../../datamart/gold/gold_combined_historical.parquet/snapshot_date=2023-03-01")
    ap.add_argument("--retrain-date", type=str, default=None,
                help="If set, load data dynamically for retraining as of this date (YYYY-MM-DD). "
                     "Overrides --data-path and --hardcode-q1-2023.")
    args = ap.parse_args()

    # >>> PROGRESS
    _p("start",
       data_path=args.data_path, n_trials=args.n_trials, total_months=args.total_months,
       val_months=args.val_months, test_months=args.test_months, date_col=args.date_col,
       label_col=args.label_col, numeric_only=args.numeric_only)

    # -------------------- Load data --------------------
    '''if args.data_path.lower().endswith(".parquet"):
        df = pd.read_parquet(args.data_path)
    else:
        df = pd.read_csv(args.data_path)'''

    # -------------------- Load data --------------------
    if args.hardcode_q1_2023:
        base_path = os.environ.get("DATAMART_ROOT", "/opt/airflow/datamart")
        base = os.path.join(base_path, "gold", "combined", "gold_combined_historical.parquet")

        # Optional: prune to only columns you truly need for training
        # (uncomment & fill as appropriate)
        # wanted_cols = [
        #     "snapshot_date", "IS_DELAYED", "FlightDate", "flight_id",
        #     # ... add actual feature columns you train on ...
        # ]
        wanted_cols = None  # keep None to read all columns

        df = _load_parquet_range_fast(
            path=base,
            date_col="snapshot_date",          # <-- change if your column differs
            start="2023-01-01",
            end="2024-12-31",
            columns=wanted_cols,
        )
    else:
        # Generic loader (file or directory) without date restriction
        # If your input is large, consider wiring --start-date/--end-date CLI and call _load_parquet_range_fast
        if os.path.isdir(args.data_path):
            try:
                dataset = ds.dataset(args.data_path, format="parquet")
                table = dataset.to_table(use_threads=True)
                df = table.to_pandas()
                _p("dir_load_arrow_ok", rows=len(df), cols=df.shape[1])
            except Exception as e:
                _p("dir_load_arrow_fallback", error=str(e))
                parts = []
                for root, _, files in os.walk(args.data_path):
                    for f in files:
                        if f.endswith(".parquet"):
                            parts.append(pd.read_parquet(os.path.join(root, f)))
                if not parts:
                    raise FileNotFoundError(f"No parquet files under directory: {args.data_path}")
                df = pd.concat(parts, ignore_index=True)
                _p("dir_load_concat_ok", rows=len(df), cols=df.shape[1])
        else:
            df = pd.read_parquet(args.data_path)
            _p("file_load_pd_ok", rows=len(df), cols=df.shape[1])
        
    df = df.reset_index(drop=True)

    # -------------------- Dynamic retrain loader --------------------
    if args.retrain_date:
        base_path = os.environ.get("DATAMART_ROOT", "/opt/airflow/datamart")
        hist_path = os.path.join(base_path, "gold", "combined", "gold_combined_historical.parquet")
        oot_dir   = os.path.join(base_path, "gold", "combined")

        retrain_date = pd.to_datetime(args.retrain_date)
        start_date = retrain_date - pd.DateOffset(months=24)
        end_date   = retrain_date - pd.DateOffset(days=1)

        _p("retrain_window", start=str(start_date.date()), end=str(end_date.date()))

        # Load from historical parquet (only overlap range)
        hist_df = _load_parquet_range_fast(
            path=hist_path,
            date_col="snapshot_date",
            start=max(start_date, pd.Timestamp("2023-01-01")).strftime("%Y-%m-%d"),
            end=min(end_date, pd.Timestamp("2024-12-31")).strftime("%Y-%m-%d"),
        )

        # Load daily OOT files (for 2025 onwards)
        oot_parts, found_dates, missing_dates = [], [], []
        daily_range = pd.date_range("2025-01-01", end_date, freq="D")

        for d in daily_range:
            fn = f"gold_combined_oot_{d.year}_{d.month:02d}_{d.day:02d}.parquet"
            fpath = os.path.join(oot_dir, fn)
            if os.path.exists(fpath):
                oot_parts.append(pd.read_parquet(fpath))
                found_dates.append(str(d.date()))
            else:
                missing_dates.append(str(d.date()))

        if oot_parts:
            oot_df = pd.concat(oot_parts, ignore_index=True)            

            # --- Harmonize OOT schema ---
            if "snapshot_date" in oot_df.columns and "FlightDate" not in oot_df.columns:
                _p("oot_fix_snapshot_date", note="Using snapshot_date as FlightDate for consistency")
                oot_df['FlightDate'] = oot_df['snapshot_date'].astype('object')

            df = pd.concat([hist_df, oot_df], ignore_index=True)

            if df["IS_DELAYED"].isna().any():
                _p("oot_unlabeled_rows_found", count=df["IS_DELAYED"].isna().sum())
                df_labeled = df.dropna(subset=["IS_DELAYED"]).copy()
                df_unlabeled = df[df["IS_DELAYED"].isna()].copy()

            _p(
                "merge_hist_oot_done",
                hist_rows=len(hist_df),
                oot_rows=len(oot_df),
                oot_files=len(found_dates),
                oot_start=found_dates[0] if found_dates else None,
                oot_end=found_dates[-1] if found_dates else None,
                missing_count=len(missing_dates),
            )

            # Optional: Log first 3 missing OOT dates if any
            if missing_dates:
                _p("oot_missing_files_sample", sample=missing_dates[:3])

        else:
            df = hist_df
            _p("oot_not_found", rows=len(df), note="No OOT files found in range")

    # >>> PROGRESS
    _p("data_loaded", rows=len(df), cols=df.shape[1])

    '''if args.hardcode_q1_2023:
        base_path = os.environ.get("DATAMART_ROOT", "/opt/airflow/datamart")
        base = os.path.join(base_path, "gold", "combined", "gold_combined_historical.parquet")

        _p("fastload", base=base)

        try:
            # Build a date range list for Q1 2023 (inclusive)
            import pandas as pd
            date_list = pd.date_range("2023-01-01", "2023-03-31", freq="D").strftime("%Y-%m-%d").tolist()

            # --- fast read with filter pushdown ---
            df = pd.read_parquet(base, filters=[("snapshot_date", "in", date_list)])
            _p("fast_filtered_load", rows=len(df), days=len(date_list))
        except Exception as e:
            _p("fallback_full_load", error=str(e))
            df = pd.read_parquet(base)

            if "snapshot_date" in df.columns:
                df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
                df = df[df["snapshot_date"].between("2023-01-01", "2023-03-31")].copy()
                _p("filtered_by_snapshot_date", rows=len(df))

        df = df.reset_index(drop=True)

    # >>> PROGRESS
    _p("data_loaded", rows=len(df), cols=df.shape[1])'''

    # Coerce date and ensure label present
    if args.date_col not in df.columns:
        raise ValueError(f"date col '{args.date_col}' not in data")
    if args.label_col not in df.columns:
        raise ValueError(f"label col '{args.label_col}' not in data")

    # AFTER — robust cast when FlightDate is object/strings, strips time/tz if present
    col = df[args.date_col]

    # cast to string first (handles mixed types), slice to date part if strings carry time
    col = col.astype(str).str.slice(0, 19)  # keeps "YYYY-MM-DD HH:MM:SS" if present

    # try general parse; if still many NaT, try strict YYYY-MM-DD as a fallback
    dt = pd.to_datetime(col, errors="coerce", utc=False)
    if dt.isna().mean() > 0.2:
        dt = pd.to_datetime(col.str.slice(0, 10), format="%Y-%m-%d", errors="coerce")

    # drop tz if any
    try:
        dt = dt.dt.tz_localize(None)
    except Exception:
        pass

    df[args.date_col] = dt

    # optional reconstruction fallback if the column couldn't be parsed at all
    if df[args.date_col].isna().all() and {"YEAR", "MONTH", "DAY_OF_MONTH"}.issubset(df.columns):
        df[args.date_col] = pd.to_datetime(
            dict(year=df["YEAR"].astype(int), month=df["MONTH"].astype(int), day=df["DAY_OF_MONTH"].astype(int)),
            errors="coerce"
        )

    df = df.dropna(subset=[args.date_col, args.label_col]).copy()
    df[args.label_col] = df[args.label_col].astype(int)

    # >>> PROGRESS
    _p("data_cleaned",
       rows=len(df),
       date_min=str(pd.to_datetime(df[args.date_col]).min().date()),
       date_max=str(pd.to_datetime(df[args.date_col]).max().date()))

    # -------------------- Dynamic Downsampling (majority class only) --------------------
    target_col = args.label_col

    # Compute class distribution
    class_counts = df[target_col].value_counts().to_dict()
    if 0 in class_counts and len(class_counts) > 1:
        cls0_count = class_counts[0]
        max_minority = max([v for k, v in class_counts.items() if k != 0])

        # Downsample class 0 to 20% of its original size
        RATIO = 0.2
        target_class0 = int(cls0_count * RATIO)

        if cls0_count > target_class0:
            _p(
                "downsampling_majority_class0",
                original_class0=cls0_count,
                target_class0=target_class0,
                ratio=RATIO,
                minority_max=max_minority,
            )

            cls0_df = (
                df[df[target_col] == 0]
                .sample(n=target_class0, random_state=42)
            )
            others_df = df[df[target_col] != 0]

            # Combine and shuffle
            df = (
                pd.concat([cls0_df, others_df], axis=0)
                .sample(frac=1.0, random_state=42)
                .reset_index(drop=True)
            )

        else:
            _p("no_downsampling_needed", class0=cls0_count, minority_max=max_minority)

        # Log new composition
        new_counts = df[target_col].value_counts().to_dict()
        _p("after_downsampling_dynamic", counts=new_counts)

    '''target_col = args.label_col

    # Compute class distribution
    class_counts = df[target_col].value_counts().to_dict()
    if 0 in class_counts and len(class_counts) > 1:
        cls0_count = class_counts[0]
        max_minority = max([v for k, v in class_counts.items() if k != 0])

        # choose ratio: keep class 0 at ~RATIO * largest minority class
        RATIO = float(os.environ.get("CLASS0_RATIO", "3.0"))
        target_class0 = int(max_minority * RATIO)

        if cls0_count > target_class0:
            _p("downsampling_majority_class0",
               original_class0=cls0_count,
               target_class0=target_class0,
               ratio=RATIO,
               minority_max=max_minority)
            cls0_df = df[df[target_col] == 0].sample(n=target_class0, random_state=42)
            others_df = df[df[target_col] != 0]
            df = pd.concat([cls0_df, others_df], axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)
        else:
            _p("no_downsampling_needed", class0=cls0_count, minority_max=max_minority)

        # Log new composition
        new_counts = df[target_col].value_counts().to_dict()
        _p("after_downsampling_dynamic", counts=new_counts)'''

    # -------------------- Splits --------------------
    train_idx, val_idx, test_idx, spans = compute_time_splits(
        df, args.date_col, args.total_months, args.val_months, args.test_months
    )

    # >>> PROGRESS
    _p("splits_ready",
       train=len(train_idx), val=len(val_idx), test=len(test_idx),
       window_start=spans["window_start"], window_end=spans["window_end"],
       val_span=f"{spans['val_start']}→{spans['val_end']}",
       test_span=f"{spans['test_start']}→{spans['test_end']}")

    excl = parse_comma_list(args.exclude_cols)

    # Log the final exclusion list
    _p("exclude_columns", count=len(excl), columns=",".join(excl))

    # Always exclude snapshot_date if it exists in the data
    if "snapshot_date" in df.columns and "snapshot_date" not in excl:
        excl.append("snapshot_date")
        _p("auto_exclude_snapshot_date", note="Added snapshot_date to exclusion list")

    X_all, y_all = split_Xy(df, args.label_col, args.date_col, excl)

    X_train, y_train = X_all.loc[train_idx], y_all[train_idx]
    X_val,   y_val   = X_all.loc[val_idx],   y_all[val_idx]
    X_test,  y_test  = X_all.loc[test_idx],  y_all[test_idx]

    # -------------------- Preprocessing --------------------
    pre, num_cols, cat_cols = build_preprocessor(X_train, numeric_only=args.numeric_only)
    pre.fit(X_train)  # fit ONLY on train
    X_tr_t = pre.transform(X_train)
    X_va_t = pre.transform(X_val)
    X_te_t = pre.transform(X_test)

    # for names after OHE
    feat_names = list(pre.get_feature_names_out())

    # >>> PROGRESS
    _p("preprocessor_fitted",
       num_features=len(num_cols), cat_features=len(cat_cols),
       transformed_dim=int(X_tr_t.shape[1]))

    # -------------------- Class imbalance weights --------------------
    w_train = class_weights_inverse_freq(y_train)
    w_val   = class_weights_inverse_freq(y_val)

    # >>> PROGRESS
    _p("weights_ready",
       w_train_mean=float(np.mean(w_train)), w_val_mean=float(np.mean(w_val)))

    # optional: limit to 50k rows for fast experimentation
    if len(X_tr_t) > 50000:
        _p("downsampling_for_speed", original_rows=len(X_tr_t))
        rs = np.random.RandomState(42)
        sel = rs.choice(len(X_tr_t), size=50000, replace=False)
        X_tr_t = X_tr_t[sel]
        y_train = y_train[sel]
        w_train = w_train[sel]

    # -------------------- Optuna search (CV on transformed train) --------------------
    study = optuna.create_study(direction="maximize")

    # >>> PROGRESS
    _p("optuna_start", trials=args.n_trials, n_splits=args.n_splits)

    study.optimize(
        make_objective(X_tr_t, y_train, w_train, args.n_splits, args.allow_max_bin),
        n_trials=args.n_trials,
        show_progress_bar=False,
        gc_after_trial=True,
    )

    # >>> PROGRESS
    _p("optuna_done", best_value=float(study.best_value), n_best_params=len(study.best_trial.params))

    best_params = study.best_trial.params.copy()
    # Fixed essentials
    best_params.update(
        {
            "objective": "multi:softprob",
            "num_class": 3,
            "tree_method": "hist",
        }
    )

    # -------------------- Refit best on full TRAIN with early stopping on VAL --------------------
    best_params["eval_metric"] = "mlogloss"
    clf = xgb.XGBClassifier(**best_params)
    
    # Cap estimators manually if you want a kind of early stopping effect
    n_estimators = best_params.get("n_estimators", 300)

    clf.fit(
        X_tr_t,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_va_t, y_val)],
        sample_weight_eval_set=[w_val],
        verbose=False,
    )

    # >>> PROGRESS
    _p("refit_complete",
       best_iteration=getattr(clf, "best_iteration", None),
       best_ntree_limit=getattr(clf, "best_ntree_limit", None))

    # Wrap as Pipeline for serving
    pipe = Pipeline([("prep", pre), ("clf", clf)])

    # -------------------- Evaluate (VAL / TEST) --------------------
    def eval_block(Xt: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        proba = clf.predict_proba(Xt)
        y_pred = np.argmax(proba, axis=1)

        metrics = {
            "roc_auc_macro_ovr": ovr_macro_auc(y, proba),
            "pr_auc_macro": macro_prauc(y, proba),
            "accuracy": accuracy_score(y, y_pred),
            "precision_macro": precision_score(y, y_pred, average="macro", zero_division=0),
            "precision_weighted": precision_score(y, y_pred, average="weighted", zero_division=0),
            "recall_macro": recall_score(y, y_pred, average="macro", zero_division=0),
            "recall_weighted": recall_score(y, y_pred, average="weighted", zero_division=0),
            "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y, y_pred, average="weighted", zero_division=0),
            "qwk": cohen_kappa_score(y, y_pred, weights="quadratic"),
        }
        cm = confusion_matrix(y, y_pred, labels=[0, 1, 2])
        return metrics, cm, proba

    val_metrics, val_cm, _ = eval_block(X_va_t, y_val)
    test_metrics, test_cm, _ = eval_block(X_te_t, y_test)

    # >>> PROGRESS
    _p("evaluated_val",
       auc_macro_ovr=float(val_metrics["roc_auc_macro_ovr"]),
       f1_macro=float(val_metrics["f1_macro"]),
       qwk=float(val_metrics["qwk"]))
    _p("evaluated_test",
       auc_macro_ovr=float(test_metrics["roc_auc_macro_ovr"]),
       f1_macro=float(test_metrics["f1_macro"]),
       qwk=float(test_metrics["qwk"]))

    # -------------------- Feature importances (after OHE) --------------------
    importances = clf.feature_importances_
    fi_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values(
        "importance", ascending=False
    )

    # >>> PROGRESS
    _p("feature_importance_ready",
       nonzero=int((fi_df['importance'] > 0).sum()),
       top1=fi_df.iloc[0]["feature"] if len(fi_df) else None)

    # -------------------- Split summaries --------------------
    def class_dist(y: np.ndarray) -> Dict[str, int]:
        vals, counts = np.unique(y, return_counts=True)
        return {str(int(v)): int(c) for v, c in zip(vals, counts)}

    month_series = month_floor(df[args.date_col])
    def mask_idx(idxs: pd.Index) -> pd.Series:
        m = pd.Series(False, index=df.index)
        m.loc[idxs] = True
        return m

    spans_ext = {
        **spans,
        "rows": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "class_dist": {
            "train": class_dist(y_train),
            "val": class_dist(y_val),
            "test": class_dist(y_test),
        },
        "months": {
            "train_min": str(month_series[mask_idx(train_idx)].min().date()) if len(train_idx) else None,
            "train_max": str(month_series[mask_idx(train_idx)].max().date()) if len(train_idx) else None,
            "val_min": str(month_series[mask_idx(val_idx)].min().date()) if len(val_idx) else None,
            "val_max": str(month_series[mask_idx(val_idx)].max().date()) if len(val_idx) else None,
            "test_min": str(month_series[mask_idx(test_idx)].min().date()) if len(test_idx) else None,
            "test_max": str(month_series[mask_idx(test_idx)].max().date()) if len(test_idx) else None,
        },
    }

    # -------------------- MLflow: log ONLY best trial --------------------
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    _p("mlflow_uri_confirmed", uri=tracking_uri)

    mlflow.set_experiment(args.mlflow_experiment)

    input_example = X_val.head(5)
    signature = infer_signature(
        input_example, pd.DataFrame(clf.predict_proba(X_va_t[:5]), columns=["p0", "p1", "p2"])
    )

    report = {
        "split_config": {
            "total_months": args.total_months,
            "val_months": args.val_months,
            "test_months": args.test_months,
            "date_col": args.date_col,
            "label_col": args.label_col,
            "exclude_cols": parse_comma_list(args.exclude_cols),
            "numeric_only": bool(args.numeric_only),
        },
        "spans": spans_ext,
        "cv": {
            "n_splits": args.n_splits,
            "best_cv_macro_auc_ovr": float(study.best_value),
        },
        "metrics": {
            "val": val_metrics,
            "test": test_metrics,
        },
        "notes": (
            "Ordinal target {0,1,2}; we include QWK to respect ordering. "
            "If performance is marginal, consider ordinal-aware models (e.g., cumulative link / CORAL) "
            "or cascading binaries.\n"
            "XGBoost with mixed numeric + OHE categorical features is suitable; "
            "Optuna + StratifiedKFold and sample weights help under class imbalance."
        ),
    }

    # Read experiment date from Airflow env
    experiment_dt = os.environ.get("EXPERIMENT_DT")
    if experiment_dt:
        _p("airflow_schedule_confirmed", experiment_dt=experiment_dt)
    else:
        experiment_dt = datetime.now().strftime("%Y-%m-%d")

    # Use date in run name
    run_name = f"xgb_multiclass_{experiment_dt}"

    # >>> PROGRESS
    _p("mlflow_start", experiment=args.mlflow_experiment)

    with mlflow.start_run(run_name=run_name):
        # Log date param & tag
        mlflow.log_param("EXPERIMENT_DT", experiment_dt)
        mlflow.set_tag("EXPERIMENT_DT", experiment_dt)

        # Params (best hyperparams only)
        mlflow.log_params(best_params)

        # Log sample sizes
        train_size, val_size, test_size = len(train_idx), len(val_idx), len(test_idx)
        mlflow.log_params({
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
        })

        # Also log class distribution summaries (optional but useful)
        mlflow.log_dict({
            "train_class_dist": spans_ext["class_dist"]["train"],
            "val_class_dist": spans_ext["class_dist"]["val"],
            "test_class_dist": spans_ext["class_dist"]["test"]
        }, artifact_file="analysis/class_distribution.json")

        mlflow.set_tags({
            "train_period": f"{spans_ext['months']['train_min']}→{spans_ext['months']['train_max']}",
            "val_period": f"{spans_ext['months']['val_min']}→{spans_ext['months']['val_max']}",
            "test_period": f"{spans_ext['months']['test_min']}→{spans_ext['months']['test_max']}",
        })

        # Metrics (VAL + TEST only)
        mlflow.log_metrics({f"val_{k}": float(v) for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in test_metrics.items()})

        # --- Log trained columns and label column ---
        trained_cols = list(X_train.columns)
        label_col = args.label_col

        mlflow.log_param("n_features", len(trained_cols))
        mlflow.log_param("label_col", label_col)

        cols_info = {
            "trained_columns": trained_cols,
            "label_column": label_col,
        }
        cols_path = os.path.join(args.outdir, "trained_columns.json")
        with open(cols_path, "w") as f:
            json.dump(cols_info, f, indent=2)
        mlflow.log_artifact(cols_path, artifact_path="analysis")

        # Confusion matrices as artifacts (json)
        with tempfile.TemporaryDirectory(dir=args.outdir) as td:
            # Feature importance
            fi_path = os.path.join(td, "feature_importance.csv")
            fi_df.to_csv(fi_path, index=False)
            mlflow.log_artifact(fi_path, artifact_path="analysis")

            # Confusion matrices
            cm_payload = {
                "val_confusion_matrix": val_cm.tolist(),
                "test_confusion_matrix": test_cm.tolist(),
                "labels": [0, 1, 2],
            }
            cm_path = os.path.join(td, "confusion_matrices.json")
            with open(cm_path, "w") as f:
                json.dump(cm_payload, f, indent=2)
            mlflow.log_artifact(cm_path, artifact_path="analysis")

            # Report
            rpt_path = os.path.join(td, "report.json")
            with open(rpt_path, "w") as f:
                json.dump(report, f, indent=2)
            mlflow.log_artifact(rpt_path, artifact_path="analysis")

        # Model (sklearn Pipeline with preprocessor + trained XGB)
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            input_example=input_example,
            signature=signature,
        )

    # >>> PROGRESS
    _p("mlflow_done")
    print("[OK] Logged best trial ONLY to MLflow.")

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
