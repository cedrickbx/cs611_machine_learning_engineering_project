import os, glob, shutil
from datetime import datetime, timedelta
from typing import Iterable, List, Dict, Tuple

import pandas as pd
import xarray as xr
from herbie import Herbie

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DateType

"""
Core functions to build a Bronze table of GFS 0.25° forecasts for JFK/LGA/EWR.

- Historical (default args): 2023-01-01 → 2024-12-31 → single Parquet file
- OOT (default args): 2025-01-01 → 2025-04-30 → one named Parquet per valid_date
"""

# ----------------------------- DEFAULT CONFIG -----------------------------

# Airports: (ID, lat, lon)
AIRPORTS = [
    ("JFK", 40.63972, -73.77889),
    ("LGA", 40.77686, -73.87407),
    ("EWR", 40.68949, -74.17454),
]

# Forecast hours to fetch (lean, high-skill set; adjust in CLI if needed)
DEFAULT_FORECAST_HOURS = (6, 12, 24, 48, 72)  # add 168 for planning if desired
RAW_PRODUCT = "pgrb2.0p25"
DEFAULT_OUTPUT_ROOT = "../datamart/bronze/forecast"

# ----------------------------- HELPERS -----------------------------

def _daterange(d0: datetime, d1: datetime) -> Iterable[datetime]:
    cur = d0
    while cur <= d1:
        yield cur
        cur += timedelta(days=1)

def _safe_first_var(ds: xr.Dataset, filt: Dict) -> Tuple[str, xr.DataArray]:
    """Return (name, dataarray) for first var matching filter attrs, else (None, None)."""
    if hasattr(ds, "filter_by_attrs"):
        sub = ds.filter_by_attrs(**filt)
        if sub.data_vars:
            vname = list(sub.data_vars)[0]
            return vname, sub[vname]
    return None, None

def fetch_gfs_points(start_date: datetime,
                     end_date: datetime,
                     cycles: Iterable[int],
                     fhours: Iterable[int],
                     airports=AIRPORTS) -> pd.DataFrame:
    """
    Use Herbie to pull GFS 0.25° and extract nearest-gridpoint values
    for JFK/LGA/EWR at requested forecast hours using regex searches
    on the GRIB index (robust to cfgrib grouping).
    """
    # Regex searches that match GFS pgrb2.0p25 variable names in the IDX:
    SEARCHES = {
        "u10":   r":UGRD:10 m above",              # 10m U wind
        "v10":   r":VGRD:10 m above",              # 10m V wind
        "t2m":   r":TMP:2 m above",                # 2m temperature
        "msl":   r":PRMSL:mean sea level",         # mean sea level pressure
        "apcp":  r":APCP:surface:",                # accumulated precip at surface
        "tcc":   r":TCDC:entire atmosphere",       # total cloud cover (entire atmos.)
        # add more if needed, e.g. ":LCDC:low cloud layer"
    }

    rows: List[Dict] = []
    for day in _daterange(start_date, end_date):
        for cc in cycles:
            init = day.replace(hour=int(cc))
            for fh in fhours:
                try:
                    H = Herbie(init, model="gfs", product=RAW_PRODUCT, fxx=int(fh))

                    # Pull each variable set separately and sample nearest grid point
                    # (Herbie subsets by byte-range using the .idx, so this stays light.)
                    ds_cache: Dict[str, xr.Dataset] = {}
                    def _get_ds(key: str):
                        if key not in ds_cache:
                            ds_cache[key] = H.xarray(SEARCHES[key])  # <- robust subset
                        return ds_cache[key]

                    # Determine valid time from any dataset we load
                    # Prefer t2m if available, else fall back to PRMSL
                    ds_time = None
                    for k in ("t2m", "msl"):
                        try:
                            ds_time = _get_ds(k)
                            break
                        except Exception:
                            pass
                    if ds_time is None:
                        # nothing available in this file; skip
                        continue

                    valid = (pd.to_datetime(ds_time.time.values) +
                             pd.to_timedelta(int(fh), unit="h"))

                    for sid, lat, lon in airports:
                        rec = {
                            "station": sid,
                            "init_time": pd.to_datetime(ds_time.time.values),
                            "lead_hr": int(fh),
                            "valid_time": valid,
                        }

                        # sample each requested field if available
                        for key in SEARCHES:
                            try:
                                dsk = _get_ds(key)
                                varname = list(dsk.data_vars)[0]  # subset returns 1–few vars
                                rec[key] = float(
                                    dsk[varname].sel(latitude=lat, longitude=lon, method="nearest").values
                                )
                            except Exception:
                                # leave missing if this var wasn't present
                                pass

                        rows.append(rec)

                    # close opened datasets
                    for ds in ds_cache.values():
                        try:
                            ds.close()
                        except Exception:
                            pass

                except Exception:
                    # Missing cycles/leads happen; skip quietly
                    pass

    return pd.DataFrame(rows)

def _write_single_file(df, dest_file: str):
    """Write a single Parquet file (coalesce + rename Spark part)."""
    tmp_dir = dest_file + ".__tmpdir__"
    df.coalesce(1).write.mode("overwrite").parquet(tmp_dir)
    part = (glob.glob(os.path.join(tmp_dir, "part-*.parquet")) or
            glob.glob(os.path.join(tmp_dir, "*.parquet")))
    if not part:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(f"No parquet part found in {tmp_dir}")
    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
    if os.path.exists(dest_file):
        os.remove(dest_file)
    shutil.move(part[0], dest_file)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"saved: {dest_file}")

def _write_named_per_day(df, out_dir: str, filename_prefix: str):
    """Write one named Parquet per valid_date using filter + rename workflow."""
    os.makedirs(out_dir, exist_ok=True)
    df = df.persist()
    _ = df.count()
    counts = df.groupBy("valid_date").agg(F.count(F.lit(1)).alias("cnt")).collect()
    per_day = {r["valid_date"]: int(r["cnt"]) for r in counts}

    for day, cnt in sorted(per_day.items()):
        if cnt == 0:
            continue
        day_str = day.strftime("%Y-%m-%d")
        day_df = df.filter(F.col("valid_date") == F.lit(day).cast(DateType()))
        dest = os.path.join(out_dir, f"{filename_prefix}{day_str.replace('-', '_')}.parquet")
        _write_single_file(day_df, dest)
    df.unpersist()